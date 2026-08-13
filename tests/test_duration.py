import json
import os
from datetime import datetime, timezone
import wave
from multiprocessing import Process
from pathlib import Path
from typing import Any

import pytest

from dub_mvp.artifacts import (
    completed_artifact_metadata,
    write_artifact_metadata,
)
from dub_mvp.duration import (
    DurationCorrectionError,
    DurationCorrector,
    DurationFitStatus,
    DurationPolicy,
    DurationRewriteResult,
    DurationStrategy,
    WavFFmpegDurationTransformer,
    build_duration_metrics,
)
from dub_mvp.localize import LocalizedSegment
from dub_mvp.manifest import RunManifest
from dub_mvp.observability import build_run_status
from dub_mvp.synthesize import SynthesisPipeline, SynthesisResult


class FailingTransformer:
    name = "failing-transformer"

    def __init__(self, message: str = "not available") -> None:
        self.message = message
        self.calls: list[str] = []

    def trim_artificial_pauses(self, *_: Any, **__: Any) -> bool:
        self.calls.append("trim")
        raise DurationCorrectionError(self.message)

    def time_stretch(self, *_: Any, **__: Any) -> None:
        self.calls.append("stretch")
        raise DurationCorrectionError(self.message)


class MustNotTransform:
    name = "must-not-transform"

    def trim_artificial_pauses(self, *_: Any, **__: Any) -> bool:
        raise AssertionError("pause trimming should not run")

    def time_stretch(self, *_: Any, **__: Any) -> None:
        raise AssertionError("time stretching should not run")


class CompactRewriter:
    provider_name = "fixture-rewriter"
    model_name = "fixture-compact"

    def __init__(self, text: str = "API संक्षेप") -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def rewrite(self, segment, **kwargs) -> DurationRewriteResult:
        self.calls.append({"segment_id": segment.segment_id, **kwargs})
        return DurationRewriteResult(
            target_text=self.text,
            meaning_preserved=True,
            required_terms_preserved=True,
            notes=["fixture semantic assertion; human review still required"],
        )


class KillDuringStretch:
    name = "kill-during-stretch"

    def trim_artificial_pauses(self, *_: Any, **__: Any) -> bool:
        raise DurationCorrectionError("no pauses")

    def time_stretch(self, *_: Any, **__: Any) -> None:
        os._exit(31)


class RecoverAfterKilledStretch:
    name = "kill-during-stretch"

    def trim_artificial_pauses(self, *_: Any, **__: Any) -> bool:
        raise DurationCorrectionError("no pauses")

    def time_stretch(
        self,
        _source: Path,
        destination: Path,
        **_: Any,
    ) -> None:
        write_wav(destination, 1000)


def write_wav(
    path: Path,
    duration_ms: int,
    *,
    leading_silence_ms: int = 0,
    trailing_silence_ms: int = 0,
    frame_rate: int = 8000,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = duration_ms * frame_rate // 1000
    leading = leading_silence_ms * frame_rate // 1000
    trailing = trailing_silence_ms * frame_rate // 1000
    voiced = max(0, frame_count - leading - trailing)
    frames = (
        b"\x00\x00" * leading
        + int(1800).to_bytes(2, "little", signed=True) * voiced
        + b"\x00\x00" * trailing
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(frame_rate)
        handle.writeframes(frames)


def segment(*, budget_ms: int = 1000) -> LocalizedSegment:
    return LocalizedSegment(
        segment_id="utt_0001",
        start_ms=200,
        end_ms=200 + budget_ms,
        duration_budget_ms=budget_ms,
        speaker_id="speaker_01",
        source_text="A compact API explanation.",
        target_text="API के बारे में एक विस्तृत व्याख्या।",
        target_text_revision=1,
        glossary_terms=["API"],
    )


def raw_artifact(
    tmp_path: Path,
    *,
    duration_ms: int,
    budget_ms: int = 1000,
    leading_silence_ms: int = 0,
    trailing_silence_ms: int = 0,
) -> tuple[LocalizedSegment, Path, Any, Path]:
    current = segment(budget_ms=budget_ms)
    speech_result = tmp_path / "speech" / "raw.result.json"
    speech_result.parent.mkdir(parents=True, exist_ok=True)
    speech_result.write_text(
        json.dumps({"utterance_id": current.segment_id, "revision": 1}),
        encoding="utf-8",
    )
    audio_path = tmp_path / "speech" / "raw.wav"
    write_wav(
        audio_path,
        duration_ms,
        leading_silence_ms=leading_silence_ms,
        trailing_silence_ms=trailing_silence_ms,
    )
    metadata = completed_artifact_metadata(
        artifact_id="raw_audio",
        kind="speech_audio",
        path=audio_path,
        root=tmp_path,
        inputs={"utterance_id": current.segment_id},
        provider="fixture-tts",
        model="fixture-model",
    )
    metadata_path = audio_path.with_name("raw.wav.meta.json")
    write_artifact_metadata(metadata_path, metadata)
    return current, speech_result, metadata, metadata_path


def run_fit(
    tmp_path: Path,
    *,
    corrector: DurationCorrector,
    duration_ms: int,
    budget_ms: int = 1000,
    leading_silence_ms: int = 0,
    trailing_silence_ms: int = 0,
    controlled_synthesizer=None,
    rewritten_synthesizer=None,
):
    current, result_path, audio, audio_metadata_path = raw_artifact(
        tmp_path,
        duration_ms=duration_ms,
        budget_ms=budget_ms,
        leading_silence_ms=leading_silence_ms,
        trailing_silence_ms=trailing_silence_ms,
    )
    return corrector.fit(
        segment=current,
        speech_result_path=result_path,
        raw_audio=audio,
        raw_audio_metadata_path=audio_metadata_path,
        run_directory=tmp_path,
        voice_id="voice_A",
        provider="fixture-tts",
        model="fixture-model",
        controlled_synthesizer=controlled_synthesizer,
        rewritten_synthesizer=rewritten_synthesizer,
    )


def test_accepts_audio_inside_both_tolerances_without_mutating_it(
    tmp_path: Path,
) -> None:
    fit, result_path, metadata_path = run_fit(
        tmp_path,
        corrector=DurationCorrector(transformer=MustNotTransform()),
        duration_ms=1080,
    )

    assert fit.status == DurationFitStatus.ACCEPTED
    assert fit.selected_strategy == DurationStrategy.ACCEPT
    assert fit.duration_error_ms == 80
    assert fit.within_primary_tolerance
    assert fit.within_hard_tolerance
    assert len(fit.attempts) == 1
    assert (tmp_path / fit.audio.path).name == "raw.wav"
    assert result_path.is_file()
    assert metadata_path.is_file()


def test_provider_controls_are_tried_before_audio_transforms(
    tmp_path: Path,
) -> None:
    calls = []

    def controlled(path: Path, rate: float, pause: float, attempt: int) -> None:
        calls.append((rate, pause, attempt))
        write_wav(path, 1100)

    fit, _, _ = run_fit(
        tmp_path,
        corrector=DurationCorrector(transformer=MustNotTransform()),
        duration_ms=1500,
        controlled_synthesizer=controlled,
    )

    assert fit.status == DurationFitStatus.CORRECTED
    assert fit.selected_strategy == DurationStrategy.PROVIDER_CONTROLS
    assert calls == [(1.1, 0.75, 1)]
    assert [item.strategy for item in fit.attempts] == [
        DurationStrategy.PROVIDER_CONTROLS
    ]


def test_real_pause_trimming_preserves_voiced_samples_and_meets_budget(
    tmp_path: Path,
) -> None:
    fit, _, _ = run_fit(
        tmp_path,
        corrector=DurationCorrector(
            transformer=WavFFmpegDurationTransformer()
        ),
        duration_ms=1600,
        leading_silence_ms=300,
        trailing_silence_ms=300,
    )

    assert fit.status == DurationFitStatus.CORRECTED
    assert fit.selected_strategy == DurationStrategy.TRIM_ARTIFICIAL_PAUSES
    assert 1070 <= fit.final_duration_ms <= 1090
    assert [item.strategy for item in fit.attempts] == [
        DurationStrategy.TRIM_ARTIFICIAL_PAUSES
    ]


def test_real_ffmpeg_time_stretch_is_mild_measured_and_pitch_preserving_path(
    tmp_path: Path,
) -> None:
    if not WavFFmpegDurationTransformer()._resolver("ffmpeg"):
        pytest.skip("ffmpeg is not installed")

    fit, _, _ = run_fit(
        tmp_path,
        corrector=DurationCorrector(
            transformer=WavFFmpegDurationTransformer(),
            policy=DurationPolicy(max_tempo_delta=0.20),
        ),
        duration_ms=1400,
    )

    assert fit.status == DurationFitStatus.CORRECTED
    assert fit.selected_strategy == DurationStrategy.MILD_TIME_STRETCH
    assert fit.within_primary_tolerance
    stretch = next(
        item
        for item in fit.attempts
        if item.strategy == DurationStrategy.MILD_TIME_STRETCH
    )
    assert stretch.tempo_factor == pytest.approx(1.2)
    assert stretch.output_duration_ms == fit.final_duration_ms


def test_rewrite_regenerates_with_recorded_revision_and_requires_review(
    tmp_path: Path,
) -> None:
    rewriter = CompactRewriter()
    regenerated = []

    def synthesize(text: str, revision: int, path: Path, attempt: int) -> None:
        regenerated.append((text, revision, attempt))
        write_wav(path, 1000)

    fit, _, _ = run_fit(
        tmp_path,
        corrector=DurationCorrector(
            transformer=FailingTransformer(),
            rewriter=rewriter,
        ),
        duration_ms=2000,
        rewritten_synthesizer=synthesize,
    )

    assert fit.status == DurationFitStatus.REVIEW_REQUIRED
    assert fit.rewritten
    assert fit.needs_human_review
    assert fit.target_text == "API संक्षेप"
    assert fit.target_text_revision == 2
    assert regenerated == [("API संक्षेप", 2, 4)]
    assert [item.strategy for item in fit.attempts] == [
        DurationStrategy.TRIM_ARTIFICIAL_PAUSES,
        DurationStrategy.MILD_TIME_STRETCH,
        DurationStrategy.COMPACT_REWRITE,
        DurationStrategy.REGENERATE_ASSIGNED_VOICE,
    ]


def test_invalid_rewrite_cannot_drop_required_term_and_violation_is_visible(
    tmp_path: Path,
) -> None:
    fit, _, _ = run_fit(
        tmp_path,
        corrector=DurationCorrector(
            transformer=FailingTransformer(),
            rewriter=CompactRewriter(text="बहुत छोटा"),
        ),
        duration_ms=2000,
        rewritten_synthesizer=lambda *_: pytest.fail(
            "invalid rewrite must not be synthesized"
        ),
    )

    assert fit.status == DurationFitStatus.UNRESOLVED
    assert not fit.within_hard_tolerance
    assert any(
        item.strategy == DurationStrategy.COMPACT_REWRITE
        and item.status.value == "failed"
        and "glossary" in (item.error or "")
        for item in fit.attempts
    )
    assert fit.attempts[-1].strategy == DurationStrategy.SURFACE_UNRESOLVED


def test_optional_tactic_failure_is_redacted_bounded_and_not_hidden(
    tmp_path: Path,
) -> None:
    corrector = DurationCorrector(
        transformer=FailingTransformer("api_key=sk-secret-duration"),
        policy=DurationPolicy(max_total_attempts=3),
    )

    fit, _, _ = run_fit(
        tmp_path,
        corrector=corrector,
        duration_ms=2000,
    )

    assert fit.status == DurationFitStatus.UNRESOLVED
    assert len(fit.attempts) == 3
    assert "sk-secret-duration" not in json.dumps(
        [item.model_dump(mode="json") for item in fit.attempts]
    )
    assert fit.attempts[-1].strategy == DurationStrategy.SURFACE_UNRESOLVED
    metrics = build_duration_metrics(
        [fit], configuration_fingerprint=corrector.configuration_fingerprint
    )
    assert metrics.unresolved_count == 1
    assert not metrics.automated_timing_gate_passed
    # Start alignment is structural; the measurable neighbour risk is zero here.
    assert metrics.next_start_overrun_count == 0
    assert metrics.maximum_next_start_overrun_ms == 0


def _die_during_fit(run_directory: str) -> None:
    root = Path(run_directory)
    run_fit(
        root,
        corrector=DurationCorrector(transformer=KillDuringStretch()),
        duration_ms=2000,
    )


def test_duration_attempt_resumes_after_real_process_death(tmp_path: Path) -> None:
    process = Process(target=_die_during_fit, args=(str(tmp_path),))
    process.start()
    process.join(timeout=10)
    assert process.exitcode == 31

    fit, _, _ = run_fit(
        tmp_path,
        corrector=DurationCorrector(transformer=RecoverAfterKilledStretch()),
        duration_ms=2000,
    )

    assert fit.status == DurationFitStatus.CORRECTED
    assert any(
        item.error_class == "interrupted" for item in fit.attempts
    )
    assert fit.attempts[-1].status.value == "completed"


class RewriteAwareSpeechProvider:
    provider_name = "fixture-tts"
    model_name = "fixture-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def synthesize(
        self,
        current,
        *,
        output_path: Path,
        voice_reference,
        target_language: str,
        revision: int,
    ) -> SynthesisResult:
        self.calls.append(
            {
                "text": current.target_text,
                "text_revision": current.target_text_revision,
                "voice": voice_reference.reference_id,
                "revision": revision,
            }
        )
        duration = 1000 if current.target_text == "API संक्षेप" else 2000
        write_wav(output_path, duration)
        return SynthesisResult(audio_path=str(output_path), duration_ms=duration)


def test_synthesis_rewrite_reuses_the_assigned_voice_and_exports_metrics(
    tmp_path: Path,
) -> None:
    current = segment()
    localized = tmp_path / "localized.json"
    localized.write_text(
        json.dumps([current.model_dump(mode="json")]), encoding="utf-8"
    )
    voices = tmp_path / "voices.json"
    voices.write_text(
        json.dumps(
            {
                "voices": [
                    {
                        "reference_id": "voice_A",
                        "path": None,
                        "consent": "stock",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    provider = RewriteAwareSpeechProvider()
    corrector = DurationCorrector(
        transformer=FailingTransformer(),
        rewriter=CompactRewriter(),
    )

    synthesized, outputs, _ = SynthesisPipeline(
        provider=provider,
        duration_corrector=corrector,
    ).run(
        localized_segments_path=localized,
        run_directory=tmp_path,
        target_language="hi",
        voice_reference_path=voices,
    )

    assert [item["voice"] for item in provider.calls] == ["voice_A", "voice_A"]
    assert synthesized[0].schema_version == 3
    assert synthesized[0].target_text == "API संक्षेप"
    assert synthesized[0].requires_timing_review
    duration_metrics = json.loads(
        Path(outputs["duration_metrics"]).read_text(encoding="utf-8")
    )
    assert duration_metrics["rewrite_count"] == 1
    assert duration_metrics["within_primary_percent"] == 100
    assert Path(outputs["duration_corrections_metadata"]).is_file()
    manifest = RunManifest(
        run_id="duration-status",
        source_path="input/source.mp4",
        source_start_ms=0,
        source_end_ms=1000,
    )
    manifest.outputs.update(outputs)
    manifest.save(tmp_path)
    status = build_run_status(tmp_path)
    duration_work = status.work_items["duration_fits"][0]
    assert duration_work.work_item_id == "utt_0001"
    assert duration_work.attempt_count == 4
    assert status.metrics["duration_timing"]["rewrite_count"] == 1
    assert status.metrics["needs_review"] == 1


def test_render_rejects_new_schema_unresolved_duration(tmp_path: Path) -> None:
    from dub_mvp.render import UnresolvedDurationError, build_render_plan
    from dub_mvp.synthesize import SynthesizedSegment

    audio = tmp_path / "unresolved.wav"
    write_wav(audio, 1500)
    unresolved = SynthesizedSegment(
        schema_version=3,
        segment_id="utt_0001",
        start_ms=0,
        end_ms=1000,
        duration_budget_ms=1000,
        speaker_id="speaker_01",
        source_text="source",
        target_text="target",
        target_text_revision=1,
        tts_audio_path=str(audio),
        tts_duration_ms=1500,
        model="fixture",
        reference_id="voice_A",
        original_tts_audio_path=str(audio),
        original_tts_duration_ms=1500,
        duration_error_ms=500,
        duration_ratio=1.5,
        duration_status="unresolved",
        duration_strategy="surface_unresolved",
        duration_correction_path="speech/duration/result.json",
        duration_correction_metadata_path="speech/duration/result.meta.json",
        requires_timing_review=True,
    )

    with pytest.raises(
        UnresolvedDurationError, match="unresolved duration violation"
    ) as captured:
        build_render_plan([unresolved], duration_ms=1000)
    assert captured.value.retryable is False


def test_render_does_not_apply_a_second_hidden_tempo_fit(tmp_path: Path) -> None:
    from dub_mvp.render import build_render_plan
    from dub_mvp.synthesize import SynthesizedSegment

    audio = tmp_path / "fitted.wav"
    write_wav(audio, 1080)
    fitted = SynthesizedSegment(
        schema_version=3,
        segment_id="utt_0001",
        start_ms=0,
        end_ms=1000,
        duration_budget_ms=1000,
        speaker_id="speaker_01",
        source_text="source",
        target_text="target",
        target_text_revision=1,
        tts_audio_path=str(audio),
        tts_duration_ms=1080,
        model="fixture",
        reference_id="voice_A",
        original_tts_audio_path=str(audio),
        original_tts_duration_ms=1080,
        duration_error_ms=80,
        duration_ratio=1.08,
        duration_status="accepted",
        duration_strategy="accept",
        duration_correction_path="speech/duration/result.json",
        duration_correction_metadata_path="speech/duration/result.meta.json",
    )

    # The 80 ms tolerance is safe here because the source timeline itself has
    # 100 ms of trailing room. Render preserves the accepted fit, but would
    # reject it if it crossed the final timeline or the next utterance.
    plan = build_render_plan([fitted], duration_ms=1100)

    assert plan.segments[0].tempo_ratio == 1.0


def test_render_rejects_tolerated_fit_that_would_overlap_next_utterance(
    tmp_path: Path,
) -> None:
    from dub_mvp.render import UnresolvedDurationError, build_render_plan
    from dub_mvp.synthesize import SynthesizedSegment

    audio = tmp_path / "fitted.wav"
    write_wav(audio, 1080)

    def fitted(identifier: str, start: int, end: int) -> SynthesizedSegment:
        return SynthesizedSegment(
            schema_version=3,
            segment_id=identifier,
            start_ms=start,
            end_ms=end,
            duration_budget_ms=end - start,
            speaker_id="speaker_01",
            source_text="source",
            target_text="target",
            target_text_revision=1,
            tts_audio_path=str(audio),
            tts_duration_ms=1080,
            model="fixture",
            reference_id="voice_A",
            original_tts_audio_path=str(audio),
            original_tts_duration_ms=1080,
            duration_error_ms=1080 - (end - start),
            duration_ratio=1080 / (end - start),
            duration_status="accepted",
            duration_strategy="accept",
            duration_correction_path=f"speech/duration/{identifier}.json",
            duration_correction_metadata_path=(
                f"speech/duration/{identifier}.meta.json"
            ),
        )

    with pytest.raises(UnresolvedDurationError, match="would overlap"):
        build_render_plan(
            [fitted("utt_0001", 0, 1000), fitted("utt_0002", 1050, 2050)],
            duration_ms=2200,
        )


def _noop_attempt(utterance_id: str) -> Any:
    from dub_mvp.duration import DurationAttempt, DurationAttemptStatus

    moment = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return DurationAttempt.model_construct(
        attempt_number=1,
        utterance_id=utterance_id,
        strategy=DurationStrategy.ACCEPT,
        status=DurationAttemptStatus.COMPLETED,
        started_at=moment,
        completed_at=moment,
        latency_seconds=0.0,
        input_duration_ms=1000,
        output_duration_ms=1000,
    )


def _fit_artifact(
    utterance_id: str,
    *,
    start_ms: int,
    budget_ms: int,
    final_duration_ms: int,
) -> Any:
    from dub_mvp.duration import DurationFitArtifact

    return DurationFitArtifact.model_construct(
        utterance_id=utterance_id,
        start_ms=start_ms,
        end_ms=start_ms + budget_ms,
        available_duration_ms=budget_ms,
        original_duration_ms=final_duration_ms,
        final_duration_ms=final_duration_ms,
        duration_error_ms=final_duration_ms - budget_ms,
        duration_ratio=final_duration_ms / budget_ms,
        primary_tolerance_ms=250,
        within_primary_tolerance=False,
        within_hard_tolerance=True,
        status=DurationFitStatus.CORRECTED,
        selected_strategy=DurationStrategy.MILD_TIME_STRETCH,
        target_text="t",
        target_text_revision=1,
        voice_id="voice_A",
        provider="fixture",
        model="fixture",
        configuration_fingerprint="f" * 64,
        source_speech_result_path="speech/r.json",
        attempts_path="speech/a.json",
        attempts=[_noop_attempt(utterance_id)],
        provider_calls=0,
        audio=None,
        audio_metadata_path="speech/r.wav.meta.json",
    )


def test_metrics_measure_audio_running_into_the_next_utterance() -> None:
    # Each utterance is 18% long: inside the hard tolerance, yet every one
    # collides with its neighbour. Tolerance alone cannot see this.
    artifacts = [
        _fit_artifact("utt_0001", start_ms=0, budget_ms=1000, final_duration_ms=1180),
        _fit_artifact("utt_0002", start_ms=1000, budget_ms=1000, final_duration_ms=1180),
        _fit_artifact("utt_0003", start_ms=2000, budget_ms=1000, final_duration_ms=1180),
    ]

    metrics = build_duration_metrics(artifacts, configuration_fingerprint="c" * 64)

    assert metrics.within_hard_count == 3
    assert metrics.next_start_overrun_count == 2
    assert metrics.maximum_next_start_overrun_ms == 180
    # A run that collides on every cue must not report a passing timing gate.
    assert not metrics.automated_timing_gate_passed


def test_metrics_report_no_overrun_when_utterances_fit_their_gaps() -> None:
    artifacts = [
        _fit_artifact("utt_0001", start_ms=0, budget_ms=1000, final_duration_ms=980),
        _fit_artifact("utt_0002", start_ms=1000, budget_ms=1000, final_duration_ms=1010),
    ]

    metrics = build_duration_metrics(artifacts, configuration_fingerprint="c" * 64)

    assert metrics.next_start_overrun_count == 0
    assert metrics.maximum_next_start_overrun_ms == 0
