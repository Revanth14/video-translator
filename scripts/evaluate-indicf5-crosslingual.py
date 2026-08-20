"""Phase 1 quality gate: does IndicF5 clone a voice across languages?

Run from the repository venv on the GPU box:

    VIDEO_TRANSLATOR_INDICF5_PYTHON=/opt/video-translator/indicf5-venv/bin/python \\
        python scripts/evaluate-indicf5-crosslingual.py

This is the decision gate for source-clone dubbing. The reference is the
speaker's own **English** audio; the targets are Hindi. Cross-script prompting
is no longer blocked — generation is pinned with ``fix_duration``, so the UTF-8
byte-ratio estimate that once mistimed everything never runs. What remains
unproven is acoustic, and no automated check can settle it: the duration numbers
below are necessary but not sufficient. **You have to listen.**

It drives ``IndicF5Provider`` rather than reimplementing inference. The earlier
smoke script kept its own copy of the inference path, and that divergence is
what hid the duration defect: what was being tested was not what shipped.

Outputs land in ``voices/samples/phase1/`` with a scoring sheet to fill in.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import wave
from pathlib import Path

from dub_mvp.indicf5 import (
    INDICF5_MAX_REFERENCE_SECONDS,
    IndicF5ReferenceError,
    indicf5_text_plan,
    validate_reference_seconds,
)
from dub_mvp.localize import LocalizedSegment
from dub_mvp.synthesize import IndicF5Provider, VoiceReference

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION = REPOSITORY_ROOT / "evaluation"
SOURCE_REFERENCE = EVALUATION / "voices" / "audio" / "hitesh-reference-001.wav"
OUTPUT_DIRECTORY = EVALUATION / "voices" / "samples" / "phase1"
TRIMMED_REFERENCE = OUTPUT_DIRECTORY / "reference-english-7350ms.wav"

# The source clip is 15 s, over the 12 s ceiling F5-TTS enforces by clipping
# during preprocessing. Trim to a window the model will actually hear whole.
REFERENCE_START_SECONDS = 0.0
REFERENCE_SECONDS = 7.35

# MUST be the verbatim transcript of the TRIMMED window, not the whole clip.
# A transcript describing audio the model was not given degrades the prompt.
# Verify reference-english-7350ms.wav by listening before trusting results.
REFERENCE_TEXT = (
    "My email is one at the rate gmail.com. And let's just say I will set up "
    "a password."
)

# Representative set: short, medium, long, punctuation-heavy, and embedded
# English technical terms. Budgets are the timeline slots generation is pinned
# to, chosen to match an unhurried natural delivery.
CASES: list[tuple[str, str, int]] = [
    ("short", "नमस्ते दोस्तों।", 1_600),
    (
        "medium",
        "यह हमारी वीडियो डबिंग प्रणाली का एक छोटा परीक्षण है।",
        4_200,
    ),
    (
        "long",
        "आज हम सीखेंगे कि किसी भी वीडियो को एक भाषा से दूसरी भाषा में कैसे "
        "बदला जाता है, और इसमें आवाज़ को कैसे बनाए रखा जाता है।",
        11_000,
    ),
    (
        "punctuation",
        "रुको! क्या तुमने यह देखा? हाँ, बिल्कुल — यही तो कमाल है।",
        5_000,
    ),
    (
        "technical",
        "पहले आप API key को environment variable में डालिए, फिर deployment "
        "script चलाइए।",
        6_500,
    ),
]

RUBRIC = {
    "hindi_intelligibility": "1-5, must be >= 4 on EVERY sample",
    "voice_similarity": "1-5, median across samples must be >= 3.5",
    "severe_filler_or_hallucination": "true/false, must be false everywhere",
    "clipped_words": "true/false, must be false everywhere",
}

# Matches DurationPolicy.hard_ratio_tolerance in src/dub_mvp/duration.py.
HARD_RATIO_TOLERANCE = 0.20


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def trim_reference() -> None:
    """Cut a model-legal reference window at IndicF5's native 24 kHz."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to trim the reference clip.")
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{REFERENCE_START_SECONDS:.3f}",
            "-i", str(SOURCE_REFERENCE),
            "-t", f"{REFERENCE_SECONDS:.3f}",
            "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le",
            str(TRIMMED_REFERENCE),
        ],
        check=True,
    )


def evaluate_case(
    provider: IndicF5Provider,
    voice_reference: VoiceReference,
    case: tuple[str, str, int],
) -> dict[str, object]:
    name, target_text, budget_ms = case
    output = OUTPUT_DIRECTORY / f"phase1-{name}.wav"
    text_plan = indicf5_text_plan(
        text=target_text,
        target_language="hi",
    )
    segment = LocalizedSegment(
        segment_id=f"phase1_{name}",
        start_ms=0,
        end_ms=budget_ms,
        duration_budget_ms=budget_ms,
        source_text=f"Phase 1 {name} case.",
        target_text=target_text,
    )
    started = time.monotonic()
    result = provider.synthesize(
        segment,
        output_path=output,
        voice_reference=voice_reference,
        target_language="hi",
        revision=1,
    )
    elapsed = time.monotonic() - started
    measured_ms = round(wav_seconds(output) * 1000)
    ratio = measured_ms / budget_ms
    evaluated: dict[str, object] = {
        "case": name,
        "path": str(output.relative_to(EVALUATION)),
        "target_text": target_text,
        "tts_text": text_plan.tts_text,
        "text_normalization_policy": text_plan.policy_version,
        "target_duration_ms": budget_ms,
        "measured_duration_ms": measured_ms,
        "duration_ratio": round(ratio, 4),
        "within_hard_gate": abs(ratio - 1.0) <= HARD_RATIO_TOLERANCE,
        "elapsed_seconds": round(elapsed, 2),
        "notes": result.notes,
        "scores": dict.fromkeys(RUBRIC),
    }
    print(
        f"{name:12s} target {budget_ms:6d}ms  measured {measured_ms:6d}ms  "
        f"ratio {ratio:.2f}  {elapsed:.1f}s"
    )
    return evaluated


def main(*, case_name: str | None = None) -> int:
    if not SOURCE_REFERENCE.is_file():
        raise SystemExit(f"Missing reference audio: {SOURCE_REFERENCE}")
    if not os.environ.get("VIDEO_TRANSLATOR_INDICF5_PYTHON"):
        raise SystemExit("Set VIDEO_TRANSLATOR_INDICF5_PYTHON first.")

    trim_reference()
    measured_reference = wav_seconds(TRIMMED_REFERENCE)
    try:
        validate_reference_seconds(measured_reference)
    except IndicF5ReferenceError as error:
        raise SystemExit(
            f"Trimmed reference is unusable: {error}\n"
            f"Adjust REFERENCE_SECONDS (ceiling {INDICF5_MAX_REFERENCE_SECONDS:g}s)."
        ) from error

    voice_reference = VoiceReference(
        reference_id="hitesh-english-source-clone",
        path=str(TRIMMED_REFERENCE),
        reference_text=REFERENCE_TEXT,
        consent="authorized evaluation reference",
    )
    provider = IndicF5Provider()

    selected_cases = (
        CASES
        if case_name is None
        else [item for item in CASES if item[0] == case_name]
    )
    if not selected_cases:
        raise SystemExit(f"Unknown Phase 1 case: {case_name}")

    provider.start_stage()
    try:
        results = [
            evaluate_case(provider, voice_reference, case)
            for case in selected_cases
        ]
    finally:
        provider.close_stage()

    sheet = {
        "schema_version": 2,
        "phase": "1-cross-lingual-quality-gate",
        "reference_audio": str(TRIMMED_REFERENCE.relative_to(EVALUATION)),
        "reference_seconds": round(measured_reference, 3),
        "reference_text": REFERENCE_TEXT,
        "reference_language": "en",
        "target_language": "hi",
        "rubric": RUBRIC,
        "results": results,
    }
    sheet_name = (
        "phase1-scoring-sheet.json"
        if case_name is None
        else f"phase1-{case_name}-scoring-sheet.json"
    )
    sheet_path = OUTPUT_DIRECTORY / sheet_name
    sheet_path.write_text(
        json.dumps(sheet, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    failed = [item["case"] for item in results if not item["within_hard_gate"]]
    print(f"\nScoring sheet: {sheet_path}")
    if failed:
        print(f"Duration gate FAILED for: {', '.join(failed)}")
    else:
        print("Duration gate passed for every case.")
    print(
        "\nDuration is necessary, not sufficient. Listen to every file and fill "
        "in `scores` before deciding whether to continue past Phase 1."
    )
    # Human scores decide the gate, so this never reports overall success.
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=[name for name, _, _ in CASES],
        help="Generate one case without replacing the complete scoring sheet.",
    )
    arguments = parser.parse_args()
    raise SystemExit(main(case_name=arguments.case))
