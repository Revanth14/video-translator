from __future__ import annotations

import pytest

from dub_mvp.indicf5 import (
    IndicF5ChunkingError,
    IndicF5DurationError,
    IndicF5ReferenceError,
    describe_prompt_scripts,
    dominant_script,
    indicf5_duration_plan,
    indicf5_text_plan,
    single_text_batch,
    speech_units,
    validate_reference_script,
    validate_reference_seconds,
)

HINDI = "नमस्ते! यह हमारी वीडियो डबिंग प्रणाली का एक छोटा परीक्षण है।"
HINDI_REFERENCE = (
    "मेरा नाम राहुल है और मैं इस वीडियो में आपको एक नई तकनीक के बारे में "
    "बताने जा रहा हूँ। यह बहुत आसान है।"
)


def test_hindi_text_plan_normalizes_evaluated_technical_terms() -> None:
    source = (
        "पहले आप API key को environment variable में डालिए, फिर deployment "
        "script चलाइए।"
    )

    plan = indicf5_text_plan(text=source, target_language="hi-IN")

    assert plan.source_text == source
    assert plan.tts_text == (
        "पहले आप एपीआई की को एनवायरनमेंट वेरिएबल में डालिए, फिर "
        "डिप्लॉयमेंट स्क्रिप्ट चलाइए।"
    )
    assert plan.policy_version == "hindi_codeswitch_v1"
    assert plan.replacement_count == 6
    assert plan.unmapped_latin_token_count == 0
    assert plan.changed


def test_hindi_text_plan_does_not_guess_unknown_latin_pronunciations() -> None:
    plan = indicf5_text_plan(
        text="GitHub खोलिए और API key डालिए।",
        target_language="hi",
    )

    assert plan.tts_text == "GitHub खोलिए और एपीआई की डालिए।"
    assert plan.replacement_count == 2
    assert plan.unmapped_latin_token_count == 1


def test_hindi_text_plan_does_not_rewrite_urls_or_identifiers() -> None:
    source = "api.example.com खोलिए और deployment_script चलाइए।"

    plan = indicf5_text_plan(text=source, target_language="hi")

    assert plan.tts_text == source
    assert plan.replacement_count == 0
    assert plan.unmapped_latin_token_count == 2


def test_text_plan_leaves_non_hindi_text_unchanged() -> None:
    plan = indicf5_text_plan(
        text="Set the API key in an environment variable.",
        target_language="en",
    )

    assert plan.tts_text == plan.source_text
    assert plan.replacement_count == 0
    assert plan.unmapped_latin_token_count == 0


def test_preserves_the_gpu_reproduction_as_one_batch() -> None:
    assert single_text_batch(HINDI) == [HINDI]


def test_returns_one_chunk_when_text_fits() -> None:
    assert single_text_batch("नमस्ते दुनिया।") == ["नमस्ते दुनिया।"]


def test_preserves_long_cross_script_text_when_duration_fits() -> None:
    target = (
        "आज हम सीखेंगे कि किसी भी वीडियो को एक भाषा से दूसरी भाषा में कैसे "
        "बदला जाता है, और इसमें आवाज़ को कैसे बनाए रखा जाता है।"
    )

    assert single_text_batch(target) == [target]


def test_rejects_empty_target_batch() -> None:
    with pytest.raises(IndicF5ChunkingError, match="empty"):
        single_text_batch("   ")


def test_speech_units_ignores_combining_marks_so_scripts_compare() -> None:
    # The core defect: Devanagari costs ~2.6 bytes per character and Latin one,
    # so byte length cannot compare text across scripts. Unit counts can.
    assert speech_units("नमस्ते") == 4
    assert speech_units("namaste") == 7
    assert speech_units("नमस्ते, 123!") == 4


def test_dominant_script_identifies_indic_and_latin_text() -> None:
    assert dominant_script(HINDI) == "devanagari"
    assert dominant_script("Hello there") == "latin"
    assert dominant_script("வணக்கம்") == "tamil"


def test_dominant_script_rejects_text_without_supported_letters() -> None:
    with pytest.raises(IndicF5ReferenceError):
        dominant_script("123 !!! ...")


def test_prompt_scripts_are_described_not_rejected() -> None:
    # Source-clone dubbing prompts Hindi with the speaker's own English audio,
    # so a cross-script pairing is the product, not an error.
    reference, target, matched = describe_prompt_scripts(
        "My email is one at the rate gmail.com.",
        HINDI,
    )

    assert (reference, target, matched) == ("latin", "devanagari", False)


def test_prompt_scripts_report_a_match() -> None:
    assert describe_prompt_scripts(
        "This is an exact reference transcript.",
        "API deployment demo mein swagat hai.",
    ) == ("latin", "latin", True)


def test_prompt_scripts_tolerate_unidentifiable_text() -> None:
    assert describe_prompt_scripts("123 !!!", HINDI) == (
        None,
        "devanagari",
        False,
    )


def test_reference_script_mismatch_is_available_for_caller_severity() -> None:
    # Preflight catches this neutral mismatch description and reports a warning;
    # synthesis uses the actual target text and records the pairing directly.
    with pytest.raises(IndicF5ReferenceError, match="latin.*devanagari"):
        validate_reference_script(
            "My email is one at the rate gmail.com.",
            target_language="hi",
        )


def test_reference_script_accepts_matching_script() -> None:
    assert validate_reference_script(HINDI_REFERENCE, target_language="hi") == (
        "devanagari"
    )


def test_reference_script_rejects_unsupported_language() -> None:
    with pytest.raises(IndicF5ReferenceError, match="does not support"):
        validate_reference_script(HINDI_REFERENCE, target_language="fr")


@pytest.mark.parametrize("seconds", [0.0, 1.5, 12.5, 15.0])
def test_reference_duration_bounds_reject_unusable_clips(seconds: float) -> None:
    with pytest.raises(IndicF5ReferenceError):
        validate_reference_seconds(seconds)


def test_reference_duration_accepts_the_documented_window() -> None:
    assert validate_reference_seconds(9.0) == 9.0


def test_duration_plan_pins_generation_to_the_timeline_budget() -> None:
    plan = indicf5_duration_plan(
        reference_text=HINDI_REFERENCE,
        reference_seconds=9.0,
        target_text=HINDI,
        target_duration_ms=4_500,
    )

    # fix_duration is the whole conditioning window, reference included.
    assert plan.fix_duration_seconds == pytest.approx(13.5)
    assert plan.target_seconds == pytest.approx(4.5)
    assert plan.scripts_match is True
    assert plan.implied_rate_scale is not None


def test_duration_plan_records_a_bad_slot_without_blocking() -> None:
    # The 11.4 s the byte heuristic asked for, against a ~4 s line. Recorded
    # as telemetry; measured output and duration correction decide acceptance.
    plan = indicf5_duration_plan(
        reference_text=HINDI_REFERENCE,
        reference_seconds=9.0,
        target_text=HINDI,
        target_duration_ms=11_445,
    )

    assert plan.fix_duration_seconds == pytest.approx(20.445)
    assert plan.implied_rate_scale is not None
    assert plan.implied_rate_scale > 2.0


def test_duration_plan_does_not_compare_rates_across_scripts() -> None:
    # Latin letters and Devanagari aksharas are not the same unit, so the
    # scale is withheld rather than computed from incomparable counts.
    plan = indicf5_duration_plan(
        reference_text="My email is one at the rate gmail.com and it works.",
        reference_seconds=9.0,
        target_text=HINDI,
        target_duration_ms=4_500,
    )

    assert plan.scripts_match is False
    assert plan.implied_rate_scale is None
    assert plan.reference_units_per_second > 0
    assert plan.target_units_per_second > 0


def test_duration_plan_notes_carry_the_policy_version() -> None:
    notes = indicf5_duration_plan(
        reference_text=HINDI_REFERENCE,
        reference_seconds=9.0,
        target_text=HINDI,
        target_duration_ms=4_500,
    ).notes()

    assert "indicf5_duration_policy=fixed_timeline_budget_v1" in notes
    assert any(note.startswith("indicf5_scripts_match=") for note in notes)


def test_duration_plan_rejects_exceeding_the_conditioning_window() -> None:
    with pytest.raises(IndicF5DurationError, match="conditioning window"):
        indicf5_duration_plan(
            reference_text=HINDI_REFERENCE,
            reference_seconds=11.0,
            target_text=HINDI,
            target_duration_ms=20_000,
        )


def test_duration_plan_rejects_an_unusable_reference_length() -> None:
    with pytest.raises(IndicF5ReferenceError):
        indicf5_duration_plan(
            reference_text=HINDI_REFERENCE,
            reference_seconds=15.0,
            target_text=HINDI,
            target_duration_ms=4_500,
        )
