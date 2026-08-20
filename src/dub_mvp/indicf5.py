from __future__ import annotations

import unicodedata

from pydantic import BaseModel, Field

INDICF5_CONTEXT_SECONDS = 25.0

# F5-TTS clips reference audio longer than 12 s inside
# ``preprocess_ref_audio_text``. Admitting a longer clip means the duration
# budget computed here describes audio the model never sees.
INDICF5_MAX_REFERENCE_SECONDS = 12.0

# Below roughly three seconds the measured speaking rate of the reference is
# too noisy to say anything useful about the speaker.
INDICF5_MIN_REFERENCE_SECONDS = 3.0

DURATION_POLICY_VERSION = "fixed_timeline_budget_v1"


class IndicF5ChunkingError(ValueError):
    """Text cannot be admitted as one unchanged IndicF5 batch."""


class IndicF5ReferenceError(ValueError):
    """A reference clip or transcript IndicF5 cannot be prompted with."""


class IndicF5DurationError(ValueError):
    """A target duration IndicF5 cannot be asked to fill."""


# Unicode block starts for every script IndicF5 is trained on, plus Latin.
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("latin", 0x0041, 0x005A),
    ("latin", 0x0061, 0x007A),
    ("latin", 0x00C0, 0x024F),
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("odia", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
)

# IndicF5's supported languages mapped to the script their text is written in.
_LANGUAGE_SCRIPTS: dict[str, str] = {
    "as": "bengali",
    "bn": "bengali",
    "en": "latin",
    "gu": "gujarati",
    "hi": "devanagari",
    "kn": "kannada",
    "ml": "malayalam",
    "mr": "devanagari",
    "or": "odia",
    "pa": "gurmukhi",
    "ta": "tamil",
    "te": "telugu",
}


def language_script(target_language: str) -> str:
    """Return the script IndicF5 expects text in for a target language."""

    code = target_language.strip().lower().replace("_", "-").split("-")[0]
    script = _LANGUAGE_SCRIPTS.get(code)
    if script is None:
        raise IndicF5ReferenceError(
            f"IndicF5 does not support target language {target_language!r}. "
            f"Supported: {', '.join(sorted(_LANGUAGE_SCRIPTS))}."
        )
    return script


def character_script(character: str) -> str | None:
    for name, start, end in _SCRIPT_RANGES:
        if start <= ord(character) <= end:
            return name
    return None


def dominant_script(text: str) -> str:
    """Return the script most of ``text``'s letters are written in.

    Digits, punctuation and whitespace are ignored because they are shared
    across scripts and say nothing about which one the text belongs to.
    """

    counts: dict[str, int] = {}
    for character in text:
        if not character.isalpha():
            continue
        name = character_script(character)
        if name is not None:
            counts[name] = counts.get(name, 0) + 1
    if not counts:
        raise IndicF5ReferenceError(
            "Text contains no letters from a script IndicF5 supports."
        )
    return max(counts, key=lambda name: counts[name])


def speech_units(text: str) -> int:
    """Count the written units that correspond to spoken sound.

    Combining marks are excluded so an Indic akshara counts once rather than
    once per matra, which makes the count comparable to a Latin letter count
    and, far more importantly, comparable between two texts in the same
    script. Raw UTF-8 byte length is not: Devanagari costs three bytes per
    character and Latin one, so a byte-ratio duration estimate built from a
    Latin reference over-predicts Indic output by roughly 2.6x.
    """

    return sum(
        1
        for character in text
        if character.isalpha() and unicodedata.category(character) not in {"Mn", "Mc"}
    )


def optional_dominant_script(text: str) -> str | None:
    """``dominant_script`` for telemetry, where unknown is an answer."""

    try:
        return dominant_script(text)
    except IndicF5ReferenceError:
        return None


def describe_prompt_scripts(
    reference_text: str,
    target_text: str,
) -> tuple[str | None, str | None, bool]:
    """Report the reference/target script pairing without judging it.

    This deliberately does not raise. Source-clone dubbing prompts Hindi output
    with the speaker's own English audio, so a cross-script pairing *is* the
    product. It used to be a hard failure because IndicF5 derives generated
    duration from a reference-to-target UTF-8 byte ratio, and Devanagari costs
    about 2.6 bytes per character against Latin's one — but generation is now
    pinned with ``fix_duration``, so that heuristic never runs. What remains is
    an acoustic question about cross-lingual prompting, which is measured by
    listening, not settled by a rule.
    """

    reference_script = optional_dominant_script(reference_text)
    target_script = optional_dominant_script(target_text)
    matched = (
        reference_script is not None
        and target_script is not None
        and reference_script == target_script
    )
    return reference_script, target_script, matched


def validate_reference_script(reference_text: str, *, target_language: str) -> str:
    """Report a reference transcript written in a different target script.

    Readiness-time approximation of :func:`describe_prompt_scripts` for callers
    that know the target language but do not yet have translated text. Callers
    decide severity: preflight treats a mismatch as advisory, because
    source-clone dubbing prompts across scripts by design.
    """

    expected = language_script(target_language)
    actual = dominant_script(reference_text)
    if actual != expected:
        raise IndicF5ReferenceError(
            f"Reference transcript is written in {actual} but target language "
            f"{target_language!r} uses {expected}. Cross-script prompting is "
            "expected for source-clone dubbing and is permitted with fixed "
            "duration; verify voice quality by listening."
        )
    return expected


def validate_reference_seconds(reference_seconds: float) -> float:
    """Reject reference audio outside the window IndicF5 prompts well from."""

    if reference_seconds <= 0:
        raise IndicF5ReferenceError("Reference audio duration must be positive.")
    if reference_seconds < INDICF5_MIN_REFERENCE_SECONDS:
        raise IndicF5ReferenceError(
            f"Reference audio is {reference_seconds:.2f}s; IndicF5 needs at "
            f"least {INDICF5_MIN_REFERENCE_SECONDS:g}s to calibrate a voice."
        )
    if reference_seconds > INDICF5_MAX_REFERENCE_SECONDS:
        raise IndicF5ReferenceError(
            f"Reference audio is {reference_seconds:.2f}s; IndicF5 clips "
            f"anything over {INDICF5_MAX_REFERENCE_SECONDS:g}s during "
            "preprocessing, so the clip the model hears would not be the clip "
            "this run measured. Trim the reference first."
        )
    return reference_seconds


class IndicF5DurationPlan(BaseModel):
    """The pinned conditioning window plus observations about the pairing.

    Everything past ``fix_duration_seconds`` is telemetry. It is recorded so a
    run can be analysed after the fact, and it blocks nothing: measured output
    duration and the existing duration-correction stage decide acceptance.
    """

    fix_duration_seconds: float = Field(gt=0)
    reference_seconds: float = Field(gt=0)
    target_seconds: float = Field(gt=0)
    reference_script: str | None = None
    target_script: str | None = None
    scripts_match: bool
    reference_units: int = Field(ge=0)
    target_units: int = Field(gt=0)
    reference_units_per_second: float = Field(ge=0)
    target_units_per_second: float = Field(gt=0)
    implied_rate_scale: float | None = Field(default=None, gt=0)

    def notes(self) -> list[str]:
        notes = [
            f"indicf5_duration_policy={DURATION_POLICY_VERSION}",
            f"indicf5_fix_duration_seconds={self.fix_duration_seconds:.3f}",
            f"indicf5_reference_seconds={self.reference_seconds:.3f}",
            f"indicf5_reference_script={self.reference_script or 'unknown'}",
            f"indicf5_target_script={self.target_script or 'unknown'}",
            f"indicf5_scripts_match={str(self.scripts_match).lower()}",
            (
                "indicf5_reference_units_per_second="
                f"{self.reference_units_per_second:.2f}"
            ),
            f"indicf5_target_units_per_second={self.target_units_per_second:.2f}",
        ]
        if self.implied_rate_scale is not None:
            notes.append(
                f"indicf5_implied_rate_scale={self.implied_rate_scale:.2f}"
            )
        return notes


def indicf5_duration_plan(
    *,
    reference_text: str,
    reference_seconds: float,
    target_text: str,
    target_duration_ms: int,
    context_seconds: float = INDICF5_CONTEXT_SECONDS,
) -> IndicF5DurationPlan:
    """Pin generation to the timeline budget and record what we observed.

    F5-TTS is duration conditioned: without ``fix_duration`` it predicts the
    generated length from a UTF-8 byte ratio and then fills whatever it
    predicted. Passing the timeline budget instead makes the utterance land on
    its slot by construction and removes the byte heuristic from the critical
    path. ``fix_duration`` is the whole conditioning window, so the reference
    length is included; the caller strips the reference span from the result.

    Only two things are hard failures here, because only two are real model
    constraints: an unusable reference length, and a window larger than the
    model can condition on. The speaking-rate comparison is *recorded*, never
    enforced — unit counts are not comparable across scripts, so a threshold on
    them would be an unmeasured heuristic replacing the one just removed.
    """

    if target_duration_ms <= 0:
        raise IndicF5DurationError("Target duration must be positive.")
    validate_reference_seconds(reference_seconds)

    target_seconds = target_duration_ms / 1000
    total_seconds = reference_seconds + target_seconds
    if total_seconds > context_seconds:
        raise IndicF5DurationError(
            f"Reference ({reference_seconds:.2f}s) plus target "
            f"({target_seconds:.2f}s) is {total_seconds:.2f}s, beyond the "
            f"{context_seconds:g}s IndicF5 conditioning window. Use a shorter "
            "reference or split the utterance upstream."
        )

    target_units = speech_units(target_text)
    if target_units <= 0:
        raise IndicF5DurationError("Target text contains no pronounceable units.")
    reference_units = speech_units(reference_text)

    reference_script, target_script, scripts_match = describe_prompt_scripts(
        reference_text, target_text
    )
    reference_rate = reference_units / reference_seconds
    target_rate = target_units / target_seconds
    # Only meaningful within one script: a Latin letter and a Devanagari
    # akshara are not the same unit of speech.
    implied_scale: float | None = None
    if scripts_match and reference_rate > 0:
        implied_scale = target_seconds / (target_units / reference_rate)

    return IndicF5DurationPlan(
        fix_duration_seconds=total_seconds,
        reference_seconds=reference_seconds,
        target_seconds=target_seconds,
        reference_script=reference_script,
        target_script=target_script,
        scripts_match=scripts_match,
        reference_units=reference_units,
        target_units=target_units,
        reference_units_per_second=reference_rate,
        target_units_per_second=target_rate,
        implied_rate_scale=implied_scale,
    )


def single_text_batch(text: str) -> list[str]:
    """Admit one unchanged batch or refuse to let IndicF5 split it.

    Each batch is generated independently and concatenated, so a mid-sentence
    split costs a prosody discontinuity at the seam. Fixed-duration synthesis
    admits text through the reference-plus-target conditioning-window check;
    deriving another ceiling from cross-script UTF-8 byte counts would restore
    the exact heuristic ``fix_duration`` removed.
    """
    normalized = " ".join(text.split())
    if not normalized:
        raise IndicF5ChunkingError("Target text cannot be empty.")
    return [normalized]
