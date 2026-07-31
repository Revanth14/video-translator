from __future__ import annotations

from decimal import Decimal, InvalidOperation


def parse_timecode_ms(value: str) -> int:
    """Parse seconds or HH:MM:SS(.mmm) into non-negative milliseconds."""
    raw = value.strip()
    if not raw:
        raise ValueError("Timecode cannot be empty.")

    parts = raw.split(":")
    if len(parts) == 1:
        hours = Decimal(0)
        minutes = Decimal(0)
        seconds = _decimal(parts[0], value)
    elif len(parts) == 3:
        hours = _decimal(parts[0], value)
        minutes = _decimal(parts[1], value)
        seconds = _decimal(parts[2], value)
        if hours != hours.to_integral_value() or hours < 0:
            raise ValueError(f"Invalid hour value in timecode: {value}")
        if minutes != minutes.to_integral_value() or not 0 <= minutes < 60:
            raise ValueError(f"Invalid minute value in timecode: {value}")
        if not 0 <= seconds < 60:
            raise ValueError(f"Invalid second value in timecode: {value}")
    else:
        raise ValueError(
            f"Expected seconds or HH:MM:SS(.mmm), received: {value}"
        )

    total_seconds = (hours * 3600) + (minutes * 60) + seconds
    if total_seconds < 0:
        raise ValueError("Timecode cannot be negative.")
    return int(total_seconds * 1000)


def format_timecode_seconds(milliseconds: int) -> str:
    if milliseconds < 0:
        raise ValueError("Timecode cannot be negative.")
    return f"{milliseconds / 1000:.3f}"


def _decimal(raw: str, original: str) -> Decimal:
    try:
        return Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(f"Invalid timecode: {original}") from error
