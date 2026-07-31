import pytest

from dub_mvp.timecode import format_timecode_seconds, parse_timecode_ms


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("0", 0),
        ("12.345", 12345),
        ("00:01:30", 90000),
        ("01:02:03.250", 3723250),
    ],
)
def test_parse_timecode_ms(value: str, expected: int) -> None:
    assert parse_timecode_ms(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "-1", "00:60:00", "00:00:60", "1:20", "nonsense"],
)
def test_parse_timecode_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        parse_timecode_ms(value)


def test_format_timecode_seconds() -> None:
    assert format_timecode_seconds(62345) == "62.345"
