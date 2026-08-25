"""Pure language-label and confidence-policy tests."""

import math

import pytest

from app.inference.language import (
    RawLanguageOutput,
    parse_language_label,
    process_language,
)
from app.schemas.response import AudioQuality
from tests.conftest import make_settings


def raw(
    label: str = "en: English",
    top: float = 0.90,
    runner_up: float = 0.05,
) -> RawLanguageOutput:
    return RawLanguageOutput(label, top, runner_up)


def test_high_confidence_good_audio_returns_language() -> None:
    result = process_language(raw(), AudioQuality.GOOD, 3.0, make_settings())

    assert result.code == "en"
    assert result.name == "English"
    assert result.confidence == 0.90


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("iw: Hebrew", ("he", "Hebrew")),
        ("jw: Javanese", ("jv", "Javanese")),
        ("ceb: Cebuano", ("ceb", "Cebuano")),
    ],
)
def test_parses_and_normalizes_language_labels(
    label: str, expected: tuple[str, str]
) -> None:
    assert parse_language_label(label) == expected


@pytest.mark.parametrize(
    "label",
    ["", "English", "e: English", "english: English", "en:", "en: Bad\nName"],
)
def test_rejects_malformed_labels(label: str) -> None:
    assert parse_language_label(label) is None


def test_degraded_quality_lowers_confidence_below_threshold() -> None:
    result = process_language(raw(top=0.80), AudioQuality.DEGRADED, 3.0, make_settings())

    assert result.code == "unknown"
    assert result.confidence == 0.0


def test_low_raw_confidence_is_unknown() -> None:
    result = process_language(raw(top=0.60), AudioQuality.GOOD, 3.0, make_settings())

    assert result.code == "unknown"


def test_close_probabilities_are_unknown() -> None:
    result = process_language(
        raw(top=0.70, runner_up=0.60), AudioQuality.GOOD, 3.0, make_settings()
    )

    assert result.code == "unknown"


@pytest.mark.parametrize(
    ("quality", "speech_seconds"),
    [(AudioQuality.INSUFFICIENT, 3.0), (AudioQuality.GOOD, 1.49)],
)
def test_insufficient_evidence_is_unknown(
    quality: AudioQuality, speech_seconds: float
) -> None:
    result = process_language(raw(), quality, speech_seconds, make_settings())

    assert result.code == "unknown"
    assert result.confidence == 0.0


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -0.1, 1.1],
)
def test_invalid_probability_is_unknown(value: float) -> None:
    result = process_language(raw(top=value), AudioQuality.GOOD, 3.0, make_settings())

    assert result.code == "unknown"
    assert math.isfinite(result.confidence)


def test_confidence_is_clamped_and_rounded() -> None:
    settings = make_settings(quality_multiplier_good=1.0)
    result = process_language(
        raw(top=0.999, runner_up=0.001), AudioQuality.GOOD, 3.0, settings
    )

    assert result.confidence == 1.0
