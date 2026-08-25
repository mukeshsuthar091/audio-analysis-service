"""Gender post-processing tests."""

from app.core.config import Settings
from app.inference.gender import process_gender
from app.schemas.response import AudioQuality, GenderPrediction


def test_high_male_good_audio(settings: Settings) -> None:
    result = process_gender(
        {"child": 0.02, "female": 0.08, "male": 0.90},
        AudioQuality.GOOD,
        settings,
    )
    assert result.prediction is GenderPrediction.MALE
    assert result.confidence == 0.9


def test_high_female_good_audio(settings: Settings) -> None:
    result = process_gender(
        {"child": 0.02, "female": 0.92, "male": 0.06},
        AudioQuality.GOOD,
        settings,
    )
    assert result.prediction is GenderPrediction.FEMALE


def test_child_highest_is_unknown(settings: Settings) -> None:
    result = process_gender(
        {"child": 0.90, "female": 0.05, "male": 0.05},
        AudioQuality.GOOD,
        settings,
    )
    assert result.prediction is GenderPrediction.UNKNOWN


def test_below_threshold_is_unknown(settings: Settings) -> None:
    result = process_gender(
        {"child": 0.10, "female": 0.30, "male": 0.60},
        AudioQuality.GOOD,
        settings,
    )
    assert result.prediction is GenderPrediction.UNKNOWN
    assert result.confidence == 0.6


def test_close_probabilities_are_unknown(settings: Settings) -> None:
    result = process_gender(
        {"child": 0.01, "female": 0.48, "male": 0.51},
        AudioQuality.GOOD,
        settings,
    )
    assert result.prediction is GenderPrediction.UNKNOWN


def test_degraded_quality_lowers_confidence(settings: Settings) -> None:
    good = process_gender(
        {"child": 0.01, "female": 0.04, "male": 0.95},
        AudioQuality.GOOD,
        settings,
    )
    degraded = process_gender(
        {"child": 0.01, "female": 0.04, "male": 0.95},
        AudioQuality.DEGRADED,
        settings,
    )
    assert degraded.confidence < good.confidence
    assert degraded.prediction is GenderPrediction.MALE


def test_insufficient_quality_is_unknown_zero(settings: Settings) -> None:
    result = process_gender(
        {"child": 0.0, "female": 0.0, "male": 1.0},
        AudioQuality.INSUFFICIENT,
        settings,
    )
    assert result.prediction is GenderPrediction.UNKNOWN
    assert result.confidence == 0.0


def test_confidence_is_clamped(settings: Settings) -> None:
    result = process_gender(
        {"child": -1.0, "female": 0.0, "male": 4.0},
        AudioQuality.GOOD,
        settings,
    )
    assert result.prediction is GenderPrediction.MALE
    assert result.confidence == 1.0

