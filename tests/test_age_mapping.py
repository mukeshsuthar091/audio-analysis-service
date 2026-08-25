"""Age mapping and confidence-policy tests."""

import pytest

from app.core.config import Settings
from app.inference.age import map_age_to_bracket, process_age
from app.schemas.response import AgeBracket, AudioQuality


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (17, AgeBracket.UNKNOWN),
        (18, AgeBracket.AGE_18_30),
        (30, AgeBracket.AGE_18_30),
        (31, AgeBracket.AGE_31_45),
        (45, AgeBracket.AGE_31_45),
        (46, AgeBracket.AGE_46_60),
        (60, AgeBracket.AGE_46_60),
        (61, AgeBracket.AGE_60_PLUS),
    ],
)
def test_age_mapping_boundaries(age: float, expected: AgeBracket) -> None:
    assert map_age_to_bracket(age) is expected


def test_age_confidence_accepts_stable_good_result(settings: Settings) -> None:
    result = process_age(0.37, AudioQuality.GOOD, 3.0, settings)
    assert result.prediction is AgeBracket.AGE_31_45
    assert result.confidence == 1.0


def test_age_boundary_uncertainty_returns_unknown(settings: Settings) -> None:
    result = process_age(0.30, AudioQuality.GOOD, 3.0, settings)
    assert result.prediction is AgeBracket.UNKNOWN
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan")])
def test_invalid_normalized_age_returns_unknown(value: float, settings: Settings) -> None:
    result = process_age(value, AudioQuality.GOOD, 3.0, settings)
    assert result.prediction is AgeBracket.UNKNOWN
    assert result.confidence == 0.0


def test_insufficient_age_is_zero_confidence(settings: Settings) -> None:
    result = process_age(0.37, AudioQuality.INSUFFICIENT, 3.0, settings)
    assert result.prediction is AgeBracket.UNKNOWN
    assert result.confidence == 0.0

