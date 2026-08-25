"""Pure age-bracket mapping and heuristic confidence calculation."""

import math

from app.core.config import Settings
from app.inference.gender import clamp_confidence, quality_multiplier
from app.schemas.response import AgeBracket, AgeBracketResult, AudioQuality

BOUNDARIES = (18.0, 30.0, 45.0, 60.0)


def map_age_to_bracket(age_years: float) -> AgeBracket:
    """Map an estimated age to the required public bracket."""

    if not math.isfinite(age_years) or age_years < 18:
        return AgeBracket.UNKNOWN
    if age_years <= 30:
        return AgeBracket.AGE_18_30
    if age_years <= 45:
        return AgeBracket.AGE_31_45
    if age_years <= 60:
        return AgeBracket.AGE_46_60
    return AgeBracket.AGE_60_PLUS


def process_age(
    normalized_age: float,
    quality: AudioQuality,
    speech_duration_seconds: float,
    settings: Settings,
) -> AgeBracketResult:
    """Return a bracket and explainable, explicitly heuristic confidence."""

    if quality is AudioQuality.INSUFFICIENT:
        return AgeBracketResult(prediction=AgeBracket.UNKNOWN, confidence=0.0)
    if not math.isfinite(normalized_age) or not 0.0 <= normalized_age <= 1.0:
        return AgeBracketResult(prediction=AgeBracket.UNKNOWN, confidence=0.0)

    age_years = normalized_age * 100.0
    bracket = map_age_to_bracket(age_years)
    if bracket is AgeBracket.UNKNOWN:
        return AgeBracketResult(prediction=AgeBracket.UNKNOWN, confidence=0.0)

    distance = min(abs(age_years - boundary) for boundary in BOUNDARIES)
    duration_factor = clamp_confidence(
        speech_duration_seconds / settings.good_speech_seconds
    )
    boundary_factor = clamp_confidence(distance / settings.age_boundary_scale_years)
    confidence = clamp_confidence(
        quality_multiplier(quality, settings)
        * (0.55 * duration_factor + 0.45 * boundary_factor)
    )
    public_confidence = round(confidence, 2)

    if (
        distance < settings.age_boundary_exclusion_years
        or confidence < settings.age_confidence_threshold
    ):
        return AgeBracketResult(
            prediction=AgeBracket.UNKNOWN, confidence=public_confidence
        )
    return AgeBracketResult(prediction=bracket, confidence=public_confidence)

