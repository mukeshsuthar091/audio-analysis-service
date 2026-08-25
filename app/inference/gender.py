"""Pure gender-presentation post-processing."""

import math
from collections.abc import Mapping

from app.core.config import Settings
from app.schemas.response import AudioQuality, GenderPrediction, GenderResult


def quality_multiplier(quality: AudioQuality, settings: Settings) -> float:
    return {
        AudioQuality.GOOD: settings.quality_multiplier_good,
        AudioQuality.DEGRADED: settings.quality_multiplier_degraded,
        AudioQuality.INSUFFICIENT: settings.quality_multiplier_insufficient,
    }[quality]


def clamp_confidence(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, value))


def process_gender(
    probabilities: Mapping[str, float],
    quality: AudioQuality,
    settings: Settings,
) -> GenderResult:
    """Map child/female/male probabilities to the conservative public contract."""

    if quality is AudioQuality.INSUFFICIENT:
        return GenderResult(prediction=GenderPrediction.UNKNOWN, confidence=0.0)

    labels = ("child", "female", "male")
    values = {
        label: clamp_confidence(float(probabilities.get(label, float("nan"))))
        for label in labels
    }
    ranked = sorted(values.items(), key=lambda item: item[1], reverse=True)
    top_label, raw_confidence = ranked[0]
    margin = raw_confidence - ranked[1][1]
    adjusted = clamp_confidence(raw_confidence * quality_multiplier(quality, settings))
    public_confidence = round(adjusted, 2)

    if (
        top_label == "child"
        or adjusted < settings.gender_confidence_threshold
        or margin < settings.gender_min_margin
    ):
        return GenderResult(
            prediction=GenderPrediction.UNKNOWN, confidence=public_confidence
        )

    prediction = (
        GenderPrediction.FEMALE if top_label == "female" else GenderPrediction.MALE
    )
    return GenderResult(prediction=prediction, confidence=public_confidence)

