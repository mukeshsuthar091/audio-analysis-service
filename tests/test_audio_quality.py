"""Audio-quality rule tests."""

from app.audio.quality import QualityMetrics, classify_quality
from app.core.config import Settings
from app.schemas.response import AudioQuality


def metrics(**overrides: float | None) -> QualityMetrics:
    values: dict[str, float | None] = {
        "total_duration_seconds": 4.0,
        "speech_duration_seconds": 3.0,
        "speech_ratio": 0.75,
        "rms_energy": 0.08,
        "peak_amplitude": 0.7,
        "clipping_ratio": 0.0,
        "silence_ratio": 0.20,
        "approximate_snr_db": 20.0,
    }
    values.update(overrides)
    return QualityMetrics(**values)  # type: ignore[arg-type]


def test_clear_speech_is_good(settings: Settings) -> None:
    assert classify_quality(metrics(), settings) is AudioQuality.GOOD


def test_short_noisy_speech_is_degraded(settings: Settings) -> None:
    result = classify_quality(
        metrics(
            speech_duration_seconds=1.8,
            speech_ratio=0.45,
            approximate_snr_db=8.0,
        ),
        settings,
    )
    assert result is AudioQuality.DEGRADED


def test_almost_no_speech_is_insufficient(settings: Settings) -> None:
    assert (
        classify_quality(
            metrics(speech_duration_seconds=0.4, speech_ratio=0.10), settings
        )
        is AudioQuality.INSUFFICIENT
    )


def test_severe_clipping_is_insufficient(settings: Settings) -> None:
    assert (
        classify_quality(metrics(clipping_ratio=0.07), settings)
        is AudioQuality.INSUFFICIENT
    )


def test_low_volume_is_degraded(settings: Settings) -> None:
    assert (
        classify_quality(metrics(rms_energy=0.01), settings)
        is AudioQuality.DEGRADED
    )

