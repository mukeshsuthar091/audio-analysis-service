"""Explainable signal-quality measurements and classification."""

from dataclasses import dataclass

import librosa
import numpy as np

from app.audio.decoder import FloatWaveform, WaveformInfo
from app.audio.vad import VADResult
from app.core.config import Settings
from app.schemas.response import AudioQuality


@dataclass(frozen=True, slots=True)
class QualityMetrics:
    total_duration_seconds: float
    speech_duration_seconds: float
    speech_ratio: float
    rms_energy: float
    peak_amplitude: float
    clipping_ratio: float
    silence_ratio: float
    approximate_snr_db: float | None


def analyze_quality(
    waveform: FloatWaveform,
    waveform_info: WaveformInfo,
    vad_result: VADResult,
    settings: Settings,
) -> tuple[AudioQuality, QualityMetrics]:
    """Measure quality and return a conservative policy classification."""

    frame_length = min(2048, waveform.size)
    hop_length = min(512, max(1, frame_length // 4))
    frame_rms = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
        center=False,
    )[0]
    silence_ratio = (
        float(np.mean(frame_rms < settings.near_silence_rms))
        if frame_rms.size
        else 1.0
    )

    mask = np.zeros(waveform.size, dtype=bool)
    for segment in vad_result.segments:
        mask[segment.start : segment.end] = True
    approximate_snr_db = approximate_snr(waveform, mask)

    metrics = QualityMetrics(
        total_duration_seconds=waveform_info.duration_seconds,
        speech_duration_seconds=vad_result.speech_duration_seconds,
        speech_ratio=vad_result.speech_ratio,
        rms_energy=waveform_info.rms_energy,
        peak_amplitude=waveform_info.peak_amplitude,
        clipping_ratio=waveform_info.clipping_ratio,
        silence_ratio=silence_ratio,
        approximate_snr_db=approximate_snr_db,
    )
    return classify_quality(metrics, settings), metrics


def approximate_snr(waveform: FloatWaveform, speech_mask: np.ndarray) -> float | None:
    """Estimate SNR from VAD speech RMS versus non-speech RMS.

    This is a rough operational indicator, not a calibrated noise measurement;
    background noise present during speech may not be represented accurately.
    """

    if not speech_mask.any() or speech_mask.all():
        return None
    speech = waveform[speech_mask]
    noise = waveform[~speech_mask]
    speech_rms = float(np.sqrt(np.mean(np.square(speech, dtype=np.float64))))
    noise_rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float64))))
    epsilon = 1e-8
    return float(20.0 * np.log10((speech_rms + epsilon) / (noise_rms + epsilon)))


def classify_quality(metrics: QualityMetrics, settings: Settings) -> AudioQuality:
    """Pure, precedence-ordered quality policy."""

    severe_noise = (
        metrics.approximate_snr_db is not None
        and metrics.approximate_snr_db < settings.severe_snr_db
    )
    if (
        metrics.speech_duration_seconds < settings.insufficient_speech_seconds
        or metrics.speech_ratio < settings.insufficient_speech_ratio
        or metrics.rms_energy < settings.near_silence_rms
        or metrics.clipping_ratio >= settings.severe_clipping_ratio
        or severe_noise
    ):
        return AudioQuality.INSUFFICIENT

    acceptable_noise = (
        metrics.approximate_snr_db is None
        or metrics.approximate_snr_db >= settings.good_snr_db
    )
    if (
        metrics.speech_duration_seconds >= settings.good_speech_seconds
        and metrics.speech_ratio >= settings.good_speech_ratio
        and metrics.rms_energy >= settings.good_rms
        and metrics.clipping_ratio < settings.good_clipping_ratio
        and acceptable_noise
    ):
        return AudioQuality.GOOD
    return AudioQuality.DEGRADED


def warmup_quality_analysis(settings: Settings) -> None:
    """Trigger librosa lazy imports/compilation before readiness is reported."""

    waveform = np.zeros(settings.sample_rate, dtype=np.float32)
    waveform_info = WaveformInfo(
        duration_seconds=1.0,
        peak_amplitude=0.0,
        rms_energy=0.0,
        clipping_ratio=0.0,
    )
    vad_result = VADResult(
        total_duration_seconds=1.0,
        speech_duration_seconds=0.0,
        speech_ratio=0.0,
        segments=[],
        speech_waveform=np.empty(0, dtype=np.float32),
    )
    try:
        analyze_quality(waveform, waveform_info, vad_result, settings)
    finally:
        waveform.fill(0.0)
        vad_result.speech_waveform.fill(0.0)
