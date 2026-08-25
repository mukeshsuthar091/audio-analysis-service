"""Silero VAD loading, timestamp processing, and speech extraction."""

import threading
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from silero_vad import get_speech_timestamps, load_silero_vad

from app.core.config import Settings

FloatWaveform = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class SpeechSegment:
    start: int
    end: int


@dataclass(slots=True)
class VADResult:
    total_duration_seconds: float
    speech_duration_seconds: float
    speech_ratio: float
    segments: list[SpeechSegment]
    speech_waveform: FloatWaveform


class VoiceActivityDetector(Protocol):
    def detect(self, waveform: FloatWaveform) -> VADResult: ...


class SileroVoiceActivityDetector:
    """Thread-safe wrapper around the stateful Silero JIT model."""

    def __init__(self, model: Any, settings: Settings) -> None:
        self._model = model
        self._settings = settings
        self._lock = threading.Lock()

    @classmethod
    def load(cls, settings: Settings) -> "SileroVoiceActivityDetector":
        model = load_silero_vad(onnx=False)
        return cls(model=model, settings=settings)

    def warmup(self) -> None:
        dummy = np.zeros(self._settings.sample_rate, dtype=np.float32)
        result = self.detect(dummy)
        result.speech_waveform.fill(0.0)
        dummy.fill(0.0)

    def detect(self, waveform: FloatWaveform) -> VADResult:
        tensor = torch.from_numpy(waveform)
        with self._lock:
            raw_segments = get_speech_timestamps(
                tensor,
                self._model,
                threshold=self._settings.vad_threshold,
                sampling_rate=self._settings.sample_rate,
                min_speech_duration_ms=self._settings.vad_min_speech_ms,
                min_silence_duration_ms=self._settings.vad_min_silence_ms,
                speech_pad_ms=self._settings.vad_speech_pad_ms,
                return_seconds=False,
            )

        segments = [
            SpeechSegment(
                start=max(0, int(item["start"])),
                end=min(waveform.size, int(item["end"])),
            )
            for item in raw_segments
            if int(item["end"]) > int(item["start"])
        ]
        return build_vad_result(waveform, segments, self._settings)


def merge_segments(
    segments: list[SpeechSegment], max_gap_samples: int
) -> list[SpeechSegment]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda segment: segment.start)
    merged = [ordered[0]]
    for segment in ordered[1:]:
        previous = merged[-1]
        if segment.start - previous.end <= max_gap_samples:
            merged[-1] = SpeechSegment(previous.start, max(previous.end, segment.end))
        else:
            merged.append(segment)
    return merged


def build_vad_result(
    waveform: FloatWaveform,
    segments: list[SpeechSegment],
    settings: Settings,
) -> VADResult:
    """Create duration metrics and a bounded, chronological speech waveform."""

    total_duration = waveform.size / settings.sample_rate
    speech_samples = sum(segment.end - segment.start for segment in segments)
    speech_duration = speech_samples / settings.sample_rate
    speech_ratio = speech_duration / total_duration if total_duration else 0.0

    merged = merge_segments(
        segments, round(settings.vad_merge_gap_ms * settings.sample_rate / 1000)
    )
    max_samples = round(settings.max_inference_seconds * settings.sample_rate)
    separator = np.zeros(
        round(settings.vad_join_silence_ms * settings.sample_rate / 1000),
        dtype=np.float32,
    )
    chunks: list[FloatWaveform] = []
    used_samples = 0
    for segment in merged:
        if used_samples >= max_samples:
            break
        if chunks and separator.size:
            available = max_samples - used_samples
            gap = separator[:available]
            chunks.append(gap)
            used_samples += gap.size
        available = max_samples - used_samples
        chunk = waveform[segment.start : segment.end][:available]
        if chunk.size:
            chunks.append(chunk)
            used_samples += chunk.size

    if chunks:
        speech_waveform = np.concatenate(chunks).astype(np.float32, copy=False)
    else:
        speech_waveform = np.empty(0, dtype=np.float32)

    return VADResult(
        total_duration_seconds=total_duration,
        speech_duration_seconds=speech_duration,
        speech_ratio=speech_ratio,
        segments=segments,
        speech_waveform=speech_waveform,
    )

