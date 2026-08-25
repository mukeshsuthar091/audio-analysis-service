"""Shared test factories and in-memory audio fixtures."""

import io
import wave
from collections.abc import AsyncIterator

import numpy as np
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from app.audio.vad import SpeechSegment, VADResult, build_vad_result
from app.core.config import Settings
from app.core.metrics import Metrics
from app.core.runtime import RuntimeState
from app.inference.model import RawModelOutput
from app.main import create_app


def make_wav(
    duration_seconds: float = 3.0,
    sample_rate: int = 16_000,
    amplitude: float = 0.2,
) -> bytes:
    sample_count = round(duration_seconds * sample_rate)
    if amplitude:
        time_axis = np.arange(sample_count, dtype=np.float64) / sample_rate
        samples = amplitude * np.sin(2.0 * np.pi * 220.0 * time_axis)
    else:
        samples = np.zeros(sample_count, dtype=np.float64)
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return output.getvalue()


class FakeVAD:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def detect(self, waveform: np.ndarray) -> VADResult:
        segments = (
            [SpeechSegment(start=0, end=waveform.size)]
            if waveform.size and float(np.max(np.abs(waveform))) > 0.01
            else []
        )
        return build_vad_result(waveform, segments, self.settings)


class FakeModel:
    async def infer(self, waveform: np.ndarray) -> RawModelOutput:
        assert waveform.size > 0
        return RawModelOutput(
            normalized_age=0.37,
            child_probability=0.02,
            female_probability=0.08,
            male_probability=0.90,
        )


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "max_upload_bytes": 1024 * 1024,
        "max_multipart_overhead_bytes": 64 * 1024,
        "decode_timeout_seconds": 5.0,
        "inference_timeout_seconds": 5.0,
        "log_level": "WARNING",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def make_test_app(settings: Settings | None = None, ready: bool = True) -> FastAPI:
    selected = settings or make_settings()

    async def factory(current: Settings, metrics: Metrics) -> RuntimeState:
        if not ready:
            return RuntimeState(settings=current, metrics=metrics)
        return RuntimeState(
            settings=current,
            metrics=metrics,
            ffmpeg_available=True,
            vad=FakeVAD(current),
            model=FakeModel(),
        )

    return create_app(settings=selected, runtime_factory=factory)


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = make_test_app()
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as test_client,
    ):
        yield test_client
