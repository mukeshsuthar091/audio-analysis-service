"""API behavior for optional spoken-language enrichment."""

from collections.abc import AsyncIterator
from uuid import uuid4

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from app.core.metrics import Metrics
from app.core.runtime import RuntimeState
from app.inference.language import LanguageInferenceTimeout, RawLanguageOutput
from app.main import create_app
from tests.conftest import FakeLanguageModel, FakeModel, FakeVAD, make_settings, make_wav


class TimedOutLanguageModel:
    async def infer(self, waveform: np.ndarray) -> RawLanguageOutput:
        del waveform
        raise LanguageInferenceTimeout("test timeout")


class FailedLanguageModel:
    async def infer(self, waveform: np.ndarray) -> RawLanguageOutput:
        del waveform
        raise RuntimeError("test failure")


async def client_for(language: object | None) -> AsyncIterator[AsyncClient]:
    settings = make_settings()

    async def factory(current, metrics: Metrics) -> RuntimeState:
        return RuntimeState(
            settings=current,
            metrics=metrics,
            ffmpeg_available=True,
            vad=FakeVAD(current),
            model=FakeModel(),
            language=language,  # type: ignore[arg-type]
        )

    app = create_app(settings=settings, runtime_factory=factory)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        yield client


async def analyze(client: AsyncClient, amplitude: float = 0.2):
    return await client.post(
        "/analyze",
        data={"contact_id": str(uuid4())},
        files={"audio": ("sample.wav", make_wav(amplitude=amplitude), "audio/wav")},
    )


@pytest.mark.asyncio
async def test_language_model_is_optional_for_readiness_and_analysis() -> None:
    async for client in client_for(None):
        ready = await client.get("/ready")
        response = await analyze(client)

    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["language_model_loaded"] is False
    assert response.status_code == 200
    assert response.json()["language"] == {
        "code": "unknown",
        "name": "unknown",
        "confidence": 0.0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("language", [TimedOutLanguageModel(), FailedLanguageModel()])
async def test_language_failure_preserves_core_results(language: object) -> None:
    async for client in client_for(language):
        response = await analyze(client)

    assert response.status_code == 200
    body = response.json()
    assert body["gender"]["prediction"] == "male"
    assert body["age_bracket"]["prediction"] == "31-45"
    assert body["language"] == {
        "code": "unknown",
        "name": "unknown",
        "confidence": 0.0,
    }


@pytest.mark.asyncio
async def test_insufficient_audio_skips_language_model() -> None:
    language = FakeLanguageModel()
    async for client in client_for(language):
        response = await analyze(client, amplitude=0.0)

    assert response.status_code == 200
    assert response.json()["audio_quality"] == "insufficient"
    assert language.calls == 0
