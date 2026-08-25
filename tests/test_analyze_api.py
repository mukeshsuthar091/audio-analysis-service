"""Full API contract tests with only the expensive ML boundary replaced."""

from uuid import uuid4

import pytest
from httpx import AsyncClient

from app.schemas.response import AnalyzeResponse
from tests.conftest import make_wav


@pytest.mark.asyncio
async def test_analyze_returns_full_valid_response(client: AsyncClient) -> None:
    contact_id = uuid4()
    response = await client.post(
        "/analyze",
        data={"contact_id": str(contact_id)},
        files={"audio": ("sample.wav", make_wav(), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    parsed = AnalyzeResponse.model_validate(response.json())
    assert parsed.contact_id == contact_id
    assert parsed.gender.prediction.value == "male"
    assert parsed.age_bracket.prediction.value == "31-45"
    assert parsed.language.code == "en"
    assert parsed.language.name == "English"
    assert parsed.language.confidence == 0.92
    assert 0.0 <= parsed.gender.confidence <= 1.0
    assert 0.0 <= parsed.age_bracket.confidence <= 1.0
    assert parsed.processing_ms >= 0
    assert response.headers["x-request-id"].startswith("req_")


@pytest.mark.asyncio
async def test_insufficient_silence_returns_200(client: AsyncClient) -> None:
    response = await client.post(
        "/analyze",
        data={"contact_id": str(uuid4())},
        files={"audio": ("silence.wav", make_wav(amplitude=0.0), "audio/wav")},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["audio_quality"] == "insufficient"
    assert body["gender"] == {"prediction": "unknown", "confidence": 0.0}
    assert body["age_bracket"] == {"prediction": "unknown", "confidence": 0.0}
    assert body["language"] == {
        "code": "unknown",
        "name": "unknown",
        "confidence": 0.0,
    }


@pytest.mark.asyncio
async def test_metrics_exposes_required_series(client: AsyncClient) -> None:
    await client.post(
        "/analyze",
        data={"contact_id": str(uuid4())},
        files={"audio": ("silence.wav", make_wav(amplitude=0.0), "audio/wav")},
    )
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert "audio_analysis_requests_total" in response.text
    assert "audio_analysis_component_ready" in response.text
    assert "audio_analysis_language_outcomes_total" in response.text
