"""Request and audio failure contract tests."""

from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from tests.conftest import make_settings, make_test_app, make_wav


@pytest.mark.asyncio
async def test_empty_audio(client: AsyncClient) -> None:
    response = await client.post(
        "/analyze",
        data={"contact_id": str(uuid4())},
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "EMPTY_AUDIO"


@pytest.mark.asyncio
async def test_corrupted_audio(client: AsyncClient) -> None:
    response = await client.post(
        "/analyze",
        data={"contact_id": str(uuid4())},
        files={"audio": ("broken.mp3", b"not audio data", "audio/mpeg")},
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "INVALID_AUDIO"


@pytest.mark.asyncio
async def test_oversized_audio() -> None:
    settings = make_settings(max_upload_bytes=512, max_multipart_overhead_bytes=4096)
    app = make_test_app(settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        response = await client.post(
            "/analyze",
            data={"contact_id": str(uuid4())},
            files={"audio": ("large.wav", b"x" * 513, "audio/wav")},
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "UPLOAD_TOO_LARGE"


@pytest.mark.asyncio
async def test_invalid_uuid(client: AsyncClient) -> None:
    response = await client.post(
        "/analyze",
        data={"contact_id": "not-a-uuid"},
        files={"audio": ("sample.wav", make_wav(), "audio/wav")},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_CONTACT_ID"


@pytest.mark.asyncio
async def test_missing_audio_field(client: AsyncClient) -> None:
    response = await client.post(
        "/analyze", files={"contact_id": (None, str(uuid4()))}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MISSING_AUDIO"


@pytest.mark.asyncio
async def test_malformed_multipart(client: AsyncClient) -> None:
    response = await client.post(
        "/analyze", content=b"bad", headers={"content-type": "multipart/form-data"}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "MALFORMED_REQUEST"


@pytest.mark.asyncio
async def test_not_ready_returns_503() -> None:
    app = make_test_app(ready=False)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        health = await client.get("/health")
        ready = await client.get("/ready")
        analyze = await client.post("/analyze")
    assert health.status_code == 200
    assert ready.status_code == 503
    assert ready.json() == {
        "status": "not_ready",
        "model_loaded": False,
        "language_model_loaded": False,
        "vad_loaded": False,
        "ffmpeg_available": False,
    }
    assert analyze.status_code == 503
