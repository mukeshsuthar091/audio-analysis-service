"""REST endpoints and request processing orchestration."""

import asyncio
import logging
import time
from pathlib import PurePath
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.api.dependencies import get_runtime
from app.api.multipart import parse_analyze_multipart
from app.audio.decoder import clear_waveform, decode_audio, validate_waveform
from app.audio.quality import analyze_quality
from app.core.exceptions import AppError
from app.core.runtime import RuntimeState
from app.inference.age import process_age
from app.inference.gender import process_gender
from app.schemas.response import (
    AgeBracket,
    AgeBracketResult,
    AnalyzeResponse,
    AudioQuality,
    GenderPrediction,
    GenderResult,
    HealthResponse,
    ReadinessResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()
RuntimeDep = Annotated[RuntimeState, Depends(get_runtime)]

MULTIPART_OPENAPI = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["contact_id", "audio"],
                    "properties": {
                        "contact_id": {"type": "string", "format": "uuid"},
                        "audio": {"type": "string", "format": "binary"},
                    },
                }
            }
        },
    }
}


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/ready", response_model=ReadinessResponse)
async def ready(runtime: RuntimeDep) -> Response:
    body = ReadinessResponse(
        status="ready" if runtime.ready else "not_ready",
        model_loaded=runtime.model_loaded,
        vad_loaded=runtime.vad_loaded,
        ffmpeg_available=runtime.ffmpeg_available,
    )
    return Response(
        content=body.model_dump_json(),
        status_code=200 if runtime.ready else 503,
        media_type="application/json",
    )


@router.get("/metrics", include_in_schema=False)
async def metrics(runtime: RuntimeDep) -> Response:
    return Response(
        content=generate_latest(runtime.metrics.registry),
        media_type=CONTENT_TYPE_LATEST,
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    openapi_extra=MULTIPART_OPENAPI,
    responses={400: {}, 413: {}, 415: {}, 422: {}, 500: {}, 503: {}, 504: {}},
)
async def analyze(request: Request, runtime: RuntimeDep) -> AnalyzeResponse:
    started = time.perf_counter()
    if not runtime.ready or runtime.vad is None or runtime.model is None:
        raise AppError(
            503,
            "SERVICE_NOT_READY",
            "Required analysis components are not available.",
        )

    parsed = await parse_analyze_multipart(request, runtime.settings)
    waveform = None
    speech_waveform = None
    decode_ms = vad_ms = quality_ms = inference_ms = 0.0
    try:
        decode_started = time.perf_counter()
        waveform = await decode_audio(parsed.audio, runtime.settings)
        decode_ms = elapsed_ms(decode_started)
        runtime.metrics.decode_latency.observe(decode_ms / 1000.0)
        waveform_info = validate_waveform(waveform, runtime.settings)

        vad_started = time.perf_counter()
        vad_result = await asyncio.to_thread(runtime.vad.detect, waveform)
        vad_ms = elapsed_ms(vad_started)
        runtime.metrics.vad_latency.observe(vad_ms / 1000.0)
        speech_waveform = vad_result.speech_waveform

        quality_started = time.perf_counter()
        quality, quality_metrics = await asyncio.to_thread(
            analyze_quality,
            waveform,
            waveform_info,
            vad_result,
            runtime.settings,
        )
        quality_ms = elapsed_ms(quality_started)
        runtime.metrics.quality_counts.labels(quality=quality.value).inc()

        if quality is AudioQuality.INSUFFICIENT:
            gender = GenderResult(
                prediction=GenderPrediction.UNKNOWN, confidence=0.0
            )
            age = AgeBracketResult(prediction=AgeBracket.UNKNOWN, confidence=0.0)
        else:
            inference_started = time.perf_counter()
            raw = await runtime.model.infer(speech_waveform)
            inference_ms = elapsed_ms(inference_started)
            runtime.metrics.inference_latency.observe(inference_ms / 1000.0)
            gender = process_gender(
                raw.gender_probabilities, quality, runtime.settings
            )
            age = process_age(
                raw.normalized_age,
                quality,
                vad_result.speech_duration_seconds,
                runtime.settings,
            )

        if gender.prediction is GenderPrediction.UNKNOWN:
            runtime.metrics.unknown_counts.labels(attribute="gender").inc()
        if age.prediction is AgeBracket.UNKNOWN:
            runtime.metrics.unknown_counts.labels(attribute="age_bracket").inc()

        processing_ms = round(elapsed_ms(started))
        response = AnalyzeResponse(
            contact_id=parsed.contact_id,
            gender=gender,
            age_bracket=age,
            processing_ms=processing_ms,
            audio_quality=quality,
        )
        logger.info(
            "analysis_completed",
            extra={
                "event_fields": {
                    "request_id": request.state.request_id,
                    "contact_id": str(parsed.contact_id),
                    "input_format": input_format_hint(
                        parsed.filename, parsed.content_type
                    ),
                    "audio_duration_ms": round(
                        quality_metrics.total_duration_seconds * 1000
                    ),
                    "speech_duration_ms": round(
                        quality_metrics.speech_duration_seconds * 1000
                    ),
                    "decode_ms": round(decode_ms, 2),
                    "vad_ms": round(vad_ms, 2),
                    "quality_ms": round(quality_ms, 2),
                    "inference_ms": round(inference_ms, 2),
                    "processing_ms": processing_ms,
                    "audio_quality": quality.value,
                    "status": "success",
                }
            },
        )
        return response
    finally:
        parsed.audio.clear()
        clear_waveform(speech_waveform)
        if waveform is not speech_waveform:
            clear_waveform(waveform)


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0


def input_format_hint(filename: str | None, content_type: str | None) -> str:
    if filename:
        suffix = PurePath(filename).suffix.lower().lstrip(".")
        if suffix and suffix.isalnum() and len(suffix) <= 10:
            return suffix
    if content_type and "/" in content_type:
        subtype = content_type.split("/", 1)[1].split(";", 1)[0]
        if subtype.replace("-", "").replace("+", "").isalnum():
            return subtype[:32]
    return "unknown"
