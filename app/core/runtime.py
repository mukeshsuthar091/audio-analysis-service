"""Shared component lifecycle and readiness state."""

import asyncio
import logging
from dataclasses import dataclass

from app.audio.decoder import verify_ffmpeg
from app.audio.quality import warmup_quality_analysis
from app.audio.vad import SileroVoiceActivityDetector, VoiceActivityDetector
from app.core.config import Settings
from app.core.metrics import Metrics
from app.inference.language import LanguageInference, LanguageModelService
from app.inference.model import AttributeInference, AttributeModelService

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeState:
    settings: Settings
    metrics: Metrics
    ffmpeg_available: bool = False
    vad: VoiceActivityDetector | None = None
    model: AttributeInference | None = None
    model_service: AttributeModelService | None = None
    language: LanguageInference | None = None
    language_model_service: LanguageModelService | None = None

    @property
    def vad_loaded(self) -> bool:
        return self.vad is not None

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    @property
    def language_model_loaded(self) -> bool:
        return self.language is not None

    @property
    def ready(self) -> bool:
        return self.ffmpeg_available and self.vad_loaded and self.model_loaded

    def update_metrics(self) -> None:
        self.metrics.readiness.labels(component="ffmpeg").set(
            int(self.ffmpeg_available)
        )
        self.metrics.readiness.labels(component="vad").set(int(self.vad_loaded))
        self.metrics.readiness.labels(component="model").set(int(self.model_loaded))
        self.metrics.readiness.labels(component="language_model").set(
            int(self.language_model_loaded)
        )

    def close(self) -> None:
        if self.model_service is not None:
            self.model_service.close()
        if self.language_model_service is not None:
            self.language_model_service.close()


async def initialize_runtime(settings: Settings, metrics: Metrics) -> RuntimeState:
    """Initialize each required component without preventing liveness startup."""

    runtime = RuntimeState(settings=settings, metrics=metrics)
    runtime.ffmpeg_available = await asyncio.to_thread(
        verify_ffmpeg, settings.ffmpeg_binary
    )
    if not runtime.ffmpeg_available:
        logger.error("ffmpeg_initialization_failed")

    try:
        vad = await asyncio.to_thread(SileroVoiceActivityDetector.load, settings)
        await asyncio.to_thread(vad.warmup)
        await asyncio.to_thread(warmup_quality_analysis, settings)
        runtime.vad = vad
    except Exception as exc:
        logger.exception(
            "vad_initialization_failed",
            extra={"event_fields": {"error_type": type(exc).__name__}},
        )

    try:
        model = await asyncio.to_thread(AttributeModelService.load, settings)
        await asyncio.to_thread(model.warmup)
        runtime.model = model
        runtime.model_service = model
    except Exception as exc:
        logger.exception(
            "model_initialization_failed",
            extra={"event_fields": {"error_type": type(exc).__name__}},
        )

    if settings.language_detection_enabled:
        language_model: LanguageModelService | None = None
        try:
            language_model = await asyncio.to_thread(LanguageModelService.load, settings)
            await asyncio.to_thread(language_model.warmup)
            runtime.language = language_model
            runtime.language_model_service = language_model
        except Exception as exc:
            if language_model is not None:
                language_model.close()
            logger.exception(
                "language_model_initialization_failed",
                extra={"event_fields": {"error_type": type(exc).__name__}},
            )

    runtime.update_metrics()
    logger.info(
        "runtime_initialized",
        extra={
            "event_fields": {
                "ffmpeg_available": runtime.ffmpeg_available,
                "vad_loaded": runtime.vad_loaded,
                "model_loaded": runtime.model_loaded,
                "language_model_loaded": runtime.language_model_loaded,
                "ready": runtime.ready,
                "device": settings.device,
                "torch_compile": settings.torch_compile,
                "torch_compile_backend": settings.torch_compile_backend,
            }
        },
    )
    return runtime
