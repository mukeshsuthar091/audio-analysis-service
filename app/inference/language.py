"""Best-effort spoken-language inference and conservative post-processing."""

from __future__ import annotations

import asyncio
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from huggingface_hub import snapshot_download
from numpy.typing import NDArray

from app.core.config import Settings
from app.inference.model import resolve_device
from app.schemas.response import AudioQuality, LanguageResult

FloatWaveform = NDArray[np.float32]
LANGUAGE_CODE_PATTERN = re.compile(r"^[a-z]{2,3}$")
LEGACY_LANGUAGE_CODES = {"iw": "he", "jw": "jv"}


@dataclass(frozen=True, slots=True)
class RawLanguageOutput:
    label: str
    top_probability: float
    runner_up_probability: float


class LanguageInference(Protocol):
    async def infer(self, waveform: FloatWaveform) -> RawLanguageOutput: ...


class LanguageInferenceTimeout(TimeoutError):
    """Raised when the optional language worker exceeds its deadline."""


def unknown_language() -> LanguageResult:
    return LanguageResult(code="unknown", name="unknown", confidence=0.0)


def parse_language_label(label: str) -> tuple[str, str] | None:
    """Parse SpeechBrain's ``code: Language`` label without trusting remote text."""

    if not isinstance(label, str) or ":" not in label:
        return None
    code, name = (part.strip() for part in label.split(":", 1))
    code = LEGACY_LANGUAGE_CODES.get(code.lower(), code.lower())
    if not LANGUAGE_CODE_PATTERN.fullmatch(code):
        return None
    if not name or len(name) > 64 or any(ord(character) < 32 for character in name):
        return None
    return code, name


def process_language(
    raw: RawLanguageOutput | None,
    quality: AudioQuality,
    speech_duration_seconds: float,
    settings: Settings,
) -> LanguageResult:
    """Convert raw language evidence into the conservative public contract."""

    if (
        raw is None
        or quality is AudioQuality.INSUFFICIENT
        or not math.isfinite(speech_duration_seconds)
        or speech_duration_seconds < settings.language_min_speech_seconds
    ):
        return unknown_language()

    parsed = parse_language_label(raw.label)
    probabilities = (raw.top_probability, raw.runner_up_probability)
    if (
        parsed is None
        or not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities)
        or raw.runner_up_probability > raw.top_probability
    ):
        return unknown_language()

    multiplier = {
        AudioQuality.GOOD: settings.quality_multiplier_good,
        AudioQuality.DEGRADED: settings.quality_multiplier_degraded,
        AudioQuality.INSUFFICIENT: settings.quality_multiplier_insufficient,
    }[quality]
    adjusted = min(1.0, max(0.0, raw.top_probability * multiplier))
    margin = raw.top_probability - raw.runner_up_probability
    if (
        adjusted < settings.language_confidence_threshold
        or margin < settings.language_min_margin
    ):
        return unknown_language()

    code, name = parsed
    return LanguageResult(code=code, name=name, confidence=round(adjusted, 2))


class LanguageModelService:
    """One loaded SpeechBrain classifier with bounded background execution."""

    def __init__(self, classifier: Any, settings: Settings) -> None:
        self._classifier = classifier
        self._settings = settings
        self._executor = ThreadPoolExecutor(
            max_workers=settings.language_inference_max_concurrency,
            thread_name_prefix="language-inference",
        )
        self._slots = asyncio.Semaphore(settings.language_inference_max_concurrency)

    @classmethod
    def load(cls, settings: Settings) -> LanguageModelService:
        # Import lazily so the core service can still report liveness when the
        # optional language dependency or checkpoint is unavailable.
        from speechbrain.inference.classifiers import EncoderClassifier

        snapshot_path = snapshot_download(
            repo_id=settings.language_model_id,
            revision=settings.language_model_revision,
            allow_patterns=[
                "hyperparams.yaml",
                "*.ckpt",
                "label_encoder.txt",
                "config.json",
            ],
        )
        device = resolve_device(settings.device)
        classifier = EncoderClassifier.from_hparams(
            source=str(Path(snapshot_path)),
            run_opts={"device": str(device)},
        )
        classifier.mods.eval()
        return cls(classifier=classifier, settings=settings)

    def warmup(self) -> None:
        dummy = np.zeros(
            round(self._settings.language_min_speech_seconds * self._settings.sample_rate),
            dtype=np.float32,
        )
        try:
            self._infer_sync(dummy)
        finally:
            dummy.fill(0.0)

    async def infer(self, waveform: FloatWaveform) -> RawLanguageOutput:
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            await asyncio.wait_for(
                self._slots.acquire(),
                timeout=self._settings.language_inference_timeout_seconds,
            )
        except TimeoutError as exc:
            raise LanguageInferenceTimeout("Language inference timed out") from exc

        owned = np.array(waveform, dtype=np.float32, copy=True)
        try:
            concurrent_future = self._executor.submit(self._infer_sync, owned)
        except BaseException:
            owned.fill(0.0)
            self._slots.release()
            raise

        def cleanup(_: object) -> None:
            owned.fill(0.0)
            loop.call_soon_threadsafe(self._slots.release)

        concurrent_future.add_done_callback(cleanup)
        wrapped = asyncio.wrap_future(concurrent_future)
        remaining = max(
            0.001,
            self._settings.language_inference_timeout_seconds
            - (loop.time() - started),
        )
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), timeout=remaining)
        except TimeoutError as exc:
            raise LanguageInferenceTimeout("Language inference timed out") from exc

    def _infer_sync(self, waveform: FloatWaveform) -> RawLanguageOutput:
        tensor = torch.from_numpy(waveform).unsqueeze(0)
        with torch.inference_mode():
            log_probabilities, _, _, labels = self._classifier.classify_batch(tensor)

        values = log_probabilities.squeeze(0)
        if values.ndim != 1 or values.numel() < 2 or len(labels) != 1:
            raise RuntimeError("Language model returned an invalid result")
        probabilities = torch.exp(values)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Language model returned an invalid result")
        top_values, _ = torch.topk(probabilities, k=2)
        return RawLanguageOutput(
            label=str(labels[0]),
            top_probability=float(top_values[0].detach().cpu().item()),
            runner_up_probability=float(top_values[1].detach().cpu().item()),
        )

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
