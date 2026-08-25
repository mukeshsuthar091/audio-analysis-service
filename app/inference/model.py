"""audEERING Wav2Vec2 model definition, loading, and bounded execution."""

import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from transformers import Wav2Vec2Config, Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from app.core.config import Settings
from app.core.exceptions import AppError

FloatWaveform = NDArray[np.float32]


class ModelHead(nn.Module):
    """Classification/regression head from the audEERING model card."""

    def __init__(self, config: Wav2Vec2Config, num_labels: int) -> None:
        super().__init__()
        hidden_size = int(config.hidden_size)
        final_dropout = float(config.final_dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(final_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    """Checkpoint-compatible age regression and gender classification model."""

    def __init__(self, config: Wav2Vec2Config) -> None:
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.post_init()

    def forward(
        self, input_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.wav2vec2(input_values)
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        age = self.age(pooled)
        gender = torch.softmax(self.gender(pooled), dim=1)
        return pooled, age, gender


@dataclass(frozen=True, slots=True)
class RawModelOutput:
    normalized_age: float
    child_probability: float
    female_probability: float
    male_probability: float

    @property
    def gender_probabilities(self) -> dict[str, float]:
        return {
            "child": self.child_probability,
            "female": self.female_probability,
            "male": self.male_probability,
        }


class AttributeInference(Protocol):
    async def infer(self, waveform: FloatWaveform) -> RawModelOutput: ...


class AttributeModelService:
    """Single loaded model with bounded, non-event-loop execution."""

    def __init__(
        self,
        model: nn.Module,
        processor: Wav2Vec2Processor,
        device: torch.device,
        settings: Settings,
    ) -> None:
        self._model = model
        self._processor = processor
        self._device = device
        self._settings = settings
        self._executor = ThreadPoolExecutor(
            max_workers=settings.inference_max_concurrency,
            thread_name_prefix="attribute-inference",
        )
        self._slots = asyncio.Semaphore(settings.inference_max_concurrency)

    @classmethod
    def load(cls, settings: Settings) -> "AttributeModelService":
        device = resolve_device(settings.device)
        processor = Wav2Vec2Processor.from_pretrained(settings.model_id)
        model = AgeGenderModel.from_pretrained(
            settings.model_id,
            use_safetensors=True,
        )
        model.to(device)
        model.eval()
        model = compile_model_if_enabled(model, settings)
        return cls(model=model, processor=processor, device=device, settings=settings)

    def warmup(self) -> None:
        dummy = np.zeros(self._settings.sample_rate, dtype=np.float32)
        try:
            self._infer_sync(dummy)
        finally:
            dummy.fill(0.0)

    async def infer(self, waveform: FloatWaveform) -> RawModelOutput:
        loop = asyncio.get_running_loop()
        started = loop.time()
        try:
            await asyncio.wait_for(
                self._slots.acquire(), timeout=self._settings.inference_timeout_seconds
            )
        except TimeoutError as exc:
            raise AppError(504, "INFERENCE_TIMEOUT", "Model inference timed out.") from exc

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
            0.001, self._settings.inference_timeout_seconds - (loop.time() - started)
        )
        try:
            return await asyncio.wait_for(asyncio.shield(wrapped), timeout=remaining)
        except TimeoutError as exc:
            raise AppError(504, "INFERENCE_TIMEOUT", "Model inference timed out.") from exc

    def _infer_sync(self, waveform: FloatWaveform) -> RawModelOutput:
        inputs = self._processor(
            waveform,
            sampling_rate=self._settings.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs["input_values"].to(self._device)
        with torch.inference_mode():
            _, age_tensor, gender_tensor = self._model(input_values)
        age = float(age_tensor.squeeze().detach().cpu().item())
        gender = gender_tensor.squeeze(0).detach().cpu().tolist()
        return map_model_output(age, gender)

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


def map_model_output(age: float, gender: list[float]) -> RawModelOutput:
    """Map the checkpoint tensor order ``female, male, child`` to named fields."""

    values = [age, *[float(value) for value in gender]]
    if len(gender) != 3 or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Model returned invalid values")
    return RawModelOutput(
        normalized_age=age,
        child_probability=gender[2],
        female_probability=gender[0],
        male_probability=gender[1],
    )


def compile_model_if_enabled(model: nn.Module, settings: Settings) -> nn.Module:
    """Apply the configured ``torch.compile`` policy after model evaluation."""

    if not settings.torch_compile:
        return model
    return torch.compile(
        model,
        backend=settings.torch_compile_backend,
        mode=settings.torch_compile_mode,
        dynamic=settings.torch_compile_dynamic,
        fullgraph=False,
    )


def resolve_device(selection: str) -> torch.device:
    if selection == "cpu":
        return torch.device("cpu")
    if selection == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
