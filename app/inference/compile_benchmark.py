"""Benchmark eager versus ``torch.compile`` model inference.

Per-run timings exclude model loading and feature preprocessing so they isolate
the effect of compiling the Wav2Vec2 forward pass.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from transformers import Wav2Vec2Processor

from app.inference.model import AgeGenderModel, resolve_device

ModelOutput = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def timing_summary(samples_ms: Sequence[float]) -> dict[str, float]:
    """Return stable, JSON-friendly latency statistics."""

    values = np.asarray(samples_ms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("At least one timing sample is required.")
    return {
        "mean_ms": round(float(statistics.fmean(samples_ms)), 2),
        "p50_ms": round(float(np.percentile(values, 50)), 2),
        "p95_ms": round(float(np.percentile(values, 95)), 2),
        "min_ms": round(float(values.min()), 2),
        "max_ms": round(float(values.max()), 2),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_forward(
    model: torch.nn.Module,
    input_values: torch.Tensor,
    device: torch.device,
) -> tuple[float, ModelOutput]:
    _synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_values)
    _synchronize(device)
    return (time.perf_counter() - started) * 1000.0, output


def _synthetic_waveform(duration_seconds: float, sample_rate: int = 16_000) -> np.ndarray:
    sample_count = int(duration_seconds * sample_rate)
    timeline = np.arange(sample_count, dtype=np.float32) / sample_rate
    waveform = (
        0.12 * np.sin(2.0 * np.pi * 180.0 * timeline)
        + 0.04 * np.sin(2.0 * np.pi * 360.0 * timeline)
    )
    return waveform.astype(np.float32, copy=False)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Load the checkpoint and compare fixed-shape eager and compiled forwards."""

    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)

    load_started = time.perf_counter()
    processor = Wav2Vec2Processor.from_pretrained(args.model_id)
    eager_model = AgeGenderModel.from_pretrained(
        args.model_id,
        use_safetensors=True,
    ).to(device)
    eager_model.eval()
    load_ms = (time.perf_counter() - load_started) * 1000.0

    waveform = _synthetic_waveform(args.duration_seconds)
    inputs = processor(
        waveform,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs.input_values.to(device)

    eager_output: ModelOutput | None = None
    for _ in range(args.warmups):
        _, eager_output = _timed_forward(eager_model, input_values, device)

    eager_samples: list[float] = []
    for _ in range(args.runs):
        elapsed_ms, eager_output = _timed_forward(eager_model, input_values, device)
        eager_samples.append(elapsed_ms)

    compile_started = time.perf_counter()
    compiled_model = torch.compile(
        eager_model,
        backend="inductor",
        mode=args.mode,
        dynamic=args.dynamic,
        fullgraph=False,
    )
    compile_wrapper_ms = (time.perf_counter() - compile_started) * 1000.0

    compile_first_call_ms, compiled_output = _timed_forward(
        compiled_model,
        input_values,
        device,
    )
    for _ in range(args.warmups):
        _, compiled_output = _timed_forward(compiled_model, input_values, device)

    compiled_samples: list[float] = []
    for _ in range(args.runs):
        elapsed_ms, compiled_output = _timed_forward(
            compiled_model,
            input_values,
            device,
        )
        compiled_samples.append(elapsed_ms)

    assert eager_output is not None
    torch.testing.assert_close(eager_output[1], compiled_output[1], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(eager_output[2], compiled_output[2], rtol=1e-4, atol=1e-5)

    eager_summary = timing_summary(eager_samples)
    compiled_summary = timing_summary(compiled_samples)
    p50_delta = eager_summary["p50_ms"] - compiled_summary["p50_ms"]
    break_even = None
    if p50_delta > 0:
        break_even = round(compile_first_call_ms / p50_delta, 1)

    return {
        "benchmark": "model_forward_only",
        "model_id": args.model_id,
        "torch_version": torch.__version__,
        "machine": platform.machine(),
        "device": str(device),
        "torch_threads": args.threads,
        "duration_seconds": args.duration_seconds,
        "runs": args.runs,
        "warmups": args.warmups,
        "compile_backend": "inductor",
        "compile_mode": args.mode,
        "compile_dynamic": args.dynamic,
        "model_load_ms": round(load_ms, 2),
        "compile_wrapper_ms": round(compile_wrapper_ms, 2),
        "compile_first_call_ms": round(compile_first_call_ms, 2),
        "eager": eager_summary,
        "compiled": compiled_summary,
        "p50_speedup": round(eager_summary["p50_ms"] / compiled_summary["p50_ms"], 3),
        "estimated_break_even_requests": break_even,
        "outputs_match": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="audeering/wav2vec2-large-robust-6-ft-age-gender",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument(
        "--dynamic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.runs <= 0 or args.warmups < 0 or args.threads <= 0:
        parser.error("duration, runs, and threads must be positive; warmups cannot be negative")
    return args


def main() -> None:
    print(json.dumps(run_benchmark(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
