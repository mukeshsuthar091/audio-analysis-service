"""Benchmark sequential versus parallel age/gender and language inference."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

import numpy as np
import torch

from app.core.config import Settings
from app.inference.compile_benchmark import timing_summary
from app.inference.language import LanguageModelService
from app.inference.model import AttributeModelService


def synthetic_waveform(duration_seconds: float, sample_rate: int) -> np.ndarray:
    timeline = np.arange(round(duration_seconds * sample_rate), dtype=np.float32)
    timeline /= sample_rate
    return (
        0.12 * np.sin(2.0 * np.pi * 180.0 * timeline)
        + 0.04 * np.sin(2.0 * np.pi * 360.0 * timeline)
    ).astype(np.float32, copy=False)


async def elapsed(awaitable: Any) -> tuple[float, Any]:
    started = time.perf_counter()
    result = await awaitable
    return (time.perf_counter() - started) * 1000.0, result


async def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    settings = Settings(
        device="cpu",
        language_inference_timeout_seconds=20.0,
    )
    waveform = synthetic_waveform(args.duration_seconds, settings.sample_rate)
    attribute = AttributeModelService.load(settings)
    language = LanguageModelService.load(settings)
    try:
        attribute.warmup()
        language.warmup()
        for _ in range(args.warmups):
            await attribute.infer(waveform)
            await language.infer(waveform)

        attribute_samples: list[float] = []
        language_samples: list[float] = []
        sequential_samples: list[float] = []
        for _ in range(args.runs):
            started = time.perf_counter()
            attribute_ms, _ = await elapsed(attribute.infer(waveform))
            language_ms, _ = await elapsed(language.infer(waveform))
            sequential_samples.append((time.perf_counter() - started) * 1000.0)
            attribute_samples.append(attribute_ms)
            language_samples.append(language_ms)

        parallel_samples: list[float] = []
        for _ in range(args.runs):
            started = time.perf_counter()
            await asyncio.gather(attribute.infer(waveform), language.infer(waveform))
            parallel_samples.append((time.perf_counter() - started) * 1000.0)

        sequential = timing_summary(sequential_samples)
        parallel = timing_summary(parallel_samples)
        p50_improved = parallel["p50_ms"] < sequential["p50_ms"]
        p95_allowed = parallel["p95_ms"] <= sequential["p95_ms"] * 1.10
        return {
            "benchmark": "age_gender_plus_language",
            "duration_seconds": args.duration_seconds,
            "runs": args.runs,
            "warmups": args.warmups,
            "torch_threads": args.threads,
            "attribute": timing_summary(attribute_samples),
            "language": timing_summary(language_samples),
            "sequential": sequential,
            "parallel": parallel,
            "parallel_meets_policy": p50_improved and p95_allowed,
            "recommended_mode": (
                "parallel" if p50_improved and p95_allowed else "sequential"
            ),
        }
    finally:
        waveform.fill(0.0)
        attribute.close()
        language.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", type=float, default=5.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.duration_seconds <= 0 or args.runs <= 0 or args.warmups < 0:
        parser.error("duration and runs must be positive; warmups cannot be negative")
    if args.threads <= 0:
        parser.error("threads must be positive")
    return args


def main() -> None:
    print(json.dumps(asyncio.run(run_benchmark(parse_args())), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
