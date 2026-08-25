"""Tests for the optional compilation policy and benchmark reporting."""

import torch

from app.inference.compile_benchmark import timing_summary
from app.inference.model import compile_model_if_enabled
from tests.conftest import make_settings


def test_compile_is_noop_when_disabled(monkeypatch) -> None:
    model = torch.nn.Linear(2, 1)

    def unexpected_compile(*args: object, **kwargs: object) -> None:
        raise AssertionError("torch.compile should not be called")

    monkeypatch.setattr(torch, "compile", unexpected_compile)

    assert compile_model_if_enabled(model, make_settings(torch_compile=False)) is model


def test_compile_uses_configured_inductor_options(monkeypatch) -> None:
    model = torch.nn.Linear(2, 1)
    compiled = torch.nn.Sequential(model)
    captured: dict[str, object] = {}

    def fake_compile(candidate: torch.nn.Module, **kwargs: object) -> torch.nn.Module:
        captured["candidate"] = candidate
        captured.update(kwargs)
        return compiled

    monkeypatch.setattr(torch, "compile", fake_compile)
    result = compile_model_if_enabled(
        model,
        make_settings(
            torch_compile=True,
            torch_compile_mode="reduce-overhead",
            torch_compile_dynamic=False,
        ),
    )

    assert result is compiled
    assert captured == {
        "candidate": model,
        "backend": "inductor",
        "mode": "reduce-overhead",
        "dynamic": False,
        "fullgraph": False,
    }


def test_timing_summary_calculates_expected_statistics() -> None:
    summary = timing_summary([10.0, 20.0, 30.0, 40.0])

    assert summary == {
        "mean_ms": 25.0,
        "p50_ms": 25.0,
        "p95_ms": 38.5,
        "min_ms": 10.0,
        "max_ms": 40.0,
    }
