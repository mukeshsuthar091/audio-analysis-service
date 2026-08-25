"""Opt-in real model loading test."""

import os

import pytest

from app.inference.language import LanguageModelService
from app.inference.model import AttributeModelService
from tests.conftest import make_settings


@pytest.mark.model
@pytest.mark.skipif(
    os.getenv("RUN_REAL_MODEL_TESTS") != "1",
    reason="Set RUN_REAL_MODEL_TESTS=1 to download and load the real checkpoint.",
)
def test_real_model_loads_and_warms() -> None:
    service = AttributeModelService.load(make_settings())
    try:
        service.warmup()
    finally:
        service.close()


@pytest.mark.model
@pytest.mark.skipif(
    os.getenv("RUN_REAL_MODEL_TESTS") != "1",
    reason="Set RUN_REAL_MODEL_TESTS=1 to download and load the real checkpoint.",
)
def test_real_language_model_loads_and_warms() -> None:
    service = LanguageModelService.load(make_settings())
    try:
        service.warmup()
    finally:
        service.close()
