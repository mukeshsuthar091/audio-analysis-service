"""Typed, centralized application configuration."""

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from ``AAS_``-prefixed environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="AAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Audio Contact Attribute Analysis Service"
    log_level: str = "INFO"

    sample_rate: int = 16_000
    max_upload_bytes: int = 10 * 1024 * 1024
    max_multipart_overhead_bytes: int = 64 * 1024
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 15.0
    preferred_duration_seconds: float = 5.0
    ffmpeg_binary: str = "ffmpeg"
    decode_timeout_seconds: float = 10.0

    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 150
    vad_min_silence_ms: int = 100
    vad_speech_pad_ms: int = 30
    vad_merge_gap_ms: int = 250
    vad_join_silence_ms: int = 20
    max_inference_seconds: float = 5.0

    insufficient_speech_seconds: float = 1.0
    good_speech_seconds: float = 2.5
    insufficient_speech_ratio: float = 0.25
    good_speech_ratio: float = 0.60
    near_silence_rms: float = 0.003
    good_rms: float = 0.02
    clipping_amplitude: float = 0.99
    good_clipping_ratio: float = 0.01
    severe_clipping_ratio: float = 0.05
    severe_snr_db: float = 3.0
    good_snr_db: float = 12.0

    model_id: str = "audeering/wav2vec2-large-robust-6-ft-age-gender"
    device: Literal["cpu", "cuda", "auto"] = "cpu"
    inference_timeout_seconds: float = 20.0
    inference_max_concurrency: int = 1
    torch_compile: bool = False
    torch_compile_backend: Literal["inductor"] = "inductor"
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = (
        "default"
    )
    torch_compile_dynamic: bool = True

    language_detection_enabled: bool = True
    language_model_id: str = "speechbrain/lang-id-voxlingua107-ecapa"
    language_model_revision: str = (
        "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"
    )
    language_min_speech_seconds: float = 1.5
    language_confidence_threshold: float = 0.65
    language_min_margin: float = 0.15
    language_inference_timeout_seconds: float = 3.0
    language_inference_max_concurrency: int = 1

    gender_confidence_threshold: float = 0.65
    gender_min_margin: float = 0.10
    quality_multiplier_good: float = 1.0
    quality_multiplier_degraded: float = 0.75
    quality_multiplier_insufficient: float = 0.0

    age_confidence_threshold: float = 0.55
    age_boundary_scale_years: float = 5.0
    age_boundary_exclusion_years: float = 1.5

    request_id_max_length: int = 128
    request_id_pattern: str = r"^[A-Za-z0-9._:-]+$"
    max_contact_id_bytes: int = 128

    @field_validator(
        "vad_threshold",
        "insufficient_speech_ratio",
        "good_speech_ratio",
        "near_silence_rms",
        "good_rms",
        "clipping_amplitude",
        "good_clipping_ratio",
        "severe_clipping_ratio",
        "gender_confidence_threshold",
        "gender_min_margin",
        "quality_multiplier_good",
        "quality_multiplier_degraded",
        "quality_multiplier_insufficient",
        "age_confidence_threshold",
        "language_confidence_threshold",
        "language_min_margin",
    )
    @classmethod
    def validate_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("value must be between 0 and 1")
        return value

    @field_validator(
        "sample_rate",
        "max_upload_bytes",
        "max_multipart_overhead_bytes",
        "inference_max_concurrency",
        "language_inference_max_concurrency",
    )
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator(
        "min_duration_seconds",
        "max_duration_seconds",
        "decode_timeout_seconds",
        "inference_timeout_seconds",
        "age_boundary_scale_years",
        "language_min_speech_seconds",
        "language_inference_timeout_seconds",
    )
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide immutable settings instance."""

    return Settings()
