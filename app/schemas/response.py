"""Public API response schemas."""

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class GenderPrediction(StrEnum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AgeBracket(StrEnum):
    AGE_18_30 = "18-30"
    AGE_31_45 = "31-45"
    AGE_46_60 = "46-60"
    AGE_60_PLUS = "60+"
    UNKNOWN = "unknown"


class AudioQuality(StrEnum):
    GOOD = "good"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"


class PredictionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)


class GenderResult(PredictionResult):
    prediction: GenderPrediction


class AgeBracketResult(PredictionResult):
    prediction: AgeBracket


class LanguageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(pattern=r"^(?:[a-z]{2,3}|unknown)$")
    name: str = Field(min_length=1, max_length=64)
    confidence: float = Field(ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contact_id: UUID
    gender: GenderResult
    age_bracket: AgeBracketResult
    language: LanguageResult
    processing_ms: int = Field(ge=0)
    audio_quality: AudioQuality


class HealthResponse(BaseModel):
    status: str = "healthy"


class ReadinessResponse(BaseModel):
    status: str
    model_loaded: bool
    language_model_loaded: bool
    vad_loaded: bool
    ffmpeg_available: bool


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
