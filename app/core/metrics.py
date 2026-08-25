"""Application-scoped Prometheus metrics."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram


class Metrics:
    """Own a registry so application factories remain safe in tests."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "audio_analysis_requests_total",
            "Analyze requests by outcome and HTTP status.",
            ("outcome", "status_code"),
            registry=self.registry,
        )
        self.request_latency = Histogram(
            "audio_analysis_request_duration_seconds",
            "End-to-end analyze request latency.",
            registry=self.registry,
        )
        self.decode_latency = Histogram(
            "audio_analysis_decode_duration_seconds",
            "FFmpeg decode latency.",
            registry=self.registry,
        )
        self.vad_latency = Histogram(
            "audio_analysis_vad_duration_seconds",
            "Voice activity detection latency.",
            registry=self.registry,
        )
        self.inference_latency = Histogram(
            "audio_analysis_inference_duration_seconds",
            "Age/gender model inference latency.",
            registry=self.registry,
        )
        self.quality_counts = Counter(
            "audio_analysis_quality_total",
            "Completed analyses by quality class.",
            ("quality",),
            registry=self.registry,
        )
        self.unknown_counts = Counter(
            "audio_analysis_unknown_predictions_total",
            "Unknown public predictions by attribute.",
            ("attribute",),
            registry=self.registry,
        )
        self.readiness = Gauge(
            "audio_analysis_component_ready",
            "Component readiness state.",
            ("component",),
            registry=self.registry,
        )
        for component in ("ffmpeg", "vad", "model"):
            self.readiness.labels(component=component).set(0)

