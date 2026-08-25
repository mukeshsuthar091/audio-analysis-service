"""FastAPI application factory and process entry point."""

import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request, Response

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.exceptions import install_exception_handlers
from app.core.logging import configure_logging
from app.core.metrics import Metrics
from app.core.runtime import RuntimeState, initialize_runtime

RuntimeFactory = Callable[[Settings, Metrics], Awaitable[RuntimeState]]


def create_app(
    settings: Settings | None = None,
    runtime_factory: RuntimeFactory = initialize_runtime,
) -> FastAPI:
    selected_settings = settings or get_settings()
    configure_logging(selected_settings.log_level)
    application_metrics = Metrics()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime = await runtime_factory(selected_settings, application_metrics)
        application.state.runtime = runtime
        try:
            yield
        finally:
            runtime.close()

    application = FastAPI(
        title=selected_settings.app_name,
        version="1.1.0",
        description=(
            "Conservative age-bracket, gender-presentation, and best-effort spoken-"
            "language estimates from short contact-side call audio. Caller audio is "
            "not intentionally persisted."
        ),
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def request_context(request: Request, call_next: Callable[..., object]) -> Response:
        supplied = request.headers.get("x-request-id", "")
        if (
            supplied
            and len(supplied) <= selected_settings.request_id_max_length
            and re.fullmatch(selected_settings.request_id_pattern, supplied)
        ):
            request_id = supplied
        else:
            request_id = f"req_{uuid4().hex}"
        request.state.request_id = request_id

        started = time.perf_counter()
        response = await call_next(request)  # type: ignore[arg-type]
        response.headers["X-Request-ID"] = request_id
        if request.url.path == "/analyze":
            status_code = str(response.status_code)
            outcome = "success" if response.status_code < 400 else "failure"
            application_metrics.requests.labels(
                outcome=outcome, status_code=status_code
            ).inc()
            application_metrics.request_latency.observe(time.perf_counter() - started)
        return response

    install_exception_handlers(application)
    application.include_router(router)
    return application


app = create_app()
