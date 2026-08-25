"""FastAPI dependency accessors."""

from fastapi import Request

from app.core.runtime import RuntimeState


def get_runtime(request: Request) -> RuntimeState:
    return request.app.state.runtime

