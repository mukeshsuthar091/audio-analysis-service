"""Internal request data structures."""

from dataclasses import dataclass
from uuid import UUID


@dataclass(slots=True)
class ParsedAnalyzeRequest:
    """Validated multipart fields kept only for the request lifetime."""

    contact_id: UUID
    audio: bytearray
    filename: str | None
    content_type: str | None

