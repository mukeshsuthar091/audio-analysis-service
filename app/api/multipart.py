"""Strict streaming multipart parser that never spools caller audio to disk."""

from dataclasses import dataclass, field
from uuid import UUID

from fastapi import Request
from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header

from app.core.config import Settings
from app.core.exceptions import AppError
from app.schemas.request import ParsedAnalyzeRequest


@dataclass(slots=True)
class _Part:
    header_field: bytearray = field(default_factory=bytearray)
    header_value: bytearray = field(default_factory=bytearray)
    headers: dict[bytes, bytes] = field(default_factory=dict)
    name: str | None = None
    filename: str | None = None
    content_type: str | None = None
    data: bytearray = field(default_factory=bytearray)


class _AnalyzeMultipartState:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.current = _Part()
        self.contact_id_bytes: bytearray | None = None
        self.audio: bytearray | None = None
        self.filename: str | None = None
        self.audio_content_type: str | None = None

    def on_part_begin(self) -> None:
        self.current = _Part()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self.current.header_field.extend(data[start:end])

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self.current.header_value.extend(data[start:end])

    def on_header_end(self) -> None:
        key = bytes(self.current.header_field).strip().lower()
        value = bytes(self.current.header_value).strip()
        if not key or len(key) > 128 or len(value) > 4096:
            raise AppError(400, "MALFORMED_MULTIPART", "The multipart request is malformed.")
        self.current.headers[key] = value
        self.current.header_field.clear()
        self.current.header_value.clear()

    def on_headers_finished(self) -> None:
        disposition = self.current.headers.get(b"content-disposition")
        if disposition is None:
            raise AppError(400, "MALFORMED_MULTIPART", "The multipart request is malformed.")
        _, options = parse_options_header(disposition)
        raw_name = options.get(b"name")
        if raw_name is None:
            raise AppError(400, "MALFORMED_MULTIPART", "The multipart request is malformed.")
        self.current.name = raw_name.decode("utf-8", errors="strict")

        raw_filename = options.get(b"filename")
        if raw_filename is not None:
            self.current.filename = raw_filename.decode("utf-8", errors="replace")[:255]
        raw_content_type = self.current.headers.get(b"content-type")
        if raw_content_type is not None:
            self.current.content_type = raw_content_type.decode(
                "ascii", errors="replace"
            )[:255]

        if self.current.name not in {"contact_id", "audio"}:
            raise AppError(
                400, "UNEXPECTED_FIELD", "The multipart request contains an unexpected field."
            )
        if self.current.name == "contact_id" and self.contact_id_bytes is not None:
            raise AppError(400, "DUPLICATE_FIELD", "The contact_id field must occur once.")
        if self.current.name == "audio":
            if self.audio is not None:
                raise AppError(400, "DUPLICATE_FIELD", "The audio field must occur once.")
            if raw_filename is None:
                raise AppError(400, "MISSING_AUDIO", "An uploaded audio file is required.")

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if self.current.name == "audio":
            if len(self.current.data) + len(chunk) > self.settings.max_upload_bytes:
                raise AppError(
                    413,
                    "UPLOAD_TOO_LARGE",
                    "The audio upload exceeds the configured size limit.",
                )
        elif len(self.current.data) + len(chunk) > self.settings.max_contact_id_bytes:
            raise AppError(422, "INVALID_CONTACT_ID", "contact_id must be a valid UUID.")
        self.current.data.extend(chunk)

    def on_part_end(self) -> None:
        if self.current.name == "contact_id":
            self.contact_id_bytes = self.current.data
        elif self.current.name == "audio":
            self.audio = self.current.data
            self.filename = self.current.filename
            self.audio_content_type = self.current.content_type

    def on_end(self) -> None:
        return None

    @property
    def callbacks(self) -> dict[str, object]:
        return {
            "on_part_begin": self.on_part_begin,
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
            "on_part_data": self.on_part_data,
            "on_part_end": self.on_part_end,
            "on_end": self.on_end,
        }

    def clear(self, preserve_audio: bool = False) -> None:
        self.current.header_field.clear()
        self.current.header_value.clear()
        if not preserve_audio or self.current.data is not self.audio:
            self.current.data.clear()
        if self.contact_id_bytes is not None:
            self.contact_id_bytes.clear()
        if not preserve_audio and self.audio is not None:
            self.audio.clear()


async def parse_analyze_multipart(
    request: Request, settings: Settings
) -> ParsedAnalyzeRequest:
    """Parse the exact v1 multipart contract from the ASGI request stream."""

    content_type = request.headers.get("content-type", "")
    media_type, options = parse_options_header(content_type.encode("latin-1"))
    if media_type != b"multipart/form-data" or b"boundary" not in options:
        raise AppError(
            400,
            "MALFORMED_REQUEST",
            "Content-Type must be multipart/form-data with a boundary.",
        )

    total_limit = settings.max_upload_bytes + settings.max_multipart_overhead_bytes
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > total_limit:
                raise AppError(
                    413,
                    "UPLOAD_TOO_LARGE",
                    "The audio upload exceeds the configured size limit.",
                )
        except ValueError as exc:
            raise AppError(
                400, "MALFORMED_REQUEST", "The request Content-Length is invalid."
            ) from exc

    state = _AnalyzeMultipartState(settings)
    parser = MultipartParser(options[b"boundary"], state.callbacks)
    total_received = 0
    success = False
    try:
        async for chunk in request.stream():
            total_received += len(chunk)
            if total_received > total_limit:
                raise AppError(
                    413,
                    "UPLOAD_TOO_LARGE",
                    "The audio upload exceeds the configured size limit.",
                )
            parser.write(chunk)
        parser.finalize()

        if state.contact_id_bytes is None:
            raise AppError(400, "MISSING_CONTACT_ID", "The contact_id field is required.")
        if state.audio is None:
            raise AppError(400, "MISSING_AUDIO", "An uploaded audio file is required.")
        if not state.audio:
            raise AppError(400, "EMPTY_AUDIO", "The uploaded audio file is empty.")
        try:
            contact_id_text = state.contact_id_bytes.decode("ascii").strip()
            contact_id = UUID(contact_id_text)
        except (UnicodeDecodeError, ValueError) as exc:
            raise AppError(
                422, "INVALID_CONTACT_ID", "contact_id must be a valid UUID."
            ) from exc

        success = True
        return ParsedAnalyzeRequest(
            contact_id=contact_id,
            audio=state.audio,
            filename=state.filename,
            content_type=state.audio_content_type,
        )
    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            400, "MALFORMED_MULTIPART", "The multipart request is malformed."
        ) from exc
    finally:
        state.clear(preserve_audio=success)
