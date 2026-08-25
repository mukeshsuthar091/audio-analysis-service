"""FFmpeg-backed in-memory decoding and waveform validation."""

import asyncio
import logging
import shutil
import struct
import subprocess
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from app.core.config import Settings
from app.core.exceptions import AppError

logger = logging.getLogger(__name__)

FloatWaveform = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class WaveformInfo:
    duration_seconds: float
    peak_amplitude: float
    rms_energy: float
    clipping_ratio: float


def verify_ffmpeg(binary: str, timeout_seconds: float = 5.0) -> bool:
    """Return whether FFmpeg exists and can execute successfully."""

    resolved = shutil.which(binary)
    if resolved is None:
        return False
    try:
        result = subprocess.run(
            [resolved, "-version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


async def decode_audio(audio: bytearray, settings: Settings) -> FloatWaveform:
    """Decode arbitrary self-describing audio to mono 16 kHz float32 PCM."""

    normalized_wav = decode_normalized_wav(audio, settings)
    if normalized_wav is not None:
        logger.debug("normalized_wav_fast_path_used")
        return normalized_wav

    try:
        process = await asyncio.create_subprocess_exec(
            settings.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            "pipe:0",
            "-map_metadata",
            "-1",
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(settings.sample_rate),
            "-acodec",
            "pcm_f32le",
            "-f",
            "f32le",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise AppError(
            503, "FFMPEG_UNAVAILABLE", "Audio decoding is temporarily unavailable."
        ) from exc

    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(input=audio), timeout=settings.decode_timeout_seconds
        )
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise AppError(504, "DECODE_TIMEOUT", "Audio decoding timed out.") from exc
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise

    if process.returncode != 0 or not stdout:
        logger.info(
            "ffmpeg_decode_rejected",
            extra={
                "event_fields": {
                    "return_code": process.returncode,
                    "stderr_bytes": min(len(stderr), 4096),
                }
            },
        )
        raise AppError(
            415, "INVALID_AUDIO", "The supplied audio could not be decoded."
        )

    if len(stdout) % np.dtype("<f4").itemsize:
        raise AppError(
            415, "INVALID_AUDIO", "The supplied audio could not be decoded."
        )

    # Copy into an owned mutable array so request cleanup can overwrite it.
    return np.frombuffer(stdout, dtype="<f4").astype(np.float32, copy=True)


def decode_normalized_wav(
    audio: bytearray, settings: Settings
) -> FloatWaveform | None:
    """Decode normalized RIFF/WAVE PCM directly, or return ``None`` for FFmpeg.

    The fast path accepts only mono 16 kHz PCM16 or IEEE float32 data. It parses
    RIFF chunks rather than trusting a filename/MIME type. Any unsupported or
    malformed WAV falls back to FFmpeg so the existing codec behavior remains.
    """

    if len(audio) < 12 or audio[:4] != b"RIFF" or audio[8:12] != b"WAVE":
        return None

    format_chunk: memoryview | None = None
    data_chunk: memoryview | None = None
    view = memoryview(audio)
    offset = 12
    try:
        while offset + 8 <= len(view):
            chunk_id = bytes(view[offset : offset + 4])
            chunk_size = struct.unpack_from("<I", view, offset + 4)[0]
            chunk_start = offset + 8
            chunk_end = chunk_start + chunk_size
            if chunk_end > len(view):
                return None
            if chunk_id == b"fmt " and format_chunk is None:
                format_chunk = view[chunk_start:chunk_end]
            elif chunk_id == b"data" and data_chunk is None:
                data_chunk = view[chunk_start:chunk_end]
            offset = chunk_end + (chunk_size & 1)

        if format_chunk is None or data_chunk is None or len(format_chunk) < 16:
            return None
        audio_format, channels, sample_rate, _, block_align, bits_per_sample = (
            struct.unpack_from("<HHIIHH", format_chunk, 0)
        )
        if audio_format == 0xFFFE and len(format_chunk) >= 40:
            # WAVE_FORMAT_EXTENSIBLE stores the real format in the sub-format GUID.
            audio_format = struct.unpack_from("<H", format_chunk, 24)[0]
        if channels != 1 or sample_rate != settings.sample_rate:
            return None

        if audio_format == 1 and bits_per_sample == 16 and block_align == 2:
            if not data_chunk or len(data_chunk) % 2:
                return None
            pcm = np.frombuffer(data_chunk, dtype="<i2")
            return (pcm.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
        if audio_format == 3 and bits_per_sample == 32 and block_align == 4:
            if not data_chunk or len(data_chunk) % 4:
                return None
            return np.frombuffer(data_chunk, dtype="<f4").astype(np.float32, copy=True)
        return None
    except (BufferError, ValueError, struct.error):
        return None
    finally:
        if format_chunk is not None:
            format_chunk.release()
        if data_chunk is not None:
            data_chunk.release()
        view.release()


def validate_waveform(waveform: FloatWaveform, settings: Settings) -> WaveformInfo:
    """Validate decoded samples and return basic signal measurements."""

    if waveform.ndim != 1 or waveform.size == 0:
        raise AppError(415, "INVALID_AUDIO", "The decoded audio is empty.")
    if not np.isfinite(waveform).all():
        raise AppError(415, "INVALID_AUDIO", "The decoded audio contains invalid samples.")

    duration = waveform.size / settings.sample_rate
    if duration < settings.min_duration_seconds or duration > settings.max_duration_seconds:
        raise AppError(
            400,
            "AUDIO_DURATION_OUT_OF_RANGE",
            (
                "Decoded audio duration must be between "
                f"{settings.min_duration_seconds:g} and {settings.max_duration_seconds:g} seconds."
            ),
        )

    absolute = np.abs(waveform)
    peak = float(np.max(absolute))
    if not np.isfinite(peak) or peak > 64.0:
        raise AppError(415, "INVALID_AUDIO", "The decoded audio amplitude is invalid.")

    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    clipping_ratio = float(np.mean(absolute >= settings.clipping_amplitude))
    return WaveformInfo(
        duration_seconds=duration,
        peak_amplitude=peak,
        rms_energy=rms,
        clipping_ratio=clipping_ratio,
    )


def clear_waveform(waveform: FloatWaveform | None) -> None:
    """Best-effort overwrite of an owned mutable waveform buffer."""

    if waveform is not None and waveform.flags.writeable:
        waveform.fill(0.0)
