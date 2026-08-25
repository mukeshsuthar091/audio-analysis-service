"""Real FFmpeg decoding integration tests."""

import asyncio
import shutil
import struct
import subprocess

import numpy as np
import pytest

from app.audio.decoder import (
    clear_waveform,
    decode_audio,
    decode_normalized_wav,
    validate_waveform,
)
from tests.conftest import make_settings, make_wav

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is unavailable"),
]


@pytest.mark.asyncio
async def test_decodes_wav_to_16khz_float() -> None:
    settings = make_settings()
    waveform = await decode_audio(bytearray(make_wav(sample_rate=8_000)), settings)
    try:
        info = validate_waveform(waveform, settings)
        assert waveform.dtype.name == "float32"
        assert waveform.ndim == 1
        assert info.duration_seconds == pytest.approx(3.0, abs=0.01)
    finally:
        clear_waveform(waveform)


@pytest.mark.asyncio
async def test_decodes_mp3() -> None:
    settings = make_settings()
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "wav",
            "-i",
            "pipe:0",
            "-f",
            "mp3",
            "pipe:1",
        ],
        input=make_wav(),
        capture_output=True,
        check=True,
        timeout=10,
    )
    audio = bytearray(result.stdout)
    waveform = await decode_audio(audio, settings)
    try:
        info = validate_waveform(waveform, settings)
        assert info.duration_seconds == pytest.approx(3.0, abs=0.1)
    finally:
        audio.clear()
        clear_waveform(waveform)


@pytest.mark.asyncio
async def test_normalized_pcm_wav_bypasses_ffmpeg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("FFmpeg must not run for normalized PCM WAV")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_if_called)
    waveform = await decode_audio(
        bytearray(make_wav(duration_seconds=1.0, sample_rate=16_000)),
        make_settings(),
    )
    try:
        assert waveform.dtype == np.float32
        assert waveform.size == 16_000
        assert np.isfinite(waveform).all()
    finally:
        clear_waveform(waveform)


def test_normalized_float_wav_uses_direct_decoder() -> None:
    samples = np.array([-1.0, -0.25, 0.0, 0.5, 1.0], dtype="<f4")
    fmt_chunk = struct.pack("<HHIIHH", 3, 1, 16_000, 64_000, 4, 32)
    data_chunk = samples.tobytes()
    riff_size = 4 + 8 + len(fmt_chunk) + 8 + len(data_chunk)
    audio = bytearray(
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack("<I", len(fmt_chunk))
        + fmt_chunk
        + b"data"
        + struct.pack("<I", len(data_chunk))
        + data_chunk
        + b"\x00"
    )

    waveform = decode_normalized_wav(audio, make_settings())

    assert waveform is not None
    try:
        np.testing.assert_allclose(waveform, samples)
    finally:
        clear_waveform(waveform)


def test_non_normalized_wav_falls_back_to_ffmpeg() -> None:
    waveform = decode_normalized_wav(
        bytearray(make_wav(duration_seconds=1.0, sample_rate=8_000)),
        make_settings(),
    )

    assert waveform is None
