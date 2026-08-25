# Local audio samples

Do not commit caller audio, private recordings, or copyrighted clips to this directory.

To record a local WAV on macOS, open QuickTime Player, choose **File → New Audio Recording**, record 3–5 seconds of speech, export it, and normalize it if needed:

```bash
ffmpeg -i recording.m4a -ac 1 -ar 16000 -c:a pcm_s16le samples/contact.wav
```

On Linux with ALSA:

```bash
arecord -d 5 -f S16_LE -r 16000 -c 1 samples/contact.wav
```

Alternatively, use a public-domain or Creative Commons speech sample whose license permits local evaluation. Keep the attribution and license information outside this repository unless redistribution is explicitly allowed.

The smoke script can generate a non-speech synthetic WAV automatically. That checks transport, decoding, VAD, and graceful `insufficient` handling; pass a local speech recording to exercise model inference:

```bash
./scripts/smoke_test.sh samples/contact.wav
```

