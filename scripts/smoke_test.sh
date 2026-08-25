#!/usr/bin/env bash
set -euo pipefail

service_url="${SERVICE_URL:-http://localhost:8000}"
sample_path="${1:-}"
temporary_dir="$(mktemp -d)"

cleanup() {
  if [[ -n "${temporary_dir}" && -d "${temporary_dir}" ]]; then
    rm -rf -- "${temporary_dir}"
  fi
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 120); do
  if curl --silent --fail "${service_url}/ready" >/dev/null; then
    break
  fi
  sleep 2
done

if ! curl --silent --fail "${service_url}/ready" >/dev/null; then
  echo "Service did not become ready: ${service_url}/ready" >&2
  exit 1
fi

if [[ -z "${sample_path}" ]]; then
  sample_path="${temporary_dir}/synthetic.wav"
  python3 - "${sample_path}" <<'PY'
import math
import struct
import sys
import wave

path = sys.argv[1]
sample_rate = 16_000
duration_seconds = 3
with wave.open(path, "wb") as output:
    output.setnchannels(1)
    output.setsampwidth(2)
    output.setframerate(sample_rate)
    frames = bytearray()
    for index in range(sample_rate * duration_seconds):
        value = int(0.12 * 32767 * math.sin(2 * math.pi * 220 * index / sample_rate))
        frames.extend(struct.pack("<h", value))
    output.writeframes(frames)
PY
fi

if [[ ! -f "${sample_path}" ]]; then
  echo "Audio sample does not exist: ${sample_path}" >&2
  exit 1
fi

contact_id="$(python3 -c 'import uuid; print(uuid.uuid4())')"
response_file="${temporary_dir}/response.json"
http_status="$(curl --silent --show-error \
  --output "${response_file}" \
  --write-out '%{http_code}' \
  --request POST "${service_url}/analyze" \
  --form "contact_id=${contact_id}" \
  --form "audio=@${sample_path}")"

python3 -m json.tool "${response_file}" || cat "${response_file}"

if [[ "${http_status}" != "200" ]]; then
  echo "Smoke test failed with HTTP ${http_status}" >&2
  exit 1
fi

echo "Smoke test passed (HTTP ${http_status})."
