#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHISPER_COMMIT="592feef04a1802b18cbeffd0fd0eb5d02570c2ec"
STAGE="$ROOT/src-tauri/resources/rubai"
WORK_PARENT="${RUBAI_BUILD_TMPDIR:-${TMPDIR:-/tmp}}"
EXPECTED_MODEL_SHA256="${RUBAI_EXPECTED_SHA256:-1b02df434902015e1464611a7748927e42fcb55c49791c85239cf713c8edc1a3}"

mkdir -p "$WORK_PARENT"
WORK="$(mktemp -d "$WORK_PARENT/handy-rubai-linux.XXXXXX")"

cleanup() {
  rm -rf "$WORK" "$STAGE"
}
trap cleanup EXIT

for tool in git cmake bun cargo glslc sha256sum; do
  command -v "$tool" >/dev/null || {
    echo "Missing required tool: $tool" >&2
    exit 1
  }
done

MODEL_PATH="${RUBAI_MODEL_PATH:-$HOME/.local/share/com.pais.handy/models/ggml-rubaistt-v2-medium-q8_0.bin}"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Set RUBAI_MODEL_PATH to the RubaiSTT v2 Medium Q8_0 model." >&2
  exit 1
fi

MODEL_SHA256="$(sha256sum "$MODEL_PATH" | awk '{print $1}')"
if [[ "$MODEL_SHA256" != "$EXPECTED_MODEL_SHA256" ]]; then
  echo "Unexpected Rubai model checksum: $MODEL_SHA256" >&2
  echo "Expected: $EXPECTED_MODEL_SHA256" >&2
  echo "Set RUBAI_EXPECTED_SHA256 only after verifying a deliberate model replacement." >&2
  exit 1
fi

git clone --quiet https://github.com/ggerganov/whisper.cpp.git "$WORK/whisper.cpp"
git -C "$WORK/whisper.cpp" checkout --quiet "$WHISPER_COMMIT"

cmake \
  -S "$ROOT/src-tauri/rubai-helper" \
  -B "$WORK/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_CPP_DIR="$WORK/whisper.cpp" \
  -DGGML_NATIVE=ON
cmake --build "$WORK/build" --config Release --parallel --target rubai-helper

mkdir -p "$STAGE"
install -m 755 "$WORK/build/rubai-helper" "$STAGE/rubai-helper"
install -m 644 "$MODEL_PATH" "$STAGE/ggml-rubaistt-v2-medium-q8_0.bin"

cd "$ROOT"
bun install --frozen-lockfile
CMAKE_POLICY_VERSION_MINIMUM=3.5 bun run tauri build --bundles deb

find "$ROOT/src-tauri/target/release/bundle/deb" -maxdepth 1 -type f -name '*.deb' -print
