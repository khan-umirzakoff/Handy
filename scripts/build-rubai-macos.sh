#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHISPER_COMMIT="592feef04a1802b18cbeffd0fd0eb5d02570c2ec"
STAGE="$ROOT/src-tauri/resources/rubai"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/handy-rubai.XXXXXX")"

cleanup() {
  rm -rf "$WORK" "$STAGE"
}
trap cleanup EXIT

for tool in git cmake bun cargo; do
  command -v "$tool" >/dev/null || {
    echo "Missing required tool: $tool" >&2
    exit 1
  }
done

MODEL_PATH="${RUBAI_MODEL_PATH:-}"
if [[ -z "$MODEL_PATH" ]]; then
  for candidate in \
    "/Applications/Handy.app/Contents/Resources/resources/rubai/ggml-rubaistt-v2-medium-q8_0.bin" \
    "/Applications/Audio-Matnga.app/Contents/Resources/ggml-rubaistt.bin"; do
    if [[ -f "$candidate" ]]; then
      MODEL_PATH="$candidate"
      break
    fi
  done
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Set RUBAI_MODEL_PATH to the RubaiSTT v2 Medium Q8_0 model." >&2
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
cp "$WORK/build/rubai-helper" "$STAGE/rubai-helper"
chmod 755 "$STAGE/rubai-helper"
cp "$MODEL_PATH" "$STAGE/ggml-rubaistt-v2-medium-q8_0.bin"

cd "$ROOT"
bun install --frozen-lockfile
HANDY_FORCE_AI_STUB=1 CMAKE_POLICY_VERSION_MINIMUM=3.5 \
  bun run tauri build --bundles app

echo "Built: $ROOT/src-tauri/target/release/bundle/macos/Handy.app"
