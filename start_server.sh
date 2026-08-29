#!/usr/bin/env bash
# DeepAnalyze Unified Local Inference Server Launcher
# Automatically detects model location and hardware acceleration (Apple Silicon Metal / NVIDIA CUDA)

# 1. Resolve Model Path
MODEL_CANDIDATES=(
  "$DEEPANALYZE_MODEL_PATH"
  "./models/deepanalyze-8b-q4_k_m.gguf"
  "$HOME/Desktop/deepanalyze/models/deepanalyze-8b-q4_k_m.gguf"
  "$HOME/models/deepanalyze-8b-q4_k_m.gguf"
  "./models/deepanalyze-8b.gguf"
)

MODEL_PATH=""
for path in "${MODEL_CANDIDATES[@]}"; do
  if [ -n "$path" ] && [ -f "$path" ]; then
    MODEL_PATH="$path"
    break
  fi
done

if [ -z "$MODEL_PATH" ]; then
  echo "❌ Error: Could not find DeepAnalyze GGUF model file."
  echo "Please place your model in ./models/ or set export DEEPANALYZE_MODEL_PATH='/path/to/model.gguf'"
  exit 1
fi

# 2. Resolve Speculative Draft Model (Qwen2.5-Coder-1.5B)
DRAFT_CANDIDATES=(
  "$DEEPANALYZE_DRAFT_MODEL_PATH"
  "./models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
  "$HOME/Desktop/deepanalyze/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
  "$HOME/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
)

DRAFT_PATH=""
for path in "${DRAFT_CANDIDATES[@]}"; do
  if [ -n "$path" ] && [ -f "$path" ]; then
    DRAFT_PATH="$path"
    break
  fi
done

SPECULATIVE_FLAGS=()
if [ -n "$DRAFT_PATH" ]; then
  echo "⚡ Speculative Draft Model (2.5x Speedup): $DRAFT_PATH"
  SPECULATIVE_FLAGS=(-md "$DRAFT_PATH" --spec-draft-n-max 8)
fi

echo "🚀 Starting DeepAnalyze Inference Server..."
echo "📦 Primary Target Model: $MODEL_PATH"
echo "🌐 Endpoint: http://127.0.0.1:8080"

# 3. Hardware-specific Flag Detection
EXTRA_FLAGS=()

# Detect macOS / Apple Silicon
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "⚡ Hardware Detected: Apple Silicon (Metal Acceleration)"
  EXTRA_FLAGS=(
    -ngl 99
    -c 16384
    -fa on
    -t 4
    --cache-type-k q8_0
    --cache-type-v q8_0
    --prompt-cache ./models/deepanalyze.cache
    --prompt-cache-all
    -a deepanalyze-8b
    --min-p 0.05
  )
else
  # Linux / CUDA / CPU fallback
  echo "⚡ Hardware Detected: Linux/Generic (CUDA/Vulkan/CPU)"
  EXTRA_FLAGS=(
    -ngl 99
    -c 16384
    --cache-type-k q8_0
    --cache-type-v q8_0
    --prompt-cache ./models/deepanalyze.cache
    --prompt-cache-all
    -a deepanalyze-8b
    --min-p 0.05
  )
fi

exec llama-server -m "$MODEL_PATH" --host 127.0.0.1 --port 8080 "${SPECULATIVE_FLAGS[@]}" "${EXTRA_FLAGS[@]}" "$@"
