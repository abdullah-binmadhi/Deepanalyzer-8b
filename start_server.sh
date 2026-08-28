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

echo "🚀 Starting DeepAnalyze Inference Server..."
echo "📦 Model: $MODEL_PATH"
echo "🌐 Port: 8080"

# 2. Hardware-specific Flag Detection
EXTRA_FLAGS=()

# Detect macOS / Apple Silicon
if [[ "$OSTYPE" == "darwin"* ]]; then
  echo "⚡ Hardware Detected: Apple Silicon (Metal Acceleration)"
  EXTRA_FLAGS=(
    -ngl 99
    -c 8192
    -fa on
    -t 4
    -b 2048
    -ub 1024
    --mlock
    --cache-type-k q8_0
    --cache-type-v q8_0
    --cache-reuse 256
  )
else
  # Linux / CUDA / CPU fallback
  echo "⚡ Hardware Detected: Linux/Generic (CUDA/Vulkan/CPU)"
  EXTRA_FLAGS=(
    -ngl 99
    -c 8192
    --cache-type-k q8_0
    --cache-type-v q8_0
  )
fi

exec llama-server -m "$MODEL_PATH" --port 8080 "${EXTRA_FLAGS[@]}" "$@"
