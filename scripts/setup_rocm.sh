#!/bin/bash

set -e

START_VLLM=0

# ----------------------------
# Parse args
# ----------------------------
for arg in "$@"; do
  case $arg in
    --vllm)
      START_VLLM=1
      shift
      ;;
  esac
done

echo "Loading environment variables..."

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

cd ..

# ----------------------------
# Check if vLLM already running
# ----------------------------
VLLM_RUNNING=0

if curl -s http://localhost:3000/v1/models > /dev/null 2>&1; then
  VLLM_RUNNING=1
  echo "vLLM already running on port 3000"
fi

# ----------------------------
# Start vLLM if needed
# ----------------------------
if [ "$START_VLLM" -eq 1 ]; then

  if [ "$VLLM_RUNNING" -eq 1 ]; then
    echo "Skipping vLLM startup (already running)"
  else
    echo "Starting vLLM in background..."

    HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0} \
    nohup vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
      --gpu-memory-utilization 0.8 \
      --swap-space 16 \
      --dtype float16 \
      --tensor-parallel-size 1 \
      --host 0.0.0.0 \
      --port 3000 \
      --max-num-seqs 128 \
      --max-num-batched-tokens 8192 \
      --max-model-len 8192 \
      --distributed-executor-backend mp \
      > vllm.log 2>&1 &

    VLLM_PID=$!
    echo "vLLM started (PID: $VLLM_PID)"

    # ----------------------------
    # Wait for readiness
    # ----------------------------
    echo "Waiting for vLLM to be ready..."

    for i in {1..60}; do
      if curl -s http://localhost:3000/v1/models > /dev/null; then
        echo "vLLM is ready"
        break
      fi

      if [ $i -eq 60 ]; then
        echo "ERROR: vLLM failed to start within timeout"
        exit 1
      fi

      echo "waiting for vLLM... ($i/60)"
      sleep 2
    done
  fi
else
  echo "Skipping vLLM startup (use --vllm to enable)"
fi

# ----------------------------
# Install dependencies
# ----------------------------
echo "Installing dependencies..."
pip install -r WhisperLive/requirements/server_rocm.txt

# ----------------------------
# Start WhisperLive
# ----------------------------
echo "Starting WhisperLive..."

cd WhisperLive
python run_server.py --enable_llm --port 9090