# WhisperLive + Synapse Fork

A realtime meeting intelligence system built on top of [WhisperLive](https://github.com/collabora/WhisperLive?utm_source=chatgpt.com).

This fork transforms WhisperLive from a realtime transcription server into a live organizational memory system for meetings.

Instead of only generating transcripts, the system continuously maintains structured meeting state in realtime:

* decisions
* action items
* commitments
* risks
* questions
* follow-ups
* evolving conversational context

The goal is not just speech-to-text.

The goal is preserving organizational understanding while conversations are happening.

---

# What We Added

## Realtime Structured Meeting Intelligence

The system continuously analyzes finalized transcript segments and converts them into structured insights.

Supported insight types:

```json
{
  "type": "update" | "flag"
}
```

### `update`

General conversational understanding:

* questions
* concerns
* decisions
* opinions
* evolving context
* important discussion changes

### `flag`

Commitments and accountability signals:

* assigned responsibilities
* ownership
* deadlines
* agreements
* obligations
* delegated tasks

---

# Persistent Meeting Memory

We added a live memory state management system that continuously tracks meeting state.

The memory layer maintains:

* rolling transcript context
* recent insights
* structured meeting state
* duplicate prevention
* incremental context updates

Unlike traditional meeting tools that repeatedly summarize the same content, this system maintains evolving memory throughout the meeting.

Core implementation:

```python
class MemoryStore:
```

Features:

* rolling context windows
* meeting session isolation
* transcript state tracking
* insight memory
* realtime context accumulation
* cleanup/finalization handling

---

# Buffered LLM Processing

Original WhisperLive processed transcription only.

We added buffered AI processing logic to avoid excessive LLM calls.

The system now:

* accumulates transcript segments
* waits until enough conversational context exists
* triggers intelligent analysis in batches
* prevents noisy per-segment inference

Current batching strategy:

```python
MIN_CHARS = 300
MAX_WAIT = 30
```

LLM calls trigger when:

* enough conversational context accumulates
  OR
* timeout threshold is reached

This significantly improves:

* coherence
* context awareness
* inference quality
* GPU efficiency

---

# Context-Aware LLM Pipeline

We implemented a structured LLM pipeline using locally served Llama models through [vLLM](https://github.com/vllm-project/vllm?utm_source=chatgpt.com).

The pipeline:

* receives rolling transcript context
* receives existing meeting memory
* avoids duplicate insights
* extracts structured realtime intelligence
* returns strict JSON outputs

Structured outputs include:

```json
{
  "type": "flag",
  "summary": "...",
  "assignee": "...",
  "assigner": "...",
  "commitment": "...",
  "implication": "..."
}
```

---

# Supabase Persistence Layer

We added persistent storage using [Supabase](https://supabase.com?utm_source=chatgpt.com).

Stored data includes:

* finalized transcripts
* structured insights
* meeting history
* commitments
* meeting memory state

Tables currently used:

* `transcripts`
* `insights`

---

# Meeting Finalization System

Meetings now properly finalize on:

* disconnect
* shutdown
* termination signals

Finalization includes:

* transcript persistence
* insight persistence
* memory cleanup
* session cleanup

Graceful shutdown support added through:

* `SIGINT`
* `SIGTERM`

---

# Authentication Layer

We added authenticated meeting access using Supabase Auth.

Features:

* token validation
* authenticated websocket connections
* protected REST endpoints
* per-user meeting ownership

---

# OpenAI-Compatible REST API

The fork now exposes an OpenAI-style transcription API.

Endpoint:

```bash
POST /v1/audio/transcriptions
```

Supports:

* JSON
* verbose_json
* text
* SRT
* VTT

Compatible with:

* OpenAI-style clients
* external applications
* browser integrations

---

# Realtime Organizational Memory Architecture

Current architecture:

```text
Browser Audio
    ↓
Whisper Transcription
    ↓
Buffered Context Accumulation
    ↓
Context-Aware LLM Processing
    ↓
Structured Insight Extraction
    ↓
Persistent Meeting Memory
    ↓
Realtime Organizational State
```

---

# Browser-Based Audio Capture

The frontend supports:

* microphone capture
* browser tab audio capture

Important behavior:

* tab sharing remains fixed
* users can freely navigate other tabs
* no switching occurs during capture

This makes it usable during:

* Google Meet
* Zoom
* browser-based meetings
* presentations

Users currently choose:

* mic OR tab audio

(not both simultaneously yet)

---

# Current Capabilities

## Implemented

* realtime transcription
* rolling conversational memory
* structured insight extraction
* commitment detection
* action item detection
* duplicate prevention
* buffered LLM inference
* Supabase persistence
* OpenAI-compatible API
* authenticated sessions
* meeting finalization
* realtime meeting state management

---

# Planned Features

## Diarization

Identify who said what during meetings.

## Meeting Replays

Replay meetings as if they were happening live.

## Autonomous Follow-Ups

AI agents remind users about commitments and deadlines.

## Collaboration Integrations

Integrations with:

* Slack
* ClickUp
* Linear
* Notion
* Jira

Example:
Tasks mentioned during meetings can automatically become assigned work items.

## Persistent Organizational Memory

Long-term searchable team intelligence across meetings.

---

# Infrastructure

## GPU Deployment

This project was deployed using AMD GPU infrastructure.

We used the pre-configured ROCm + vLLM image provided through the AMD cloud environment to serve local Llama models efficiently.

---

# Setup

## Clone Repository

```bash
git clone https://github.com/al-ameen36/WhisperLive
cd WhisperLive
git checkout dev
```

---

# Install Dependencies

```bash
pip install "fastapi[standard]" pyngrok openai-whisper onnxruntime-rocm supabase
```

---

# Environment Variables

Export the required environment variables.

Check:

```bash
.env.example
```

Required variables include:

* `SUPABASE_URL`
* `SUPABASE_KEY`
* `SUPABASE_SERVICE_ROLE_KEY`
* `NGROK_AUTHTOKEN`
* `NGROK_DOMAIN`

---

# Serve Local Llama Model

Using vLLM:

```bash
HIP_VISIBLE_DEVICES=0 vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --gpu-memory-utilization 0.8 \
    --swap-space 16 \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --host 0.0.0.0 \
    --port 3000 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 8192 \
    --max-model-len 8192 \
    --distributed-executor-backend "mp"
```

---

# Start Server

In a separate terminal:

```bash
python run_server.py --enable_llm --llm_model meta-llama/Llama-3.3-70B-Instruct
```

---

# Default Runtime Arguments

```bash
# Network
--port 9090
--cors-origins None

# Backend
--backend whisper
--faster_whisper_custom_model_path None
--trt_model_path None
--trt_multilingual false
--trt_py_session false
--cache_path ~/.cache/whisper-live/

# Client limits
--max_clients 4
--max_connection_time 86400
--no_single_model false

# Voice Activity Detection
--no_vad false

# Audio input
--raw_pcm_input false

# Batch inference
--batch_inference false
--batch_max_size 8
--batch_window_ms 50

# REST API
--enable_rest false
--rest_port 8000

# Performance
--omp_num_threads 1

# LLM
--enable_llm false
--llm_host localhost
--llm_port 3000
--llm_buffer_size 3
--llm_model meta-llama/Meta-Llama-3-8B-Instruct
```

---

# Tech Stack

Core technologies used:

* [WhisperLive](https://github.com/collabora/WhisperLive?utm_source=chatgpt.com)
* [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper?utm_source=chatgpt.com)
* [vLLM](https://github.com/vllm-project/vllm?utm_source=chatgpt.com)
* [Meta Llama 3](https://www.llama.com?utm_source=chatgpt.com)
* [Supabase](https://supabase.com?utm_source=chatgpt.com)
* [FastAPI](https://fastapi.tiangolo.com?utm_source=chatgpt.com)
* [Pyngrok](https://pyngrok.readthedocs.io?utm_source=chatgpt.com)

---

# Vision

Meetings are where organizations think.

This project turns conversations into persistent organizational memory in realtime.
