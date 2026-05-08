import os
import logging
from pyngrok import ngrok as pyngrok
from whisper_live.server import TranscriptionServer
from whisper_live.memory import MemoryStore
from whisper_live.llm import call_llm_async
import time
from collections import defaultdict
from arguments import args


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # ── Ngrok ─────────────────────────────────────────────────────────────────

    pyngrok.set_auth_token(os.environ.get("NGROK_AUTHTOKEN"))
    tunnel = pyngrok.connect(
        args.port, proto="http", hostname=os.environ.get("NGROK_DOMAIN")
    )
    logging.info(f"Ngrok tunnel: {tunnel.public_url}")

    # ── Server ────────────────────────────────────────────────────────────────

    server = TranscriptionServer()
    memory = MemoryStore()

    # ── LLM Callback ────────────────────────────────────────────────────────

    LLM_PORT = int(os.environ.get("LLM_PORT", "3000"))
    LLM_HOST = os.environ.get("LLM_HOST", "localhost")
    LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

    LAST_LLM_CALL = defaultdict(lambda: 0)
    PENDING_TEXT = defaultdict(list)

    def should_trigger_llm(meeting_id, text):
        now = time.time()
        last = LAST_LLM_CALL[meeting_id]

        # cooldown (prevents spam)
        if now - last < 6:
            return False

        # strong signal: long speech chunk
        if len(text) > 300:
            return True

        # strong signal: question detected
        if "?" in text:
            return True

        # fallback: buffer-based trigger
        if len(PENDING_TEXT[meeting_id]) >= args.llm_buffer_size:
            return True

        return False

    def flush_pending(meeting_id):
        text = " ".join(PENDING_TEXT[meeting_id]).strip()
        PENDING_TEXT[meeting_id].clear()
        return text

    def on_statement_finalized(segment, meeting_id):
        memory.add(meeting_id, segment)

        if not args.enable_llm:
            return

        text = segment.get("text", "").strip()
        if not text:
            return

        # accumulate streaming text
        PENDING_TEXT[meeting_id].append(text)

        combined_text = " ".join(PENDING_TEXT[meeting_id]).strip()

        # decision gate (ALL logic centralized here)
        if not should_trigger_llm(meeting_id, combined_text):
            return

        # flush only when actually triggering
        context_text = flush_pending(meeting_id)

        text = segment.get("text", "").strip()
        if not text:
            return

        # accumulate streaming text
        PENDING_TEXT[meeting_id].append(text)

        combined_text = " ".join(PENDING_TEXT[meeting_id]).strip()

        # decision gate (ALL logic centralized here)
        if not should_trigger_llm(meeting_id, combined_text):
            return

        # flush only when actually triggering
        context_text = flush_pending(meeting_id)

        context = memory.get_context(meeting_id, 20)

        LAST_LLM_CALL[meeting_id] = time.time()
        LAST_LLM_CALL[meeting_id] = time.time()

        call_llm_async(
            host=LLM_HOST,
            port=LLM_PORT,
            model=LLM_MODEL,
            context=context,
            new_text=context_text,
            existing_memory=memory.get_recent_insights(meeting_id),
        )

    server.run(
        "0.0.0.0",
        port=args.port,
        backend=args.backend,
        faster_whisper_custom_model_path=args.faster_whisper_custom_model_path,
        whisper_tensorrt_path=args.trt_model_path,
        trt_multilingual=args.trt_multilingual,
        trt_py_session=args.trt_py_session,
        single_model=not args.no_single_model,
        max_clients=args.max_clients,
        max_connection_time=args.max_connection_time,
        cache_path=args.cache_path,
        rest_port=args.rest_port,
        enable_rest=args.enable_rest,
        cors_origins=args.cors_origins,
        batch_enabled=args.batch_inference,
        batch_max_size=args.batch_max_size,
        batch_window_ms=args.batch_window_ms,
        raw_pcm_input=args.raw_pcm_input,
        use_vad=not args.no_vad,
        on_statement_finalized=on_statement_finalized,
    )
