import os
import time
import logging
from collections import defaultdict
from pyngrok import ngrok as pyngrok
from whisper_live.server import TranscriptionServer
from whisper_live.memory import MemoryStore
from whisper_live.llm import call_llm_async
from arguments import args

logging.basicConfig(level=logging.INFO)

LLM_HOST = os.environ.get("LLM_HOST", "localhost")
LLM_PORT = int(os.environ.get("LLM_PORT", "3000"))
LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

LAST_LLM_CALL = defaultdict(lambda: 0)

if __name__ == "__main__":
    pyngrok.set_auth_token(os.environ.get("NGROK_AUTHTOKEN"))
    tunnel = pyngrok.connect(
        args.port, proto="http", hostname=os.environ.get("NGROK_DOMAIN")
    )
    logging.info(f"Ngrok tunnel: {tunnel.public_url}")

    server = TranscriptionServer()
    memory = MemoryStore()

    def on_statement_finalized(segment, meeting_id):
        text = (segment.get("text") or "").strip()
        if not text:
            return

        memory.add(meeting_id, segment)

        now = time.time()
        if now - LAST_LLM_CALL[meeting_id] < 15:
            return

        LAST_LLM_CALL[meeting_id] = now
        context = memory.get_context(meeting_id, 25)
        logging.info(f"[LLM] firing with {len(context)} segments")

        call_llm_async(
            host=LLM_HOST,
            port=LLM_PORT,
            model=LLM_MODEL,
            context=context,
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
