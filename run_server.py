import os
import logging
import signal
import sys
from pyngrok import ngrok as pyngrok
from whisper_live.server import TranscriptionServer
from whisper_live.pipeline_handler import (
    on_statement_finalized,
    finalize_meeting,
)
from whisper_live.cli_arguments import args

logging.basicConfig(level=logging.INFO)

ACTIVE_MEETINGS = set()


def handle_statement(segment, meeting_id):
    ACTIVE_MEETINGS.add(meeting_id)
    on_statement_finalized(segment, meeting_id)


def shutdown_handler(signum, frame):
    logging.info("[SYSTEM] shutting down")

    for meeting_id in list(ACTIVE_MEETINGS):
        try:
            finalize_meeting(meeting_id)

        except Exception as e:
            logging.exception(f"[SYSTEM] failed to finalize {meeting_id}: {e}")

    sys.exit(0)


if __name__ == "__main__":
    # graceful shutdown hooks
    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    # ngrok
    pyngrok.set_auth_token(os.environ.get("NGROK_AUTHTOKEN"))

    tunnel = pyngrok.connect(
        args.port,
        proto="http",
        hostname=os.environ.get("NGROK_DOMAIN"),
    )

    logging.info(f"Ngrok tunnel: {tunnel.public_url}")

    # whisper server

    server = TranscriptionServer()

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
        on_statement_finalized=handle_statement,
    )
