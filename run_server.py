import argparse
import os
import logging
from pyngrok import ngrok as pyngrok
from whisper_live.server import TranscriptionServer
from whisper_live.memory import MemoryStore
from whisper_live.llm import call_llm_async


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhisperLive Transcription Server")

    # Network
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=9090,
        help="Websocket port to run the server on.",
    )
    parser.add_argument(
        "--cors-origins",
        type=str,
        default=None,
        help="Comma-separated list of allowed CORS origins. "
        "Defaults to localhost/127.0.0.1 on the WebSocket port.",
    )

    # Backend
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        default="faster_whisper",
        help='Backend to use: ["tensorrt", "faster_whisper", "openvino", "whisper"]',
    )
    parser.add_argument(
        "--faster_whisper_custom_model_path",
        "-fw",
        type=str,
        default=None,
        help="Path to a custom Faster Whisper model.",
    )
    parser.add_argument(
        "--trt_model_path",
        "-trt",
        type=str,
        default=None,
        help="Whisper TensorRT model path.",
    )
    parser.add_argument(
        "--trt_multilingual",
        "-m",
        action="store_true",
        help="TensorRT only: set if model is multilingual.",
    )
    parser.add_argument(
        "--trt_py_session",
        action="store_true",
        help="TensorRT only: use Python session instead of C++ session.",
    )
    parser.add_argument(
        "--cache_path",
        "-c",
        type=str,
        default="~/.cache/whisper-live/",
        help="Path to cache converted ctranslate2 models.",
    )

    # Client limits
    parser.add_argument(
        "--max_clients",
        type=int,
        default=4,
        help="Maximum number of simultaneous clients.",
    )
    parser.add_argument(
        "--max_connection_time",
        type=int,
        default=86400,
        help="Max duration (seconds) a client can stay connected.",
    )
    parser.add_argument(
        "--no_single_model",
        "-nsm",
        action="store_true",
        help="Each connection instantiates its own model (only for custom model paths).",
    )

    # VAD
    parser.add_argument(
        "--no_vad",
        action="store_true",
        help="Disable Voice Activity Detection (VAD). VAD is enabled by default.",
    )

    # Audio input
    parser.add_argument(
        "--raw_pcm_input",
        action="store_true",
        help="Expect raw PCM int16 audio from clients instead of float32.",
    )

    # Batch inference
    parser.add_argument(
        "--batch_inference",
        action="store_true",
        help="Enable batched GPU inference for concurrent sessions.",
    )
    parser.add_argument(
        "--batch_max_size",
        type=int,
        default=8,
        help="Maximum batch size for batched inference (default: 8).",
    )
    parser.add_argument(
        "--batch_window_ms",
        type=int,
        default=50,
        help="Max time (ms) to wait for a batch to fill (default: 50).",
    )

    # REST API
    parser.add_argument(
        "--enable_rest",
        action="store_true",
        help="Enable the OpenAI-compatible REST API endpoint.",
    )
    parser.add_argument(
        "--rest_port", type=int, default=8000, help="Port for the REST API server."
    )

    # Performance
    parser.add_argument(
        "--omp_num_threads",
        "-omp",
        type=int,
        default=1,
        help="Number of threads to use for OpenMP.",
    )

    # LLM
    parser.add_argument(
        "--enable_llm",
        action="store_true",
        help="Enable LLM processing of finalized segments.",
    )
    parser.add_argument(
        "--llm_host", type=str, default="localhost", help="LLM server host."
    )
    parser.add_argument("--llm_port", type=int, default=3000, help="LLM server port.")
    parser.add_argument(
        "--llm_buffer_size",
        type=int,
        default=3,
        help="Number of segments to buffer before calling LLM.",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Model name to use for the LLM call.",
    )

    args = parser.parse_args()

    # ── Validation ────────────────────────────────────────────────────────────
    if args.backend == "tensorrt" and args.trt_model_path is None:
        raise ValueError(
            "Please provide a valid TensorRT model path via --trt_model_path."
        )

    # ── Environment ───────────────────────────────────────────────────────────
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(args.omp_num_threads)

    if not os.environ.get("HF_TOKEN"):
        logging.warning(
            "HF_TOKEN is not set — gated HuggingFace models will be unavailable."
        )

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

    def on_statement_finalized(segment, meeting_id):
        memory.add(meeting_id, segment)

        if not args.enable_llm:
            return

        context = memory.get_context(meeting_id, 20)

        LLM_HOST = os.environ.get("LLM_HOST")
        LLM_PORT = os.environ.get("LLM_PORT")
        LLM_MODEL = os.environ.get("LLM_MODEL")

        call_llm_async(
            host=LLM_HOST,
            port=LLM_PORT,
            model=LLM_MODEL,
            context=context,
            new_text=segment["text"],
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
