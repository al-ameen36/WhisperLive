import os
import time
import json
import requests
import threading
from pyngrok import ngrok
from whisper_live.server import TranscriptionServer

# --- CONFIGURATION ---
PORT = 9090
NGROK_TOKEN = os.environ.get('NGROK_AUTH_TOKEN')
HF_TOKEN = os.environ.get('HF_TOKEN')
# Use the URL from your vLLM notebook
VLLM_API_URL = "https://exception-shelve-backer.ngrok-free.dev/v1/chat/completions"
MODEL_ID = "casperhansen/llama-3-8b-instruct-awq"

os.environ["HF_TOKEN"] = HF_TOKEN
ngrok.set_auth_token(NGROK_TOKEN)

# ---------------- CLEANUP ----------------
print("🧹 Killing old process on port...")
os.system(f"fuser -k {PORT}/tcp")

# ---------------- SERVER ----------------
def run_whisper_server():
    try:
        print("🚀 Starting WhisperLive server...")

        server = TranscriptionServer()
        server.use_vad=True

        server.run(
            host="0.0.0.0",
            port=PORT,
        )

    except Exception as e:
        print(f"❌ SERVER CRASHED: {e}")


# Start server thread
server_thread = threading.Thread(target=run_whisper_server, daemon=True)
server_thread.start()

# ---------------- WAIT FOR INIT ----------------
print("⏳ Waiting for model to load...")
time.sleep(15)

# ---------------- NGROK ----------------
try:
    print("🌐 Creating ngrok tunnel...")

    # Clean old tunnels safely
    for t in ngrok.get_tunnels():
        ngrok.disconnect(t.public_url)

    public_tunnel = ngrok.connect(PORT, "http", domain="artistic-preferably-sole.ngrok-free.app")

    ws_url = public_tunnel.public_url.replace("https", "wss")

    print("\n✅ SERVER READY")
    print(f"HTTP URL: {public_tunnel.public_url}")
    print(f"WS URL:   {ws_url}")

    # IMPORTANT: do NOT block Colab indefinitely with join()
    # Instead, keep process alive safely
    while True:
        time.sleep(60)

except KeyboardInterrupt:
    print("\n🛑 Shutting down...")

except Exception as e:
    print(f"❌ NGROK ERROR: {e}")