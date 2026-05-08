import threading
import http.client
import json
import logging


def call_llm_async(
    host,
    port,
    model,
    context,
    new_text,
    callback=None,
):
    def run():
        conn = None

        try:
            context_text = "\n".join([f"- {segment['text']}" for segment in context])

            prompt = f"""
You are monitoring a live meeting transcript.

Your responsibilities:
- maintain awareness of the conversation
- identify important discussion points
- identify decisions
- identify action items
- produce concise realtime updates

Keep responses short and useful.

MEETING CONTEXT:
{context_text}

NEW TRANSCRIPT:
{new_text}
"""

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an AI assistant helping users stay "
                            "synchronized during live meetings."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "stream": False,
                "temperature": 0.3,
            }

            conn = http.client.HTTPConnection(
                host,
                int(port),
                timeout=20,
            )

            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload),
                headers={
                    "Content-Type": "application/json",
                },
            )

            response = conn.getresponse()

            if response.status != 200:
                error_body = response.read().decode()

                logging.error(
                    f"LLM request failed (status={response.status}): {error_body}"
                )
                return

            raw = response.read().decode()

            data = json.loads(raw)

            output = data["choices"][0]["message"]["content"].strip()

            logging.info(
                "\n"
                "================ LLM RESPONSE ================\n"
                f"{output}\n"
                "=============================================="
            )

            if callback:
                callback(output)

        except Exception as e:
            logging.exception(f"LLM async call failed: {e}")

        finally:
            if conn:
                conn.close()

    threading.Thread(
        target=run,
        daemon=True,
    ).start()
