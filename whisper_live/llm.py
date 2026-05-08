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
    existing_memory=None,
    callback=None,
):
    def run():
        conn = None
        memory_snapshot = existing_memory or []

        try:
            new_text_clean = (new_text or "").strip()

            if len(new_text_clean) < 120:
                return

            if new_text_clean.count("...") > 2:
                return

            filler_words = ["um", "uh", "yeah", "right", "mmm"]
            if len(new_text_clean) < 200 and all(
                w in new_text_clean.lower() for w in filler_words
            ):
                return

            context_text = "\n".join(
                f"- {s.get('text', '')}" for s in context if s and s.get("text")
            )

            memory_text = "\n".join(
                f"- {item.get('type', '')}: {item.get('items', [])}"
                for item in memory_snapshot
            )

            prompt = f"""
You are a strict meeting state extractor.

Only extract explicit, complete, and meaningful information.

If uncertain, return no new data.

Return JSON only.

EMPTY FORMAT:
{{
  "type": "meeting_update",
  "summary": "",
  "has_new_data": false,
  "topics": [],
  "data": []
}}

EXISTING MEMORY:
{memory_text}

RECENT CONTEXT:
{context_text}

NEW TRANSCRIPT:
{new_text_clean}
"""

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return only valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "stream": False,
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }

            conn = http.client.HTTPConnection(host, int(port), timeout=20)

            conn.request(
                "POST",
                "/v1/chat/completions",
                body=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )

            response = conn.getresponse()

            if response.status != 200:
                logging.error(response.read().decode())
                return

            raw = response.read().decode()
            data = json.loads(raw)

            output = data["choices"][0]["message"]["content"].strip()

            try:
                parsed = json.loads(output)
            except Exception:
                logging.error(f"Invalid JSON:\n{output}")
                return

            if not parsed.get("has_new_data"):
                return

            if not parsed.get("data"):
                return

            if callback:
                callback(parsed)

        except Exception as e:
            logging.exception(f"LLM async call failed: {e}")

        finally:
            if conn:
                conn.close()

    threading.Thread(target=run, daemon=True).start()
