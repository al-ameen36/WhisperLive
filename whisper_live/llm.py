import http.client
import json
import logging
import threading


def call_llm_async(host, port, model, context, existing_memory=None, callback=None):
    def run():
        conn = None
        try:
            context_text = "\n".join(
                f"- {s.get('text', '')}" for s in context if s and s.get("text")
            )
            memory_text = "\n".join(
                f"- {item.get('type', '')}: {item.get('items', [])}"
                for item in (existing_memory or [])
            )

            if not context_text.strip():
                logging.warning("[LLM] empty context, skipping")
                return

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

RECENT TRANSCRIPT:
{context_text}
"""

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
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
                logging.error(
                    f"[LLM] bad status {response.status}: {response.read().decode()}"
                )
                return

            data = json.loads(response.read().decode())
            output = data["choices"][0]["message"]["content"].strip()

            try:
                parsed = json.loads(output)
            except Exception:
                logging.error(f"[LLM] invalid JSON response:\n{output}")
                return

            logging.info(f"[LLM] response:\n{json.dumps(parsed, indent=2)}")

            if callback and parsed.get("has_new_data"):
                callback(parsed)

        except Exception as e:
            logging.exception(f"[LLM] call failed: {e}")
        finally:
            if conn:
                conn.close()

    threading.Thread(target=run, daemon=True).start()
