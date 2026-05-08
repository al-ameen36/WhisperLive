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

        try:
            existing_memory = existing_memory or []

            # ---- compact context (avoid token bloat) ----
            context_text = "\n".join(f"- {s.get('text', '')}" for s in context)

            memory_text = json.dumps(existing_memory)

            prompt = f"""
You are maintaining LIVE STRUCTURED STATE of a meeting.

CRITICAL RULE:
You are NOT a summarizer.
You are a state updater.

Only extract NEW information that is NOT already present in EXISTING MEMORY.

If nothing new exists, return:
{{
  "type": "meeting_update",
  "summary": "",
  "has_new_data": false,
  "topics": [],
  "data": []
}}

VALID DATA TYPES:
- decision
- action_item
- question
- important_point
- risk
- follow_up

OUTPUT RULES:
- STRICT JSON ONLY
- NO markdown
- NO explanations
- NO extra text

EXISTING MEMORY (already known facts):
{memory_text}

RECENT CONTEXT:
{context_text}

NEW TRANSCRIPT:
{new_text}
"""

            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You output ONLY valid JSON meeting updates.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "stream": False,
                "temperature": 0.2,
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
                error_body = response.read().decode()
                logging.error(
                    f"LLM request failed (status={response.status}): {error_body}"
                )
                return

            raw = response.read().decode()

            data = json.loads(raw)

            output = data["choices"][0]["message"]["content"].strip()

            try:
                parsed = json.loads(output)
            except Exception:
                logging.error(f"Invalid JSON returned by LLM:\n{output}")
                return

            # ---- suppress empty updates ----
            if not parsed.get("has_new_data", False):
                return

            logging.info(
                "\n================ LLM RESPONSE ================\n"
                f"{json.dumps(parsed, indent=2)}\n"
                "=============================================="
            )

            if callback:
                callback(parsed)

        except Exception as e:
            logging.exception(f"LLM async call failed: {e}")

        finally:
            if conn:
                conn.close()

    threading.Thread(target=run, daemon=True).start()
