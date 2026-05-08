import http.client
import json
import logging
import threading
from typing import Callable, Optional, List, Dict, Any


def call_llm_async(
    host: str,
    port: int,
    model: str,
    context: List[Dict[str, Any]],
    existing_memory: Optional[List[Dict[str, Any]]] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    def run():
        conn = None
        try:
            context_text = "\n".join(
                f"- {s.get('text', '')}" for s in context if s and s.get("text")
            )

            if not context_text.strip():
                logging.warning("[LLM] empty context, skipping")
                return

            memory_text = "\n".join(
                f"- [{item.get('type', '')}] {item.get('summary', '')} | "
                f"topics: {item.get('topics', [])} | "
                f"action_items: {item.get('action_items', [])}"
                for item in (existing_memory or [])
            )

            prompt = f"""
You are a strict meeting state extractor.
Only extract explicit, complete, and meaningful information.
Return JSON only.

FORMAT:
{{
  "type": one of: "decisions" | "questions" | "action_items" | "risks" | "followups" | "general",
  "summary": "2-3 sentences of what is being discussed right now",
  "has_new_data": true or false,
  "topics": [],
  "action_items": []
}}

RULES:
- type: pick the single most dominant thing happening in this transcript segment
- summary: always fill this if there is anything meaningful being said
- topics: high level subjects mentioned
- action_items: only include this field if tasks were explicitly assigned, otherwise omit it
- set has_new_data to false only if the transcript contains nothing meaningful

EXISTING MEMORY (already reported — do not repeat, rephrase, or re-summarize any of this):
{memory_text}

RECENT TRANSCRIPT (only extract what is NEW relative to existing memory):
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

            if parsed.get("has_new_data") and callback:
                callback(parsed)

        except Exception as e:
            logging.exception(f"[LLM] call failed: {e}")
        finally:
            if conn:
                conn.close()

    threading.Thread(target=run, daemon=True).start()
