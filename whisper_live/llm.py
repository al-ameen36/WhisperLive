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
            if not context:
                return

            context_text = "\n".join(
                (
                    f"[{s.get('start_ts', 0):.2f} - "
                    f"{s.get('end_ts', 0):.2f}] "
                    f"{s.get('text', '')}"
                )
                for s in context
                if s and s.get("text")
            )

            if not context_text.strip():
                logging.warning("[LLM] empty context")
                return

            memory_text = json.dumps(existing_memory or [], indent=2)

            start_time = context[0].get("start_ts", 0)
            end_time = context[-1].get("end_ts", 0)

            prompt = f"""
You are a strict, high-precision meeting state extractor.

Your job is NOT to summarize generally.
Your job is to extract ONLY new, concrete, verifiable information from the transcript.

You must decide whether the transcript contains NEW meaningful information compared to existing memory.

If nothing new or nothing meaningful is said, return ONLY:
{{
  "has_new_data": false
}}

----------------------------------------
OUTPUT FORMAT (ONLY if new data exists)
----------------------------------------
{{
  "type": one of: "decisions", "questions", "action_items", "risks", "followups", "general",
  "summary": "2-3 sentences of concrete factual content",
  "has_new_data": true,
  "topics": [],
  "action_items": [],
  "start_time": {start_time},
  "end_time": {end_time}
}}

----------------------------------------
STRICT RULES
----------------------------------------

1. NO VAGUE LANGUAGE
Do NOT use phrases like:
- "they are discussing"
- "they are talking about"
- "there is a conversation about"
- "updates on"
- "general discussion"

Every sentence must contain concrete information.

2. CONCRETE CONTENT REQUIREMENT
Your summary MUST contain at least one:
- decision
- task
- named feature
- requirement
- measurable detail
- technical issue

3. NO MEMORY REPETITION
Do NOT repeat or paraphrase information already present in EXISTING MEMORY.

4. TYPE RULE
- type MUST be exactly one value
- do NOT return multiple values
- do NOT use separators like "|"

----------------------------------------
EXISTING MEMORY
----------------------------------------
{memory_text}

----------------------------------------
RECENT TRANSCRIPT
----------------------------------------
{context_text}
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
                "response_format": {
                    "type": "json_object",
                },
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
