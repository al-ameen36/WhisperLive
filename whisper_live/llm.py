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
    def fmt(ts: float) -> str:
        ts = float(ts)
        m = int(ts // 60)
        s = int(ts % 60)
        return f"{m:02d}:{s:02d}"

    def run():
        conn = None
        try:
            if not context:
                logging.warning("[LLM] empty context, skipping")
                return

            context_text = "\n".join(
                f"- {s.get('text', '')}" for s in context if s and s.get("text")
            )

            if not context_text.strip():
                logging.warning("[LLM] empty context text, skipping")
                return

            memory_text = json.dumps(existing_memory or [], indent=2)

            # -----------------------------
            # AUDIO TIME WINDOW
            # -----------------------------
            start_ts = context[0].get("start_ts", context[0].get("start", 0))
            end_ts = context[-1].get("end_ts", context[-1].get("end", 0))

            start_time = fmt(start_ts)
            end_time = fmt(end_ts)

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
  "type": "..."  // set type as one of "decisions", "questions", "action_items", "risks", "followups", "general"
  "summary": "2-3 sentences of concrete, factual content",
  "has_new_data": true,
  "topics": [],
  "action_items": [],
  "start_time": "{start_time}",
  "end_time": "{end_time}"
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
Your summary MUST include at least one of:
- a decision
- a task
- a named feature/product/topic
- a measurable detail
- a clear action or requirement

If you cannot extract at least ONE concrete fact → return has_new_data=false.

3. NO MEMORY REPEATING
Do NOT restate anything already present in EXISTING MEMORY.
Only extract NEW information that appears in RECENT TRANSCRIPT.

4. MINIMUM INFORMATION THRESHOLD
Only produce output if BOTH are true:
- At least 1 new fact is present
- That fact is not already in memory

Otherwise return has_new_data=false.

5. SUMMARY QUALITY RULE
The summary must be:
- specific
- self-contained (understandable without context)
- fact-dense
- free of filler words

Bad example:
"They are discussing project updates."

Good example:
"The team decided to delay the API release to next week due to missing authentication tests."

----------------------------------------
EXISTING MEMORY (DO NOT REPEAT OR PARAPHRASE):
{memory_text}

RECENT TRANSCRIPT (ONLY NEW INFORMATION):
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

            # -----------------------------------
            # HARD GATE: suppress useless output
            # -----------------------------------
            if not parsed.get("has_new_data", False):
                logging.info("[LLM] no new data — skipping callback")
                return

            if not parsed.get("summary") and not parsed.get("action_items"):
                logging.info("[LLM] empty structured output — skipping")
                return

            if callback:
                callback(parsed)

        except Exception as e:
            logging.exception(f"[LLM] call failed: {e}")
        finally:
            if conn:
                conn.close()

    threading.Thread(target=run, daemon=True).start()
