import http.client
import json
import logging
import threading

from typing import Callable, Optional, List, Dict, Any


SYSTEM_PROMPT = """You are a neutral agreement referee monitoring a live meeting.

Your job is to detect two things:
1. What is concretely happening in the meeting (update).
2. Whether a commitment, obligation, or agreement is being formed between parties (flag).
3. Not to make up any information. stick to the information you were provided.

Return ONLY valid JSON. No explanation. No markdown."""


def build_prompt(
    context_text: str, memory_text: str, start_time: float, end_time: float
) -> str:
    return f"""
EXISTING MEMORY (already captured — do not repeat):
{memory_text}

RECENT TRANSCRIPT:
{context_text}

---

Return a JSON object with this exact shape:

{{
  "type": "update" or "flag",
  "summary": "1-2 sentences. Plain language. What happened in this segment.",
  "has_new_data": true or false,
  "assignee": null,
  "assigner": null,
  "commitment": null,
  "implication": null,
  "start_time": {start_time},
  "end_time": {end_time}
}}

---

TYPE RULES:

Set type to "flag" when the transcript contains ANY of the following:
- A task, deliverable, or responsibility is assigned to a named person or implied party.
- A deadline or timeline is attached to a person or team.
- Someone agrees to something explicitly ("I will", "we'll handle", "I can do that", "I commit", "yes I can").
- Scope or budget is defined and attributed to someone.
- A decision is made that binds one or more parties going forward.
- Ownership of an outcome is stated or strongly implied.

Set type to "update" for everything else that contains real content.

---

has_new_data RULES:

Set has_new_data to TRUE when the transcript contains ANY of:
- A question being asked
- An opinion or position being stated
- A concern or hesitation being raised
- A topic being introduced
- Any expression of agreement, disagreement, or uncertainty
- Any reference to a deadline, event, product, person, or named thing
- Any statement about what someone will, should, or needs to do

Set has_new_data to FALSE only when the segment is exclusively:
- Pure filler ("uh", "um", "okay", "right", "yeah", "thanks")
- An exact repetition of something already in memory word for word
- Completely inaudible or empty

DEFAULT TO TRUE. Only set false when you are certain the segment has zero information content.

---

FIELD RULES:

summary:
- Always populate when has_new_data is true.
- Capture the actual substance of what was said, even if conversational.
- Do not require numbers or named features. Capture intent, questions, concerns, and positions.
- Never use "they discussed" or "there was talk of". Say what was actually said.
- Do not make vague summaries

assignee:
- The person receiving the commitment, task, or obligation.
- Extract the name directly from the transcript if mentioned.
- If no name is mentioned but a person is clearly implied, use [Person].
- null if type is "update".

assigner:
- The person creating or delegating the commitment.
- Extract the name directly from the transcript if mentioned.
- If no name is clearly implied, use [Person].
- null if type is "update".

commitment:
- One sentence. The exact obligation being formed, stated as fact.
- Example: "John will deliver the API spec by Friday."
- null if type is "update".

implication:
- One sentence. What accepting this commitment means in practice.
- Example: "Accepting this means John is on record as responsible for the API spec by Friday."
- null if type is "update".
"""


def call_llm_async(
    host: str,
    port: int,
    model: str,
    context: List[Dict[str, Any]],
    existing_memory: Optional[List[str]] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
):
    def run():
        conn = None

        try:
            if not context:
                return

            context_text = "\n".join(
                f"[{s.get('start_ts', 0):.2f} - {s.get('end_ts', 0):.2f}] {s.get('text', '')}"
                for s in context
                if s and s.get("text")
            )

            if not context_text.strip():
                logging.warning("[LLM] empty context")
                return

            memory_text = (
                "\n".join(f"- {m}" for m in (existing_memory or [])) or "None yet."
            )

            start_time = context[0].get("start_ts", 0)
            end_time = context[-1].get("end_ts", 0)

            prompt = build_prompt(context_text, memory_text, start_time, end_time)

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
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
