from whisper_live.memory import MemoryStore
import time
import logging
from collections import defaultdict

from whisper_live.db.repository import insert_transcript, insert_insight
from whisper_live.llm import call_llm_async


MIN_CHARS = 300
MAX_WAIT = 30

LAST_LLM_CALL = defaultdict(lambda: 0.0)
PENDING_CHARS = defaultdict(lambda: 0)

memory = MemoryStore()


def on_statement_finalized(segment, meeting_id):
    text = (segment.get("text") or "").strip()
    if not text:
        return

    # timestamp normalization
    segment["start_ts"] = float(segment.get("start", 0))
    segment["end_ts"] = float(segment.get("end", 0))

    # persist raw transcript
    insert_transcript(meeting_id, segment)

    # memory update
    memory.add(meeting_id, segment)

    # batching logic
    now = time.time()

    PENDING_CHARS[meeting_id] += len(text)

    chars = PENDING_CHARS[meeting_id]
    last_call = LAST_LLM_CALL[meeting_id]

    enough_chars = chars >= MIN_CHARS
    timed_out = (now - last_call) >= MAX_WAIT and chars > 0

    if not (enough_chars or timed_out):
        logging.info(f"[LLM] waiting — {chars}/{MIN_CHARS}")
        return

    new_context, new_index = memory.get_new_context(meeting_id)

    if not new_context:
        return

    LAST_LLM_CALL[meeting_id] = now
    PENDING_CHARS[meeting_id] = 0

    start_ts = new_context[0]["start_ts"]
    end_ts = new_context[-1]["end_ts"]

    # -------------------------
    # LLM call
    # -------------------------
    def handle_insight(insight):
        insight["start_ts"] = start_ts
        insight["end_ts"] = end_ts

        memory.add_insight(meeting_id, insight)
        insert_insight(meeting_id, insight)

    call_llm_async(
        host="localhost",
        port=3000,
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        context=new_context,
        existing_memory=memory.get_recent_insights(meeting_id),
        callback=handle_insight,
    )
