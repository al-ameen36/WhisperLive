import time
import logging

from collections import defaultdict

from whisper_live.memory import MemoryStore
from whisper_live.llm import call_llm_async
from whisper_live.db.repository import (
    save_transcript,
    save_insight,
)

MIN_CHARS = 300
MAX_WAIT = 30

LAST_LLM_CALL = defaultdict(lambda: 0.0)
LAST_SEGMENT_INDEX = defaultdict(lambda: 0)
PENDING_CHARS = defaultdict(lambda: 0)

memory = MemoryStore()


def on_statement_finalized(segment, meeting_id):
    text = (segment.get("text") or "").strip()

    if not text:
        return

    print(segment)

    # normalize timestamps
    segment["start_ts"] = float(segment.get("start", 0))
    segment["end_ts"] = float(segment.get("end", 0))

    # update memory
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

    new_context, new_index = memory.get_new_context(
        meeting_id,
        LAST_SEGMENT_INDEX[meeting_id],
    )

    if not new_context:
        return

    LAST_LLM_CALL[meeting_id] = now
    LAST_SEGMENT_INDEX[meeting_id] = new_index
    PENDING_CHARS[meeting_id] = 0

    logging.info(
        f"[LLM] firing — {chars} chars, "
        f"{len(new_context)} new segments "
        f"(timed_out={timed_out})"
    )

    # insight callback
    def handle_insight(insight):
        memory.add_insight(
            meeting_id,
            insight,
        )

        save_insight(
            meeting_id,
            insight,
        )

    # llm call
    call_llm_async(
        host="localhost",
        port=3000,
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        context=new_context,
        existing_memory=memory.get_recent_insights(meeting_id),
        callback=handle_insight,
    )


def finalize_meeting(meeting_id):
    transcript = memory.get_full_transcript(meeting_id)

    save_transcript(
        meeting_id,
        transcript,
    )
    memory.clear(meeting_id)

    logging.info(f"[MEETING] finalized {meeting_id}")
