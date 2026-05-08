from whisper_live.db.supabase_client import supabase


def insert_transcript(meeting_id, segment):
    supabase.table("transcripts").insert(
        {
            "meeting_id": meeting_id,
            "text": segment.get("text"),
            "start_ts": float(segment.get("start_ts", 0)),
            "end_ts": float(segment.get("end_ts", 0)),
        }
    ).execute()


def insert_insight(meeting_id, insight):
    supabase.table("insights").insert(
        {
            "meeting_id": meeting_id,
            "type": insight.get("type"),
            "summary": insight.get("summary"),
            "topics": insight.get("topics", []),
            "action_items": insight.get("action_items", []),
            "start_ts": float(insight.get("start_ts", 0)),
            "end_ts": float(insight.get("end_ts", 0)),
        }
    ).execute()
