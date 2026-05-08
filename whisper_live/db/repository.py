from whisper_live.db.supabase_client import supabase


def save_transcript(meeting_id, transcript):
    supabase.table("transcripts").insert(
        {
            "meeting_id": meeting_id,
            "transcript": transcript,
        }
    ).execute()


def save_insight(meeting_id, insight):
    if not insight:
        return

    payload = {
        "meeting_id": meeting_id,
        "type": insight.get("type"),
        "summary": insight.get("summary"),
        "topics": insight.get("topics", []),
        "action_items": insight.get("action_items", []),
        "start_ts": float(insight.get("start_ts", 0)),
        "end_ts": float(insight.get("end_ts", 0)),
    }

    supabase.table("insights").insert(payload).execute()
