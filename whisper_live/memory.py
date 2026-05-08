from collections import defaultdict, deque
import threading
from typing import Any, Dict, List


class MeetingSession:
    def __init__(self):
        self.transcript: List[Dict[str, Any]] = []
        self.context = deque(maxlen=50)
        self.insights: List[Dict[str, Any]] = []
        self.subscribers = set()
        self.lock = threading.Lock()


class MemoryStore:
    def __init__(self):
        self.sessions = defaultdict(MeetingSession)

    # transcript state
    def add(self, meeting_id, segment):
        session = self.sessions[meeting_id]

        with session.lock:
            session.transcript.append(segment)
            session.context.append(segment)

    def get_context(self, meeting_id, n=20):
        session = self.sessions[meeting_id]

        with session.lock:
            return list(session.context)[-n:]

    def get_new_context(self, meeting_id, last_index=0):
        session = self.sessions[meeting_id]

        with session.lock:
            new_context = session.transcript[last_index:]
            new_index = len(session.transcript)

        return new_context, new_index

    def get_full_transcript(self, meeting_id):
        session = self.sessions[meeting_id]

        with session.lock:
            lines = []

            for segment in session.transcript:
                text = (segment.get("text") or "").strip()

                if not text:
                    continue

                start_ts = float(segment.get("start_ts", 0))
                end_ts = float(segment.get("end_ts", 0))

                lines.append(f"[{start_ts:.2f} - {end_ts:.2f}] {text}")

            return "\n".join(lines)

    # insights state
    def add_insight(self, meeting_id, insight):
        session = self.sessions[meeting_id]

        with session.lock:
            session.insights.append(insight)

    def get_recent_insights(self, meeting_id, n=10):
        session = self.sessions[meeting_id]

        with session.lock:
            return session.insights[-n:]

    def get_all_insights(self, meeting_id):
        session = self.sessions[meeting_id]

        with session.lock:
            return list(session.insights)

    # cleanup
    def clear(self, meeting_id):
        if meeting_id in self.sessions:
            del self.sessions[meeting_id]
