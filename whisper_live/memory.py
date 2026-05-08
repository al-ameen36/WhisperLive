from collections import defaultdict, deque
import threading
import json
from typing import Any, Dict, List, Tuple


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
        self.lock = threading.Lock()

    def add(self, meeting_id, segment):
        session = self.sessions[meeting_id]

        with session.lock:
            session.transcript.append(segment)
            session.context.append(segment)

        self.broadcast_transcript(session, segment)

    def add_insight(self, meeting_id, insight):
        session = self.sessions[meeting_id]

        with session.lock:
            session.insights.append(insight)

        self.broadcast_insight(session, insight)

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

    def get_recent_insights(self, meeting_id, n=10):
        session = self.sessions[meeting_id]
        with session.lock:
            return session.insights[-n:]

    def broadcast_transcript(self, session, segment):
        msg = json.dumps(
            {
                "type": "transcript_update",
                "segment": segment,
            }
        )
        self._broadcast(session, msg)

    def broadcast_insight(self, session, insight):
        msg = json.dumps(
            {
                "type": "insight_update",
                "insight": insight,
            }
        )
        self._broadcast(session, msg)

    def _broadcast(self, session, msg):
        dead = []

        for ws in session.subscribers:
            try:
                ws.send(msg)
            except Exception:
                dead.append(ws)

        for d in dead:
            session.subscribers.discard(d)
