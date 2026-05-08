from collections import defaultdict, deque
import threading
import json


class MeetingSession:
    def __init__(self):
        self.transcript = []
        self.context = deque(maxlen=50)
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

        self.broadcast(session, segment)

    def get_context(self, meeting_id, n=20):
        session = self.sessions[meeting_id]
        return list(session.context)[-n:]

    def broadcast(self, session, segment):
        msg = json.dumps({"type": "transcript_update", "segment": segment})

        dead = []
        for ws in session.subscribers:
            try:
                ws.send(msg)
            except Exception:
                dead.append(ws)

        for d in dead:
            session.subscribers.discard(d)
