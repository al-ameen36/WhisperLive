import unittest
from unittest import mock
import sys
import os

# Mock dependencies not available in this environment
sys.modules['torch'] = mock.MagicMock()
sys.modules['whisper'] = mock.MagicMock()

from whisper_live.backend.whisper_backend import ServeClientWhisper

class TestWhisperBackend(unittest.TestCase):
    def test_instantiation(self):
        mock_websocket = mock.MagicMock()
        with mock.patch('whisper_live.backend.whisper_backend.threading.Thread'):
            client = ServeClientWhisper(
                websocket=mock_websocket,
                client_uid="test_whisper",
                model="base"
            )
            self.assertEqual(client.client_uid, "test_whisper")
            self.assertEqual(client.model_size_or_path, "base")

if __name__ == "__main__":
    unittest.main()
