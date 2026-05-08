import os
import json
import logging
import threading
import torch
import whisper
import numpy as np
from whisper_live.backend.base import ServeClientBase


class ServeClientWhisper(ServeClientBase):
    SINGLE_MODEL = None
    SINGLE_MODEL_LOCK = threading.Lock()

    def __init__(
        self,
        websocket,
        task="transcribe",
        device=None,
        language=None,
        client_uid=None,
        model="large",
        initial_prompt=None,
        vad_parameters=None,
        use_vad=True,
        single_model=False,
        send_last_n_segments=10,
        no_speech_thresh=0.45,
        clip_audio=False,
        same_output_threshold=7,
        cache_path="~/.cache/whisper-live/",
        translation_queue=None,
        on_statement_finalized=None,
    ):
        """
        Initialize a ServeClient instance for OpenAI Whisper.
        """
        super().__init__(
            client_uid,
            websocket,
            send_last_n_segments,
            no_speech_thresh,
            clip_audio,
            same_output_threshold,
            translation_queue,
            on_statement_finalized,
        )
        self.model_size_or_path = model
        self.language = "en" if self.model_size_or_path.endswith(".en") else language
        self.task = task
        self.initial_prompt = initial_prompt
        self.vad_parameters = vad_parameters or {"threshold": 0.5}
        self.use_vad = use_vad

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using Device={device} for OpenAI Whisper")

        try:
            if single_model:
                with ServeClientWhisper.SINGLE_MODEL_LOCK:
                    if ServeClientWhisper.SINGLE_MODEL is None:
                        self.create_model(device)
                        ServeClientWhisper.SINGLE_MODEL = self.transcriber
                    else:
                        self.transcriber = ServeClientWhisper.SINGLE_MODEL
            else:
                self.create_model(device)
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.websocket.send(
                json.dumps(
                    {
                        "uid": self.client_uid,
                        "status": "ERROR",
                        "message": f"Failed to load model: {str(self.model_size_or_path)}",
                    }
                )
            )
            self.websocket.close()
            return

        # threading
        self.trans_thread = threading.Thread(target=self.speech_to_text)
        self.trans_thread.start()
        self.websocket.send(
            json.dumps(
                {
                    "uid": self.client_uid,
                    "message": self.SERVER_READY,
                    "backend": "whisper",
                }
            )
        )

    def create_model(self, device):
        """
        Instantiates a new model using openai-whisper.
        """
        logging.info(f"Loading OpenAI Whisper model: {self.model_size_or_path}")
        self.transcriber = whisper.load_model(self.model_size_or_path, device=device)

    def transcribe_audio(self, input_sample):
        """
        Transcribes the provided audio sample using OpenAI Whisper.
        """
        if ServeClientWhisper.SINGLE_MODEL:
            with ServeClientWhisper.SINGLE_MODEL_LOCK:
                result = self.transcriber.transcribe(
                    input_sample,
                    initial_prompt=self.initial_prompt,
                    language=self.language,
                    task=self.task,
                    fp16=torch.cuda.is_available(),
                )
        else:
            result = self.transcriber.transcribe(
                input_sample,
                initial_prompt=self.initial_prompt,
                language=self.language,
                task=self.task,
                fp16=torch.cuda.is_available(),
            )

        if self.language is None and result.get("language"):
            self.language = result["language"]
            self.websocket.send(
                json.dumps({"uid": self.client_uid, "language": self.language})
            )

        # OpenAI Whisper returns a dict with 'segments'
        # We need to adapt these to the format expected by update_segments
        # base.py expects objects with .text, .start, .end, .no_speech_prob

        class SegmentWrapper:
            def __init__(self, s):
                self.text = s.get("text", "")
                self.start = s.get("start", 0.0)
                self.end = s.get("end", 0.0)
                self.no_speech_prob = s.get("no_speech_prob", 0.0)

        wrapped_segments = [SegmentWrapper(s) for s in result.get("segments", [])]
        return wrapped_segments

    def handle_transcription_output(self, result, duration):
        """
        Handle the transcription output, updating the transcript and sending data to the client.
        """
        segments = []
        if len(result):
            last_segment = self.update_segments(result, duration)
            segments = self.prepare_segments(last_segment)

        if len(segments):
            self.send_transcription_to_client(segments)
