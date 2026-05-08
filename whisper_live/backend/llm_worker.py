import threading
import queue
import time
import json
import logging
import http.client

class LLMWorker:
    """
    Worker that buffers transcription segments and sends them to an LLM.
    Uses standard library http.client to avoid extra dependencies.
    """
    def __init__(self, host="localhost", port=3000, model="meta-llama/Meta-Llama-3-8B-Instruct", buffer_size=3, timeout=10.0, system_prompt=None):
        self.host = host
        self.port = port
        self.model = model
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.system_prompt = system_prompt or "The following are transcription segments from a meeting. Please summarize the conversation"
        
        self.queue = queue.Queue()
        self.buffer = []
        self.last_segment_time = time.time()
        self.exit = False
        
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        logging.info(f"LLMWorker started, target {host}:{port}, buffer_size={buffer_size}")

    def add_segment(self, segment):
        """Add a segment to the queue."""
        logging.info(f"LLMWorker: Segment received: {segment.get('text', '')[:50]}...")
        self.queue.put(segment)

    def _run(self):
        while not self.exit:
            try:
                # Use a small timeout on get to allow checking for fallback/exit
                segment = self.queue.get(timeout=1.0)
                
                if segment is None:
                    # Final flush
                    if self.buffer:
                        self._process_buffer()
                    break
                
                logging.info(f"LLMWorker: Buffering segment ({len(self.buffer) + 1}/{self.buffer_size})")
                self.buffer.append(segment)
                self.last_segment_time = time.time()
                
                if len(self.buffer) >= self.buffer_size:
                    self._process_buffer()
                
            except queue.Empty:
                # Check for time-based fallback
                if self.buffer and (time.time() - self.last_segment_time) >= self.timeout:
                    logging.info("LLMWorker: Timeout reached, flushing buffer.")
                    self._process_buffer()
            except Exception as e:
                logging.error(f"LLMWorker error in loop: {e}")

    def _process_buffer(self):
        if not self.buffer:
            return
            
        text_to_process = " ".join([s.get("text", "").strip() for s in self.buffer])
        self.buffer = [] # Clear buffer
        
        if not text_to_process:
            return
            
        logging.info(f"LLMWorker: Processing {text_to_process[:50]}...")
        
        # Start a thread to make the actual HTTP call so we don't block the worker loop
        # (even though it's already a background thread, this allows the worker to keep 
        #  collecting segments if the LLM is slow)
        threading.Thread(target=self._make_llm_call, args=(text_to_process,), daemon=True).start()

    def _make_llm_call(self, text):
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=30)
            
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                "stream": False
            }
            
            headers = {'Content-Type': 'application/json'}
            conn.request("POST", "/v1/chat/completions", json.dumps(payload), headers)
            
            response = conn.getresponse()
            data = response.read().decode()
            
            if response.status == 200:
                result = json.loads(data)
                answer = result['choices'][0]['message']['content']
                logging.info(f"LLM Response: {answer[:100]}...")
                # Here we could potentially send this back to the client if needed
            else:
                logging.error(f"LLM API error: {response.status} - {data}")
            
            conn.close()
        except Exception as e:
            logging.error(f"Failed to connect to LLM at {self.host}:{self.port}: {e}")

    def cleanup(self):
        """Signal the worker to exit."""
        self.exit = True
        self.queue.put(None)
        self.thread.join(timeout=2.0)
