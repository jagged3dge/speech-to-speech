import threading
import sounddevice as sd
import numpy as np

import time
import logging
from queue import Empty
from audio.ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


class LocalAudioStreamer:
    def __init__(
        self,
        input_queue,
        output_queue,
        sample_rate=16000,
        channels=1,
        blocksize=512,
        latency="low"
    ):
        self.list_play_chunk_size = blocksize

        self.stop_event = threading.Event()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.sample_rate = sample_rate
        self.channels = channels
        self.blocksize = blocksize
        self.latency = latency
        self.running = False

        # Initialize ring buffers for duplex audio
        self.input_buffer = RingBuffer(size=self.blocksize * 8, dtype=np.int16)
        self.output_buffer = RingBuffer(size=self.blocksize * 8, dtype=np.int16)

        # Thread-safe flags
        self.stream_lock = threading.Lock()
        self._stream = None

    def audio_callback(self, indata, outdata, frames, time, status):
        if status:
            logger.warning(f"Audio callback status: {status}")

        # Handle input audio
        if indata is not None:
            self.input_buffer.write(indata.flatten())
            self.input_queue.put(indata.tobytes())

        # Handle output audio
        try:
            while not self.output_queue.empty():
                audio_chunk = np.frombuffer(self.output_queue.get_nowait(), dtype=np.int16)
                self.output_buffer.write(audio_chunk)
        except Empty:
            pass

        # Fill output buffer
        if outdata is not None:
            out_data = self.output_buffer.read(frames)
            outdata[:] = out_data.reshape(-1, self.channels)

    def start(self):
        """Start the audio stream with duplex mode"""
        with self.stream_lock:
            if self._stream is not None:
                return

            self.running = True
            try:
                self._stream = sd.Stream(
                    samplerate=self.sample_rate,
                    blocksize=self.blocksize,
                    channels=self.channels,
                    dtype=np.int16,
                    callback=self.audio_callback,
                    latency=self.latency
                )
                self._stream.start()
                logger.info("Audio stream started in full-duplex mode")
            except Exception as e:
                logger.error(f"Failed to start audio stream: {e}")
                self.running = False
                raise

    def stop(self):
        """Stop the audio stream"""
        with self.stream_lock:
            self.running = False
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
                logger.info("Audio stream stopped")

    def __del__(self):
        self.stop()
