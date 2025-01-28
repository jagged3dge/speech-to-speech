from threading import Event
from parler_tts import ParlerTTSStreamer
import logging

logger = logging.getLogger(__name__)

class InterruptibleParlerTTSStreamer(ParlerTTSStreamer):
    """
    A TTS streamer that supports interruption during audio generation.
    Extends ParlerTTSStreamer to add stop functionality.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stop_event = Event()
        self._is_streaming = True

    def stop(self):
        """Signal the streamer to stop generating audio"""
        logger.debug("Stopping TTS streamer")
        self._stop_event.set()
        self._is_streaming = False

    def put(self, value):
        """Override put to check for stop signal"""
        if self._stop_event.is_set():
            raise StopIteration
        super().put(value)

    def __iter__(self):
        """Override iterator to handle interruption"""
        return self

    def __next__(self):
        """Get next audio chunk, checking for interruption"""
        if self._stop_event.is_set():
            self._is_streaming = False
            raise StopIteration
        try:
            return super().__next__()
        except StopIteration:
            self._is_streaming = False
            raise

    @property
    def is_streaming(self):
        """Check if streamer is currently active"""
        return self._is_streaming
