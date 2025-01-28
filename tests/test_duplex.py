import pytest
import numpy as np
import torch
import threading
from threading import Event
from queue import Queue
import sounddevice as sd
from VAD.vad_handler import VADHandler
from TTS.parler_handler import ParlerTTSHandler
from TTS.melo_handler import MeloTTSHandler
from TTS.chat_tts_handler import ChatTTSHandler
from connections.local_audio_streamer import LocalAudioStreamer
from audio.ring_buffer import RingBuffer

class TestDuplexFunctionality:
    @pytest.fixture
    def setup_test_environment(self):
        """Setup basic test environment with queues and events"""
        return {
            'should_listen': Event(),
            'interrupt_event': Event(),
            'input_queue': Queue(),
            'output_queue': Queue()
        }

    def test_ring_buffer_operations(self):
        """Test ring buffer read/write operations"""
        buffer_size = 1024
        buffer = RingBuffer(size=buffer_size, dtype=np.float32)
        
        # Test write operation
        test_data = np.ones(512, dtype=np.float32)
        buffer.write(test_data)
        assert buffer.write_idx == 512
        
        # Test read operation
        read_data = buffer.read(512)
        assert np.array_equal(read_data, test_data)
        assert buffer.read_idx == 512

    def test_vad_interruption(self, setup_test_environment):
        """Test VAD detection and interruption triggering"""
        env = setup_test_environment
        vad = VADHandler(
            should_listen=env['should_listen'],
            interrupt_event=env['interrupt_event'],
            enable_duplex=True,
            interrupt_threshold=0.8,
            sample_rate=16000
        )
        
        # Create synthetic speech audio (high amplitude)
        speech_audio = np.sin(np.linspace(0, 1000, 16000)) * 0.9
        vad.process(speech_audio.astype(np.int16).tobytes())
        
        assert env['interrupt_event'].is_set(), "Interruption not detected for clear speech"

    def test_audio_streamer_duplex(self, setup_test_environment):
        """Test LocalAudioStreamer duplex operation"""
        env = setup_test_environment
        streamer = LocalAudioStreamer(
            input_queue=env['input_queue'],
            output_queue=env['output_queue'],
            sample_rate=16000,
            channels=1,
            blocksize=512
        )
        
        try:
            streamer.start()
            # Generate test audio
            test_audio = np.sin(np.linspace(0, 1000, 16000)).astype(np.int16)
            env['output_queue'].put(test_audio.tobytes())
            
            # Allow some time for processing
            Event().wait(0.1)
            
            # Check that audio is being processed
            assert not env['input_queue'].empty(), "No audio input received"
            
        finally:
            streamer.stop()

    def test_tts_interruption(self, setup_test_environment):
        """Test TTS handler interruption behavior"""
        env = setup_test_environment
        
        # Test each TTS handler
        handlers = [
            (ParlerTTSHandler, {"model_name": "parler-tts/parler-mini-v1-jenny"}),
            (MeloTTSHandler, {"language": "en"}),
            (ChatTTSHandler, {"stream": True})
        ]
        
        for HandlerClass, kwargs in handlers:
            handler = HandlerClass(
                stop_event=Event(),
                queue_in=Queue(),
                queue_out=Queue(),
                setup_args=(env['should_listen'], env['interrupt_event']),
                setup_kwargs=kwargs
            )
            
            # Start TTS processing
            test_thread = threading.Thread(
                target=lambda: list(handler.process("This is a test sentence."))
            )
            test_thread.start()
            
            # Simulate interruption
            env['interrupt_event'].set()
            test_thread.join(timeout=2.0)
            
            assert not test_thread.is_alive(), f"{HandlerClass.__name__} failed to handle interruption"
            assert env['should_listen'].is_set(), f"{HandlerClass.__name__} failed to set should_listen"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
