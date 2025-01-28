import numpy as np
from ctypes import *
import logging
import time
from .ring_buffer import RingBuffer

logger = logging.getLogger(__name__)


class AcousticEchoCanceller:
    def __init__(self, sample_rate=16000, frame_size=1024, filter_length=2048):
        """Initialize AEC with SpeexDSP."""
        self.enabled = True
        try:
            # Load the SpeexDSP library
            self.speex = CDLL('libspeexdsp.so')
            self.sample_rate = sample_rate
            self.frame_size = frame_size
            self.filter_length = filter_length

            # Initialize ring buffers
            self.mic_buffer = RingBuffer(filter_length * 2)
            self.ref_buffer = RingBuffer(filter_length * 2)

            # Pre-allocate processing buffers
            self.mic_process = np.zeros(frame_size, dtype=np.float32)
            self.ref_process = np.zeros(frame_size, dtype=np.float32)
            self.out_process = np.zeros(frame_size, dtype=np.float32)

            self.reset_count = 0
            self.max_resets = 3
            self.reset_window = 5.0  # seconds
            self.last_reset_time = time.time()
            self.stability_window = []
            self.max_window_size = 100

            self._initialize_speex()
            self._warmup()
        except OSError as e:
            logger.error(f"Failed to load SpeexDSP library: {e}")
            logger.warning(
                "AEC will be disabled and processing will fall back to pass-through mode")
            self.enabled = False

    def _initialize_speex(self):
        """Initialize SpeexDSP echo canceller state."""
        if not self.enabled:
            return

        try:
            # Define function signatures
            self.speex.speex_echo_state_init.restype = c_void_p
            self.speex.speex_echo_state_init.argtypes = [c_int, c_int]
            self.speex.speex_echo_ctl.restype = c_int
            self.speex.speex_echo_ctl.argtypes = [c_void_p, c_int, c_void_p]
            self.speex.speex_echo_cancellation.restype = None
            self.speex.speex_echo_cancellation.argtypes = [
                c_void_p, POINTER(c_float), POINTER(c_float), POINTER(c_float)]
            self.speex.speex_echo_state_destroy.restype = None
            self.speex.speex_echo_state_destroy.argtypes = [c_void_p]

            # Initialize echo canceller state
            self.state = self.speex.speex_echo_state_init(
                self.frame_size, self.filter_length)
            if not self.state:
                raise RuntimeError(
                    "Failed to initialize SpeexDSP echo canceller")

            # Configure sample rate
            SPEEX_ECHO_SET_SAMPLING_RATE = 24  # Typically defined in speex_echo.h
            sample_rate_c = c_int(self.sample_rate)
            result = self.speex.speex_echo_ctl(
                self.state, SPEEX_ECHO_SET_SAMPLING_RATE, byref(sample_rate_c))
            if result != 0:
                raise RuntimeError(
                    f"Failed to set sample rate: error {result}")

        except Exception as e:
            logger.error(f"Error during SpeexDSP initialization: {e}")
            self.enabled = False
            if hasattr(self, 'state') and self.state:
                self.speex.speex_echo_state_destroy(self.state)
                self.state = None

    def _warmup(self):
        """Warm up the echo canceller with silence."""
        silence = np.zeros(self.frame_size, dtype=np.float32)
        for _ in range(5):
            self.process(silence, silence)

    def _check_stability(self, output):
        """Monitor AEC stability through output statistics"""
        if len(self.stability_window) >= self.max_window_size:
            self.stability_window.pop(0)
        
        # Calculate signal statistics
        energy = np.mean(np.square(output))
        self.stability_window.append(energy)
        
        if len(self.stability_window) < 10:
            return True
            
        # Check for instability indicators
        avg_energy = np.mean(self.stability_window)
        energy_variance = np.var(self.stability_window)
        
        return energy_variance < 0.1 and avg_energy < 1.0

    def process(self, mic_signal, speaker_signal):
        if not self.enabled:
            return mic_signal

        try:
            # Input validation and normalization
            mic_signal = np.nan_to_num(mic_signal, nan=0.0, posinf=0.0, neginf=0.0)
            speaker_signal = np.nan_to_num(speaker_signal, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Clip signals to prevent floating point errors
            mic_signal = np.clip(mic_signal, -1.0, 1.0)
            speaker_signal = np.clip(speaker_signal, -1.0, 1.0)

            if len(mic_signal) != self.frame_size or len(speaker_signal) != self.frame_size:
                logger.warning("Input size mismatch")
                return mic_signal

            # Process through ring buffers
            try:
                self.mic_buffer.write(mic_signal)
                self.ref_buffer.write(speaker_signal)
            except Exception as e:
                logger.error(f"Ring buffer error: {e}")
                return mic_signal

            # Get processing frames
            np.copyto(self.mic_process, self.mic_buffer.read(self.frame_size))
            np.copyto(self.ref_process, self.ref_buffer.read(self.frame_size))

            # Process through SpeexDSP
            mic_ptr = self.mic_process.ctypes.data_as(POINTER(c_float))
            ref_ptr = self.ref_process.ctypes.data_as(POINTER(c_float))
            out_ptr = self.out_process.ctypes.data_as(POINTER(c_float))

            self.speex.speex_echo_cancellation(self.state, mic_ptr, ref_ptr, out_ptr)

            # Post-process output
            output = np.copy(self.out_process)
            output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
            output = np.clip(output, -1.0, 1.0)

            # Check stability and reset if needed
            if not self._check_stability(output):
                current_time = time.time()
                if current_time - self.last_reset_time > self.reset_window:
                    self.reset_count = 0
                
                if self.reset_count < self.max_resets:
                    logger.warning("AEC showing instability, attempting reset...")
                    self.reset()
                    self.reset_count += 1
                    self.last_reset_time = current_time
                else:
                    logger.warning("Too many resets, falling back to pass-through mode")
                    self.enabled = False
                    return mic_signal

            return output

        except Exception as e:
            logger.error(f"Critical error in AEC processing: {e}")
            return mic_signal

    def reset(self):
        """Reset the AEC state if processing quality degrades."""
        if not self.enabled:
            return False

        try:
            self._initialize_speex()
            self._warmup()
            return True
        except Exception as e:
            logger.error(f"Failed to reset AEC: {e}")
            self.enabled = False
            return False

    def __del__(self):
        """Clean up SpeexDSP resources."""
        if hasattr(self, 'state') and self.state and self.enabled:
            try:
                self.speex.speex_echo_state_destroy(self.state)
                self.state = None
            except Exception as e:
                logger.error(f"Error cleaning up AEC resources: {e}")
