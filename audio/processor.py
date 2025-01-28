import time
import numpy as np
import sounddevice as sd
from queue import Queue
from .aec import AcousticEchoCanceller
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, sample_rate=16000, frame_size=1024, channels=1, dtype=np.float32):
        # Update default frame size to match common audio buffer sizes
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.channels = channels
        self.dtype = dtype
        
        # Initialize AEC with matching frame size
        self.aec = None
        try:
            self.aec = AcousticEchoCanceller(
                sample_rate=sample_rate,
                frame_size=frame_size  # Ensure AEC uses same frame size
            )
            logger.info(f"AEC initialized successfully with frame size {frame_size}")
        except Exception as e:
            logger.error(f"Error during AEC initialization: {e}")
            logger.warning("Falling back to pass-through mode (no AEC).")
            self.aec = None
        
        # Buffers for audio processing
        self.mic_buffer = Queue()
        self.speaker_buffer = Queue()
        
        self.error_count = 0
        self.max_errors = 5
        self.recovery_needed = False
        
        # Add buffer settings
        self.buffer_size = 5  # Increased for stability
        self.min_buffer_size = 2  # Increased minimum samples
        self.underflow_count = 0
        self.max_underflows = 10
        self.underflow_reset_time = time.time()
        
    def _ensure_frame_size(self, data):
        """Ensure data matches expected frame size"""
        if len(data) < self.frame_size:
            # Pad with zeros if too short
            return np.pad(data, (0, self.frame_size - len(data)), 'constant')
        elif len(data) > self.frame_size:
            # Truncate if too long
            return data[:self.frame_size]
        return data

    def _sanitize_audio(self, data):
        """Clean up audio data by removing NaN/Inf and limiting amplitude"""
        if not isinstance(data, np.ndarray):
            return np.zeros(self.frame_size, dtype=self.dtype)
            
        # Replace NaN/Inf with zeros
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip to prevent floating point errors
        data = np.clip(data, -1.0, 1.0)
        
        return data

    def process_microphone(self, indata, frames=None, time_info=None, status=None):
        """Callback for microphone input with error recovery"""
        # Initialize processed_data at the start to avoid UnboundLocalError
        processed_data = None
        
        try:
            current_time = time.time()  # Use time module instead of callback time_info
            if current_time - self.underflow_reset_time > 5.0:
                self.underflow_count = 0
                self.underflow_reset_time = current_time

            if status:
                if status.output_underflow:
                    self.underflow_count += 1
                    if self.underflow_count > self.max_underflows:
                        logger.warning("Excessive underflows, increasing buffer size")
                        self.buffer_size = min(10, self.buffer_size + 1)
                        self.underflow_count = 0
                else:
                    logger.warning(f"Microphone input status: {status}")
                    self.error_count += 1
            
            # Convert and clean input data
            processed_data = self._ensure_numpy_array(indata)
            processed_data = self._sanitize_audio(processed_data)
            processed_data = self._ensure_frame_size(processed_data)
            
            if not self.aec:
                return self._float32_to_bytes(processed_data)

            # Only process if we have enough reference samples
            if self.speaker_buffer.qsize() >= self.min_buffer_size:
                ref_signal = self.speaker_buffer.get()
                ref_signal = self._sanitize_audio(ref_signal)
                ref_signal = self._ensure_frame_size(ref_signal)
                processed = self.aec.process(processed_data, ref_signal)
                
                # Validate AEC output
                if np.any(np.isnan(processed)) or np.any(np.isinf(processed)):
                    logger.debug("AEC produced invalid output, using original signal")
                    return self._float32_to_bytes(processed_data)
                return self._float32_to_bytes(processed)
            
            return self._float32_to_bytes(processed_data)

        except Exception as e:
            logger.error(f"Error in microphone processing: {e}")
            self.error_count += 1
            fallback_data = processed_data if processed_data is not None else np.zeros(self.frame_size, dtype=np.float32)
            return self._float32_to_bytes(self._sanitize_audio(fallback_data))

    def _ensure_numpy_array(self, data):
        """Convert input data to numpy array if needed and ensure correct shape"""
        try:
            if isinstance(data, bytes):
                # Convert bytes to numpy array ensuring proper scaling
                arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(data, np.ndarray):
                if data.dtype == np.int16:
                    arr = data.astype(np.float32) / 32768.0
                else:
                    arr = data
            else:
                logger.warning(f"Unexpected input type: {type(data)}")
                arr = np.array(data, dtype=np.float32)
            
            # Ensure array is properly shaped
            if arr.ndim == 2:
                arr = arr.reshape(-1)  # Flatten 2D arrays
            elif arr.ndim > 2:
                logger.warning(f"Unexpected array dimensions: {arr.ndim}")
                arr = arr.reshape(-1)  # Flatten multi-dimensional arrays
                
            return self._ensure_frame_size(arr)
            
        except Exception as e:
            logger.error(f"Error converting input data: {e}")
            return np.zeros(self.frame_size, dtype=np.float32)

    def _float32_to_bytes(self, data):
        """Convert float32 array to int16 bytes"""
        try:
            # Clip to [-1, 1] range
            data = np.clip(data, -1.0, 1.0)
            # Convert to int16 range and then to bytes
            return (data * 32768).astype(np.int16).tobytes()
        except Exception as e:
            logger.error(f"Error converting to bytes: {e}")
            return np.zeros(self.frame_size, dtype=np.int16).tobytes()

    def attempt_recovery(self):
        """Attempt to recover from errors by resetting components"""
        try:
            # Reset AEC
            if self.aec and not self.aec.reset():
                return False
            
            # Clear buffers
            while not self.mic_buffer.empty():
                self.mic_buffer.get()
            while not self.speaker_buffer.empty():
                self.speaker_buffer.get()
                
            logger.info("Successfully recovered audio processing")
            return True
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            return False

    def process_output(self, outdata, frames=None, time_info=None, status=None):
        """Callback for speaker output"""
        try:
            if status and not status.output_underflow:
                logger.warning(f"Speaker output status: {status}")
            
            # Process output data
            processed = self._ensure_numpy_array(outdata)
            processed = self._sanitize_audio(processed)
            processed = self._ensure_frame_size(processed)
            
            # Only update reference buffer if we have valid data
            if not np.all(processed == 0):
                self.speaker_buffer.put(processed)
            
            # Dynamic buffer size management
            while self.speaker_buffer.qsize() > self.buffer_size:
                self.speaker_buffer.get()
                
        except Exception as e:
            logger.error(f"Error in output processing: {e}")

    def start_processing(self):
        """Start audio processing streams with retry mechanism"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self.input_stream = sd.InputStream(
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    dtype=self.dtype,
                    blocksize=self.frame_size,
                    callback=self.process_microphone
                )
                
                self.output_stream = sd.OutputStream(
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    dtype=self.dtype,
                    blocksize=self.frame_size,
                    callback=self.process_output
                )
                
                self.input_stream.start()
                self.output_stream.start()
                return
            
            except Exception as e:
                retry_count += 1
                logger.error(f"Failed to start audio processing (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(1)  # Wait before retrying
                else:
                    raise

    def stop_processing(self):
        """Stop audio processing streams with error handling"""
        try:
            logger.debug("Stopping audio processing streams")
            if hasattr(self, 'input_stream'):
                self.input_stream.stop()
                self.input_stream.close()
            if hasattr(self, 'output_stream'):
                self.output_stream.stop()
                self.output_stream.close()
        except Exception as e:
            logger.error(f"Error during audio processing cleanup: {e}")
            # Attempt force cleanup
            for attr in ['input_stream', 'output_stream']:
                if hasattr(self, attr):
                    try:
                        stream = getattr(self, attr)
                        stream.stop()
                        stream.close()
                    except:
                        pass
