import numpy as np
import sounddevice as sd
from queue import Queue
from .aec import AcousticEchoCanceller
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, sample_rate=16000, frame_size=160, channels=1, dtype=np.float32):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.channels = channels
        self.dtype = dtype
        
        # Initialize AEC
        self.aec = AcousticEchoCanceller(
            sample_rate=sample_rate,
            frame_size=frame_size
        )
        
        # Buffers for audio processing
        self.mic_buffer = Queue()
        self.speaker_buffer = Queue()
        
        self.error_count = 0
        self.max_errors = 5
        self.recovery_needed = False
        
    def process_microphone(self, indata, frames, time, status):
        """Callback for microphone input with error recovery"""
        try:
            if status:
                logger.warning(f"Microphone input status: {status}")
                self.error_count += 1
            
            if self.error_count >= self.max_errors:
                logger.warning("Too many errors, attempting recovery")
                self.recovery_needed = True
                self.error_count = 0
                return indata.flatten()

            if self.recovery_needed:
                if self.attempt_recovery():
                    self.recovery_needed = False
                else:
                    return indata.flatten()

            if self.speaker_buffer.qsize() > 0:
                ref_signal = self.speaker_buffer.get()
                processed = self.aec.process(indata.flatten(), ref_signal)
            else:
                processed = indata.flatten()
            self.mic_buffer.put(processed)

        except Exception as e:
            logger.error(f"Error in microphone processing: {e}")
            self.error_count += 1
            return indata.flatten()

    def attempt_recovery(self):
        """Attempt to recover from errors by resetting components"""
        try:
            # Reset AEC
            if not self.aec.reset():
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

    def process_output(self, outdata, frames, time, status):
        """Callback for speaker output"""
        if status:
            logger.warning(f"Speaker output status: {status}")
        # Store output audio for AEC reference
        self.speaker_buffer.put(outdata.flatten())

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
