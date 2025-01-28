import os
import signal
import socket
import threading
from queue import Queue, Empty
from dataclasses import dataclass, field
import time
import sounddevice as sd
import numpy as np
from transformers import HfArgumentParser
from audio.processor import AudioProcessor
from audio.ring_buffer import RingBuffer
import logging
import struct

logger = logging.getLogger(__name__)


@dataclass
class ListenAndPlayArguments:
    send_rate: int = field(default=16000, metadata={
                           "help": "In Hz. Default is 16000."})
    recv_rate: int = field(default=16000, metadata={
                           "help": "In Hz. Default is 16000."})
    list_play_chunk_size: int = field(
        default=1024,
        metadata={
            "help": "The size of data chunks (in bytes). Default is 1024."},
    )
    host: str = field(
        default="localhost",
        metadata={
            "help": "The hostname or IP address for listening and playing. Default is 'localhost'."
        },
    )
    send_port: int = field(
        default=12345,
        metadata={"help": "The network port for sending data. Default is 12345."},
    )
    recv_port: int = field(
        default=12346,
        metadata={"help": "The network port for receiving data. Default is 12346."},
    )
    use_aec: bool = field(
        default=False,
        metadata={"help": "Enable AEC processing. Default is False."},
    )
    enable_duplex: bool = field(
        default=False,
        metadata={"help": "Enable full-duplex audio processing. Default is False."},
    )
    input_buffer_size: int = field(
        default=4096,
        metadata={"help": "Size of the input audio ring buffer. Default is 4096."},
    )
    output_buffer_size: int = field(
        default=4096,
        metadata={"help": "Size of the output audio ring buffer. Default is 4096."},
    )
    latency: str = field(
        default="low",
        metadata={
            "help": "Audio latency setting ('low', 'high'). Default is 'low'."},
    )


class AudioClient:
    def __init__(self, args):
        self.send_rate = args.send_rate
        self.recv_rate = args.recv_rate
        self.list_play_chunk_size = args.list_play_chunk_size
        self.host = args.host
        self.send_port = args.send_port
        self.recv_port = args.recv_port
        self.audio_processor = None
        if args.use_aec:
            self.audio_processor = AudioProcessor(
                sample_rate=args.send_rate,
                frame_size=args.list_play_chunk_size
            )
        self.enable_duplex = args.enable_duplex
        if self.enable_duplex:
            self.input_buffer = RingBuffer(
                size=args.input_buffer_size, dtype=np.int16)
            self.output_buffer = RingBuffer(
                size=args.output_buffer_size, dtype=np.int16)
        self.latency = args.latency
        self.running = True  # Add running flag
        self.stop_event = threading.Event()
        self.cleanup_lock = threading.Lock()  # Add lock for cleanup synchronization
        self.cleanup_done = False
        self.reconnect_delay = 1.0  # Initial reconnect delay
        self.max_reconnect_delay = 30.0  # Maximum reconnect delay
        self.connection_retry = True  # Flag to control connection retry attempts
        self.socket_lock = threading.Lock()
        self.initial_connection = True  # Flag for initial connection attempt
        self.last_error_time = 0  # Add timestamp for error rate limiting
        self.error_cooldown = 5  # Minimum seconds between repeated error messages
        self.connected = False  # Track connection state
        # Add reentrant lock for connection state
        self.connection_lock = threading.RLock()
        self.reconnecting = False  # Flag to track reconnection state
        self.sample_rate = args.recv_rate
        self.frame_duration_ms = 20  # Match server's frame duration
        self.samples_per_frame = (self.sample_rate * self.frame_duration_ms // 1000)
        self.bytes_per_sample = 2  # int16 = 2 bytes
        self.chunk_size = self.samples_per_frame * self.bytes_per_sample

    def process_audio(self, audio_data):
        """Process audio with AEC if enabled"""
        if self.audio_processor:
            processed = self.audio_processor.process_microphone(audio_data)
            # Ensure we return numpy array
            if isinstance(processed, (bytes, bytearray)):
                return np.frombuffer(processed, dtype=np.int16)
            return processed
        return audio_data

    def connect_socket(self, sock, host, port, purpose=""):
        """Helper to connect a socket with retry logic"""
        try:
            # Simple socket configuration - match test environment
            sock.settimeout(None)  # No timeout - blocking mode
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.connect((host, port))
            logger.info(f"Connected {purpose} socket to {host}:{port}")
            return True
        except Exception as e:
            logger.error(f"{purpose} connection failed: {e}")
            return False

    def connect_to_server(self):
        """Establish both send and receive connections with retry"""
        while self.connection_retry and not self.stop_event.is_set():
            try:
                # Clean up any existing sockets
                self._cleanup_sockets()

                # Create new sockets
                self.send_socket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)
                self.recv_socket = socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM)

                # Connect both sockets
                if not self.connect_socket(self.send_socket, self.host, self.send_port, "send"):
                    self._cleanup_sockets()
                    self.stop_event.wait(1.0)  # Wait 1 second between retries
                    continue

                if not self.connect_socket(self.recv_socket, self.host, self.recv_port, "receive"):
                    self._cleanup_sockets()
                    self.stop_event.wait(1.0)  # Wait 1 second between retries
                    continue

                return True

            except Exception as e:
                logger.error(f"Connection attempt failed: {e}")
                self._cleanup_sockets()
                self.stop_event.wait(1.0)

        return False

    def _cleanup_sockets(self):
        """Helper method to cleanup sockets with proper locking"""
        with self.socket_lock:
            for sock_attr in ['send_socket', 'recv_socket']:
                if hasattr(self, sock_attr):
                    sock = getattr(self, sock_attr)
                    if sock:
                        try:
                            # Force immediate close
                            linger_struct = struct.pack('ii', 1, 0)
                            sock.setsockopt(socket.SOL_SOCKET,
                                            socket.SO_LINGER, linger_struct)
                            sock.shutdown(socket.SHUT_RDWR)
                        except:
                            pass
                        sock.close()
                    setattr(self, sock_attr, None)

    def start(self):
        """Start audio processing with connection management"""
        if self.audio_processor:
            self.audio_processor.start_processing()

        # Initial connection - will keep retrying until successful or stopped
        if not self.connect_to_server():
            if self.initial_connection:
                logger.info("Connection attempts stopped by user")
            else:
                logger.error("Failed to connect to server")
            return False

        print("Recording and streaming...")

        self.stop_event = threading.Event()
        self.recv_queue = Queue()
        self.send_queue = Queue()

        # Update stream configuration for duplex mode
        stream_kwargs = {
            "samplerate": self.sample_rate,
            "channels": 1,
            "dtype": np.int16,
            "blocksize": self.samples_per_frame,  # Use calculated frame size
            "latency": self.latency
        }

        try:
            if self.enable_duplex:
                self.audio_stream = sd.Stream(
                    callback=self.audio_callback,
                    device=None,
                    **stream_kwargs
                )
                self.audio_stream.start()
            else:
                # Original separate streams
                self.send_stream = sd.RawInputStream(
                    callback=self.callback_send,
                    **stream_kwargs
                )
                self.recv_stream = sd.RawOutputStream(
                    callback=self.callback_recv,
                    **stream_kwargs
                )
                self.send_stream.start()
                self.recv_stream.start()

            # Start network threads
            self.send_thread = threading.Thread(target=self.send, args=(self.stop_event, self.send_queue))
            self.recv_thread = threading.Thread(target=self.recv, args=(self.stop_event, self.recv_queue))
            self.send_thread.start()
            self.recv_thread.start()
            
            # Create input thread for handling keyboard interrupt
            self.input_thread = threading.Thread(target=self._wait_for_input)
            self.input_thread.daemon = True
            self.input_thread.start()

            return True

        except sd.PortAudioError as e:
            logger.error(f"Error initializing audio streams: {e}")
            if hasattr(self, 'audio_stream'):
                self.audio_stream.close()
            if hasattr(self, 'send_stream'):
                self.send_stream.close()
            if hasattr(self, 'recv_stream'):
                self.recv_stream.close()
            return False

    def audio_callback(self, indata, outdata, frames, time, status):
        """Combined callback for duplex audio"""
        try:
            # Handle input (recording)
            if indata is not None:
                audio_data = indata.flatten().astype(np.int16)
                processed_data = self.process_audio(audio_data)
                if processed_data is not None:
                    if isinstance(processed_data, np.ndarray):
                        data_bytes = processed_data.tobytes()
                    else:
                        data_bytes = processed_data
                    self.send_queue.put(data_bytes)

            # Handle output (playback)
            if outdata is not None:
                try:
                    if not self.recv_queue.empty():
                        data = self.recv_queue.get_nowait()
                        audio_array = np.frombuffer(data, dtype=np.int16)
                        # Ensure we have exactly the right number of samples
                        if len(audio_array) == frames:
                            outdata[:] = audio_array.reshape(-1, 1)
                        else:
                            # Pad or truncate as needed
                            out_buffer = np.zeros(frames, dtype=np.int16)
                            n_samples = min(len(audio_array), frames)
                            out_buffer[:n_samples] = audio_array[:n_samples]
                            outdata[:] = out_buffer.reshape(-1, 1)
                    else:
                        outdata.fill(0)
                except Empty:
                    outdata.fill(0)
                except Exception as e:
                    logger.error(f"Error processing output audio: {e}")
                    outdata.fill(0)
        except Exception as e:
            logger.error(f"Error in audio callback: {e}", exc_info=True)
            if outdata is not None:
                outdata.fill(0)

    def recv(self, stop_event, recv_queue):
        """Receive raw audio data continuously"""
        while not stop_event.is_set():
            try:
                if self.recv_socket is None:
                    time.sleep(0.01)  # Reduced sleep time
                    continue

                # Read exact chunk
                chunk = self.recv_socket.recv(self.chunk_size)
                if not chunk:
                    logger.info("Connection closed by server")
                    self.stop()
                    break
                
                if len(chunk) == self.chunk_size:
                    recv_queue.put(chunk)

            except Exception as e:
                logger.error(f"Receive error: {e}")
                break

        self._cleanup_sockets()

    def send(self, stop_event, send_queue):
        """Send raw audio data without length prefix"""
        while not stop_event.is_set():
            try:
                data = send_queue.get(timeout=0.5)
                if data is not None and self.send_socket is not None:
                    self.send_socket.sendall(data)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Send error: {e}")
                break

        self._cleanup_sockets()

    def callback_send(self, indata, frames, time, status):
        """Callback for sending audio in non-duplex mode"""
        if status:
            logger.warning(f"Input stream status: {status}")

        try:
            # indata is raw bytes in non-duplex mode
            audio_data = np.frombuffer(indata, dtype=np.int16)
            processed_data = self.process_audio(audio_data)
            if processed_data is not None:
                # Handle both numpy arrays and raw bytes
                if isinstance(processed_data, np.ndarray):
                    data_bytes = processed_data.tobytes()
                else:
                    data_bytes = processed_data
                self.send_queue.put(data_bytes)
                logger.debug(
                    f"Queued {len(processed_data)} samples for sending")
        except Exception as e:
            # Add stack trace
            logger.error(f"Error in send callback: {e}", exc_info=True)

    def callback_recv(self, outdata, frames, time, status):
        """Callback for receiving audio in non-duplex mode"""
        if status:
            logger.warning(f"Output stream status: {status}")

        try:
            if not self.recv_queue.empty():
                data = self.recv_queue.get()
                if isinstance(data, bytes):
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    outdata[:] = audio_array.reshape(-1, 1)
                    logger.debug(f"Playing {len(audio_array)} samples")
                else:
                    outdata.fill(0)
            else:
                outdata.fill(0)
        except Exception as e:
            logger.error(f"Error in receive callback: {e}")
            outdata.fill(0)

    def _wait_for_input(self):
        """Handle keyboard input in separate thread"""
        try:
            input("Press Enter to stop...\n")
        except EOFError:
            pass
        finally:
            self.stop_event.set()

    def stop(self):
        """Clean shutdown sequence"""
        if self.cleanup_done:
            return

        logger.info("Shutting down...")

        # Stop everything
        self.running = False
        self.stop_event.set()

        try:
            # 1. Stop audio streams
            logger.debug('Stopping audio streams...')
            for attr in ['audio_stream', 'send_stream', 'recv_stream']:
                if hasattr(self, attr):
                    stream = getattr(self, attr)
                    if stream:
                        stream.stop()
                        stream.close()
                        setattr(self, attr, None)
            logger.debug('Audio streams stopped')

            # 2. Stop audio processor
            logger.debug('Stopping audio processor...')
            if self.audio_processor:
                self.audio_processor.stop_processing()
                self.audio_processor = None
            logger.debug('Audio processor stopped')

            # 3. Close sockets
            logger.debug('Closing sockets...')
            self._cleanup_sockets()
            logger.debug('Sockets closed')

            # 4. Clear queues
            logger.debug('Clearing queues...')
            if (hasattr(self, 'send_queue') and hasattr(self, 'recv_queue')):
                for queue in [self.send_queue, self.recv_queue]:
                    while not queue.empty():
                        try:
                            queue.get_nowait()
                        except Empty:
                            break
            logger.debug('Queues cleared')

            # Join network threads to avoid long wait
            logger.debug('Closing network threads...')
            for thread_attr in ['send_thread', 'recv_thread', 'input_thread']:
                if hasattr(self, thread_attr):
                    t = getattr(self, thread_attr)
                    if t and t.is_alive() and t != threading.current_thread():
                        t.join(timeout=1.0)
            logger.debug('Network threads closed')
            # Mark cleanup as done
            self.cleanup_done = True
            logger.info("Shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def run(self):
        """Main client loop with error handling"""
        def force_stop(signum=None, frame=None):
            """Force stop handler"""
            try:
                self.stop()
            except:
                os._exit(1)

        # Register signal handlers
        signal.signal(signal.SIGINT, force_stop)
        signal.signal(signal.SIGTERM, force_stop)

        try:
            print("Attempting to connect to server...")
            if not self.start():
                return

            print("Recording and streaming...")
            print("Press Enter or Ctrl+C to stop...")

            # Wait for stop event with timeout to check periodically
            while not self.stop_event.is_set():
                time.sleep(0.2)

        except KeyboardInterrupt:
            force_stop()
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            force_stop()
        finally:
            self.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = HfArgumentParser((ListenAndPlayArguments,))
    (listen_and_play_kwargs,) = parser.parse_args_into_dataclasses()
    client = AudioClient(listen_and_play_kwargs)
    client.run()
