import socket
from rich.console import Console
import logging
import numpy as np

logger = logging.getLogger(__name__)

console = Console()


class SocketSender:
    """
    Handles sending generated audio packets to the clients.
    """

    def __init__(self, stop_event, queue_in, host="0.0.0.0", port=12346):
        self.stop_event = stop_event
        self.queue_in = queue_in
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.sample_rate = 16000  # Add explicit sample rate
        self.bytes_per_sample = 2  # int16 = 2 bytes
        self.frame_duration_ms = 20  # 20ms frames
        self.chunk_size = (self.sample_rate * self.frame_duration_ms // 1000) * self.bytes_per_sample

    def run(self):
        """Run the socket sender with improved error handling"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            logger.info(f"Listening for connections on {self.host}:{self.port}")

            while not self.stop_event.is_set():
                self.client_socket, address = self.server_socket.accept()
                logger.info(f"Connection established from {address}")

                try:
                    buffer = bytearray()
                    while not self.stop_event.is_set():
                        data = self.queue_in.get()
                        if isinstance(data, bytes) and data == b"END":
                            break

                        # Accumulate data in buffer
                        if isinstance(data, np.ndarray):
                            buffer.extend((data * 32768).astype(np.int16).tobytes())
                        else:
                            buffer.extend(data)

                        # Send complete chunks
                        while len(buffer) >= self.chunk_size:
                            chunk = buffer[:self.chunk_size]
                            buffer = buffer[self.chunk_size:]
                            self.client_socket.sendall(chunk)

                        # Send remaining data if buffer not empty
                        if buffer:
                            padding = bytes(self.chunk_size - len(buffer))
                            self.client_socket.sendall(buffer + padding)
                            buffer.clear()

                except Exception as e:
                    logger.error(f"Error in sender loop: {e}")
                finally:
                    self.client_socket.close()

        except Exception as e:
            logger.error(f"Error in socket sender: {e}")
        finally:
            if self.client_socket:
                self.client_socket.close()
            if self.server_socket:
                self.server_socket.close()
            logger.info("Socket sender stopped")
