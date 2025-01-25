import socket
import threading
from queue import Queue
from dataclasses import dataclass, field
import sounddevice as sd
from transformers import HfArgumentParser
from audio.processor import AudioProcessor


@dataclass
class ListenAndPlayArguments:
    send_rate: int = field(default=16000, metadata={"help": "In Hz. Default is 16000."})
    recv_rate: int = field(default=16000, metadata={"help": "In Hz. Default is 16000."})
    list_play_chunk_size: int = field(
        default=1024,
        metadata={"help": "The size of data chunks (in bytes). Default is 1024."},
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

    def process_audio(self, audio_data):
        """Process audio with AEC if enabled"""
        if self.audio_processor:
            return self.audio_processor.process_microphone(audio_data)
        return audio_data

    def start(self):
        """Start audio processing"""
        if self.audio_processor:
            self.audio_processor.start_processing()
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.send_socket.connect((self.host, self.send_port))

        self.recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.recv_socket.connect((self.host, self.recv_port))

        print("Recording and streaming...")

        self.stop_event = threading.Event()
        self.recv_queue = Queue()
        self.send_queue = Queue()

        def callback_recv(outdata, frames, time, status):
            if not self.recv_queue.empty():
                data = self.recv_queue.get()
                outdata[: len(data)] = data
                outdata[len(data) :] = b"\x00" * (len(outdata) - len(data))
            else:
                outdata[:] = b"\x00" * len(outdata)

        def callback_send(indata, frames, time, status):
            if self.recv_queue.empty():
                data = bytes(indata)
                self.send_queue.put(data)

        def send(stop_event, send_queue):
            while not stop_event.is_set():
                data = send_queue.get()
                self.send_socket.sendall(data)

        def recv(stop_event, recv_queue):
            def receive_full_chunk(conn, chunk_size):
                data = b""
                while len(data) < chunk_size:
                    packet = conn.recv(chunk_size - len(data))
                    if not packet:
                        return None  # Connection has been closed
                    data += packet
                return data

            while not stop_event.is_set():
                data = receive_full_chunk(self.recv_socket, self.list_play_chunk_size * 2)
                if data:
                    self.recv_queue.put(data)

        self.send_stream = sd.RawInputStream(
            samplerate=self.send_rate,
            channels=1,
            dtype="int16",
            blocksize=self.list_play_chunk_size,
            callback=callback_send,
        )
        self.recv_stream = sd.RawOutputStream(
            samplerate=self.recv_rate,
            channels=1,
            dtype="int16",
            blocksize=self.list_play_chunk_size,
            callback=callback_recv,
        )
        threading.Thread(target=self.send_stream.start).start()
        threading.Thread(target=self.recv_stream.start).start()

        self.send_thread = threading.Thread(target=send, args=(self.stop_event, self.send_queue))
        self.send_thread.start()
        self.recv_thread = threading.Thread(target=recv, args=(self.stop_event, self.recv_queue))
        self.recv_thread.start()

    def stop(self):
        """Stop audio processing"""
        if self.audio_processor:
            self.audio_processor.stop_processing()
        self.stop_event.set()
        # Given that socket::recv is blocking in receive_data_chunk, shut it down to allow the thread to continue.
        self.recv_socket.shutdown(socket.SHUT_RDWR)
        self.recv_thread.join()
        self.send_thread.join()
        self.send_socket.close()
        self.recv_socket.close()
        print("Connection closed.")

    def run(self):
        """Main client loop"""
        try:
            self.start()
            input("Press Enter to stop...")
        except KeyboardInterrupt:
            print("Finished streaming.")
        finally:
            self.stop()


if __name__ == "__main__":
    parser = HfArgumentParser((ListenAndPlayArguments,))
    (listen_and_play_kwargs,) = parser.parse_args_into_dataclasses()
    client = AudioClient(listen_and_play_kwargs)
    client.run()
