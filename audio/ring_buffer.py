import numpy as np
from threading import Lock

class RingBuffer:
    def __init__(self, size, dtype=np.float32):
        """Initialize ring buffer with given size."""
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.write_idx = 0
        self.read_idx = 0
        self.lock = Lock()

    def write(self, data):
        """Write data to ring buffer."""
        with self.lock:
            data_len = len(data)
            if data_len > self.size:
                data = data[-self.size:]
                data_len = self.size
            
            # First write position until end of buffer
            first_write = min(data_len, self.size - self.write_idx)
            self.buffer[self.write_idx:self.write_idx + first_write] = data[:first_write]
            
            # Write remaining data at beginning of buffer
            if first_write < data_len:
                remaining = data_len - first_write
                self.buffer[:remaining] = data[first_write:]
            
            self.write_idx = (self.write_idx + data_len) % self.size

    def read(self, size):
        """Read specified amount of data from ring buffer."""
        with self.lock:
            if size > self.size:
                raise ValueError("Read size cannot be larger than buffer size")
            
            # First read from current position to end of buffer
            first_read = min(size, self.size - self.read_idx)
            data = np.copy(self.buffer[self.read_idx:self.read_idx + first_read])
            
            # Read remaining from beginning of buffer if needed
            if first_read < size:
                remaining = size - first_read
                data = np.concatenate([data, self.buffer[:remaining]])
            
            self.read_idx = (self.read_idx + size) % self.size
            return data
