# Acoustic Echo Cancellation (AEC)

## Overview
The Acoustic Echo Cancellation (AEC) module uses SpeexDSP to remove audio feedback and echo from the speech pipeline. This is particularly useful in scenarios where speaker output may be picked up by the microphone.

## Features
- Real-time echo cancellation using SpeexDSP
- Automatic fallback to pass-through mode on errors
- Thread-safe ring buffer implementation
- Adaptive filter with configurable parameters
- Error recovery and self-healing capabilities

## Configuration

### Command Line Arguments
```bash
# Enable AEC
--use_aec

# Configure filter length (default: 2048)
--aec_filter_length 2048
```

### System Requirements

#### Ubuntu/Debian
```bash
sudo apt-get install libspeexdsp-dev
```

#### macOS
```bash
brew install speexdsp
```

#### Windows
```bash
pacman -S mingw-w64-x86_64-speexdsp  # Using MSYS2
```

## Troubleshooting

### Common Issues

1. **Library Not Found Error**
   ```
   Error: Failed to load SpeexDSP library
   ```
   Solution: Ensure SpeexDSP is properly installed for your system.

2. **Audio Glitches**
   - Check if frame size matches your audio configuration
   - Verify sample rate settings
   - Monitor CPU usage for performance issues

3. **High Latency**
   - Reduce filter length for lower latency
   - Adjust frame size for better performance
   - Consider buffer size adjustments

### Error Recovery
The AEC module includes automatic error recovery:
- Detects and handles invalid audio data
- Resets processing on critical errors
- Falls back to pass-through mode when necessary

## Performance Optimization

### Buffer Settings
- Ring buffer size: `filter_length * 2`
- Recommended frame sizes: 160-512 samples
- Filter length: 1024-4096 (trade-off between quality and latency)

### Memory Usage
- Pre-allocated processing buffers
- Efficient ring buffer implementation
- Automatic cleanup of resources

## Integration Example

```python
from audio.processor import AudioProcessor

# Initialize with AEC
processor = AudioProcessor(
    sample_rate=16000,
    frame_size=160,
    channels=1
)

# Start processing
processor.start_processing()

# Process audio
# ... audio processing loop ...

# Cleanup
processor.stop_processing()
```

## Monitoring and Logging
The AEC module provides detailed logging:
- Error conditions and recovery attempts
- Processing statistics
- Resource management events

Use the `--log_level debug` argument for detailed debugging information.
