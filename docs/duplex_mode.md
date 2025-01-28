# Full-Duplex Mode with Interruption Support

## Overview
The full-duplex mode enables natural conversational flow by allowing users to interrupt the AI's speech, similar to human-to-human conversations.

## Server-Client Setup

### Server Side
Run the pipeline with duplex mode enabled:

```bash
# Basic server setup
python s2s_pipeline.py \
    --enable_duplex \
    --recv_host 0.0.0.0 \
    --send_host 0.0.0.0

# With custom sensitivity
python s2s_pipeline.py \
    --enable_duplex \
    --interrupt_threshold 0.8 \
    --recv_host 0.0.0.0 \
    --send_host 0.0.0.0 \
    --tts melo \
    --buffer_size 4096

# With AEC (Echo Cancellation)
python s2s_pipeline.py \
    --enable_duplex \
    --use_aec \
    --recv_host 0.0.0.0 \
    --send_host 0.0.0.0
```

### Client Side
Run the client with matching duplex settings:

```bash
# Basic client setup
python listen_and_play.py \
    --enable_duplex \
    --host <server_ip>

# Optimized for low latency
python listen_and_play.py \
    --enable_duplex \
    --host <server_ip> \
    --input_buffer_size 4096 \
    --output_buffer_size 4096 \
    --latency low

# With echo cancellation
python listen_and_play.py \
    --enable_duplex \
    --host <server_ip> \
    --use_aec
```

## Configuration Parameters

### Server Parameters
| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| `--enable_duplex` | Enable duplex mode | False | bool |
| `--interrupt_threshold` | Interruption sensitivity | 0.8 | 0.0-1.0 |
| `--buffer_size` | Audio buffer size | 4096 | int |
| `--use_aec` | Enable echo cancellation | False | bool |

### Client Parameters
| Parameter | Description | Default | Range |
| --------- | ----------- | ------- | ----- |
| `--enable_duplex` | Enable duplex mode | False | bool |
| `--input_buffer_size` | Input buffer size | 4096 | int |
| `--output_buffer_size` | Output buffer size | 4096 | int |
| `--latency` | Audio latency setting | "low" | "low"/"high" |

## Common Configurations

### High Responsiveness Setup
```bash
# Server
python s2s_pipeline.py --enable_duplex --interrupt_threshold 0.6 --buffer_size 2048

# Client
python listen_and_play.py --enable_duplex --input_buffer_size 2048 --output_buffer_size 2048 --latency low
```

### High Quality Setup
```bash
# Server
python s2s_pipeline.py --enable_duplex --interrupt_threshold 0.8 --buffer_size 4096 --use_aec

# Client
python listen_and_play.py --enable_duplex --input_buffer_size 4096 --output_buffer_size 4096 --use_aec
```

## Debugging Tips

### Audio Issues
- Start with default buffer sizes (4096)
- Adjust `interrupt_threshold` based on environment noise
- Enable `--use_aec` if experiencing echo
- Try different `--latency` settings on client

### Connection Issues
- Ensure server is using `0.0.0.0` for hosts
- Verify client using correct server IP
- Check network firewall settings
- Monitor server logs with `--log_level debug`

## Performance Notes
- Lower buffer sizes = Lower latency but higher CPU usage
- Higher `interrupt_threshold` = Fewer false triggers but less responsive
- AEC adds slight processing overhead but improves quality
- Local mode typically has lower latency than network mode
