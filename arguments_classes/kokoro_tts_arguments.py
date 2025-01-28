from dataclasses import dataclass, field

@dataclass
class KokoroTTSHandlerArguments:
    """Arguments for Kokoro TTS Handler"""
    kokoro_device: str = field(
        default="cuda",
        metadata={"help": "Device to run Kokoro TTS on (cpu/cuda)"}
    )
    kokoro_library: str = field(
        default="kokoro-library",
        metadata={"help": "Path to Kokoro TTS library files"}
    )
    kokoro_blocksize: int = field(
        default=512,
        metadata={"help": "Audio block size for processing"}
    )
