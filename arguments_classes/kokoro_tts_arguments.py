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
    kokoro_voice: str = field(
        default="af_bella",
        metadata={
            "help": "Voice to use for TTS output",
            "choices": ["af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", "am_puck"]
        }
    )
    
