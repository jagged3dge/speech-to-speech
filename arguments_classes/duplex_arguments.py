from dataclasses import dataclass, field

@dataclass
class DuplexArguments:
    """
    Arguments for controlling full-duplex audio processing
    """
    enable_duplex: bool = field(
        default=False,
        metadata={"help": "Enable full-duplex audio processing (simultaneous listening and speaking)"}
    )
    interrupt_threshold: float = field(
        default=0.8,
        metadata={"help": "Voice activity threshold to trigger interruption (0.0-1.0)"}
    )
    buffer_size: int = field(
        default=4096,
        metadata={"help": "Size of the audio ring buffer for duplex processing"}
    )
