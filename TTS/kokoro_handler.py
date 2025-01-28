import logging
import numpy as np
import torch
import librosa
from typing import Generator
from rich.console import Console

from baseHandler import BaseHandler
from kokoro.pipeline import KPipeline  # Correct import path

console = Console()

logger = logging.getLogger(__name__)


class KokoroTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device=None,
        library=None,  # Not used directly, kept for compatibility
        language="en",
        gen_kwargs=None,
        blocksize=512,
    ):
        """
        Proper initialization using KPipeline
        """
        self.should_listen = should_listen
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.blocksize = blocksize
        self.gen_kwargs = gen_kwargs or {}

        # Map language codes ('en' -> 'a' for American English)
        self.lang_code = 'a'  # Default to American English
        if isinstance(language, str) and language.lower() in ['b', 'en-gb']:
            self.lang_code = 'b'

        # Initialize pipeline with proper configuration
        self.pipeline = KPipeline(
            lang_code=self.lang_code,
            device=self.device,
            **self.gen_kwargs
        )

        # Set default generation parameters
        self.default_params = {
            'voice': 'af_bella',  # Default voice
            'speed': 1.0,
            'split_pattern': r'\n+',  # Split on newlines
        }
        self.default_params.update(self.gen_kwargs)

        self.warmup()

    def warmup(self):
        """Initialize the model with a short text"""
        logger.info(f"Initializing Kokoro TTS on {self.device}")
        try:
            # Run a short generation to load all components
            list(self.pipeline("Warmup", voice=self.default_params['voice'], speed=1.0))
            if self.device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Warmup failed: {str(e)}")
            raise

    def process(self, llm_sentence) -> Generator[np.ndarray, None, None]:
        try:
            if isinstance(llm_sentence, tuple):
                llm_sentence, language_code = llm_sentence

            console.print(f"[green]ASSISTANT: {llm_sentence}")

            audio_chunks = []
            sample_rate = 24000  # Kokoro TTS uses a fixed 24kHz sample rate

            # Process text through the pipeline
            for result in self.pipeline(
                llm_sentence,
                voice=self.default_params['voice'],
                speed=self.default_params['speed'],
                split_pattern=self.default_params['split_pattern']
            ):
                # Validate pipeline output structure
                if len(result) != 3:
                    logger.warning(f"Skipping invalid pipeline output: {result}")
                    continue

                _, _, audio_tensor = result

                # Handle tensor/numpy conversion
                if isinstance(audio_tensor, torch.Tensor):
                    # Process tensor dimensions
                    if audio_tensor.ndim == 2:
                        audio_tensor = audio_tensor.squeeze(0)
                    audio_np = audio_tensor.cpu().detach().numpy()
                else:
                    # Already a numpy array
                    if audio_tensor.ndim == 2:
                        audio_tensor = np.squeeze(audio_tensor, axis=0)
                    audio_np = audio_tensor

                # Skip empty audio segments
                if audio_np.size == 0:
                    continue

                audio_chunks.append(audio_np)

            if not audio_chunks:
                logger.error("No audio generated from pipeline")
                return

            # Concatenate and process final audio
            full_audio = np.concatenate(audio_chunks)
            full_audio = librosa.to_mono(full_audio)
            
            # Resample while still in floating-point (Kokoro uses 24kHz)
            target_sr = 16000
            if sample_rate != target_sr:
                full_audio = librosa.resample(
                    full_audio,
                    orig_sr=sample_rate,
                    target_sr=target_sr
                )
            
            # Convert to int16 after resampling
            full_audio = librosa.util.normalize(full_audio) * 32767.0
            full_audio = full_audio.astype(np.int16)

            # Generate blocks
            for i in range(0, len(full_audio), self.blocksize):
                chunk = full_audio[i:i + self.blocksize]
                if len(chunk) < self.blocksize:
                    chunk = np.pad(chunk, (0, self.blocksize - len(chunk)))
                yield chunk

        except Exception as e:
            logger.error(f"TTS Generation Error: {str(e)}", exc_info=True)
            raise
        finally:
            self.should_listen.set()
