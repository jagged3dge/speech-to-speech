from melo.api import TTS
import logging
from baseHandler import BaseHandler
import librosa
import numpy as np
from rich.console import Console
import torch
from threading import Thread
from time import perf_counter
from TTS.streamer import InterruptibleParlerTTSStreamer

logger = logging.getLogger(__name__)

console = Console()

WHISPER_LANGUAGE_TO_MELO_LANGUAGE = {
    "en": "EN",
    "fr": "FR",
    "es": "ES",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}

WHISPER_LANGUAGE_TO_MELO_SPEAKER = {
    "en": "EN-BR",
    "fr": "FR",
    "es": "ES",
    "zh": "ZH",
    "ja": "JP",
    "ko": "KR",
}


class MeloTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        interrupt_event=None,
        device="mps",
        language="en",
        speaker_to_id="en",
        gen_kwargs={},  # Unused
        blocksize=512,
    ):
        self.should_listen = should_listen
        self.interrupt_event = interrupt_event  # Store interrupt event
        self.device = device
        self.language = language
        self.chunk_size = blocksize
        self.model = TTS(
            language=WHISPER_LANGUAGE_TO_MELO_LANGUAGE[self.language], device=device
        )
        self.speaker_id = self.model.hps.data.spk2id[
            WHISPER_LANGUAGE_TO_MELO_SPEAKER[speaker_to_id]
        ]
        self.blocksize = blocksize
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.tts_to_file("text", self.speaker_id, quiet=True)

    def generate_audio(self, text):
        """Generate audio from text using Melo TTS"""
        try:
            logger.debug(f"Generating audio for text: {text}")
            audio = self.model.tts_to_file(text, self.speaker_id, quiet=True)
            if audio is None:
                logger.error("Failed to generate audio")
                return None
            return audio
        except Exception as e:
            logger.error(f"Error in generate_audio: {str(e)}")
            logger.exception("Full traceback:")
            return None

    def process(self, llm_sentence):
        language_code = None
        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence

        console.print(f"[green]ASSISTANT: {llm_sentence}")
        logger.debug(f"Processing text: {llm_sentence}")
        logger.debug(f"Language code: {language_code}")

        try:
            # Generate audio
            logger.debug("Starting Melo TTS generation...")
            audio_out = self.generate_audio(llm_sentence)
            if audio_out is None:
                logger.error("Melo TTS generation failed")
                self.should_listen.set()
                return

            logger.debug(f"Generated audio shape: {audio_out.shape}")

            # Process chunks with interruption check
            chunk_size = self.chunk_size
            audio_data = librosa.resample(audio_out, orig_sr=44100, target_sr=16000)
            audio_data = (audio_data * 32768).astype(np.int16)
            
            for i in range(0, len(audio_data), chunk_size):
                if self.interrupt_event and self.interrupt_event.is_set():
                    logger.info("Melo TTS interrupted during playback")
                    self.interrupt_event.clear()
                    self.should_listen.set()
                    break

                chunk = audio_data[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad last chunk if needed
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                # logger.debug(f"Sending audio chunk {i//chunk_size + 1}/{len(audio_data)//chunk_size + 1}")
                yield chunk.tobytes()

        except Exception as e:
            logger.error(f"Error in Melo TTS processing: {e}")
            logger.exception("Full traceback:")
        finally:
            self.should_listen.set()
