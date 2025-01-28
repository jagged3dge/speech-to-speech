from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
import librosa
from rich.console import Console
from baseHandler import BaseHandler
import logging
from threading import Thread

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

console = Console()

WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE = {
    "en": "eng", # English
    "fr": "fra", # French
    "es": "spa", # Spanish
    "ko": "kor", # Korean
    "hi": "hin", # Hindi
    "ar": "ara", # Arabic
    "ar": "hyw", # Armenian
    "az": "azb", # Azerbaijani
    "bu": "bul", # Bulgarian
    "ca": "cat", # Catalan
    "nl": "nld", # Dutch
    "fi": "fin", # Finnish
    "fr": "fra", # French
    "de": "deu", # German
    "el": "ell", # Greek
    "he": "heb", # Hebrew
    "hu": "hun", # Hungarian
    "is": "isl", # Icelandic
    "id": "ind", # Indonesian
    "ka": "kan", # Kannada
    "kk": "kaz", # Kazakh
    "lv": "lav", # Latvian
    "zl": "zlm", # Malay
    "ma": "mar", # Marathi
    "fa": "fas", # Persian
    "po": "pol", # Polish
    "pt": "por", # Portuguese
    "ro": "ron", # Romanian
    "ru": "rus", # Russian
    "sw": "swh", # Swahili
    "sv": "swe", # Swedish
    "tg": "tgl", # Tagalog
    "ta": "tam", # Tamil
    "th": "tha", # Thai
    "tu": "tur", # Turkish
    "uk": "ukr", # Ukrainian
    "ur": "urd", # Urdu
    "vi": "vie", # Vietnamese
    "cy": "cym", # Welsh
}

class FacebookMMSTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        interrupt_event=None,
        device="cuda",
        torch_dtype="float32",
        language="en",
        stream=True,
        chunk_size=512,  # Already has chunk_size parameter
        **kwargs
    ):
        self.should_listen = should_listen
        self.interrupt_event = interrupt_event
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        self.stream = stream
        self.chunk_size = chunk_size  # This is already correct
        self.language = language

        self.load_model(self.language)
        self.warmup()

    def load_model(self, language_code):
        try:
            model_name = f"facebook/mms-tts-{WHISPER_LANGUAGE_TO_FACEBOOK_LANGUAGE[language_code]}"
            logger.info(f"Loading model: {model_name}")
            self.model = VitsModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.language = language_code
        except KeyError:
            logger.warning(f"Unsupported language: {language_code}. Falling back to English.")
            self.load_model("en")

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        output = self.generate_audio("Hello, this is a test")

    def generate_audio(self, text):
        """Generate audio from text using Facebook MMS TTS"""
        if not text:
            logger.warning("Received empty text input")
            return None

        try:
            logger.debug(f"Generating audio for text: {text}")
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(self.device).long()
            attention_mask = inputs.attention_mask.to(self.device)
            
            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            return output.waveform[0].cpu().numpy()
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

        if language_code is not None and self.language != language_code:
            try:
                logger.info(f"Switching language from {self.language} to {language_code}")
                self.load_model(language_code)
            except KeyError:
                console.print(f"[red]Language {language_code} not supported. Using {self.language} instead.")

        try:
            # Generate audio
            logger.debug("Starting TTS generation...")
            audio_out = self.generate_audio(llm_sentence)
            if audio_out is None:
                logger.error("TTS generation failed")
                self.should_listen.set()
                return

            logger.debug(f"Generated audio shape: {audio_out.shape}")

            # Process chunks with interruption check
            chunk_size = self.chunk_size
            audio_data = (audio_out * 32768).astype(np.int16)
            
            for i in range(0, len(audio_data), chunk_size):
                if self.interrupt_event and self.interrupt_event.is_set():
                    logger.info("TTS interrupted during playback")
                    self.interrupt_event.clear()
                    self.should_listen.set()
                    break

                chunk = audio_data[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad last chunk if needed
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                # logger.debug(f"Sending audio chunk {i//chunk_size + 1}: shape={chunk.shape}, dtype={chunk.dtype}")
                yield chunk

        except Exception as e:
            logger.error(f"Error in TTS processing: {e}")
            logger.exception("Full traceback:")
        finally:
            self.should_listen.set()
