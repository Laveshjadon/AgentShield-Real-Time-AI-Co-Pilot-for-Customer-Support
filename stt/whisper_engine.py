"""
this runs whisper for stt stuff. 
using faster-whisper so it doesn't take forever.

how to use:
    from stt.whisper_engine import Transcriber
    stt = Transcriber()
    result = stt.transcribe(audio_file_path)
"""

import time
from faster_whisper import WhisperModel
from config.settings import Settings
from config.logger import get_logger

logger = get_logger("stt.whisper")
settings = Settings()

class Transcriber:
    def __init__(self):
        """
        setting up whisper.
        will download the model the first time you run it (base is like 140MB).
        """
        model_size = settings.WHISPER_MODEL
        device = settings.WHISPER_DEVICE

        
        compute_type = "default" if device == "cpu" else "float16"

        logger.info(f"Loading Whisper '{model_size}' model on {device.upper()} (compute: {compute_type})...")

        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            logger.info("Whisper model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise e

    def transcribe(self, audio_data_or_path, language=None) -> dict:
        """
        turns the audio into text.
        added an error boundary so it doesn't crash on bad inputs.

        args:
            audio_data_or_path: file path or just the numpy array
            language: 'en', 'hi', or leave None to guess
        """
        
        if audio_data_or_path is None or (
            hasattr(audio_data_or_path, '__len__') and len(audio_data_or_path) < 1600
        ):
            return {"transcript": "", "language": None, "error": "audio_too_short"}

        start_time = time.time()
        try:
            
            
            segments, info = self.model.transcribe(
                audio_data_or_path,
                beam_size=1,
                language=language,
                vad_filter=False  
            )

            
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text)

            final_text = " ".join(text_parts).strip()
            elapsed = time.time() - start_time

            
            logger.info(
                f"[STT] stt_complete | latency_ms={elapsed * 1000:.0f} "
                f"| transcript_length={len(final_text)} | language={info.language}"
            )

            return {"transcript": final_text, "language": info.language, "error": None}

        except Exception as e:
            
            logger.error(f"[STT] Transcription failed: {type(e).__name__}: {e}")
            return {"transcript": "", "language": None, "error": str(e)}


if __name__ == "__main__":
    import os
    import soundfile as sf
    import numpy as np

    print("\n--- Testing Whisper Engine ---")
    stt = Transcriber()

    
    sample_rate = 16000
    t = np.linspace(0, 2, int(sample_rate * 2), endpoint=False)
    
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    test_file = "test_tone.wav"
    sf.write(test_file, audio, sample_rate)

    print(f"\nTranscribing dummy audio file...")
    result = stt.transcribe(test_file)
    print(f"Result: {result}")

    
    if os.path.exists(test_file):
        os.remove(test_file)
