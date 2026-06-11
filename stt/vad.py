"""
this uses silero vad to figure out if there's actual talking in the audio.
otherwise whisper just makes up random words from background noise, 
which wastes so much cpu it's not even funny.

how to use it:
    from stt.vad import VADEngine
    vad = VADEngine()
    has_speech = vad.contains_speech(audio_bytes)
"""

import torch
import numpy as np
from config.logger import get_logger

logger = get_logger("stt.vad")


class VADEngine:
    def __init__(self, threshold: float = 0.5):
        """
        setting up the silero vad model.
        threshold is 0 to 1, higher means it's super picky about what counts as speech.
        """
        self.threshold = threshold
        logger.info("Loading Silero VAD model...")
        
        try:
            
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True
            )
            (self.get_speech_timestamps, _, self.read_audio, *_) = utils
            logger.info("Silero VAD model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            raise

    def contains_speech(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        checks the raw audio and returns True if someone is talking.
        make sure you give it 16kHz float32 audio.
        """
        try:
            
            audio_tensor = torch.from_numpy(audio_data).float()
            
            
            if len(audio_tensor.shape) > 1 and audio_tensor.shape[1] > 1:
                audio_tensor = torch.mean(audio_tensor, dim=1)
                
            
            if len(audio_tensor.shape) > 1:
                audio_tensor = audio_tensor.squeeze()

            
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.model, 
                sampling_rate=sample_rate,
                threshold=self.threshold
            )
            
            
            return len(speech_timestamps) > 0
            
        except Exception as e:
            logger.warning(f"VAD error, defaulting to True: {e}")
            return True


if __name__ == "__main__":
    print("\n--- Testing VAD Engine ---")
    vad = VADEngine()
    
    
    silent_audio = np.zeros(16000, dtype=np.float32)
    has_speech = vad.contains_speech(silent_audio)
    
    print(f"Silent audio contains speech: {has_speech} (Expected: False)")
    
    
    noisy_audio = np.random.uniform(-1, 1, 16000).astype(np.float32)
    has_speech = vad.contains_speech(noisy_audio)
    
    print(f"White noise contains speech: {has_speech} (Expected: False or True depending on noise pattern)")
