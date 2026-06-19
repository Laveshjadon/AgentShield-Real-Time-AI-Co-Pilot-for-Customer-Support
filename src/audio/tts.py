"""Converts AI responses to English or Hindi speech using Edge TTS."""

import os
import uuid
import asyncio
import edge_tts
from config.logger import get_logger
from config.settings import Settings

logger = get_logger("tts.engine")
settings = Settings()

class TTSEngine:
    def __init__(self):
        """
        setting up the tts engine
        change the voices down below if you want
        """
        
        self.english_voice = "en-US-ChristopherNeural" 
        self.hindi_voice = "hi-IN-MadhurNeural"       
        
        
        self.output_dir = "data/audio_out"
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Edge TTS Engine initialized successfully.")

    def _is_hindi(self, text: str) -> bool:
        """
        kinda hacky way to check if the text is in hindi
        just looking for those devanagari characters basically
        """
        
        for char in text:
            if '\u0900' <= char <= '\u097F':
                return True
        return False

    async def synthesize(self, text: str, force_language: str = None) -> str:
        """
        turns text into an mp3 file
        tries to guess the language unless you force it
        """
        if not text or not text.strip():
            logger.warning("Empty text passed to TTS.")
            return None

        
        voice = self.english_voice
        if force_language == "hi" or self._is_hindi(text):
            voice = self.hindi_voice
            
        logger.info(f"Synthesizing audio (Voice: {voice}): '{text[:30]}...'")
        
        try:
            
            filename = f"response_{uuid.uuid4().hex[:8]}.mp3"
            filepath = os.path.join(self.output_dir, filename)
            
            
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(filepath)
            
            logger.debug(f"Audio saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None


if __name__ == "__main__":
    print("\n--- Testing TTS Engine ---")
    
    async def run_test():
        tts = TTSEngine()
        
        print("\n1. Testing English Voice...")
        en_file = await tts.synthesize("Hello! This is AgentShield. How can I help you today?")
        print(f"English audio saved to: {en_file}")
        
        print("\n2. Testing Hindi Voice...")
        hi_file = await tts.synthesize("नमस्ते! मैं एजेंट शील्ड हूँ। मैं आपकी कैसे मदद कर सकता हूँ?")
        print(f"Hindi audio saved to: {hi_file}")
        
    asyncio.run(run_test())
