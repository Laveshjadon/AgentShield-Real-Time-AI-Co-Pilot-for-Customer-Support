# --- FROM: audio/stt/audio_converter.py ---
"""
Validate and convert uploaded audio files for transcription.

Whisper-style models expect 16 kHz mono WAV input. This module also enforces
file size and duration limits before conversion.
"""

import os
import tempfile
import logging
from typing import Optional

import ffmpeg

from config.logger import get_logger

logger = get_logger("stt.audio_converter")


class AudioConversionError(Exception):
    """Raised when FFmpeg conversion fails."""
    pass


class AudioValidationError(Exception):
    """Raised when uploaded audio violates size or duration limits."""
    pass


class AudioConverter:
    """Convert uploaded audio to 16 kHz mono WAV with managed temp files."""

    def __init__(self, max_size_bytes: int = 25 * 1024 * 1024, max_duration_seconds: float = 600.0):
        self.max_size_bytes = max_size_bytes
        self.max_duration_seconds = max_duration_seconds
        self._temp_files = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def cleanup(self):
        """Delete temporary conversion files."""
        for file_path in self._temp_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temporary audio file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {file_path}: {e}")
        self._temp_files.clear()

    def validate_size(self, input_path: str):
        """Validate the uploaded file size."""
        file_size = os.path.getsize(input_path)
        if file_size > self.max_size_bytes:
            raise AudioValidationError(
                f"File size {file_size / (1024*1024):.1f}MB exceeds the maximum allowed "
                f"limit of {self.max_size_bytes / (1024*1024):.1f}MB."
            )

    def validate_duration(self, input_path: str):
        """Validate audio duration using FFprobe."""
        try:
            probe = ffmpeg.probe(input_path)
            
            format_info = probe.get("format", {})
            duration_str = format_info.get("duration")
            
            if not duration_str:
                
                streams = probe.get("streams", [])
                audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
                if not audio_streams:
                    raise AudioValidationError("No audio streams found in file.")
                
                duration_str = audio_streams[0].get("duration")
                
            if duration_str:
                duration = float(duration_str)
                if duration > self.max_duration_seconds:
                    raise AudioValidationError(
                        f"Audio duration {duration:.1f}s exceeds the maximum allowed "
                        f"limit of {self.max_duration_seconds}s."
                    )
        except ffmpeg.Error as e:
            err_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            logger.error(f"FFprobe failed to analyze file: {err_msg}")
            raise AudioConversionError(f"Failed to analyze audio file format: {err_msg}")
        except ValueError:
            logger.warning("Could not parse duration from ffprobe output. Skipping duration validation.")

    def convert(self, input_path: str) -> str:
        """
        validates the file, runs ffmpeg, and returns the path to the wav file.
        took me a sec to figure out the right ffmpeg flags for whisper.
        """
        self.validate_size(input_path)
        self.validate_duration(input_path)

        try:
            
            fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="agentshield_converted_")
            os.close(fd)
            self._temp_files.append(output_path)

            logger.info(f"Converting audio from {input_path} to {output_path}")

            
            
            
            (
                ffmpeg
                .input(input_path)
                .output(output_path, format='wav', acodec='pcm_s16le', ac=1, ar='16k')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            return output_path

        except ffmpeg.Error as e:
            err_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
            logger.error(f"FFmpeg conversion failed: {err_msg}")
            raise AudioConversionError(f"Failed to convert audio file: {err_msg}")
        except Exception as e:
            logger.error(f"Unexpected error during audio conversion: {e}")
            raise AudioConversionError(f"Unexpected conversion error: {e}")


# --- FROM: audio/stt/vad.py ---
"""
this uses silero vad to figure out if there's actual talking in the audio.
otherwise whisper just makes up random words from background noise, 
which wastes so much cpu it's not even funny.

how to use it:
    vad = VADEngine()
    has_speech = vad.contains_speech(audio_bytes)
"""

import torch
import numpy as np
from config.logger import get_logger

logger = get_logger("stt.vad")


_VAD_WINDOW_SAMPLES = 512  # silero processes 512 samples (32ms) per forward pass at 16kHz


class VADEngine:
    def __init__(self, threshold: float = 0.3):
        """
        setting up the silero vad model.
        threshold is 0 to 1 — 0.3 works well for real mic input.
        lowered from 0.5 because get_speech_timestamps was the old approach
        (batch mode) which needs speech start AND end in the same chunk.
        we now call the model directly on 512-sample windows so it works
        correctly with streaming audio.
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
            self.model.eval()
            logger.info("Silero VAD model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            raise

    def contains_speech(self, audio_data: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        two-stage detection so we're not dependent on any one method:

        stage 1 — energy gate (instant, zero-latency):
            if RMS < 0.005, it's definitely silence. skip silero entirely.
            this avoids running the model on background noise.

        stage 2 — silero model (per-window, streaming-safe):
            call model(chunk, sr) on each 512-sample window rather than
            get_speech_timestamps(). the old get_speech_timestamps() is a BATCH
            utility that needs both speech onset and offset in the same audio
            segment. for 250ms streaming chunks, speech always crosses boundaries
            so it always returned [] → False → nothing ever transcribed.
            calling the model directly avoids this completely.
        """
        try:
            audio_data = audio_data.astype(np.float32)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            # stage 1: energy gate
            rms = float(np.sqrt(np.mean(audio_data ** 2)))
            if rms < 0.005:
                return False  # definitely silence, don't bother silero

            # stage 2: silero per-window
            with torch.no_grad():
                for start in range(0, len(audio_data) - _VAD_WINDOW_SAMPLES + 1, _VAD_WINDOW_SAMPLES):
                    window = audio_data[start : start + _VAD_WINDOW_SAMPLES]
                    tensor = torch.from_numpy(window).unsqueeze(0)  # (1, 512)
                    result = self.model(tensor, sample_rate)
                    prob = result.item() if isinstance(result, torch.Tensor) else float(result)
                    if prob > self.threshold:
                        logger.info(f"[VAD] speech detected: rms={rms:.4f} silero_prob={prob:.3f}")
                        return True

            logger.debug(f"[VAD] no speech: rms={rms:.4f}")
            return False

        except Exception as e:
            # if silero errors, fall back to pure energy — rms > 0.01 is speech
            logger.warning(f"[VAD] silero error ({e}), using energy fallback")
            rms = float(np.sqrt(np.mean(audio_data.astype(np.float32) ** 2)))
            return rms > 0.01


if __name__ == "__main__":
    print("\n--- Testing VAD Engine ---")
    vad = VADEngine()
    
    
    silent_audio = np.zeros(16000, dtype=np.float32)
    has_speech = vad.contains_speech(silent_audio)
    
    print(f"Silent audio contains speech: {has_speech} (Expected: False)")
    
    
    noisy_audio = np.random.uniform(-1, 1, 16000).astype(np.float32)
    has_speech = vad.contains_speech(noisy_audio)
    
    print(f"White noise contains speech: {has_speech} (Expected: False or True depending on noise pattern)")


# --- FROM: audio/stt/whisper_engine.py ---
"""
this runs whisper for stt stuff. 
using faster-whisper so it doesn't take forever.

how to use:
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


# --- FROM: audio/stt/streaming_normalizer.py ---
import asyncio
import logging
import numpy as np
from typing import Optional

from config.logger import get_logger

logger = get_logger("stt.streaming_normalizer")

class StreamingAudioNormalizer:
    """Normalize streaming audio to 16 kHz float32 mono with FFmpeg."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.process: Optional[asyncio.subprocess.Process] = None
        self._buffer = bytearray()
        self._stderr_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the FFmpeg normalization subprocess."""
        cmd = [
            "ffmpeg",
            "-v", "warning",          
            "-i", "pipe:0",           
            "-f", "f32le",            
            "-acodec", "pcm_f32le",   
            "-ar", str(self.sample_rate), 
            "-ac", "1",               
            "pipe:1"                  
        ]
        
        try:
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            self._stderr_task = asyncio.create_task(self._read_stderr())
            logger.info("FFmpeg streaming normalizer started successfully.")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg subprocess: {e}")
            raise

    async def _read_stderr(self):
        """Read and log FFmpeg diagnostics."""
        if not self.process or not self.process.stderr:
            return
        try:
            while True:
                line = await self.process.stderr.readline()
                if not line:
                    break
                logger.warning(f"FFmpeg: {line.decode().strip()}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"FFmpeg stderr reading stopped: {e}")

    async def feed_chunk(self, chunk: bytes):
        """Decode raw bytes with FFmpeg, returning no samples for invalid input."""
        if not self.process or self.process.returncode is not None:
            logger.warning("Tried to feed audio chunk, but FFmpeg process is dead.")
            return

        try:
            self.process.stdin.write(chunk)
            await self.process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError) as e:
            logger.error(f"Pipe broke while feeding FFmpeg. It might have crashed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error feeding audio chunk: {e}")

    async def read_normalized(
        self,
        max_bytes: int = 8192,
        timeout: float = 0.05,
    ) -> np.ndarray:
        """
        pulls the normalized float32 pcm out of stdout.
        makes sure we only grab full floats (4 bytes each).
        """
        if not self.process or self.process.returncode is not None:
            return np.array([], dtype=np.float32)

        try:
            try:
                data = await asyncio.wait_for(
                    self.process.stdout.read(max_bytes),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                return np.array([], dtype=np.float32)
            if not data:
                return np.array([], dtype=np.float32)
                
            self._buffer.extend(data)
            
            
            floats_to_read = len(self._buffer) // 4
            if floats_to_read == 0:
                return np.array([], dtype=np.float32)
                
            bytes_to_process = floats_to_read * 4
            process_bytes = self._buffer[:bytes_to_process]
            
            
            self._buffer = self._buffer[bytes_to_process:]
            
            
            return np.frombuffer(process_bytes, dtype='<f4')
            
        except Exception as e:
            logger.error(f"Error reading from FFmpeg stdout: {e}")
            return np.array([], dtype=np.float32)

    async def close(self):
        """Close all subprocess streams and wait for FFmpeg to exit."""
        process = self.process
        if process:
            if process.returncode is None:
                try:
                    if process.stdin:
                        process.stdin.close()
                        await process.stdin.wait_closed()

                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg did not exit within the timeout; terminating it.")
                    process.kill()
                    await process.wait()
                except Exception as e:
                    logger.error(f"Error while closing FFmpeg: {e}")

        if self._stderr_task:
            try:
                await asyncio.wait_for(self._stderr_task, timeout=1.0)
            except asyncio.TimeoutError:
                self._stderr_task.cancel()
                try:
                    await self._stderr_task
                except asyncio.CancelledError:
                    pass
            except asyncio.CancelledError:
                pass
            self._stderr_task = None

        self.process = None
        self._buffer.clear()
        logger.info("FFmpeg streaming normalizer shut down.")


