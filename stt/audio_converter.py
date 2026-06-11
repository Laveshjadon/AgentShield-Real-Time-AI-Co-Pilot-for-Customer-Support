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
