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
