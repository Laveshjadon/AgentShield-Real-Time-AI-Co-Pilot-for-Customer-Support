import pytest
import asyncio
import io
import wave
import numpy as np

from stt.streaming_normalizer import StreamingAudioNormalizer

def create_dummy_wav(sample_rate=44100, channels=2, duration_sec=1.0):
    """Create a stereo WAV file in memory for FFmpeg decoding tests."""
    num_samples = int(sample_rate * duration_sec)
    
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    
    
    if channels == 2:
        audio_data = np.column_stack((audio_data, audio_data)).flatten()
        
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2) 
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
        
    return buf.getvalue()

@pytest.mark.asyncio
async def test_streaming_normalizer():
    
    dummy_wav_bytes = create_dummy_wav(44100, 2, 1.0)
    
    normalizer = StreamingAudioNormalizer(sample_rate=16000)
    await normalizer.start()
    
    try:
        
        chunk_size = 1024
        output_arrays = []
        
        for i in range(0, len(dummy_wav_bytes), chunk_size):
            chunk = dummy_wav_bytes[i:i+chunk_size]
            await normalizer.feed_chunk(chunk)
            
            
            out = await normalizer.read_normalized(max_bytes=8192, timeout=0.1)
            if out.size > 0:
                output_arrays.append(out)
                
        
        await asyncio.sleep(0.5)
        
        
        if normalizer.process.stdin:
            normalizer.process.stdin.close()
            
        while True:
            out = await normalizer.read_normalized(max_bytes=8192, timeout=1.0)
            if out.size == 0:
                break
            output_arrays.append(out)
            
        final_audio = np.concatenate(output_arrays) if output_arrays else np.array([], dtype=np.float32)
        
        
        
        assert final_audio.dtype == np.float32
        assert len(final_audio) > 15000, f"Expected ~16000 samples, got {len(final_audio)}"
        
    finally:
        await normalizer.close()
