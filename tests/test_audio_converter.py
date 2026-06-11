import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock

import ffmpeg

from stt.audio_converter import AudioConverter, AudioConversionError, AudioValidationError

@pytest.fixture
def dummy_file():
    fd, path = tempfile.mkstemp(suffix=".mp3")
    with os.fdopen(fd, 'w') as f:
        f.write("dummy audio data")
    yield path
    if os.path.exists(path):
        os.remove(path)

def test_validate_size_exceeded(dummy_file):
    
    converter = AudioConverter(max_size_bytes=1)
    with pytest.raises(AudioValidationError, match="exceeds the maximum allowed limit"):
        converter.validate_size(dummy_file)

def test_validate_size_ok(dummy_file):
    
    converter = AudioConverter(max_size_bytes=1024*1024)
    converter.validate_size(dummy_file)

@patch("stt.audio_converter.ffmpeg.probe")
def test_validate_duration_exceeded(mock_probe, dummy_file):
    
    mock_probe.return_value = {"format": {"duration": "1000.0"}}
    
    converter = AudioConverter(max_duration_seconds=600.0)
    with pytest.raises(AudioValidationError, match="exceeds the maximum allowed limit"):
        converter.validate_duration(dummy_file)

@patch("stt.audio_converter.ffmpeg.probe")
def test_validate_duration_ok(mock_probe, dummy_file):
    
    mock_probe.return_value = {"format": {"duration": "300.0"}}
    
    converter = AudioConverter(max_duration_seconds=600.0)
    converter.validate_duration(dummy_file)

@patch("ffmpeg.probe")
@patch("subprocess.Popen")
def test_convert_success_and_cleanup(mock_popen, mock_probe, dummy_file):
    mock_probe.return_value = {"format": {"duration": "10.0"}}
    
    
    mock_process = MagicMock()
    mock_process.communicate.return_value = (b"stdout", b"stderr")
    mock_process.poll.return_value = 0
    mock_popen.return_value = mock_process
    
    with AudioConverter() as converter:
        output_path = converter.convert(dummy_file)
        assert os.path.exists(output_path)
        assert output_path.endswith(".wav")
        assert len(converter._temp_files) == 1
        assert output_path in converter._temp_files
            
    assert not os.path.exists(output_path)
    assert len(converter._temp_files) == 0

@patch("ffmpeg.probe")
@patch("subprocess.Popen")
def test_convert_ffmpeg_error(mock_popen, mock_probe, dummy_file):
    mock_probe.return_value = {"format": {"duration": "10.0"}}
    
    
    mock_process = MagicMock()
    mock_process.communicate.return_value = (b"stdout", b"Invalid data found")
    mock_process.poll.return_value = 1
    mock_popen.return_value = mock_process
    
    with pytest.raises(AudioConversionError, match="Invalid data found"):
        with AudioConverter() as converter:
            converter.convert(dummy_file)
