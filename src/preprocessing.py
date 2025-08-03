import os
import subprocess
from pathlib import Path
from typing import List
from math import ceil

from pydub import AudioSegment, exceptions as pydub_exceptions

from src.data_models import TimeSpan, AudioChunk

MAX_FILE_SIZE_MB = 25
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024

def normalise(path: Path) -> Path:
    """
    Converts any input audio/video file to a mono, 16kHz WAV file using ffmpeg.
    This is the standard format required by Whisper.
    """
    if not path.exists():
        raise FileNotFoundError(f"Input file not found at {path}")

    normalised_path = path.parent / f"{path.stem}_normalised.wav"

    command = [
        'ffmpeg',
        '-i', str(path),       # Input file
        '-ar', '16000',        # Audio sample rate: 16kHz
        '-ac', '1',            # Audio channels: 1 (mono)
        '-c:a', 'pcm_s16le',   # Audio codec: PCM 16-bit little-endian
        str(normalised_path),
        '-y',                  # Overwrite output file if it exists
        '-hide_banner',        # Hide unnecessary ffmpeg banner info
        '-loglevel', 'error'   # Only show errors
    ]

    print(f"Normalising audio with ffmpeg for file: {path.name}")
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"Successfully normalised audio to {normalised_path}")
        return normalised_path
    except subprocess.CalledProcessError as e:
        error_message = f"ffmpeg failed for {path.name}. Error: {e.stderr}"
        print(f"ERROR: {error_message}")
        raise RuntimeError(error_message) from e
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please ensure ffmpeg is installed and in your system's PATH.")

def detect_speech(path: Path) -> List[TimeSpan]:
    """
    Detects speech regions in the audio file.
    NOTE: For the MVP, this remains a stub. A real implementation would use
    a VAD model like Silero VAD.
    """
    print("INFO: (Stub) Skipping actual speech detection for MVP.")
    try:
        audio = AudioSegment.from_wav(path)
        duration_sec = len(audio) / 1000
        # Return a single span covering the whole audio
        return [TimeSpan(start=0.0, end=duration_sec)]
    except pydub_exceptions.CouldntDecodeError:
        print(f"Warning: Could not decode {path} to get duration. Returning dummy span.")
        return [TimeSpan(start=0.0, end=60.0)] # Fallback

def split_for_whisper(audio_path: Path, speech_spans: List[TimeSpan]) -> List[AudioChunk]:
    """
    Splits the normalised audio into chunks suitable for the Whisper API (<25MB).
    This implementation splits based on file size, which is simpler for the MVP.
    A more advanced version would use the speech_spans to create more logical splits.
    """
    print(f"Checking if audio needs splitting for Whisper (<{MAX_FILE_SIZE_MB}MB)...")

    try:
        file_size = os.path.getsize(audio_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found at {audio_path} for splitting.")

    if file_size <= MAX_FILE_SIZE:
        print("Audio file is small enough, no splitting required.")
        try:
            audio = AudioSegment.from_wav(audio_path)
            duration_sec = len(audio) / 1000
        except pydub_exceptions.CouldntDecodeError:
            duration_sec = 0 # Should not happen if normalise worked
        return [AudioChunk(path=audio_path, timestamp=TimeSpan(start=0, end=duration_sec))]

    print("Audio file is larger than 25MB, splitting into parts...")
    try:
        audio = AudioSegment.from_wav(audio_path)
    except pydub_exceptions.CouldntDecodeError as e:
        raise RuntimeError(f"Could not read audio file {audio_path} with pydub for splitting.") from e

    duration_ms = len(audio)
    num_parts = ceil(file_size / MAX_FILE_SIZE)
    chunk_length_ms = ceil(duration_ms / num_parts)

    chunks = []
    for i in range(num_parts):
        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, duration_ms)

        audio_chunk = audio[start_ms:end_ms]
        chunk_path = audio_path.with_name(f"{audio_path.stem}_part_{i+1}.wav")

        print(f"Exporting chunk {i+1}/{num_parts} ({start_ms/1000:.2f}s to {end_ms/1000:.2f}s) to {chunk_path}")
        audio_chunk.export(chunk_path, format="wav")

        chunks.append(
            AudioChunk(
                path=chunk_path,
                timestamp=TimeSpan(start=start_ms / 1000, end=end_ms / 1000)
            )
        )

    return chunks
