from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

# General-purpose time span
@dataclass
class TimeSpan:
    """Represents a time interval with a start and end point in seconds."""
    start: float
    end: float

# §1. Ingestion & Pre-processing
@dataclass
class AudioChunk:
    """Represents a chunk of the source audio with its global time offset."""
    path: Path
    timestamp: TimeSpan

# §2. Speaker Diarisation
@dataclass
class SpeakerTurn:
    """Represents a continuous segment of speech from a single speaker."""
    start: float
    end: float
    speaker_id: str

# §3. ASR (Whisper)
@dataclass
class WhisperSegment:
    """Represents a segment of transcribed text from Whisper."""
    start: float
    end: float
    text: str

# §5. Post-processing & Merging (Final Output Object)
@dataclass
class TranscriptUnit:
    """
    Represents a single unit in the final transcript.
    This can be a line of speech or a descriptive event.
    """
    start: float
    end: float
    speaker: Optional[str]  # Speaker is None for non-speech events
    type: Literal["speech", "event"]
    text: str  # The transcribed text or the event description
