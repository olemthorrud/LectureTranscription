from pathlib import Path
from typing import List
from src.data_models import TimeSpan, TranscriptUnit, WhisperSegment

def tag_acoustic_events(audio_path: Path, non_speech_spans: List[TimeSpan]) -> List[TranscriptUnit]:
    """
    Detects acoustic events (e.g., music, applause) in the non-speech segments.

    For the MVP, this returns a dummy event. In a real implementation, this
    would use a model like YAMNet on the non-speech audio portions.
    """
    print(f"INFO: (Stub) Tagging acoustic events for {audio_path}...")
    if not non_speech_spans:
        return []

    # Placeholder: Simulate finding theme music in the first non-speech span.
    event_span = non_speech_spans[0]
    return [
        TranscriptUnit(
            start=event_span.start,
            end=event_span.end,
            speaker=None,
            type="event",
            text="*theme music plays*"
        )
    ]

def tag_visual_events(segments: List[WhisperSegment]) -> List[TranscriptUnit]:
    """
    Detects "visual" events by analyzing dialogue for cues (e.g., "look at this slide").

    For the MVP, this uses a simple keyword search. In a real implementation,
    this could involve more sophisticated NLP or an LLM prompt.
    """
    print("INFO: (Stub) Tagging visual events from dialogue...")
    events = []
    for segment in segments:
        if "check out this image" in segment.text.lower():
            # Create an event that happens right at the end of the speech
            events.append(
                TranscriptUnit(
                    start=segment.end,
                    end=segment.end, # Events can be instantaneous
                    speaker=None,
                    type="event",
                    text="*Jamie shows an image, presumably of a horse*"
                )
            )
    return events
