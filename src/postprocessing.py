from typing import List
from src.data_models import TranscriptUnit, WhisperSegment, SpeakerTurn

def merge_transcripts(
    whisper_segments: List[WhisperSegment],
    speaker_turns: List[SpeakerTurn],
    events: List[TranscriptUnit]
) -> List[TranscriptUnit]:
    """
    Merges ASR segments, speaker turns, and tagged events into a final,
    ordered list of TranscriptUnit objects.

    The logic involves:
    1. Assigning speaker labels to each Whisper segment.
    2. Combining speech and event units.
    3. Sorting all units by their start time.
    4. (Optional) Collapsing consecutive segments from the same speaker.
    """
    print("INFO: (Stub) Merging transcription results...")

    speech_units = []
    for seg in whisper_segments:
        # Find the speaker for the segment by checking for temporal overlap.
        # This is a simplified approach; a more robust one would find the speaker
        # with the maximum overlap.
        assigned_speaker = "UNKNOWN"
        for turn in speaker_turns:
            # Check if the segment's midpoint falls within a speaker's turn
            segment_midpoint = seg.start + (seg.end - seg.start) / 2
            if turn.start <= segment_midpoint < turn.end:
                assigned_speaker = turn.speaker_id
                break

        speech_units.append(
            TranscriptUnit(
                start=seg.start,
                end=seg.end,
                speaker=assigned_speaker,
                type="speech",
                text=seg.text
            )
        )

    # Combine speech and event units and sort them chronologically
    all_units = sorted(speech_units + events, key=lambda x: x.start)

    # In a real implementation, you might add logic here to collapse
    # consecutive speech units from the same speaker into a single entry.

    return all_units
