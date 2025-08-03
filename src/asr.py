import openai
from typing import List

from .config import OPENAI_API_KEY
from .data_models import AudioChunk, WhisperSegment

# Initialize the OpenAI client.
# If the API key is not available, the client will be None, and
# transcription calls will fail with a clear error message.
try:
    if OPENAI_API_KEY:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    else:
        client = None
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    client = None

def transcribe_chunk(chunk: AudioChunk) -> List[WhisperSegment]:
    """
    Transcribes a single audio chunk using the OpenAI Whisper API.

    It requires the `OPENAI_API_KEY` to be set in the environment.
    """
    if not client:
        raise ValueError("OpenAI API client is not initialized. Please check your API key.")

    print(f"Transcribing chunk '{chunk.path.name}' with Whisper API...")

    try:
        with open(chunk.path, 'rb') as audio_file:
            # Call the Whisper API with verbose JSON response format to get segments.
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        # The response object from the openai library is a Pydantic model, not a raw dict.
        # The 'segments' attribute contains the list of transcribed segments.
        segments_data = response.segments

        print(f"Successfully transcribed {chunk.path.name}. Received {len(segments_data)} segments.")

        # Convert the API response into our internal WhisperSegment data model.
        # The API returns timestamps relative to the start of the chunk, so we
        # must add the chunk's global start time to get the correct overall timestamp.
        whisper_segments = [
            WhisperSegment(
                start=chunk.timestamp.start + seg['start'],
                end=chunk.timestamp.start + seg['end'],
                text=seg['text'].strip()
            )
            for seg in segments_data
        ]
        return whisper_segments

    except openai.APIError as e:
        error_message = f"OpenAI API Error during transcription of {chunk.path.name}: {e}"
        print(f"ERROR: {error_message}")
        raise RuntimeError(error_message) from e
    except FileNotFoundError:
        raise FileNotFoundError(f"Audio chunk file not found at {chunk.path}")
