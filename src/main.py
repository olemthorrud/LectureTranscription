import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from typing import Optional, List

# Import the pipeline modules and data models
from . import preprocessing
from . import diarisation
from . import asr
from . import event_tagging
from . import postprocessing
from .data_models import TranscriptUnit, TimeSpan

# --- App Initialization ---
app = FastAPI(
    title="Podcast Transcription Service",
    description="A service to transcribe audio files with speaker labels and event tagging, based on a detailed specification.",
    version="0.1.0",
)

# --- Temporary Storage & Job DB ---
# In a production environment, this would be replaced with a proper object store (like S3)
# and a database (like Postgres or Redis).
TEMP_STORAGE_PATH = Path("./temp_storage")
TEMP_STORAGE_PATH.mkdir(exist_ok=True)
jobs_db = {} # In-memory dictionary to act as a simple job database.

# --- API Endpoints ---

@app.post("/transcriptions", status_code=202, response_model=dict)
async def create_transcription_job(
    file: UploadFile = File(..., description="Audio or video file to transcribe."),
    webhook_url: Optional[str] = Form(None, description="Optional URL for webhook notifications."),
    output_format: str = Form("json", description="Desired output format: 'json', 'txt', or 'srt'."),
):
    """
    Accepts a file and starts the transcription pipeline.

    For this MVP, the pipeline runs synchronously and blocks the request,
    but it is architected to be moved to a background worker easily.
    """
    job_id = str(uuid.uuid4())
    input_path = TEMP_STORAGE_PATH / f"{job_id}_{file.filename}"

    # 1. Save the uploaded file to our temporary storage.
    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()

    jobs_db[job_id] = {"status": "processing", "result": None}

    # 2. Execute the stubbed transcription pipeline.
    try:
        # Pre-processing
        normalised_path = preprocessing.normalise(input_path)
        speech_spans = preprocessing.detect_speech(normalised_path)
        audio_chunks = preprocessing.split_for_whisper(normalised_path, speech_spans)

        # Diarisation
        speaker_turns = diarisation.diarise_speakers(normalised_path)

        # ASR
        all_segments = [seg for chunk in audio_chunks for seg in asr.transcribe_chunk(chunk)]

        # Event Tagging
        # A real implementation would need the audio duration to find non-speech spans accurately.
        # We'll create some dummy spans for the stub.
        non_speech_spans = [TimeSpan(start=10.2, end=12.0)]
        acoustic_events = event_tagging.tag_acoustic_events(normalised_path, non_speech_spans)
        visual_events = event_tagging.tag_visual_events(all_segments)
        all_events = acoustic_events + visual_events

        # Post-processing
        final_transcript = postprocessing.merge_transcripts(
            whisper_segments=all_segments,
            speaker_turns=speaker_turns,
            events=all_events
        )

        # 3. Store the final result and update job status.
        jobs_db[job_id] = {"status": "completed", "result": [unit.__dict__ for unit in final_transcript]}

    except Exception as e:
        jobs_db[job_id] = {"status": "failed", "error": str(e)}
        # In a real app, you would have more robust logging here.
        print(f"ERROR: Job {job_id} failed with exception: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during transcription: {e}")

    return {
        "job_id": job_id,
        "status": "processing_started",
        "message": f"Job started. Check status at /transcriptions/{job_id}",
    }


@app.get("/transcriptions/{job_id}", response_model=dict)
def get_transcription_result(job_id: str):
    """
    Retrieves the status or the final result of a transcription job.
    """
    job = jobs_db.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job with ID '{job_id}' not found.")

    if job["status"] == "completed":
        # Structure the response to match the spec's example output
        return {
            "podcast_id": job_id,
            "duration_sec": 60.0, # Placeholder duration
            "transcript": job["result"],
        }

    return {"job_id": job_id, "status": job["status"]}
