# Podcast Transcription Service

This repository contains the source code for a podcast transcription service, built according to the provided specification.

## Overview

The service takes an audio or video file and produces a detailed, turn-by-turn transcript with speaker labels and descriptive event tags.

The high-level architecture is as follows:

- **Ingestion API**: A FastAPI-based API for submitting transcription jobs.
- **Processing Pipeline**: A series of workers that perform:
    1.  **Preprocessing**: Audio normalization, voice activity detection (VAD), and chunking.
    2.  **Speaker Diarisation**: Identifying different speakers using `pyannote.audio`.
    3.  **ASR**: Transcribing audio chunks using OpenAI's Whisper.
    4.  **Event Tagging**: Detecting non-verbal sounds and visual cues from dialogue.
    5.  **Post-processing**: Merging and formatting the final transcript.

## Getting Started

1.  Install dependencies:
    ```bash
    pip install -e .
    ```

2.  Set up your environment variables (e.g., in a `.env` file):
    ```
    OPENAI_API_KEY=your_api_key_here
    ```

3.  Run the FastAPI server:
    ```bash
    uvicorn src.main:app --reload
    ```
