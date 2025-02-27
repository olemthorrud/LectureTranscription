import os
import sys
import openai
import subprocess
from pydub import AudioSegment
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("API_KEY")

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB limit for Whisper API

def extract_audio(video_path, audio_path):
    """Extracts audio from a video file using ffmpeg."""
    command = [
        'ffmpeg', '-i', video_path, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', audio_path, '-y'
    ]
    subprocess.run(command, check=True)

def get_required_parts(audio_path):
    """Determines how many parts are needed to fit within Whisper's 25MB limit."""
    file_size = os.path.getsize(audio_path)
    num_parts = max(1, (file_size // MAX_FILE_SIZE) + 1)  # Ensure at least 1 part
    return num_parts

def split_audio(audio_path, num_parts):
    """Splits the audio into the required number of parts."""
    audio = AudioSegment.from_wav(audio_path)
    chunk_length = len(audio) // num_parts
    split_files = []

    for i in range(num_parts):
        start = i * chunk_length
        end = (i + 1) * chunk_length if i < num_parts - 1 else len(audio)
        chunk = audio[start:end]
        chunk_path = f"{audio_path.replace('.wav', '')}_part{i+1}.wav"
        chunk.export(chunk_path, format="wav")
        split_files.append(chunk_path)

    os.remove(audio_path)  # Remove original large audio file after splitting
    return split_files

def transcribe_audio(audio_path, api_key):
    """Transcribes a single audio file."""
    client = openai.OpenAI(api_key=api_key)

    with open(audio_path, 'rb') as audio_file:
        try:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return response.text
        except openai.APIError as e:
            print(f"API Error: {e}")
            return None

def process_transcription(audio_parts, output_path, api_key):
    """Processes transcription sequentially, ensuring correct order."""
    with open(output_path, 'w', encoding='utf-8') as output_file:
        for part in audio_parts:
            print(f"Transcribing {part}...")
            transcription = transcribe_audio(part, api_key)

            if transcription:
                output_file.write(transcription + "\n")

            os.remove(part)  # Cleanup after processing

def main():
    if len(sys.argv) != 2:
        print("Usage: python transcription.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        sys.exit(1)

    base_name = os.path.splitext(video_path)[0]
    audio_path = f"{base_name}_audio.wav"
    transcription_path = f"{base_name}.txt"

    print("Extracting audio...")
    extract_audio(video_path, audio_path)

    print("Determining required parts...")
    num_parts = get_required_parts(audio_path)
    print(f"Splitting into {num_parts} parts...")

    audio_parts = split_audio(audio_path, num_parts)

    print("Starting transcription...")
    process_transcription(audio_parts, transcription_path, api_key)

    print(f"Done! Transcription saved to {transcription_path}")

if __name__ == "__main__":
    main()