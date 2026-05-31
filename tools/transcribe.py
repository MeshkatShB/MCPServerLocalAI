# tools/transcribe.py

import base64
import tempfile
import whisper

# Load the Whisper model once
CACHE_DIR = "./cache"
model = whisper.load_model("base", device="cuda:0", download_root=CACHE_DIR)  # Options: tiny, base, small, medium, large

# def transcribe_audio(audio_base64: str) -> str:
#     """
#     Transcribes base64-encoded audio using OpenAI's Whisper model.

#     Args:
#         audio_base64 (str): Base64-encoded audio data.

#     Returns:
#         str: Transcribed text.
#     """
#     # Decode the base64 audio
#     audio_data = base64.b64decode(audio_base64)

#     # Write the audio data to a temporary file
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
#         temp_audio_file.write(audio_data)
#         temp_audio_file.flush()

#         # Transcribe the audio file
#         result = model.transcribe(temp_audio_file.name)

#     return result["text"]


def transcribe_audio(file_path: str) -> str:
    """
    Transcribes the audio file at the given path using OpenAI's Whisper model.

    Args:
        file_path (str): The path to the audio file.

    Returns:
        str: The transcribed text.
    """
    result = model.transcribe(file_path)
    return result["text"]

file_path = "./harvard.wav"
transcribe_audio(file_path)
