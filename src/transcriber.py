import os 
import whisper
import yt_dlp
from typing import Any, cast


def download_and_transcribe(yt_url: str) -> list[dict[str, Any]]:
    """
    Downloads a YouTube video and transcribes its audio using the Whisper model.
    """

    ydl_opts: Any = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloads/audio.%(ext)s',
    }

    os.makedirs('downloads', exist_ok=True)

    with yt_dlp.YoutubeDL(params=ydl_opts) as ydl:
        ydl.download([yt_url])

    audio_path: str = 'downloads/audio.mp3'

    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    segments = result.get("segments", [])

    if not isinstance(segments, list):
        raise TypeError("Transcription result did not include a valid 'segments' list.")

    return cast(list[dict[str, Any]], segments)

