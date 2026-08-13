import os
import tempfile
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI(title="My Auto Recap API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not configured")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "My Auto Recap API"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured"
        )

    allowed_types = {
        "video/mp4",
        "video/quicktime",
        "video/x-matroska",
        "video/webm",
        "audio/mpeg",
        "audio/mp4",
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
        "audio/ogg",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )

    video_path = None
    audio_path = None

    try:
        # Save uploaded video/audio temporarily
        suffix = os.path.splitext(file.filename or "")[1] or ".mp4"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:
            temp.write(await file.read())
            video_path = temp.name

        # Create temporary MP3 file
        audio_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )
        audio_path = audio_file.name
        audio_file.close()

        # Extract audio and compress it.
        # 16 kHz mono, 48 kbps MP3 keeps speech quality
        # while making long videos much smaller.
        command = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "libmp3lame",
            "-b:a", "48k",
            audio_path,
        ]

        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail="FFmpeg audio extraction failed: "
                       + process.stderr[-2000:]
            )

        # Check compressed audio size
        audio_size = os.path.getsize(audio_path)

        # Groq free-tier direct upload limit is 25 MB.
        if audio_size > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Audio is still larger than 25 MB after compression. "
                    "Please use a shorter video."
                )
            )

        # Send compressed audio to Groq Whisper
        with open(audio_path, "rb") as audio_file:

            result = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                temperature=0.0,
            )

        return {
            "success": True,
            "filename": file.filename,
            "text": result.text
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if video_path:
            try:
                os.remove(video_path)
            except Exception:
                pass

        if audio_path:
            try:
                os.remove(audio_path)
            except Exception:
                pass
