import os
import tempfile
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import imageio_ffmpeg
from faster_whisper import WhisperModel


app = FastAPI(title="My Auto Recap API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# GROQ
# Used temporarily for Myanmar recap generation.
# We will replace this later with a free solution.
# --------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY is not configured")

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# --------------------------------------------------
# FREE WHISPER
# --------------------------------------------------

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "tiny"
)

print("Loading Free Whisper model:", WHISPER_MODEL)

whisper_model = WhisperModel(
    WHISPER_MODEL,
    device="cpu",
    compute_type="int8"
)

print("Free Whisper model loaded.")


class RecapRequest(BaseModel):
    text: str


# --------------------------------------------------
# HOME
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "My Auto Recap API",
        "transcription": "Free faster-whisper"
    }


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# --------------------------------------------------
# GENERATE MYANMAR RECAP
# TEMPORARILY USING GROQ
# --------------------------------------------------

def generate_recap(text: str):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured"
        )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """
You are a professional Myanmar movie recap writer.

Convert the transcript into a natural Burmese movie recap.

Rules:
- Write only in Burmese.
- Do not include English.
- Keep the important story events.
- Make the narration smooth and easy to understand.
- Do not invent events.
- Do not use bullet points.
- Write as continuous narration.
"""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        temperature=0.3,
        max_tokens=4000
    )

    return response.choices[0].message.content


# --------------------------------------------------
# FREE TRANSCRIPTION FUNCTION
# --------------------------------------------------

def transcribe_audio(audio_path: str):

    segments, info = whisper_model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True
    )

    transcript_parts = []
    timestamped_segments = []

    for segment in segments:

        text = segment.text.strip()

        if text:

            transcript_parts.append(text)

            timestamped_segments.append({
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": text
            })

    return {
        "text": " ".join(transcript_parts),
        "segments": timestamped_segments,
        "language": info.language,
        "language_probability": round(
            info.language_probability,
            4
        ),
        "duration": round(
            info.duration,
            2
        )
    }


# --------------------------------------------------
# TRANSCRIBE VIDEO
# FREE WHISPER
# --------------------------------------------------

@app.post("/transcribe")
async def transcribe_video(
    file: UploadFile = File(...)
):

    suffix = os.path.splitext(
        file.filename or ""
    )[1] or ".mp4"

    input_path = None
    audio_path = None

    try:

        # Save uploaded video

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            temp.write(await file.read())
            input_path = temp.name


        # Get FFmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


        # Create audio file

        audio_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        audio_path = audio_temp.name
        audio_temp.close()


        # Extract audio

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                input_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                audio_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )


        # FREE WHISPER

        result = transcribe_audio(
            audio_path
        )


        return {
            "success": True,
            "filename": file.filename,
            "language": result["language"],
            "language_probability": result[
                "language_probability"
            ],
            "duration": result["duration"],
            "text": result["text"],
            "segments": result["segments"],
            "transcription_engine": "faster-whisper"
        }


    except subprocess.CalledProcessError:

        raise HTTPException(
            status_code=500,
            detail="FFmpeg failed to extract audio"
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        if input_path:

            try:
                os.remove(input_path)
            except Exception:
                pass


        if audio_path:

            try:
                os.remove(audio_path)
            except Exception:
                pass


# --------------------------------------------------
# RECAP
# --------------------------------------------------

@app.post("/recap")
def recap(request: RecapRequest):

    recap_text = generate_recap(
        request.text
    )

    return {
        "success": True,
        "recap": recap_text
    }


# --------------------------------------------------
# AUTO RECAP
# FREE WHISPER + TEMPORARY GROQ RECAP
# --------------------------------------------------

@app.post("/auto-recap")
async def auto_recap(
    file: UploadFile = File(...)
):

    input_suffix = os.path.splitext(
        file.filename or ""
    )[1] or ".mp4"

    input_path = None
    audio_path = None

    try:

        # Save video

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=input_suffix
        ) as temp:

            temp.write(await file.read())
            input_path = temp.name


        # FFmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


        # Audio file

        audio_temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        audio_path = audio_temp.name
        audio_temp.close()


        # Extract audio

        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                input_path,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-acodec",
                "pcm_s16le",
                audio_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )


        # FREE WHISPER

        transcription = transcribe_audio(
            audio_path
        )

        transcript = transcription["text"]


        # Generate Myanmar recap

        recap_text = generate_recap(
            transcript
        )


        return {
            "success": True,
            "filename": file.filename,
            "language": transcription["language"],
            "transcript": transcript,
            "segments": transcription["segments"],
            "recap": recap_text,
            "transcription_engine": "faster-whisper"
        }


    except subprocess.CalledProcessError:

        raise HTTPException(
            status_code=500,
            detail="FFmpeg failed to extract audio"
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


    finally:

        if input_path:

            try:
                os.remove(input_path)
            except Exception:
                pass


        if audio_path:

            try:
                os.remove(audio_path)
            except Exception:
                pass
