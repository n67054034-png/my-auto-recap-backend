import os
import tempfile
import subprocess

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import imageio_ffmpeg


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


class RecapRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "status": "online",
        "service": "My Auto Recap API"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


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


@app.post("/recap")
def recap(request: RecapRequest):

    recap_text = generate_recap(request.text)

    return {
        "success": True,
        "recap": recap_text
    }


@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured"
        )

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        temp.write(await file.read())
        temp_path = temp.name

    try:

        with open(temp_path, "rb") as audio_file:

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

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        try:
            os.remove(temp_path)
        except Exception:
            pass


@app.post("/auto-recap")
async def auto_recap(file: UploadFile = File(...)):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured"
        )

    input_suffix = os.path.splitext(
        file.filename or ""
    )[1] or ".mp4"

    input_path = None
    audio_path = None

    try:

        # Save uploaded video
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=input_suffix
        ) as temp:

            temp.write(await file.read())
            input_path = temp.name


        # Get ffmpeg bundled with imageio-ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


        # Convert video audio to small MP3
        audio_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        )

        audio_path = audio_file.name
        audio_file.close()


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
                "-b:a",
                "48k",
                audio_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )


        # Transcribe compressed audio
        with open(audio_path, "rb") as audio:

            result = client.audio.transcriptions.create(
                file=audio,
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                temperature=0.0
            )


        transcript = result.text


        # Generate Burmese recap
        recap_text = generate_recap(transcript)


        return {
            "success": True,
            "filename": file.filename,
            "transcript": transcript,
            "recap": recap_text
        }


    except subprocess.CalledProcessError as e:

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
