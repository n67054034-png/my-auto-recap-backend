import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

Convert the provided transcript into a natural Burmese movie recap.

Rules:
- Write only in Burmese.
- Do not include English.
- Keep the important story events.
- Make the narration smooth and easy to understand.
- Do not add events that are not in the transcript.
- Do not use bullet points.
- Write as a continuous movie recap narration.
- Do not mention that you are an AI.
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
        "audio/mpeg",
        "audio/mp4",
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type"
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


@app.post("/recap")
def recap(request: RecapRequest):

    recap_text = generate_recap(request.text)

    return {
        "success": True,
        "recap": recap_text
    }


@app.post("/auto-recap")
async def auto_recap(file: UploadFile = File(...)):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured"
        )

    allowed_types = {
        "video/mp4",
        "video/quicktime",
        "video/x-matroska",
        "audio/mpeg",
        "audio/mp4",
        "audio/wav",
        "audio/x-wav",
        "audio/webm",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type"
        )

    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp:

        temp.write(await file.read())
        temp_path = temp.name

    try:

        # STEP 1: Video -> Text
        with open(temp_path, "rb") as audio_file:

            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                response_format="verbose_json",
                temperature=0.0,
            )

        transcript_text = transcription.text

        # STEP 2: Text -> Myanmar Movie Recap
        recap_text = generate_recap(transcript_text)

        return {
            "success": True,
            "filename": file.filename,
            "transcript": transcript_text,
            "recap": recap_text
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
