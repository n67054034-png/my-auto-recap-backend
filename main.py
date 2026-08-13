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


# =========================
# HOME
# =========================

@app.get("/")
def home():
    return {
        "status": "online",
        "service": "My Auto Recap API"
    }


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# =========================
# TRANSCRIBE VIDEO
# =========================

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
        "audio/flac",
        "audio/ogg",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type"
        )

    suffix = os.path.splitext(
        file.filename or ""
    )[1] or ".mp4"

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


# =========================
# RECAP REQUEST MODEL
# =========================

class RecapRequest(BaseModel):
    text: str


# =========================
# GENERATE BURMESE RECAP
# =========================

@app.post("/recap")
def generate_recap(request: RecapRequest):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured"
        )

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text is required"
        )

    prompt = f"""
You are a professional Burmese movie recap writer.

Convert the following movie/video transcript into a natural,
easy-to-understand Burmese movie recap.

IMPORTANT RULES:

1. Write ONLY in Burmese.
2. Do NOT write English sentences.
3. Do NOT translate word-by-word.
4. Make it sound like a natural Burmese movie recap narrator.
5. Keep the important story events.
6. Remove unnecessary repetition.
7. Keep character names when they are clearly known.
8. Explain events in the correct chronological order.
9. Do not invent major events that are not in the transcript.
10. Make the narration interesting and easy to follow.
11. Do not add a title unless necessary.
12. Do not use bullet points.
13. Write as continuous narration suitable for a TikTok/movie recap voice-over.

TRANSCRIPT:

{request.text}
"""

    try:

        result = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert Burmese movie recap writer. "
                        "Always produce natural Burmese narration."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_completion_tokens=8192,
        )

        recap_text = result.choices[0].message.content

        return {
            "success": True,
            "recap": recap_text
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
    )
