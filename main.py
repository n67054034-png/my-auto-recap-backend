import os
import tempfile

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


# =========================
# MOVIE RECAP GENERATOR
# =========================

@app.post("/recap")
async def generate_recap(text: str):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not configured"
        )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text is required"
        )

    prompt = f"""
You are a professional movie recap writer.

Rewrite the following movie transcript into a natural Burmese
movie recap narration.

IMPORTANT RULES:

1. Write ONLY in Burmese language.
2. Do not write English.
3. Do not translate word-for-word.
4. Make it sound like a human movie recap narrator.
5. Keep the story events in the correct order.
6. Do not invent major events that are not in the transcript.
7. Keep important character names if they are clearly mentioned.
8. Make the narration interesting and easy to understand.
9. Remove unnecessary repeated dialogue and filler.
10. Do not add headings.
11. Do not add explanations about what you are doing.
12. Return ONLY the finished Burmese movie recap.

TRANSCRIPT:

{text}
"""

    try:

        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert Burmese movie recap writer."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=8000,
        )

        recap = completion.choices[0].message.content

        return {
            "success": True,
            "recap": recap
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
)
