"""FastAPI server.  python main.py serve   ->  POST /ask (multipart form)

curl -F "text=What schemes can a farmer in Bihar get?" localhost:8000/ask
curl -F "audio=@question.wav" -F "lang=hi" localhost:8000/ask
curl -F "text=Am I eligible?" -F "image=@card.jpg" localhost:8000/ask
"""
from fastapi import FastAPI, File, Form, UploadFile

from src.agent import build_graph
from src.speech import get_speech
from src.pipeline import Assistant
from src.retriever import HybridRetriever

app = FastAPI(title="Scheme Assistant")
_assistant: Assistant | None = None


def get_assistant() -> Assistant:
    global _assistant
    if _assistant is None:  # heavy models load once, on first request
        _assistant = Assistant(build_graph(HybridRetriever()), get_speech())
    return _assistant


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
async def ask(text: str | None = Form(None), lang: str | None = Form(None), speak: bool = Form(True),
              audio: UploadFile | None = File(None), image: UploadFile | None = File(None)):
    audio_bytes = await audio.read() if audio else None
    image_bytes = await image.read() if image else None
    return get_assistant().ask(
        text=text, audio=audio_bytes, image=image_bytes, lang=lang, speak=speak,
        image_mime=(image.content_type if image else None) or "image/jpeg",
        audio_format=(audio.filename.rsplit(".", 1)[-1].lower() if audio and audio.filename and "." in audio.filename else "wav"),
    )
