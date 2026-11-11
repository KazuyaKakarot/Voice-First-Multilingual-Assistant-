"""End-to-end assistant: voice/text/image in, grounded answer (text + voice) out, with per-stage latency.

audio -> ASR -> detect language -> translate to English --\
image -> vision model (runs in parallel with the voice path) -> agent -> translate back -> TTS
text  ---------------------------------------------------/
"""
import base64
import time
from concurrent.futures import ThreadPoolExecutor

from src import vision
from src.langutil import detect_lang, strip_citations


class Assistant:
    def __init__(self, graph, bhashini=None):
        self.graph = graph
        self.bhashini = bhashini
        self.pool = ThreadPoolExecutor(max_workers=2)

    @staticmethod
    def _speak(timed, bh, text, lang, notes):
        """Speech output is optional: if it fails, the text answer is still returned with a note."""
        try:
            return timed("tts", bh.tts, text, lang)
        except Exception as e:
            notes.append(f"Spoken answer unavailable: {e}")
            return None

    def ask(self, text: str | None = None, audio: bytes | None = None, image: bytes | None = None,
            lang: str | None = None, speak: bool = True, image_mime: str = "image/jpeg",
            audio_format: str = "wav") -> dict:
        t_start = time.perf_counter()
        timings: dict[str, float] = {}
        notes: list[str] = []

        def timed(name, fn, *args, **kwargs):
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0) * 1000

        bh = self.bhashini if (self.bhashini and self.bhashini.available) else None

        # Vision runs concurrently with the voice path (they're independent), hiding its latency.
        image_future = self.pool.submit(timed, "vision", vision.image_context, image, image_mime) if image else None

        # 1. speech -> text
        if audio is not None:
            if bh is None:
                raise RuntimeError("Audio input needs a speech provider: set LLM_API_KEY (free mode) or Bhashini credentials in .env.")
            lang = lang or "hi"
            text = timed("asr", bh.asr, audio, lang, audio_format)
        if not text:
            raise ValueError("Provide text, audio, or both an image and a question.")
        lang = lang or detect_lang(text)

        # 2. non-English -> English (retrieval index is English)
        question_en = text
        if lang != "en":
            if bh is None:
                notes.append("No speech/translation provider configured: treating the question as English.")
                lang = "en"
            else:
                question_en = timed("translate_in", bh.translate, text, lang, "en")

        # 3. agent
        image_ctx = image_future.result() if image_future else None
        result = timed("agent", self.graph.invoke,
                       {"question": question_en, "image_context": image_ctx or "", "trace": [], "timings": {}})
        answer_en = result["answer"]
        for node, ms in result.get("timings", {}).items():
            timings[f"agent.{node}"] = ms

        # 4. English -> user's language, then speech
        spoken_en = strip_citations(answer_en)
        answer_local, audio_out = answer_en, None
        if lang != "en" and bh is not None:
            answer_local = timed("translate_out", bh.translate, answer_en, "en", lang)
            if speak:
                audio_out = self._speak(timed, bh, strip_citations(answer_local), lang, notes)
        elif speak and bh is not None:
            audio_out = self._speak(timed, bh, spoken_en, "en", notes)

        total_ms = (time.perf_counter() - t_start) * 1000
        return {
            "language": lang,
            "question": text,
            "question_en": question_en,
            "image_context": image_ctx,
            "queries": result.get("queries", []),
            "filter_state": result.get("filter_state"),
            "answer": answer_local,
            "answer_en": answer_en,
            "audio_b64": base64.b64encode(audio_out).decode() if audio_out else None,
            "sources": [{"n": i, "url": d.get("url"), "title": d.get("title"), "section": d.get("section"),
                         "verified_at": d.get("verified_at")} for i, d in enumerate(result.get("relevant", []), 1)],
            "trace": result.get("trace", []),
            "timings_ms": {k: round(v) for k, v in timings.items()},
            "total_ms": round(total_ms),
            "notes": notes,
        }
