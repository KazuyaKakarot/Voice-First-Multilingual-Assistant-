"""Free speech stack that needs no Bhashini account.

  speech -> text : Whisper, hosted by Groq (same LLM_API_KEY if LLM_BASE_URL is Groq)
  translation    : your LLM (temperature 0, translation-only prompt)
  text -> speech : edge-tts (Microsoft's free online voices; an unofficial library, so it can break)

Same three methods as the Bhashini client, so the pipeline treats both the same way.
"""
import asyncio
import concurrent.futures
import io

import openai
from openai import OpenAI

from src import config, llm

LANG_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "ta": "Tamil",
              "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "or": "Odia"}
# edge-tts has no Punjabi or Odia voice. Punjabi falls back to gTTS (Google Translate's voice); Odia has none free,
# so its answers are shown as text only.
GTTS_LANGS = {"en": "en", "hi": "hi", "bn": "bn", "mr": "mr", "gu": "gu", "ta": "ta", "te": "te", "kn": "kn",
              "ml": "ml", "pa": "pa"}
WHISPER_FALLBACKS = ["whisper-large-v3-turbo", "whisper-large-v3"]
VOICES = {"en": "en-IN-NeerjaNeural", "hi": "hi-IN-SwaraNeural", "bn": "bn-IN-TanishaaNeural",
          "mr": "mr-IN-AarohiNeural", "gu": "gu-IN-DhwaniNeural", "ta": "ta-IN-PallaviNeural",
          "te": "te-IN-ShrutiNeural", "kn": "kn-IN-SapnaNeural", "ml": "ml-IN-SobhanaNeural"}


class SpeechError(RuntimeError):
    pass


def audio_ext(audio: bytes) -> str:
    """File extension for audio bytes: wav, mp3, or wav as the safe default."""
    if audio[:4] == b"RIFF":
        return "wav"
    if audio[:3] == b"ID3" or (len(audio) > 1 and audio[0] == 0xFF and audio[1] & 0xE0 == 0xE0):
        return "mp3"
    return "wav"


class FreeSpeech:
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self._client = None

    @property
    def available(self) -> bool:
        return bool(config.LLM_API_KEY or config.SPEECH_API_KEY)

    def _asr_client(self) -> OpenAI:
        if self._client is None:
            if not config.SPEECH_API_KEY:
                raise SpeechError("No key for speech-to-text. Set SPEECH_API_KEY (a Groq key) in .env.")
            self._client = OpenAI(api_key=config.SPEECH_API_KEY, base_url=config.SPEECH_BASE_URL, timeout=self.timeout)
        return self._client

    def asr(self, audio: bytes, lang: str, audio_format: str = "wav", sampling_rate: int = 16000) -> str:
        ext = audio_ext(audio) if audio_format in ("", None) else audio_format.lstrip(".").lower()
        base = {"file": (f"audio.{ext}", io.BytesIO(audio)), "temperature": 0}
        if lang in LANG_NAMES:
            base["language"] = lang
        models = list(dict.fromkeys([config.WHISPER_MODEL] + WHISPER_FALLBACKS))
        last = None
        for model in models:                       # a retired model name should not break voice input
            base["file"][1].seek(0)
            try:
                resp = self._asr_client().audio.transcriptions.create(model=model, **base)
                return (resp.text or "").strip()
            except openai.NotFoundError as e:
                last = e
            except openai.AuthenticationError as e:
                raise SpeechError(f"The speech key was rejected. Whisper needs a Groq key: set SPEECH_API_KEY in .env. ({e})") from None
            except openai.OpenAIError as e:
                raise SpeechError(f"speech-to-text failed: {e}") from None
        raise SpeechError(f"No Whisper model worked (tried {', '.join(models)}). Check the current model list of your "
                          f"provider and set WHISPER_MODEL in .env. ({last})")

    def translate(self, text: str, src: str, tgt: str) -> str:
        if src == tgt or not text.strip():
            return text
        out = llm.chat(
            f"You are a translator. Translate the user's text from {LANG_NAMES.get(src, src)} to "
            f"{LANG_NAMES.get(tgt, tgt)}. Keep numbers, names, URLs and citation markers like [1] unchanged. "
            "Output only the translation, with no notes.", text, model=config.TRANSLATE_MODEL or None)
        return out.strip()

    def _edge(self, text: str, voice: str) -> bytes:
        import edge_tts

        async def synth() -> bytes:
            chunks = []
            async for part in edge_tts.Communicate(text, voice).stream():
                if part["type"] == "audio":
                    chunks.append(part["data"])
            return b"".join(chunks)

        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False
        if in_loop:                              # already inside an event loop: use a helper thread
            with concurrent.futures.ThreadPoolExecutor(1) as pool:
                return pool.submit(lambda: asyncio.run(synth())).result()
        return asyncio.run(synth())

    def _gtts(self, text: str, lang: str) -> bytes:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(text=text, lang=GTTS_LANGS[lang]).write_to_fp(buf)
        return buf.getvalue()

    def tts(self, text: str, lang: str, gender: str = "female") -> bytes:
        """Try edge-tts first, then gTTS. Both are free online services; neither is an official API."""
        errors = []
        engines = []
        if lang in VOICES:
            engines.append(("edge-tts", lambda: self._edge(text, VOICES[lang])))
        if lang in GTTS_LANGS:
            engines.append(("gTTS", lambda: self._gtts(text, lang)))
        if not engines:
            raise SpeechError(f"No free voice for '{lang}'. The answer is shown as text.")
        for name, run in engines:
            try:
                data = run()
                if data:
                    return data
                errors.append(f"{name}: no audio returned")
            except ImportError:
                errors.append(f"{name}: package not installed (pip install -r requirements.txt)")
            except Exception as e:
                errors.append(f"{name}: {e}")
        raise SpeechError("text-to-speech failed (" + "; ".join(errors) + "). Check your internet connection.")
