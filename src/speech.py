"""Choose the speech provider.  SPEECH_PROVIDER in .env:
  bhashini : Bhashini only (needs credentials)
  free     : Groq Whisper + LLM translation + edge-tts (no account beyond your LLM key)
  auto     : Bhashini if credentials are set, otherwise free   (default)
"""
from src import config
from src.bhashini import Bhashini
from src.speech_free import FreeSpeech


def get_speech():
    provider = config.SPEECH_PROVIDER
    if provider not in ("auto", "bhashini", "free"):
        raise ValueError("SPEECH_PROVIDER must be auto, bhashini or free")
    if provider == "free":
        return FreeSpeech()
    bhashini = Bhashini()
    if provider == "bhashini" or bhashini.available:
        return bhashini
    return FreeSpeech()
