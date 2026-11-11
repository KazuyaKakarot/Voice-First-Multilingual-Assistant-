"""First-contact smoke test for your Bhashini credentials.  Run:  python main.py bhashini-check --lang hi

Runs one sentence through each service on its own, so a failure points at exactly one stage:
  1. translate English -> your language
  2. text-to-speech of that sentence (saved so you can listen to it)
  3. speech-to-text of the audio just produced (a round trip)
  4. translate the transcript back to English
"""
import io
import time
import wave
from pathlib import Path

SAMPLE_EN = "How can a farmer apply for a government scheme?"


def wav_sample_rate(audio: bytes) -> int | None:
    try:
        with wave.open(io.BytesIO(audio), "rb") as w:
            return w.getframerate()
    except (wave.Error, EOFError):
        return None


def run_check(client, lang: str = "hi", wav_path=None) -> list[dict]:
    """Return one result dict per stage: {stage, ok, ms, detail}. Later stages are skipped after a failure."""
    results: list[dict] = []
    state: dict = {}

    def step(name, fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            ok = True
        except Exception as e:  # report any failure (HTTP, parsing, bad audio) as a failed stage
            detail, ok = f"{type(e).__name__}: {e}", False
        results.append({"stage": name, "ok": ok, "ms": round((time.perf_counter() - t0) * 1000), "detail": detail})
        return ok

    def translate_out():
        state["local"] = client.translate(SAMPLE_EN, "en", lang)
        return state["local"]

    def tts():
        state["audio"] = client.tts(state["local"], lang)
        if wav_path:
            Path(wav_path).write_bytes(state["audio"])
        return f"{len(state['audio'])} bytes of audio" + (f", saved to {wav_path}" if wav_path else "")

    def asr():
        rate = wav_sample_rate(state["audio"]) or 16000     # tell ASR the real sampling rate of the TTS output
        state["heard"] = client.asr(state["audio"], lang, "wav", rate)
        return f"{state['heard']}  (audio at {rate} Hz)"

    def translate_back():
        return client.translate(state["heard"], lang, "en")

    for name, fn in [(f"translate en->{lang}", translate_out), ("text-to-speech", tts),
                     ("speech-to-text", asr), (f"translate {lang}->en", translate_back)]:
        if not step(name, fn):
            break
    return results
