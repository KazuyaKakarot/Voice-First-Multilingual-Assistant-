"""Bhashini client: speech-to-text (ASR), translation (NMT), text-to-speech (TTS).

Bhashini's flow has two calls:
  1. Pipeline CONFIG  -> returns, per task, a serviceId, plus the compute callbackUrl and inference key.
  2. Pipeline COMPUTE -> POST to callbackUrl with that key; returns the task output.
Request/response shapes follow https://bhashini.gitbook.io/bhashini-apis. The config-call URL and your
pipeline ID come from your Bhashini onboarding, so they are configurable (see .env.example).

NOTE: written from the public docs and tested against mocked responses only. Run it once against the
live API with your own credentials and adjust field names if Bhashini has changed them.
"""
import base64
import io
import wave

import requests

from src import config
from src.langutil import split_sentences


class BhashiniError(RuntimeError):
    pass


def merge_wav(clips: list[bytes]) -> bytes:
    """Join WAV clips into one valid WAV (each clip has its own header, so bytes can't just be appended)."""
    if not clips:
        return b""
    if len(clips) == 1:
        return clips[0]
    out = io.BytesIO()
    writer = None
    for clip in clips:
        with wave.open(io.BytesIO(clip), "rb") as r:
            if writer is None:
                writer = wave.open(out, "wb")
                writer.setparams(r.getparams())
            writer.writeframes(r.readframes(r.getnframes()))
    writer.close()
    return out.getvalue()


class Bhashini:
    def __init__(self, user_id: str = "", ulca_key: str = "", pipeline_id: str = "", timeout: int = 60):
        self.user_id = user_id or config.BHASHINI_USER_ID
        self.ulca_key = ulca_key or config.BHASHINI_ULCA_API_KEY
        self.pipeline_id = pipeline_id or config.BHASHINI_PIPELINE_ID
        self.timeout = timeout
        self._cfg_cache: dict[tuple, dict] = {}

    @property
    def available(self) -> bool:
        return bool(self.user_id and self.ulca_key and self.pipeline_id)

    # ---- call 1: pipeline config ------------------------------------------
    def _config(self, task: str, src: str, tgt: str | None = None) -> dict:
        key = (task, src, tgt)
        if key in self._cfg_cache:
            return self._cfg_cache[key]
        language = {"sourceLanguage": src}
        if tgt:
            language["targetLanguage"] = tgt
        body = {
            "pipelineTasks": [{"taskType": task, "config": {"language": language}}],
            "pipelineRequestConfig": {"pipelineId": self.pipeline_id},
        }
        resp = requests.post(
            config.BHASHINI_CONFIG_URL, json=body, timeout=self.timeout,
            headers={"userID": self.user_id, "ulcaApiKey": self.ulca_key, "Content-Type": "application/json"},
        )
        if not resp.ok:
            raise BhashiniError(f"config call failed ({resp.status_code}): {resp.text[:200]}")
        data = resp.json()
        try:
            service_id = data["pipelineResponseConfig"][0]["config"][0]["serviceId"]
            endpoint = data["pipelineInferenceAPIEndPoint"]
            result = {
                "serviceId": service_id,
                "url": endpoint["callbackUrl"],
                "auth": {endpoint["inferenceApiKey"]["name"]: endpoint["inferenceApiKey"]["value"]},
            }
        except (KeyError, IndexError, TypeError) as e:
            raise BhashiniError(f"unexpected config response: {e}") from e
        self._cfg_cache[key] = result
        return result

    # ---- call 2: pipeline compute -----------------------------------------
    def _compute(self, cfg: dict, task_config: dict, task: str, input_data: dict) -> dict:
        body = {
            "pipelineTasks": [{"taskType": task, "config": {**task_config, "serviceId": cfg["serviceId"]}}],
            "inputData": input_data,
        }
        resp = requests.post(cfg["url"], json=body, timeout=self.timeout,
                             headers={**cfg["auth"], "Content-Type": "application/json"})
        if not resp.ok:
            raise BhashiniError(f"{task} compute failed ({resp.status_code}): {resp.text[:200]}")
        try:
            return resp.json()["pipelineResponse"][0]
        except (KeyError, IndexError, ValueError) as e:
            raise BhashiniError(f"unexpected {task} response: {e}") from e

    # ---- public API -------------------------------------------------------
    def asr(self, audio: bytes, lang: str, audio_format: str = "wav", sampling_rate: int = 16000) -> str:
        cfg = self._config("asr", lang)
        out = self._compute(
            cfg,
            {"language": {"sourceLanguage": lang}, "audioFormat": audio_format, "samplingRate": sampling_rate},
            "asr",
            {"input": [{"source": None}], "audio": [{"audioContent": base64.b64encode(audio).decode()}]},
        )
        return out["output"][0]["source"].strip()

    def translate(self, text: str, src: str, tgt: str) -> str:
        if src == tgt or not text.strip():
            return text
        cfg = self._config("translation", src, tgt)
        parts = []
        for piece in split_sentences(text):
            out = self._compute(
                cfg,
                {"language": {"sourceLanguage": src, "targetLanguage": tgt}},
                "translation",
                {"input": [{"source": piece}], "audio": [{"audioContent": None}]},
            )
            parts.append(out["output"][0]["target"].strip())
        return " ".join(parts)

    def tts(self, text: str, lang: str, gender: str = "female") -> bytes:
        cfg = self._config("tts", lang)
        clips = []
        for piece in split_sentences(text):
            out = self._compute(
                cfg,
                {"language": {"sourceLanguage": lang}, "gender": gender},
                "tts",
                {"input": [{"source": piece}], "audio": [{"audioContent": None}]},
            )
            clips.append(base64.b64decode(out["audio"][0]["audioContent"]))
        return merge_wav(clips)
