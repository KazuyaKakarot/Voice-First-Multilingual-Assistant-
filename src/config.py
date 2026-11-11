import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# ---- data locations -------------------------------------------------------
DATA_DIR = ROOT / "data"
SCHEMES_DIR = DATA_DIR / "schemes"           # curated scheme JSON files (the main data source)
DOCS_DIR = DATA_DIR / "docs"                 # optional extras: loose .md / .txt / .pdf scheme files
CHUNKS_PATH = DATA_DIR / "chunks.json"
GOLDEN_PATH = DATA_DIR / "golden.jsonl"       # evaluation questions (see data/golden.example.jsonl)
RESULTS_DIR = ROOT / "results"               # evaluation and latency outputs
QDRANT_PATH = str(DATA_DIR / "qdrant_db")
CACHE_DIR = str(DATA_DIR / "llm_cache")
COLLECTION = "schemes"

# ---- retrieval ------------------------------------------------------------
# Queries are translated to English by Bhashini, so an English embedder is enough
# and stays light on an 8GB machine.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
CHUNK_WORDS = 220
CHUNK_OVERLAP = 40
CANDIDATES = 20
TOP_K = 5

# ---- LLM (any OpenAI-compatible API) ---------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
# Vision-capable model on the same provider (used only when an image is uploaded)
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.8-27b")
# Judge for evaluation. Use a DIFFERENT model than LLM_MODEL when you can: models rate their own output too kindly.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "") or LLM_MODEL

# ---- agent loop bounds ----------------------------------------------------
MAX_RETRIEVAL_RETRIES = 2
MAX_GENERATION_RETRIES = 1

# ---- Bhashini ---------------------------------------------------------------
SPEECH_PROVIDER = os.getenv("SPEECH_PROVIDER", "auto").strip().lower()
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
# Speech-to-text can use its own key, so the main LLM can be Gemini while Whisper stays on Groq.
SPEECH_API_KEY = os.getenv("SPEECH_API_KEY", "") or LLM_API_KEY
SPEECH_BASE_URL = os.getenv("SPEECH_BASE_URL", "") or (
    "https://api.groq.com/openai/v1" if os.getenv("SPEECH_API_KEY") else LLM_BASE_URL)
# Optional separate (often stronger) model for translation; empty = same as LLM_MODEL.
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "")
BHASHINI_USER_ID = os.getenv("BHASHINI_USER_ID", "")
BHASHINI_ULCA_API_KEY = os.getenv("BHASHINI_ULCA_API_KEY", "")
BHASHINI_PIPELINE_ID = os.getenv("BHASHINI_PIPELINE_ID", "")
# Pipeline *config* call URL. Verify against the current Bhashini docs / your onboarding email.
BHASHINI_CONFIG_URL = os.getenv(
    "BHASHINI_CONFIG_URL",
    "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline",
)
