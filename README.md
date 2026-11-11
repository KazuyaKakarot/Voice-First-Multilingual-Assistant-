# Voice-first multilingual government scheme assistant (agentic RAG)

Speak or upload a photo in an Indian language. Get a grounded, cited answer, spoken back in the same
language. Runs on an 8GB laptop: embeddings and reranker run locally (ONNX), the LLM and Bhashini are APIs.

Scheme data is **curated once and stored in `data/schemes/`** (no web scraping), which keeps running cost
and complexity low and makes every answer traceable to a source you checked yourself.

## Architecture

```
OFFLINE   data/schemes/*.json -> validate -> section-aware chunks -> embed -> Qdrant + BM25
          (curated, one file       (schema,     (title | section: text,
           per scheme)              dates)       metadata: state, verified date)

ONLINE    voice --> Bhashini ASR --\
          text  -------------------+-> detect language -> Bhashini translate -> English question
          image --> vision model --/   (runs in parallel with the voice path)
                                                    |
              +-----------------------  LangGraph agent  -----------------------+
              |  route -> rewrite -> retrieve -> grade -> generate -> check     |
              |            ^   multi-query +      |  nothing relevant:          |
              |            |   state filter       v  retry (max 2)              |
              |            +---------------- rewrite again                      |
              +-----------------------------------------------------------------+
                                                    |
          English answer -> Bhashini translate -> Bhashini TTS -> spoken answer + text + sources + timings
```

| Stage | What it does | Why |
|---|---|---|
| validate | Checks each scheme file: required fields, id format, URL, dates, state/level consistency, duplicates; warns on stale dates and thin sections | Bad data is caught before it is indexed |
| chunk | One chunk per section, prefixed `Scheme title \| section:` | A chunk saying "18 years or above" still knows its scheme |
| rewrite | LLM makes 1-3 queries + extracts the user's state | Recall: direct + synonym + per-sub-question queries |
| retrieve | Dense + BM25 per query, fused by RRF, state filter (state + all-India) | Hybrid beats either alone |
| rerank | Cross-encoder scores candidates against the **original** question | Rewrites are for recall; the original is the real intent |
| grade / check | LLM drops irrelevant chunks, then verifies the answer is grounded | Self-correction and abstaining instead of guessing |
| vision | Extracts scheme-relevant facts from an image, never personal identifiers | Multimodal input without storing PII |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # LLM key required; Bhashini credentials optional (text-only without them)
python tests/test_offline.py                          # 24 tests, no network needed
```

## Build your scheme dataset

One JSON file per scheme in `data/schemes/` (copy `_TEMPLATE.json`; files starting with `_` are ignored):

```json
{
  "id": "pm-kisan",                     // lowercase-with-hyphens, unique
  "name": "Scheme name as officially titled",
  "level": "central",                   // "central" or "state"
  "state": "all-india",                 // "all-india" for central schemes, else e.g. "bihar"
  "category": "agriculture",
  "ministry": "Administering ministry or department",
  "summary": "What it is and who it is for.",
  "eligibility": ["rule one", "rule two"],        // string or list of strings
  "benefits": "What the beneficiary gets.",
  "documents": ["document one", "document two"],
  "how_to_apply": "Where and how to apply.",
  "source_url": "https://the-official-page",
  "last_verified": "2026-10-25"         // the day you checked it against the source
}
```

Rules that keep the data trustworthy: copy or paraphrase from the **official** page, never from memory;
set `last_verified` to the day you actually checked; start with 30-50 schemes across a few categories
(agriculture, education, health, housing, women and child, employment) and your target states.

Optional extras: drop `.md` / `.txt` / `.pdf` scheme files into `data/docs/` (for example PDFs from a public
dataset). They are section-split by heading-like lines and loaded as all-India with no verified date, so use
them for breadth and keep your hand-verified JSON for the schemes you demo.

## Run

```bash
python main.py import-csv schemes.csv --verified 2026-10-25   # optional: bulk-load a myscheme-style CSV
python main.py validate        # check the data; fix every ERROR, read the warnings
python main.py ingest          # validate + chunk + embed + index (run again after any data change)
python main.py ask "I am a farmer in Bihar with 2 acres. Which schemes can I get?"
python main.py ask --audio question.wav --lang hi          # voice in, spoken answer saved to answer.wav
python main.py ask "Am I eligible?" --image card.jpg
python main.py serve                                        # FastAPI: POST /ask (text / audio / image)
python main.py ui                                           # Streamlit app (voice, text, image)
python main.py new-scheme pm-kisan                          # blank scheme file to fill in
python main.py bhashini-check --lang hi                     # test Bhashini credentials stage by stage
python main.py eval                                         # retrieval ablation + naive RAG vs agent (needs data/golden.jsonl)
python main.py bench questions.txt --no-cache               # p50/p95 latency per stage
docker build -t scheme-assistant . && docker run -v $(pwd)/data:/app/data --env-file .env -p 8000:8000 scheme-assistant
```

Step-by-step build order with commands: see `IMPLEMENTATION_GUIDE.md`.

Every answer prints the detected language, the rewritten queries, the agent trace, sources with their
verified dates, and a per-stage latency table. The same data is in `last_result.json`.

## Latency

Targets for a free-tier setup (measure your own and put p50/p95 in this table):

| Stage | Target | Measured |
|---|---|---|
| ASR | 0.8-1.5s | |
| Translate in | 0.3-0.5s | |
| Rewrite (LLM) | 0.5-1s | |
| Retrieve + rerank | 0.2-0.5s | |
| Vision (image only, parallel with voice path) | 1-2s | |
| Generate + grade + check (LLM) | 1.5-3s | |
| Translate out + TTS | 1-2s | |
| **Total** | **~5-8s** | |

Levers: LLM calls are disk-cached; vision runs in parallel with ASR; the grade/check steps are the
first thing to drop if you need speed or lower cost (measure the quality cost first).

## Notes and limits

- **Bhashini client is untested against the live API.** It follows the public docs (config call, then
  compute call) and is verified only against mocked responses. Run one real request with your credentials
  and check field names. The config-call URL and pipeline ID come from your Bhashini onboarding.
- Scheme amounts and rules change. Every answer carries the verified date and tells the user to confirm
  on the official site. `validate` warns when a scheme has not been re-checked for 180 days.
- Language detection from text is script-based (Devanagari defaults to Hindi). Pass `--lang` for Marathi etc.
- Audio is sent as wav/flac at 16kHz; convert other formats first (e.g. ffmpeg).
- No auth or rate limiting on the FastAPI app. Add both before exposing it publicly.

## Layout

```
main.py             CLI
src/schemes.py      scheme schema, validation     src/pipeline.py   voice/text/image -> answer + timings
src/cleaner.py      text normalize + sectioning   src/agent.py      LangGraph agent
src/chunking.py     section-aware chunks          src/retriever.py  hybrid search + RRF + rerank
src/ingest.py       validate, embed, index        src/bhashini.py   ASR / translate / TTS client
src/vision.py       image -> facts                src/llm.py        cached OpenAI-compatible client
src/app.py          FastAPI                       tests/            offline tests
data/schemes/       your curated scheme files
```
