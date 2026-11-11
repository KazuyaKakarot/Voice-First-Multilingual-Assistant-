# Implementation guide: 8 steps, command and code for each

Run everything from the project root. Tick each step's "done when" before moving on.

## Step 1. Set up
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # set LLM_API_KEY (Groq or Gemini, OpenAI-compatible)
python tests/test_offline.py                           # 24 tests, no network
```
Code: `src/config.py` (all settings), `src/llm.py` (LLM client: disk cache, retries, call counter).
Done when: all 24 tests pass and `.env` has an LLM key.

## Step 2. Build the scheme dataset (the real work; budget 2-3 days)
Have a CSV export instead (columns like slug, scheme_name, level, state, benefits, eligibility ...)? Skip the hand-typing:
```bash
python main.py import-csv schemes.csv --verified 2026-10-25 --limit 300   # trial run; drop --limit for all rows
```
Code: `src/import_csv.py`. `--verified` = the date the file was exported or checked.

```bash
python main.py new-scheme pm-kisan        # creates data/schemes/pm-kisan.json with empty fields
# open it, fill every field from the OFFICIAL page, set last_verified to today
python main.py validate                   # fix every ERROR, read warnings
```
Code: `src/schemes.py` (`new_scheme_file`, `validate`, `load_all`). Schema example: `data/schemes/_TEMPLATE.json`.
Rules: copy/paraphrase from the official site only; 30-50 schemes across agriculture, education, health,
housing, women and child, employment, plus 2-3 states. Done when: `validate` shows 0 errors.

## Step 3. Ingest
```bash
python main.py ingest                     # validate -> chunk -> embed -> Qdrant + BM25
```
Code: `src/ingest.py`, `src/chunking.py` (one chunk per section, prefixed `title | section:`), `src/retriever.py`.
Done when: it prints the chunk count. Re-run after every data change.

## Step 4. Get English text Q&A right (before any voice)
```bash
python main.py ask "I am a farmer in Bihar. Which schemes can I get?"
python main.py ask "Is there a scheme for a spaceship licence?"      # must decline, not invent
```
Code: `src/agent.py` (route -> rewrite -> retrieve -> grade -> generate -> check), `src/pipeline.py`.
Read the printed trace and sources. Fix data gaps first, prompts second.
Done when: 10 hand-tried questions give correct, cited answers and the unanswerable one is declined.

## Step 5. Evaluation set and harness
```bash
cp data/golden.example.jsonl data/golden.jsonl     # replace with 40-60 questions about YOUR schemes
python main.py eval --limit 5 --no-judge           # quick plumbing check, no judge cost
python main.py eval                                 # full run -> results/eval_*.csv and .md
```
Golden format (one JSON per line): `id`, `question`, `answerable`, `expected_scheme_ids`, `reference_answer`.
Include about 25% unanswerable questions, some state-specific, some vague.
Code: `src/evaluation.py`. It produces (1) retrieval ablation: dense only -> hybrid -> hybrid + rerank
(hit@k, MRR); (2) naive RAG vs agent: correctness, faithfulness, correct/false abstention, latency p50/p95,
LLM calls per question. Set `JUDGE_MODEL` in `.env` to a different model than the generator if you can.
Done when: you have a results table you can put in the README.

## Step 6. Add Bhashini voice
```bash
# put ULCA user id, API key and pipeline id in .env, then:
python main.py bhashini-check --lang hi             # 4 stages, stops at the first failure
python main.py ask --audio question.wav --lang hi   # 16kHz wav; answer audio saved to answer.wav
```
Code: `src/bhashini.py` (client), `src/bhashini_check.py` (smoke test), `src/langutil.py`.
This client has never touched the live API. If `bhashini-check` fails, the failing stage tells you which
field name or URL to fix in `src/bhashini.py`. Done when: all 4 stages show OK.

## Step 7. Image input and Streamlit UI
```bash
python main.py ask "Am I eligible?" --image card.jpg
python main.py ui                                    # opens http://localhost:8501
```
Code: `src/vision.py` (extracts scheme-relevant facts, never personal identifiers), `ui/streamlit_app.py`.
Done when: you can speak a question in the browser, attach a photo, and see answer, audio, sources and latency.

## Step 8. Measure, package, report
```bash
printf "Which schemes can a farmer in Bihar get?\nWhat documents are needed for ...?\n" > questions.txt
python main.py bench questions.txt --no-cache        # p50/p95 per stage -> results/latency_*.md
docker build -t scheme-assistant .
docker run -v $(pwd)/data:/app/data --env-file .env -p 8000:8000 scheme-assistant
```
Code: `src/bench.py`, `src/stats.py`, `Dockerfile`. Always use `--no-cache` for latency, otherwise repeated
LLM calls come back from disk and the numbers are too good.
Then: paste the eval and latency tables into the README, record a 2-minute demo, add the architecture diagram.
Done when: README has real numbers and the project runs from a clean clone.

## What is verified and what is not
Tested offline with stubs (24 tests): validation, chunking, retrieval fusion and switches, agent paths,
evaluation maths, bench, bhashini-check logic, new-scheme helper, Streamlit flow, pipeline.
NOT verified: live Bhashini calls, live LLM quality, Docker build (no Docker daemon in the build environment),
and real latency numbers. Those come from you running steps 4-8.
