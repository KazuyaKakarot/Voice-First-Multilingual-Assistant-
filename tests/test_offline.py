"""Offline tests: no network, no model downloads.   Run:  python tests/test_offline.py"""
import base64
import io
import json
import sys
import openai
import tempfile
import time
import wave
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import speech, speech_free, golden_gen, import_csv, bench, bhashini, bhashini_check, chunking, cleaner, config, evaluation, ingest, llm, pipeline, retriever, schemes, stats, vision  # noqa: E402
from src.agent import build_graph  # noqa: E402
from src.langutil import detect_lang, split_sentences, strip_citations  # noqa: E402


REAL_LLM_CHAT, REAL_LLM_CHAT_JSON = llm.chat, llm.chat_json


def make_wav(n_frames=100, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(b"\x01\x00" * n_frames)
    return buf.getvalue()


# ---------------------------------------------------------------- cleaning
def test_cleaning():
    assert cleaner.normalize("a​  b\xa0c\n\n\n\nd") == "a b c\n\nd"
    assert cleaner.classify_heading("Who can apply?") == "eligibility"
    assert cleaner.classify_heading("Benefits") == "benefits"
    assert cleaner.classify_heading("Documents Required") == "documents"
    assert cleaner.classify_heading("How to Apply") == "application"
    assert cleaner.classify_heading("Background") == "about"
    secs = cleaner.split_sections("Intro text\n\n## Eligibility\nSmall farmers\n\n## Benefits\nRs 6000")
    assert [s["type"] for s in secs] == ["about", "eligibility", "benefits"], secs
    assert secs[1]["text"] == "Small farmers"


# ---------------------------------------------------------------- chunking
def test_chunking():
    doc = {"title": "PM Test Scheme", "url": "http://x", "state": "all-india", "category": "agri",
           "verified_at": "2026-10-01",
           "sections": [{"type": "eligibility", "heading": "E", "text": " ".join(f"w{i}" for i in range(500))},
                        {"type": "benefits", "heading": "B", "text": "short text"}]}
    chunks = chunking.chunk_document(doc)
    assert len(chunks) > 3
    assert chunks[0]["text"].startswith("PM Test Scheme | eligibility:")
    assert chunks[-1]["section"] == "benefits" and chunks[-1]["verified_at"] == "2026-10-01"
    assert [c["id"] for c in chunks] == list(range(len(chunks)))
    # overlap: last words of chunk 0 reappear at the start of chunk 1
    assert chunks[0]["text"].split()[-1] in chunks[1]["text"]
    # every word is covered
    covered = set(" ".join(c["text"] for c in chunks).split())
    assert all(f"w{i}" in covered for i in range(500))


# ---------------------------------------------------------------- language helpers
def test_langutil():
    assert detect_lang("मुझे योजना चाहिए") == "hi"
    assert detect_lang("எனக்கு திட்டம் வேண்டும்") == "ta"
    assert detect_lang("what scheme") == "en"
    assert strip_citations("You get Rs 6000 [1]. See [2] **now**.") == "You get Rs 6000. See now."
    long = ". ".join(f"Sentence number {i}" for i in range(100)) + "."
    pieces = split_sentences(long, max_chars=200)
    assert all(len(p) <= 200 for p in pieces) and len(pieces) > 5
    assert " ".join(pieces).count("Sentence") == 100


# ---------------------------------------------------------------- retrieval pure functions
def test_retrieval_functions():
    fused = retriever.rrf([[1, 2, 3], [3, 2, 9]])
    assert max(fused, key=fused.get) in (2, 3) and fused[2] > fused[1] and fused[9] < fused[2]
    chunks = [{"state": "bihar"}, {"state": "kerala"}, {"state": "all-india"}, {"state": "bihar"}]
    assert retriever.filter_by_state(chunks, [0, 1, 2, 3], "Bihar", min_keep=2) == [0, 2, 3]
    assert retriever.filter_by_state(chunks, [0, 1, 2, 3], None, min_keep=2) == [0, 1, 2, 3]
    # too few survive -> fall back to unfiltered rather than starving the agent
    assert retriever.filter_by_state(chunks, [0, 1, 2, 3], "bihar", min_keep=4) == [0, 1, 2, 3]


# ---------------------------------------------------------------- bhashini (mocked HTTP)
class FakeResp:
    def __init__(self, data, ok=True, status=200):
        self._d, self.ok, self.status_code, self.text = data, ok, status, str(data)

    def json(self):
        return self._d


def test_bhashini():
    calls = []
    wav = make_wav()

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers))
        if url == config.BHASHINI_CONFIG_URL:
            assert headers["userID"] == "uid" and headers["ulcaApiKey"] == "key"
            task = json["pipelineTasks"][0]["taskType"]
            return FakeResp({
                "pipelineResponseConfig": [{"taskType": task, "config": [{"serviceId": f"svc-{task}"}]}],
                "pipelineInferenceAPIEndPoint": {"callbackUrl": "https://compute.test/run",
                                                 "inferenceApiKey": {"name": "Authorization", "value": "tok"}}})
        task = json["pipelineTasks"][0]["taskType"]
        assert headers["Authorization"] == "tok"
        assert json["pipelineTasks"][0]["config"]["serviceId"] == f"svc-{task}"
        if task == "asr":
            assert base64.b64decode(json["inputData"]["audio"][0]["audioContent"]) == b"AUDIO"
            return FakeResp({"pipelineResponse": [{"output": [{"source": " नमस्ते "}]}]})
        if task == "translation":
            src = json["inputData"]["input"][0]["source"]
            return FakeResp({"pipelineResponse": [{"output": [{"source": src, "target": f"T({src})"}]}]})
        return FakeResp({"pipelineResponse": [{"audio": [{"audioContent": base64.b64encode(wav).decode()}]}]})

    bhashini.requests.post = fake_post
    bh = bhashini.Bhashini("uid", "key", "pipe")
    assert bh.available and not bhashini.Bhashini("", "", "").available

    assert bh.asr(b"AUDIO", "hi") == "नमस्ते"
    assert bh.translate("hello. world.", "en", "en") == "hello. world."      # same language: no call
    n_before = len(calls)
    assert bh.translate("hello", "hi", "en") == "T(hello)"
    long = " ".join(f"This is sentence {i}." for i in range(60))
    out = bh.translate(long, "hi", "en")
    assert out.count("T(") > 1                                                  # chunked into several requests
    config_calls = [c for c in calls if c[0] == config.BHASHINI_CONFIG_URL]
    assert len(config_calls) == 2, "config call should be cached per (task, src, tgt)"   # asr + translation hi->en
    assert len(calls) > n_before

    audio = bh.tts(long, "hi")
    with wave.open(io.BytesIO(audio), "rb") as w:
        assert w.getnframes() > 100, "multi-sentence TTS must merge into one valid WAV"

    # HTTP error surfaces as BhashiniError
    bhashini.requests.post = lambda *a, **k: FakeResp({}, ok=False, status=401)
    try:
        bhashini.Bhashini("u", "k", "p").asr(b"x", "hi")
        raise AssertionError("expected BhashiniError")
    except bhashini.BhashiniError:
        pass


# ---------------------------------------------------------------- agent
class StubRetriever:
    def __init__(self):
        self.calls = []

    def search(self, queries, question, k=5, state=None):
        self.calls.append({"queries": queries, "question": question, "state": state})
        time.sleep(0.01)
        return [{"url": "https://pmkisan.test", "title": "PM Test", "section": "eligibility",
                 "verified_at": "2026-10-01", "text": f"chunk for {queries[0]}"}]


def stub_llm(grades=None, rewrite=None, grounded=True, route=True):
    """Install fake llm.chat / llm.chat_json that dispatch on the system prompt."""
    grades = list(grades or [[1]])
    seen = {"gen_systems": []}

    def chat_json(system, user, default):
        if "lookup" in system:
            return {"retrieve": route}
        if "search queries" in system:
            return rewrite or {"queries": ["farmer scheme bihar", "kisan subsidy"], "state": "Bihar"}
        if "grade" in system:
            return {"relevant": grades.pop(0) if len(grades) > 1 else grades[0]}
        return {"grounded": grounded}

    def chat(system, user):
        seen["gen_systems"].append(system)
        return "You can get Rs 6000 a year [1]. Confirm on the official site."

    llm.chat_json, llm.chat = chat_json, chat
    return seen


def test_agent_paths():
    # happy path, multi-query + state filter + timings
    stub_llm()
    r = StubRetriever()
    out = build_graph(r).invoke({"question": "farmer in Bihar, what scheme?", "trace": [], "timings": {}})
    assert r.calls[0]["queries"] == ["farmer scheme bihar", "kisan subsidy"]
    assert r.calls[0]["state"] == "bihar"
    assert r.calls[0]["question"] == "farmer in Bihar, what scheme?"          # rerank uses the ORIGINAL question
    assert out["trace"][-1] == "check: grounded=True"
    assert set(out["timings"]) == {"route", "rewrite", "retrieve", "grade", "generate", "check"}
    assert out["timings"]["retrieve"] >= 10

    # nothing relevant first time -> rewrite and retry, then succeed
    stub_llm(grades=[[], [1]])
    r = StubRetriever()
    out = build_graph(r).invoke({"question": "q", "trace": [], "timings": {}})
    assert len(r.calls) == 2 and out["timings"]["rewrite"] > 0

    # never relevant -> bounded retries then abstain
    stub_llm(grades=[[]])
    r = StubRetriever()
    out = build_graph(r).invoke({"question": "q", "trace": [], "timings": {}})
    assert len(r.calls) == 3 and "couldn't find" in out["answer"]

    # ungrounded -> regenerate once with the stricter prompt, then stop
    seen = stub_llm(grounded=False)
    out = build_graph(StubRetriever()).invoke({"question": "q", "trace": [], "timings": {}})
    assert out["generation_tries"] == 2
    assert "ONLY facts" in seen["gen_systems"][-1]

    # chit-chat -> no retrieval
    stub_llm(route=False)
    r = StubRetriever()
    out = build_graph(r).invoke({"question": "hello", "trace": [], "timings": {}})
    assert r.calls == [] and "check: skipped" in out["trace"][-1]

    # image context reaches rewrite/generate input
    captured = {}

    def chat_json(system, user, default):
        if "search queries" in system:
            captured["rewrite_input"] = user
            return {"queries": ["x"], "state": None}
        return {"retrieve": True} if "lookup" in system else ({"relevant": [1]} if "grade" in system else {"grounded": True})

    llm.chat_json = chat_json
    build_graph(StubRetriever()).invoke({"question": "Am I eligible?", "image_context": "land: 2 acres",
                                         "trace": [], "timings": {}})
    assert "land: 2 acres" in captured["rewrite_input"]


# ---------------------------------------------------------------- full pipeline
class FakeBhashini:
    available = True

    def __init__(self):
        self.log = []

    def asr(self, audio, lang, fmt="wav"):
        self.log.append(("asr", lang))
        time.sleep(0.02)
        return "मैं बिहार का किसान हूँ, मुझे कौन सी योजना मिलेगी?"

    def translate(self, text, src, tgt):
        self.log.append(("translate", src, tgt))
        return "I am a farmer from Bihar, which scheme?" if tgt == "en" else f"[{tgt}] {text}"

    def tts(self, text, lang, gender="female"):
        self.log.append(("tts", lang, text))
        return b"WAVDATA"


def test_pipeline_voice_to_voice():
    stub_llm()
    vision.image_context = lambda image, mime="image/jpeg": (time.sleep(0.05), "Uploaded card: Bihar")[1]
    bh = FakeBhashini()
    a = pipeline.Assistant(build_graph(StubRetriever()), bh)
    res = a.ask(audio=b"RIFF", image=b"IMG", lang="hi")

    assert res["language"] == "hi"
    assert res["question_en"].startswith("I am a farmer")
    assert res["image_context"] == "Uploaded card: Bihar"
    assert res["answer"].startswith("[hi] ")
    assert base64.b64decode(res["audio_b64"]) == b"WAVDATA"
    tts_text = [e for e in bh.log if e[0] == "tts"][0][2]
    assert "[1]" not in tts_text, "citations must not be spoken"
    assert [e[0] for e in bh.log] == ["asr", "translate", "translate", "tts"]
    for key in ("asr", "translate_in", "vision", "agent", "translate_out", "tts", "agent.retrieve"):
        assert key in res["timings_ms"], key
    # vision (50ms) ran in parallel with ASR (20ms), so total < asr+vision+agent summed
    serial = sum(v for k, v in res["timings_ms"].items() if "." not in k)
    assert res["total_ms"] <= serial, (res["total_ms"], serial)
    assert res["sources"][0]["url"] == "https://pmkisan.test"

    # text-only English: no Bhashini calls for translation, TTS in English
    bh2 = FakeBhashini()
    res = pipeline.Assistant(build_graph(StubRetriever()), bh2).ask(text="What can a farmer get?")
    assert res["language"] == "en" and [e[0] for e in bh2.log] == ["tts"]

    # no Bhashini configured: Hindi text degrades gracefully with a note
    res = pipeline.Assistant(build_graph(StubRetriever()), bhashini.Bhashini("", "", "")).ask(text="किसान योजना क्या है?")
    assert res["notes"] and res["audio_b64"] is None

    # audio without Bhashini is a clear error
    try:
        pipeline.Assistant(build_graph(StubRetriever()), None).ask(audio=b"x")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "speech provider" in str(e)


# ---------------------------------------------------------------- curated scheme data
def good_scheme(**over):
    s = {"id": "pm-test", "name": "PM Test Scheme", "level": "central", "state": "all-india",
         "category": "Agriculture", "ministry": "Ministry of Testing",
         "summary": "Income support for small farmer families across the country.",
         "eligibility": ["Farmer families owning cultivable land", "Institutional landholders are excluded from benefits"],
         "benefits": "Financial assistance is paid in instalments directly to the bank account of each family.",
         "documents": ["Proof of identity", "Land ownership records", "Bank account details of beneficiary"],
         "how_to_apply": "Apply online through the official portal or at a common service centre near you.",
         "source_url": "https://example.gov.in/pm-test", "last_verified": "2026-10-20"}
    s.update(over)
    return s


def test_scheme_validation():
    today = date(2026, 10, 25)
    assert schemes.validate(good_scheme(), today) == ([], [])

    def errs(**over):
        return schemes.validate(good_scheme(**over), today)[0]

    def warns(**over):
        return schemes.validate(good_scheme(**over), today)[1]

    assert any("id must be" in e for e in errs(id="PM Test"))
    assert any("source_url" in e for e in errs(source_url="example.gov.in"))
    assert any("YYYY" in e or "2026-10-25" in e for e in errs(last_verified="25/10/2026"))
    assert any("future" in e for e in errs(last_verified="2026-12-01"))
    assert any("not 'all-india'" in e for e in errs(level="state", state="all-india"))
    assert any("central scheme" in e for e in errs(level="central", state="bihar"))
    assert any("level must be" in e for e in errs(level="local"))
    bad = good_scheme()
    del bad["name"]
    assert any("missing 'name'" in e for e in schemes.validate(bad, today)[0])
    empty = {k: v for k, v in good_scheme().items()
             if k not in ("summary", "eligibility", "benefits", "documents", "how_to_apply")}
    assert any("at least one of" in e for e in schemes.validate(empty, today)[0])

    assert any("days old" in w for w in warns(last_verified="2026-01-01"))
    assert any("very short" in w for w in warns(benefits="Pays money."))
    assert any("no 'eligibility'" in w for w in warns(eligibility=""))


def test_scheme_loading():
    today = date(2026, 10, 25)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "pm-test.json").write_text(json.dumps(good_scheme()))
        (d / "_TEMPLATE.json").write_text("this is not even json")          # ignored: starts with "_"
        (d / "many.json").write_text(json.dumps([
            good_scheme(id="second-scheme", name="Second"),
            good_scheme(id="third scheme"),                                  # invalid id
            good_scheme(id="pm-test"),                                       # duplicate of pm-test.json
        ]))
        (d / "broken.json").write_text("{ not valid")
        valid, errors, warnings = schemes.load_all(d, today)
        assert sorted(s["id"] for s in valid) == ["pm-test", "second-scheme"]
        assert any("broken.json" in e and "invalid JSON" in e for e in errors)
        assert any("duplicate id" in e for e in errors)
        assert any("id must be" in e for e in errors)
        assert not any("_TEMPLATE" in e for e in errors)
    assert schemes.load_all(Path("/nonexistent-dir"), today) == ([], [], [])


def test_scheme_to_document_and_chunks():
    doc = schemes.to_document(good_scheme(), 7)
    assert doc["id"] == 7 and doc["state"] == "all-india" and doc["category"] == "agriculture"
    assert doc["verified_at"] == "2026-10-20" and doc["url"] == "https://example.gov.in/pm-test"
    assert [s["type"] for s in doc["sections"]] == ["about", "eligibility", "benefits", "documents", "application"]
    assert doc["sections"][0]["text"].startswith("Administered by: Ministry of Testing.")
    assert "- Proof of identity" in doc["sections"][3]["text"]            # lists become bullets
    chunks = chunking.chunk_document(doc, start_id=100)
    assert [c["id"] for c in chunks] == list(range(100, 100 + len(chunks)))
    assert chunks[1]["text"].startswith("PM Test Scheme | eligibility:")
    assert all(c["scheme_id"] == "pm-test" and c["verified_at"] == "2026-10-20" for c in chunks)


def test_plain_text_sections():
    text = ("Scheme Name\nPM Test\nDetails\nIncome support for farmers.\nEligibility Criteria:\nOwns land.\n"
            "Eligible farmers must hold land.\nBenefits\nCash support.\nDocuments Required\nAadhaar\n"
            "How to Apply\nOnline.")
    secs = cleaner.split_plain_sections(text)
    assert [s["type"] for s in secs] == ["about", "about", "eligibility", "benefits", "documents", "application"]
    assert "Eligible farmers must hold land." in secs[2]["text"], "a sentence must not be mistaken for a heading"
    assert not cleaner.is_plain_heading("Eligible farmers must apply today.")
    assert cleaner.is_plain_heading("Documents Required:") and cleaner.is_plain_heading("How to Apply")


def test_load_documents_combines_curated_and_loose():
    old = config.SCHEMES_DIR, config.DOCS_DIR
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        config.SCHEMES_DIR, config.DOCS_DIR = Path(a), Path(b)
        try:
            (Path(a) / "pm-test.json").write_text(json.dumps(good_scheme(last_verified=date.today().isoformat())))
            (Path(b) / "Some_Loose-Scheme.txt").write_text("Eligibility\nAny adult resident may apply for this scheme.\n")
            docs, errors, warnings = ingest.load_documents()
        finally:
            config.SCHEMES_DIR, config.DOCS_DIR = old
    assert errors == [] and len(docs) == 2
    assert [d["id"] for d in docs] == [0, 1]
    assert docs[1]["title"] == "Some Loose Scheme" and docs[1]["state"] == "all-india"
    assert any("loose file" in w for w in warnings)


# ---------------------------------------------------------------- stats, llm wrapper, retriever switches
def test_stats():
    assert stats.percentile([], 50) is None and stats.mean([]) is None
    assert stats.percentile([5], 95) == 5
    assert stats.percentile([1, 2, 3, 4, 5], 50) == 3
    assert stats.percentile([10, 20], 50) == 15
    assert abs(stats.percentile(list(range(1, 101)), 95) - 95.05) < 1e-9
    assert stats.mean([1, 2, 3]) == 2


def test_llm_stats_and_cache_toggle():
    llm.chat, llm.chat_json = REAL_LLM_CHAT, REAL_LLM_CHAT_JSON
    real_call, old_dir, old_cache = llm._call, config.CACHE_DIR, llm._cache
    calls = {"n": 0}

    def fake_call(model, messages):
        calls["n"] += 1
        return '{"ok": true}'

    with tempfile.TemporaryDirectory() as d:
        config.CACHE_DIR, llm._cache, llm._call = d, None, fake_call
        try:
            llm.stats.update(calls=0, cache_hits=0)
            assert llm.chat("s", "u") == llm.chat("s", "u")
            assert calls["n"] == 1 and llm.stats == {"calls": 2, "cache_hits": 1}
            llm.set_cache_enabled(False)
            llm.chat("s", "u")
            assert calls["n"] == 2 and llm.stats["cache_hits"] == 1       # cache bypassed when disabled
            llm.set_cache_enabled(True)
            llm.chat("s", "u", model="other-model")
            assert calls["n"] == 3                                         # a different model is a different key
            assert llm.chat_json("s", "j", default={}, model="m2") == {"ok": True}
        finally:
            llm.set_cache_enabled(True)
            if llm._cache is not None:
                llm._cache.close()
            config.CACHE_DIR, llm._cache, llm._call = old_dir, old_cache, real_call


def test_retriever_stage_switches():
    class Reranker:
        def rerank(self, query, texts):
            return [len(t) for t in texts]        # longer text scores higher

    r = object.__new__(retriever.HybridRetriever)
    r.chunks = [{"text": t, "scheme_id": f"s{i}", "state": "all-india"} for i, t in enumerate(["a", "bbbb", "cc"])]
    r._dense = lambda q, n: [0, 1, 2]
    r._sparse = lambda q, n: [2, 1, 0]
    r.reranker = Reranker()

    def texts(**kw):
        return [d["text"] for d in r.search(["q"], "q", k=2, **kw)]

    assert texts(dense=True, sparse=False, rerank=False) == ["a", "bbbb"]
    assert texts(dense=False, sparse=True, rerank=False) == ["cc", "bbbb"]
    assert texts(dense=True, sparse=True, rerank=False) == ["a", "cc"]
    assert texts() == ["bbbb", "cc"]                                       # defaults = hybrid + rerank
    try:
        texts(dense=False, sparse=False)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------- evaluation harness
def test_golden_loading():
    good = (
        '# comment line\n'
        '{"id": "q1", "question": "Who can apply?", "answerable": true, "expected_scheme_ids": ["pm-test"], "reference_answer": "Farmers"}\n'
        '\n'
        '{"id": "q2", "question": "Spaceship?", "answerable": false}\n'
    )
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "golden.jsonl"
        path.write_text(good)
        assert [g["id"] for g in evaluation.load_golden(path)] == ["q1", "q2"]

        def problems(text):
            path.write_text(text)
            try:
                evaluation.load_golden(path)
            except ValueError as e:
                return str(e)
            raise AssertionError("expected ValueError")

        assert "invalid JSON" in problems("{not json}")
        assert "missing 'question'" in problems('{"id": "x", "answerable": false}')
        assert "expected_scheme_ids" in problems('{"id": "x", "question": "q", "answerable": true, "reference_answer": "a"}')
        assert "reference_answer" in problems('{"id": "x", "question": "q", "answerable": true, "expected_scheme_ids": ["a"]}')
        assert "duplicate id" in problems('{"id": "x", "question": "q", "answerable": false}\n{"id": "x", "question": "q", "answerable": false}')
        assert "empty" in problems("# only a comment\n")
    try:
        evaluation.load_golden("/nonexistent/golden.jsonl")
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError:
        pass


def test_hit_and_reciprocal_rank():
    assert evaluation.hit_and_rr(["a", "b", "c"], ["b"], 5) == (True, 0.5)
    assert evaluation.hit_and_rr(["a", "a", "b"], ["b"], 5) == (True, 0.5)       # chunks of one scheme count once
    assert evaluation.hit_and_rr(["a", "b", "c"], ["c"], 2) == (False, 0.0)      # outside top-k
    assert evaluation.hit_and_rr([], ["a"], 5) == (False, 0.0)
    assert evaluation.hit_and_rr(["", "a"], ["a"], 5) == (True, 1.0)             # blank ids are ignored


class EvalRetriever:
    """Dense-only puts the right scheme second; hybrid puts it first; unrelated questions get only a wrong scheme."""

    def search(self, queries, question, k=5, state=None, dense=True, sparse=True, rerank=True, **kw):
        base = {"url": "https://example.gov.in", "section": "eligibility", "verified_at": "2026-10-01"}
        right = {**base, "text": "PM Test | eligibility: farmers", "scheme_id": "pm-test", "title": "PM Test"}
        wrong = {**base, "text": "Other | benefits: students", "scheme_id": "other", "title": "Other"}
        if "farmer" not in question.lower():
            return [wrong]
        return [wrong, right] if not sparse else [right, wrong]


def install_eval_stubs():
    def chat_json(system, user, default, model=None):
        llm.stats["calls"] += 1
        if "lookup" in system:
            return {"retrieve": True}
        if "search queries" in system:
            return {"queries": ["farmer scheme"], "state": None}
        if "grade" in system:
            return {"relevant": [] if "spaceship" in user.lower() else [1]}
        if '"abstained"' in system:
            return {"abstained": "could not find" in user.lower() or "couldn't find" in user.lower()}
        if '"faithful"' in system:
            return {"faithful": True}
        if '"correct"' in system:
            return {"correct": "Rs 6000" in user.split("Answer:")[-1]}
        return {"grounded": True}

    def chat(system, user, model=None):
        llm.stats["calls"] += 1
        return "You get Rs 6000 [1]."

    llm.chat_json, llm.chat = chat_json, chat


def test_evaluation_end_to_end():
    install_eval_stubs()
    golden = [
        {"id": "q1", "question": "Which farmer scheme pays money?", "answerable": True,
         "expected_scheme_ids": ["pm-test"], "reference_answer": "Rs 6000"},
        {"id": "q2", "question": "What documents does the farmer need?", "answerable": True,
         "expected_scheme_ids": ["pm-test"], "reference_answer": "Rs 6000"},
        {"id": "q3", "question": "Is there a spaceship licence scheme?", "answerable": False,
         "expected_scheme_ids": [], "reference_answer": ""},
    ]
    retr = EvalRetriever()
    ablation = evaluation.eval_retrieval(golden, retr, k=5)
    assert ablation["dense only"]["hit_at_k"] == 1.0 and ablation["dense only"]["mrr"] == 0.5
    assert ablation["hybrid + rerank"]["mrr"] == 1.0 and ablation["hybrid + rerank"]["n"] == 2

    result = evaluation.run_eval(golden, retr, build_graph(retr), k=5, progress=None)
    naive, agent = result["systems"]["naive RAG"], result["systems"]["agent"]
    assert naive["questions"] == agent["questions"] == 3 and agent["answerable"] == 2 and agent["unanswerable"] == 1
    assert naive["mrr"] == 0.5 and agent["mrr"] == 1.0
    assert naive["correctness"] == agent["correctness"] == 1.0
    assert naive["correct_abstention"] == 0.0, "the naive baseline answers even when it should decline"
    assert agent["correct_abstention"] == 1.0
    assert agent["false_abstention"] == 0.0 and naive["false_abstention"] == 0.0
    assert naive["faithfulness"] == agent["faithfulness"] == 1.0
    assert naive["avg_llm_calls"] == 1.0, "judge calls must not be counted in the system's own cost"
    assert agent["avg_llm_calls"] > naive["avg_llm_calls"]
    assert naive["latency_p50_ms"] is not None

    md = result["markdown"]
    assert "Retrieval ablation" in md and "Naive RAG vs agent" in md and "| Correct abstention" in md

    with tempfile.TemporaryDirectory() as d:
        csv_path, md_path = evaluation.save_results(result, d)
        lines = csv_path.read_text().strip().splitlines()
        assert len(lines) == 1 + 6                                         # header + 3 questions x 2 systems
        assert "Naive RAG vs agent" in md_path.read_text()

    # --limit and --no-judge
    quick = evaluation.run_eval(golden, retr, build_graph(retr), k=5, limit=1, judge=False, progress=None)
    assert quick["systems"]["agent"]["questions"] == 1
    assert quick["systems"]["agent"]["correctness"] is None and quick["systems"]["agent"]["hit_at_k"] == 1.0


def test_abstaining_on_answerable_question_counts_as_incorrect():
    install_eval_stubs()
    llm.chat = lambda system, user, model=None: "I could not find that."
    g = [{"id": "q1", "question": "Which farmer scheme?", "answerable": True,
          "expected_scheme_ids": ["pm-test"], "reference_answer": "Rs 6000"}]
    rows = evaluation.evaluate_system("naive RAG", lambda q: evaluation.run_naive(q, EvalRetriever()), g)
    assert rows[0]["abstained"] is True and rows[0]["correct"] is False
    assert evaluation.summarize(rows)["naive RAG"]["false_abstention"] == 1.0


# ---------------------------------------------------------------- latency benchmark
def test_bench():
    class FakeAssistant:
        def __init__(self):
            self.n = 0

        def ask(self, text, speak=False):
            self.n += 1
            return {"total_ms": self.n * 1000, "timings_ms": {"agent": self.n * 800, "agent.retrieve": self.n * 100}}

    with tempfile.TemporaryDirectory() as d:
        qfile = Path(d) / "q.txt"
        qfile.write_text("# comment\nWhat is X?\n\nWhat is Y?\nWhat is Z?\nAnd W?\nAnd V?\n")
        questions = bench.load_questions(qfile)
        assert len(questions) == 5
        rows = bench.run_bench(FakeAssistant(), questions, progress=None)
        summary = bench.summarize(rows)
        assert summary["TOTAL"]["p50"] == 3000 and summary["TOTAL"]["n"] == 5
        assert abs(summary["TOTAL"]["p95"] - 4800) < 1e-6
        assert summary["agent.retrieve"]["p50"] == 300
        md = bench.format_summary(summary)
        assert "| TOTAL | 5 | 3.00s | 4.80s |" in md
        csv_path, md_path = bench.save(rows, md, d)
        assert csv_path.exists() and md_path.read_text().startswith("| Stage")


# ---------------------------------------------------------------- bhashini smoke test, new-scheme helper
def test_bhashini_check():
    audio = make_wav(rate=22050)
    assert bhashini_check.wav_sample_rate(audio) == 22050 and bhashini_check.wav_sample_rate(b"nope") is None

    class Client:
        def __init__(self, fail_tts=False):
            self.fail_tts, self.asr_rate = fail_tts, None

        def translate(self, text, src, tgt):
            return f"[{tgt}] {text}" if tgt != "en" else "translated back"

        def tts(self, text, lang, gender="female"):
            if self.fail_tts:
                raise bhashini.BhashiniError("tts compute failed (500)")
            return audio

        def asr(self, a, lang, fmt="wav", rate=16000):
            self.asr_rate = rate
            return "heard text"

    with tempfile.TemporaryDirectory() as d:
        client = Client()
        results = bhashini_check.run_check(client, "hi", Path(d) / "out.wav")
        assert [r["ok"] for r in results] == [True] * 4 and len(results) == 4
        assert client.asr_rate == 22050, "ASR must be told the real sampling rate of the TTS audio"
        assert (Path(d) / "out.wav").read_bytes() == audio

    failed = bhashini_check.run_check(Client(fail_tts=True), "ta")
    assert [r["stage"] for r in failed] == ["translate en->ta", "text-to-speech"]
    assert failed[0]["ok"] and not failed[1]["ok"] and "BhashiniError" in failed[1]["detail"]


def test_new_scheme_file():
    with tempfile.TemporaryDirectory() as d:
        path = schemes.new_scheme_file("pm-kisan", d, today=date(2026, 10, 25))
        data = json.loads(path.read_text())
        assert data["id"] == "pm-kisan" and data["name"] == "Pm Kisan" and data["last_verified"] == "2026-10-25"
        valid, errors, _ = schemes.load_all(d, date(2026, 10, 25))
        assert valid == [] and any("source_url" in e for e in errors), "a blank skeleton must not validate"
        data.update(category="agriculture", source_url="https://example.gov.in/x", summary="Income support for farmers across the country.",
                    eligibility=["Farmer families owning cultivable land in any state"],
                    benefits="Financial assistance is paid in instalments directly to the bank account.")
        path.write_text(json.dumps(data))
        valid, errors, _ = schemes.load_all(d, date(2026, 10, 25))
        assert errors == [] and len(valid) == 1
        for bad in ("Bad Name", "-x", "a--b"):
            try:
                schemes.new_scheme_file(bad, d)
                raise AssertionError("expected ValueError for " + bad)
            except ValueError:
                pass
        try:
            schemes.new_scheme_file("pm-kisan", d)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass


# ---------------------------------------------------------------- streamlit front end
def test_streamlit_ui():
    import os

    from streamlit.testing.v1 import AppTest

    os.environ["SCHEME_TEST_ROOT"] = str(Path(__file__).resolve().parent.parent)

    def app():
        import os
        import sys
        sys.path.insert(0, os.environ["SCHEME_TEST_ROOT"])
        from ui import streamlit_app

        class Fake:
            def ask(self, **kw):
                if kw["text"] == "boom":
                    raise RuntimeError("LLM_API_KEY is empty")
                return {"answer": "Fake answer about " + kw["text"], "audio_b64": None, "language": "en",
                        "question_en": kw["text"], "image_context": None, "queries": ["q1"], "filter_state": "bihar",
                        "sources": [{"n": 1, "title": "PM Test", "url": "https://example.gov.in", "section": "eligibility",
                                     "verified_at": "2026-10-20"}],
                        "trace": ["route: retrieve=True"], "timings_ms": {"agent": 900, "agent.retrieve": 100},
                        "total_ms": 950, "notes": []}

        streamlit_app.main(lambda: Fake())

    at = AppTest.from_function(app, default_timeout=30).run()
    assert not at.exception, at.exception
    at.button[0].click().run()
    assert any("Type or record" in w.value for w in at.warning)            # empty submit is refused politely
    at.text_area[0].set_value("farmer scheme").run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert any("Fake answer about farmer scheme" in m.value for m in at.markdown)
    assert any("PM Test" in m.value and "2026-10-20" in m.value for m in at.markdown)
    at.text_area[0].set_value("boom").run()
    at.button[0].click().run()
    assert any("RuntimeError" in e.value for e in at.error)               # failures are shown, not stack traces


def test_import_csv():
    header = ("slug,scheme_name,level,state,ministry,department,categories,sub_categories,tags,brief_description,"
              "detailed_description,benefits,eligibility,exclusions,application_mode,application_process,"
              "documents_required,scheme_open_date,scheme_close_date,source_url\n")
    rows = (
        'pm-test,PM Test,Central,None,Ministry of Test,Dept of Test,Agriculture; Rural,Farm aid,Farmer; Aid,'
        '"Aid for farmers across the country.","Intro | Longer text about the scheme.",'
        '"- Rs 6000 a year. | - Paid in three instalments.",'
        '"Must own land; Must be a farmer",Institutional landholders,Online; Offline,"Step 1: Register. | Step 2: Verify.",'
        '"Aadhaar; Bank passbook",2019-02-24,2020-01-01,https://example.gov.in/pm-test\n'
        'st-test,State Test,State/ UT,Bihar,,Dept of Bihar,Health,,,"Health aid.",None,,"Resident of Bihar",,,,,,,https://example.gov.in/st\n'
    )
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "My Schemes.csv"
        src.write_text(header + rows, encoding="utf-8")
        out, n = import_csv.import_csv(src, out_dir=Path(d) / "out", verified="2026-10-25")
        assert n == 2 and out.name == "imported_my_schemes.json"
        a, b = json.loads(out.read_text(encoding="utf-8"))
        assert a["level"] == "central" and a["state"] == "all-india" and a["category"] == "agriculture"
        assert a["benefits"] == ["Rs 6000 a year.", "Paid in three instalments."]
        assert a["eligibility"] == ["Must own land", "Must be a farmer", "Not eligible: Institutional landholders"]
        assert a["documents"] == ["Aadhaar", "Bank passbook"] and a["how_to_apply"][0] == "Application mode: Online; Offline."
        assert "Scheme closed on 2020-01-01." in a["summary"] and "Tags: Farmer; Aid." in a["summary"]
        assert b["level"] == "state" and b["state"] == "bihar" and b["ministry"] == "Dept of Bihar" and b["benefits"] == []
        valid, errors, _ = schemes.load_all(Path(d) / "out", date(2026, 10, 25))
        assert errors == [] and len(valid) == 2
        bad = Path(d) / "bad.csv"
        bad.write_text("a,b\n1,2\n")
        try:
            import_csv.import_csv(bad, out_dir=Path(d) / "out")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
        try:
            import_csv.import_csv(src, out_dir=Path(d) / "out", verified="26-10-2026")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_llm_errors_are_friendly_and_not_retried():
    import httpx
    import openai

    llm.chat, llm.chat_json = REAL_LLM_CHAT, REAL_LLM_CHAT_JSON
    calls = {"n": 0}

    class Completions:
        def __init__(self, exc):
            self.exc = exc

        def create(self, **kw):
            calls["n"] += 1
            raise self.exc

    class Client:
        def __init__(self, exc):
            self.chat = type("C", (), {"completions": Completions(exc)})()

    req = httpx.Request("POST", "https://x.test/v1/chat/completions")
    old_get = llm._get_client
    try:
        for exc_cls, status, needle in ((openai.NotFoundError, 404, "does not exist"),
                                        (openai.AuthenticationError, 401, "key was rejected")):
            exc = exc_cls("boom", response=httpx.Response(status, request=req), body=None)
            llm._get_client = lambda exc=exc: Client(exc)
            calls["n"] = 0
            try:
                llm._call("gone-model", [{"role": "user", "content": "hi"}])
                raise AssertionError("expected RuntimeError")
            except RuntimeError as e:
                assert needle in str(e) and "gone-model" in str(e) + "gone-model"
            assert calls["n"] == 1, "a 4xx error must not be retried"
    finally:
        llm._get_client = old_get


def test_make_golden():
    with tempfile.TemporaryDirectory() as d:
        sdir = Path(d) / "schemes"
        sdir.mkdir()
        mk = lambda i: {"id": f"s{i}", "name": f"Scheme {i}", "level": "central", "state": "all-india", "category": "x",
                        "summary": "A scheme that helps people across the whole country with money.",
                        "eligibility": ["Any resident of the country may apply for this scheme"],
                        "benefits": "Money is paid every year directly into the bank account.",
                        "source_url": "https://example.gov.in", "last_verified": date.today().isoformat()}
        (sdir / "a.json").write_text(json.dumps([mk(i) for i in range(1, 6)]))
        faq = Path(d) / "faq.csv"
        lines = ["scheme_slug,scheme_name,faq_number,question,answer"]
        for i in range(1, 8):                                     # s6, s7 are not in the data
            lines.append(f's{i},Scheme {i},1,What is the objective?,"It pays [?]6000 to people with money every year."')
            lines.append(f's{i},Scheme {i},2,Is it free?,Yes.')      # too short: skipped
        faq.write_text("\n".join(lines))
        g = golden_gen.make_golden(faq, n=8, seed=3, scheme_dir=sdir)
        ans = [x for x in g if x["answerable"]]
        assert len(ans) == 5 and len(g) == 7        # only 5 schemes exist, so 5 answerable + 2 unanswerable
        assert all(x["expected_scheme_ids"][0] in {"s1", "s2", "s3", "s4", "s5"} for x in ans)
        assert all(x["question"].startswith("About Scheme") for x in ans)
        assert all("\u20b96000" in x["reference_answer"] and "[?]" not in x["reference_answer"] for x in ans)
        assert len({x["expected_scheme_ids"][0] for x in ans}) == len(ans), "one FAQ per scheme"
        assert golden_gen.make_golden(faq, n=8, seed=3, scheme_dir=sdir) == g, "same seed, same set"
        out = golden_gen.write_golden(g, Path(d) / "golden.jsonl")
        assert len(evaluation.load_golden(out)) == len(g), "output must load in the eval harness"
        try:
            golden_gen.write_golden(g, out)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass
        (sdir / "a.json").write_text("[]")
        try:
            golden_gen.make_golden(faq, scheme_dir=sdir)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_free_speech_and_provider_choice():
    FS = speech_free.FreeSpeech
    assert speech_free.audio_ext(b"RIFF....WAVE") == "wav" and speech_free.audio_ext(b"ID3\x03") == "mp3"
    assert speech_free.audio_ext(b"\xff\xfb\x90") == "mp3" and speech_free.audio_ext(b"zzzz") == "wav"

    # translate: goes through the LLM with a translation-only prompt, skips identical languages and empty text
    seen = {}
    llm.chat = lambda system, user, model=None: (seen.update(system=system, user=user) or "  नमस्ते  ")
    fs = FS()
    assert fs.translate("hello", "en", "hi") == "नमस्ते" and "Hindi" in seen["system"] and seen["user"] == "hello"
    seen.clear()
    assert fs.translate("hello", "en", "en") == "hello" and fs.translate("  ", "en", "hi") == "  " and not seen

    # asr: calls the Whisper endpoint with language and file name; errors become SpeechError
    calls = {}

    class Transcriptions:
        def create(self, **kw):
            calls.update(kw)
            return type("R", (), {"text": " heard it "})()

    fs._client = type("C", (), {"audio": type("A", (), {"transcriptions": Transcriptions()})()})()
    assert fs.asr(b"RIFFxxxx", "hi", "wav") == "heard it"
    assert calls["language"] == "hi" and calls["file"][0] == "audio.wav" and calls["model"] == config.WHISPER_MODEL

    class Broken:
        def create(self, **kw):
            import httpx
            raise openai.NotFoundError("gone", response=httpx.Response(404, request=httpx.Request("POST", "https://x.test")), body=None)

    fs._client = type("C", (), {"audio": type("A", (), {"transcriptions": Broken()})()})()
    try:
        fs.asr(b"RIFFxxxx", "hi")
        raise AssertionError("expected SpeechError")
    except speech_free.SpeechError as e:
        assert "WHISPER_MODEL" in str(e)

    # asr: a retired Whisper model falls back to the next one
    tried = []

    class Fallback:
        def create(self, **kw):
            import httpx
            tried.append(kw["model"])
            if kw["model"] != "whisper-large-v3":
                raise openai.NotFoundError("gone", response=httpx.Response(404, request=httpx.Request("POST", "https://x.test")), body=None)
            return type("R", (), {"text": "ok"})()

    fs._client = type("C", (), {"audio": type("A", (), {"transcriptions": Fallback()})()})()
    assert fs.asr(b"RIFFxxxx", "hi") == "ok" and tried[-1] == "whisper-large-v3" and len(tried) >= 2
    fs._client = type("C", (), {"audio": type("A", (), {"transcriptions": Broken()})()})()

    # separate translate model is passed through
    got = {}
    llm.chat = lambda system, user, model=None: (got.update(model=model) or "x")
    old_tm = config.TRANSLATE_MODEL
    config.TRANSLATE_MODEL = "better-model"
    fs.translate("hello", "en", "hi")
    config.TRANSLATE_MODEL = old_tm
    assert got["model"] == "better-model"

    # tts: edge-tts first, gTTS as fallback (Punjabi only has gTTS), clear error when nothing works
    order = []
    fs._edge = lambda text, voice: (order.append("edge") or b"")
    fs._gtts = lambda text, lang: (order.append("gtts") or b"ID3mp3")
    assert fs.tts("hi", "hi") == b"ID3mp3" and order == ["edge", "gtts"]
    order.clear()
    assert fs.tts("hi", "pa") == b"ID3mp3" and order == ["gtts"]
    try:
        fs.tts("hi", "or")
        raise AssertionError("expected SpeechError")
    except speech_free.SpeechError as e:
        assert "No free voice" in str(e)
    fs._gtts = lambda text, lang: (_ for _ in ()).throw(OSError("offline"))
    try:
        fs.tts("hi", "hi")
        raise AssertionError("expected SpeechError")
    except speech_free.SpeechError as e:
        assert "offline" in str(e) and "internet" in str(e)

    # speech key is separate from the LLM key
    old = (config.SPEECH_API_KEY, config.SPEECH_BASE_URL)
    config.SPEECH_API_KEY = ""
    try:
        FS()._asr_client()
        raise AssertionError("expected SpeechError")
    except speech_free.SpeechError as e:
        assert "SPEECH_API_KEY" in str(e)
    config.SPEECH_API_KEY, config.SPEECH_BASE_URL = old

    # the pipeline keeps the text answer when speech output fails
    class Fake(FS):
        available = True

        def tts(self, text, lang, gender="female"):
            raise speech_free.SpeechError("no voice")

    llm.chat_json = lambda s, u, default, model=None: default
    res = pipeline.Assistant(build_graph(StubRetriever()), Fake()).ask(text="What is the scheme?", speak=True)
    assert res["audio_b64"] is None and any("Spoken answer unavailable" in n for n in res["notes"]) and res["answer"]

    # provider choice
    old = (config.SPEECH_PROVIDER, config.BHASHINI_USER_ID, config.BHASHINI_ULCA_API_KEY, config.BHASHINI_PIPELINE_ID)
    try:
        config.BHASHINI_USER_ID = config.BHASHINI_ULCA_API_KEY = config.BHASHINI_PIPELINE_ID = ""
        config.SPEECH_PROVIDER = "auto"
        assert isinstance(speech.get_speech(), FS), "no Bhashini credentials: fall back to free"
        config.BHASHINI_USER_ID, config.BHASHINI_ULCA_API_KEY, config.BHASHINI_PIPELINE_ID = "u", "k", "p"
        assert isinstance(speech.get_speech(), bhashini.Bhashini)
        config.SPEECH_PROVIDER = "free"
        assert isinstance(speech.get_speech(), FS)
        config.SPEECH_PROVIDER = "nonsense"
        try:
            speech.get_speech()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    finally:
        config.SPEECH_PROVIDER, config.BHASHINI_USER_ID, config.BHASHINI_ULCA_API_KEY, config.BHASHINI_PIPELINE_ID = old


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nALL {len(tests)} PASSED")
