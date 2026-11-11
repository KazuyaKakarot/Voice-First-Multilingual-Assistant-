"""Evaluation harness.  Run:  python main.py eval

Two experiments over a golden question set (data/golden.jsonl):

1. Retrieval ablation (no LLM cost): dense only -> hybrid -> hybrid + rerank, scored by hit@k and MRR.
2. System comparison: a naive RAG baseline (dense top-k -> answer) versus the full agent, scored by
   retrieval hit@k, answer correctness, faithfulness, correct/false abstention, latency and LLM calls.

Correctness, faithfulness and abstention are judged by an LLM (config.JUDGE_MODEL). Judge verdicts are
only as good as the judge, so spot-check some of them in the CSV this writes.
"""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from src import config, llm
from src.stats import mean, percentile

REQUIRED_KEYS = ["id", "question", "answerable"]

RETRIEVAL_CONFIGS = {
    "dense only": dict(dense=True, sparse=False, rerank=False),
    "hybrid (dense + BM25)": dict(dense=True, sparse=True, rerank=False),
    "hybrid + rerank": dict(dense=True, sparse=True, rerank=True),
}

NAIVE_SYSTEM = (
    "Answer the question using only the numbered context. "
    "If the context does not contain the answer, say you could not find it."
)


# ------------------------------------------------------------------ golden set
def load_golden(path=None) -> list[dict]:
    """One JSON object per line. Answerable questions need expected_scheme_ids and reference_answer."""
    path = Path(path or config.GOLDEN_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy data/golden.example.jsonl to data/golden.jsonl and write your own questions.")
    items, problems, seen = [], [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"line {n}: invalid JSON ({e})")
            continue
        for key in REQUIRED_KEYS:
            if key not in item:
                problems.append(f"line {n}: missing '{key}'")
        if item.get("answerable"):
            if not item.get("expected_scheme_ids"):
                problems.append(f"line {n}: an answerable question needs expected_scheme_ids")
            if not str(item.get("reference_answer", "")).strip():
                problems.append(f"line {n}: an answerable question needs a reference_answer")
        if item.get("id") in seen:
            problems.append(f"line {n}: duplicate id '{item.get('id')}'")
        seen.add(item.get("id"))
        items.append(item)
    if problems:
        raise ValueError("problems in the golden set:\n" + "\n".join(problems))
    if not items:
        raise ValueError("the golden set is empty")
    return items


# ------------------------------------------------------------------ retrieval metrics
def hit_and_rr(retrieved_ids: list[str], expected_ids: list[str], k: int) -> tuple[bool, float]:
    """Did any expected scheme appear in the top-k distinct schemes, and at what reciprocal rank?"""
    ranked: list[str] = []
    for sid in retrieved_ids:
        if sid and sid not in ranked:
            ranked.append(sid)
    for rank, sid in enumerate(ranked[:k], start=1):
        if sid in expected_ids:
            return True, 1.0 / rank
    return False, 0.0


def eval_retrieval(golden: list[dict], retriever, k: int = config.TOP_K) -> dict:
    answerable = [g for g in golden if g["answerable"]]
    out = {}
    for name, flags in RETRIEVAL_CONFIGS.items():
        hits, rrs = [], []
        for g in answerable:
            docs = retriever.search([g["question"]], g["question"], k=k, **flags)
            hit, rr = hit_and_rr([d.get("scheme_id", "") for d in docs], g["expected_scheme_ids"], k)
            hits.append(1.0 if hit else 0.0)
            rrs.append(rr)
        out[name] = {"n": len(answerable), "hit_at_k": mean(hits), "mrr": mean(rrs)}
    return out


# ------------------------------------------------------------------ the two systems
def _format_context(docs: list[dict]) -> str:
    return "\n\n".join(f"[{i}] {d['text']}" for i, d in enumerate(docs, start=1))


def run_naive(question: str, retriever, k: int = config.TOP_K) -> dict:
    """Baseline: dense top-k, no rewriting, no grading, no grounding check."""
    docs = retriever.search([question], question, k=k, dense=True, sparse=False, rerank=False)
    answer = llm.chat(NAIVE_SYSTEM, f"Context:\n{_format_context(docs)}\n\nQuestion: {question}")
    return {"answer": answer, "retrieved": docs, "context_docs": docs}


def run_agent(question: str, graph) -> dict:
    state = graph.invoke({"question": question, "trace": [], "timings": {}})
    return {"answer": state["answer"], "retrieved": state.get("docs", []),
            "context_docs": state.get("relevant", [])}


# ------------------------------------------------------------------ LLM judge
def _verdict(system: str, user: str, key: str):
    out = llm.chat_json(system, user, default={}, model=config.JUDGE_MODEL)
    value = out.get(key)
    return value if isinstance(value, bool) else None


def judge_abstained(answer: str):
    return _verdict(
        "Does the answer say that it could not find, or does not know, the requested information, "
        'instead of giving an answer? Reply {"abstained": true} or {"abstained": false}.',
        f"Answer:\n{answer}", "abstained")


def judge_faithful(answer: str, contexts: list[str]):
    context = "\n\n".join(contexts) if contexts else "(no context was retrieved)"
    return _verdict(
        "You check an answer against a context. Reply {\"faithful\": true} if every factual claim in the answer "
        "is supported by the context, otherwise {\"faithful\": false}. An answer that says it could not find "
        "the information is faithful.",
        f"Context:\n{context}\n\nAnswer:\n{answer}", "faithful")


def judge_correct(answer: str, reference: str):
    return _verdict(
        "Compare the answer with the reference answer. Reply {\"correct\": true} if the answer contains the key "
        'facts of the reference and does not contradict it, otherwise {"correct": false}.',
        f"Reference answer:\n{reference}\n\nAnswer:\n{answer}", "correct")


# ------------------------------------------------------------------ running a system over the golden set
def evaluate_system(name: str, runner, golden: list[dict], k: int = config.TOP_K, judge: bool = True,
                    progress=None) -> list[dict]:
    rows = []
    for i, g in enumerate(golden, start=1):
        calls_before = llm.stats["calls"]
        t0 = time.perf_counter()
        res = runner(g["question"])
        latency_ms = (time.perf_counter() - t0) * 1000
        llm_calls = llm.stats["calls"] - calls_before          # measured before judging, so judge calls don't count

        row = {"system": name, "id": g["id"], "question": g["question"], "answerable": bool(g["answerable"]),
               "answer": res["answer"], "hit": None, "rr": None, "correct": None, "faithful": None,
               "abstained": None, "latency_ms": round(latency_ms), "llm_calls": llm_calls}
        if g["answerable"]:
            hit, rr = hit_and_rr([d.get("scheme_id", "") for d in res["retrieved"]], g["expected_scheme_ids"], k)
            row["hit"], row["rr"] = hit, rr
        if judge:
            row["abstained"] = judge_abstained(res["answer"])
            row["faithful"] = judge_faithful(res["answer"], [d["text"] for d in res["context_docs"]])
            if g["answerable"]:
                # Declining to answer an answerable question is wrong; no need to spend a judge call on it.
                row["correct"] = False if row["abstained"] else judge_correct(res["answer"], g["reference_answer"])
        rows.append(row)
        if progress:
            progress(f"[{name}] {i}/{len(golden)}")
    return rows


# ------------------------------------------------------------------ summary
def _rate(items: list[dict], key: str):
    values = [r[key] for r in items if r[key] is not None]
    return sum(1 for v in values if v) / len(values) if values else None


def summarize(rows: list[dict]) -> dict:
    out = {}
    for system in dict.fromkeys(r["system"] for r in rows):
        rs = [r for r in rows if r["system"] == system]
        ans = [r for r in rs if r["answerable"]]
        unans = [r for r in rs if not r["answerable"]]
        latencies = [r["latency_ms"] for r in rs]
        out[system] = {
            "questions": len(rs), "answerable": len(ans), "unanswerable": len(unans),
            "hit_at_k": _rate(ans, "hit"),
            "mrr": mean([r["rr"] for r in ans if r["rr"] is not None]),
            "correctness": _rate(ans, "correct"),
            "faithfulness": _rate(rs, "faithful"),
            "correct_abstention": _rate(unans, "abstained"),
            "false_abstention": _rate(ans, "abstained"),
            "latency_p50_ms": percentile(latencies, 50),
            "latency_p95_ms": percentile(latencies, 95),
            "avg_llm_calls": mean([r["llm_calls"] for r in rs]),
        }
    return out


def _fmt(value, kind="rate"):
    if value is None:
        return "n/a"
    if kind == "rate":
        return f"{value:.2f}"
    if kind == "ms":
        return f"{value / 1000:.1f}s"
    return f"{value:.1f}"


def format_summary(retrieval: dict, systems: dict, k: int) -> str:
    lines = [f"## Retrieval ablation (answerable questions, top-{k})", "",
             f"| Retrieval setup | Questions | hit@{k} | MRR |", "|---|---|---|---|"]
    for name, m in retrieval.items():
        lines.append(f"| {name} | {m['n']} | {_fmt(m['hit_at_k'])} | {_fmt(m['mrr'])} |")
    names = list(systems)
    lines += ["", "## Naive RAG vs agent", "",
              "| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    rows = [
        ("Questions (answerable / unanswerable)", lambda s: f"{s['answerable']} / {s['unanswerable']}"),
        (f"Retrieval hit@{k}", lambda s: _fmt(s["hit_at_k"])),
        ("Answer correctness", lambda s: _fmt(s["correctness"])),
        ("Faithfulness", lambda s: _fmt(s["faithfulness"])),
        ("Correct abstention (unanswerable)", lambda s: _fmt(s["correct_abstention"])),
        ("False abstention (answerable)", lambda s: _fmt(s["false_abstention"])),
        ("Latency p50", lambda s: _fmt(s["latency_p50_ms"], "ms")),
        ("Latency p95", lambda s: _fmt(s["latency_p95_ms"], "ms")),
        ("LLM calls per question", lambda s: _fmt(s["avg_llm_calls"], "num")),
    ]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(systems[n]) for n in names) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ orchestration
def run_eval(golden: list[dict], retriever, graph, k: int = config.TOP_K, limit: int | None = None,
             judge: bool = True, progress=print) -> dict:
    if limit:
        golden = golden[:limit]
    retrieval = eval_retrieval(golden, retriever, k)
    rows = evaluate_system("naive RAG", lambda q: run_naive(q, retriever, k), golden, k, judge, progress)
    rows += evaluate_system("agent", lambda q: run_agent(q, graph), golden, k, judge, progress)
    systems = summarize(rows)
    return {"retrieval": retrieval, "rows": rows, "systems": systems,
            "markdown": format_summary(retrieval, systems, k)}


def save_results(result: dict, out_dir=None) -> tuple[Path, Path]:
    out_dir = Path(out_dir or config.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path, md_path = out_dir / f"eval_{stamp}.csv", out_dir / f"eval_{stamp}.md"
    fields = list(result["rows"][0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["rows"])
    md_path.write_text(result["markdown"] + "\n", encoding="utf-8")
    return csv_path, md_path
