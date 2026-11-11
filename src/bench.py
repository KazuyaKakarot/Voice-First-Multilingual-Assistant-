"""Latency benchmark.  Run:  python main.py bench questions.txt --no-cache

Asks every question through the full assistant and reports p50 / p95 per stage, ready to paste into the
README latency table. Use --no-cache, otherwise repeated LLM calls come back instantly from disk and the
numbers are too good to be true.
"""
import csv
from datetime import datetime
from pathlib import Path

from src import config
from src.stats import percentile


def load_questions(path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]


def run_bench(assistant, questions: list[str], speak: bool = False, progress=print) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, start=1):
        result = assistant.ask(text=q, speak=speak)
        rows.append({"question": q, "TOTAL": result["total_ms"], **result["timings_ms"]})
        if progress:
            progress(f"{i}/{len(questions)}  {result['total_ms']} ms  {q[:60]}")
    return rows


def summarize(rows: list[dict]) -> dict:
    stages = [k for k in dict.fromkeys(k for r in rows for k in r) if k not in ("question", "TOTAL")]
    summary = {}
    for stage in stages + ["TOTAL"]:
        values = [r[stage] for r in rows if stage in r]
        summary[stage] = {"n": len(values), "p50": percentile(values, 50), "p95": percentile(values, 95)}
    return summary


def format_summary(summary: dict) -> str:
    lines = ["| Stage | Runs | p50 | p95 |", "|---|---|---|---|"]
    for stage, m in summary.items():
        lines.append(f"| {stage} | {m['n']} | {m['p50'] / 1000:.2f}s | {m['p95'] / 1000:.2f}s |")
    return "\n".join(lines)


def save(rows: list[dict], markdown: str, out_dir=None) -> tuple[Path, Path]:
    out_dir = Path(out_dir or config.RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path, md_path = out_dir / f"latency_{stamp}.csv", out_dir / f"latency_{stamp}.md"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown + "\n", encoding="utf-8")
    return csv_path, md_path
