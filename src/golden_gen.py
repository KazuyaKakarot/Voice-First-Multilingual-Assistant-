"""Draft a golden evaluation set from an FAQ CSV (columns: scheme_slug, scheme_name, faq_number, question, answer).

Run:  python main.py make-golden schemes_faqs.csv --n 60 --seed 1

Answerable questions are sampled FAQs of schemes that are in data/schemes/ (so the expected scheme exists). Each
question is prefixed with the scheme name because many FAQs are generic ("What is the objective?"). Unanswerable
questions come from a fixed out-of-scope list. REVIEW THE OUTPUT before trusting it: delete odd questions and
check a few reference answers against the official pages.
"""
import csv
import json
import random
import sys
from pathlib import Path

from src import config, schemes

csv.field_size_limit(sys.maxsize)

OUT_OF_SCOPE = [
    "Is there a scheme that pays for a private spaceship licence?",
    "Which scheme gives free gold jewellery to every citizen?",
    "How do I apply for a scheme that gives a free Tesla car?",
    "Which scheme will pay my credit card bills?",
    "Is there a government scheme for buying a private island?",
    "What is the capital of France?",
    "Which scheme gives a free luxury yacht to fishermen?",
    "Can I get a scheme that pays for a Hollywood film production?",
    "Which scheme provides free cryptocurrency to students?",
    "How do I get a scheme that guarantees winning the lottery?",
    "Is there a scheme for a personal helicopter for farmers?",
    "Which scheme will pay my overseas holiday to Switzerland?",
]
MIN_ANSWER, MAX_ANSWER = 25, 500


def fix_rupee(text: str) -> str:
    """The FAQ export lost the rupee sign and wrote it as "[?]" (e.g. "[?]6000/-")."""
    return text.replace("[?]", "\u20b9")


def load_faqs(path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and not {"scheme_slug", "question", "answer"} <= set(rows[0]):
        raise ValueError(f"unexpected columns {list(rows[0])}; need scheme_slug, question, answer")
    return rows


def make_golden(faq_path, n: int = 60, seed: int = 1, unanswerable_share: float = 0.25, scheme_dir=None) -> list[dict]:
    known = {s["id"]: s for s in schemes.load_all(scheme_dir)[0]}
    n_un = min(len(OUT_OF_SCOPE), round(n * unanswerable_share))
    n_ans = n - n_un
    rng = random.Random(seed)

    by_scheme: dict[str, list[dict]] = {}
    for r in load_faqs(faq_path):
        slug, q, a = r["scheme_slug"].strip(), fix_rupee(r["question"].strip()), fix_rupee(r["answer"].strip())
        r["question"], r["answer"] = q, a
        if slug in known and q and MIN_ANSWER <= len(a) <= MAX_ANSWER:
            by_scheme.setdefault(slug, []).append(r)
    if not by_scheme:
        raise ValueError("none of the FAQ rows match a scheme in data/schemes/. Import the schemes first.")

    slugs = sorted(by_scheme)
    rng.shuffle(slugs)
    picked = []
    for slug in slugs[:n_ans]:            # at most one FAQ per scheme, so the set covers many schemes
        picked.append(rng.choice(by_scheme[slug]))

    golden = []
    for i, r in enumerate(picked, start=1):
        name = known[r["scheme_slug"].strip()]["name"]
        golden.append({"id": f"f{i:03d}", "question": f"About {name}: {r['question'].strip()}", "answerable": True,
                       "expected_scheme_ids": [r["scheme_slug"].strip()], "reference_answer": r["answer"].strip()})
    for j, q in enumerate(rng.sample(OUT_OF_SCOPE, n_un), start=1):
        golden.append({"id": f"u{j:03d}", "question": q, "answerable": False, "expected_scheme_ids": [], "reference_answer": ""})
    return golden


def write_golden(golden: list[dict], path=None) -> Path:
    path = Path(path or config.GOLDEN_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"{path} already exists; move it or pass --out")
    path.write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in golden) + "\n", encoding="utf-8")
    return path
