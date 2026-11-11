"""Convert a myscheme-style CSV export into scheme JSON that `validate` and `ingest` understand.

Run:  python main.py import-csv path/to/schemes.csv [--verified 2026-10-26] [--limit 300]

Writes one list file, data/schemes/imported_<csvname>.json. Your hand-written files in data/schemes/ stay separate;
a duplicate id between the two is reported by `validate`.
"""
import csv
import json
import re
import sys
from datetime import date
from pathlib import Path

from src import config

csv.field_size_limit(sys.maxsize)

NOISE = {"", "none", "null", "nan", "n/a", "na", "-"}


def clean(value) -> str:
    text = re.sub(r"[ \t]+", " ", (value or "").replace("\r", "")).strip()
    return "" if text.lower() in NOISE else text


def split_items(value: str, sep: str) -> list[str]:
    """Split a delimited cell into items, dropping bullets and empties."""
    items = []
    for part in value.split(sep):
        part = re.sub(r"^\s*[-•*]\s*", "", part).strip()
        if clean(part):
            items.append(part)
    return items


def parse_date(value: str) -> str | None:
    value = clean(value)
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError:
        return None


def convert_row(row: dict, verified: str, today: date | None = None) -> dict:
    """One CSV row -> one scheme dict in the project's schema."""
    today = today or date.today()
    central = clean(row.get("level")).lower() == "central"
    state = "all-india" if central else clean(row.get("state")).lower() or "all-india"
    categories = [c.strip() for c in clean(row.get("categories")).split(";") if c.strip()]

    owner = [clean(row.get("ministry")), clean(row.get("department"))]
    owner = ", ".join(dict.fromkeys(o for o in owner if o))

    summary = [clean(row.get("brief_description")), clean(row.get("detailed_description")).replace(" | ", "\n")]
    extras = []
    for label, key in (("Categories", "categories"), ("Topics", "sub_categories"), ("Tags", "tags")):
        if clean(row.get(key)):
            extras.append(f"{label}: {clean(row[key])}.")
    opened, closes = parse_date(row.get("scheme_open_date")), parse_date(row.get("scheme_close_date"))
    if opened:
        extras.append(f"Scheme opened on {opened}.")
    if closes:
        past = date.fromisoformat(closes) < today
        extras.append(f"Scheme {'closed' if past else 'closes'} on {closes}.")

    eligibility = split_items(clean(row.get("eligibility")), ";")
    exclusions = split_items(clean(row.get("exclusions")), ";")
    if exclusions:
        eligibility += [f"Not eligible: {e}" for e in exclusions]

    how = split_items(clean(row.get("application_process")), " | ")
    mode = clean(row.get("application_mode"))
    if mode:
        how.insert(0, f"Application mode: {mode}.")

    return {
        "id": clean(row.get("slug")),
        "name": clean(row.get("scheme_name")),
        "level": "central" if central else "state",
        "state": state,
        "category": (categories[0] if categories else "general").lower(),
        "ministry": owner,
        "summary": "\n".join(p for p in summary + extras if p),
        "eligibility": eligibility,
        "benefits": split_items(clean(row.get("benefits")), " | "),
        "documents": split_items(clean(row.get("documents_required")), ";"),
        "how_to_apply": how,
        "source_url": clean(row.get("source_url")),
        "last_verified": verified,
    }


def import_csv(csv_path, out_dir=None, verified: str | None = None, limit: int | None = None) -> tuple[Path, int]:
    verified = verified or date.today().isoformat()
    date.fromisoformat(verified)                     # fail early on a typo
    out_dir = Path(out_dir or config.SCHEMES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "slug" not in rows[0]:
        raise ValueError(f"unexpected columns {list(rows[0])[:6]}...; this importer expects a 'slug' column")
    if limit:
        rows = rows[:limit]
    schemes = [convert_row(r, verified) for r in rows]
    stem = re.sub(r"[^a-z0-9]+", "_", Path(csv_path).stem.lower()).strip("_")
    path = out_dir / f"imported_{stem}.json"
    path.write_text(json.dumps(schemes, ensure_ascii=False, indent=1), encoding="utf-8")
    return path, len(schemes)
