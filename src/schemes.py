"""Curated scheme dataset: one JSON file per scheme (or a list per file) in data/schemes/.

Files whose names start with "_" (e.g. _TEMPLATE.json) are ignored. Run `python main.py validate`
to check your data before indexing it.
"""
import json
import re
from datetime import date
from pathlib import Path

from src import config
from src.cleaner import normalize

# (json field, section type used for chunk labels, heading shown in the document)
CONTENT_FIELDS = [
    ("summary", "about", "Overview"),
    ("eligibility", "eligibility", "Eligibility"),
    ("benefits", "benefits", "Benefits"),
    ("documents", "documents", "Documents Required"),
    ("how_to_apply", "application", "How to Apply"),
]
REQUIRED = ["id", "name", "state", "category", "source_url", "last_verified"]
MIN_WORDS = 8        # shorter sections get a warning
STALE_DAYS = 180     # older last_verified dates get a warning


def field_text(value) -> str:
    """A content field may be a string or a list of strings (rendered as bullets)."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "\n".join(f"- {str(v).strip()}" for v in value if str(v).strip())
    return str(value).strip()


def validate(s: dict, today: date | None = None) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for one scheme. Errors block indexing; warnings don't."""
    today = today or date.today()
    errors: list[str] = []
    warnings: list[str] = []

    for key in REQUIRED:
        if not str(s.get(key, "")).strip():
            errors.append(f"missing '{key}'")

    sid = str(s.get("id", ""))
    if sid and not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", sid):
        errors.append("id must be lowercase letters, digits and hyphens, e.g. 'pm-kisan'")

    url = str(s.get("source_url", ""))
    if url and not url.startswith(("http://", "https://")):
        errors.append("source_url must start with http:// or https://")

    level = s.get("level")
    state = str(s.get("state", "")).strip().lower()
    if level not in (None, "central", "state"):
        errors.append("level must be 'central' or 'state'")
    if level == "state" and state == "all-india":
        errors.append("a state scheme needs a specific state, not 'all-india'")
    if level == "central" and state and state != "all-india":
        errors.append("a central scheme should use state 'all-india'")

    verified = s.get("last_verified")
    if verified:
        try:
            age = (today - date.fromisoformat(str(verified))).days
            if age < 0:
                errors.append("last_verified is in the future")
            elif age > STALE_DAYS:
                warnings.append(f"last_verified is {age} days old; re-check it against the official site")
        except ValueError:
            errors.append("last_verified must be a date like 2026-10-25")

    filled = {k: field_text(s.get(k)) for k, _, _ in CONTENT_FIELDS if field_text(s.get(k))}
    if not filled:
        errors.append("needs at least one of: summary, eligibility, benefits, documents, how_to_apply")
    for key, text in filled.items():
        if len(text.split()) < MIN_WORDS:
            warnings.append(f"'{key}' is very short ({len(text.split())} words)")
    for key in ("eligibility", "benefits"):
        if key not in filled:
            warnings.append(f"no '{key}' text (users ask about this most)")
    return errors, warnings


def load_all(directory=None, today: date | None = None) -> tuple[list[dict], list[str], list[str]]:
    """Load and validate every scheme file. Returns (valid schemes, errors, warnings)."""
    directory = Path(directory or config.SCHEMES_DIR)
    schemes, errors, warnings = [], [], []
    seen: dict[str, str] = {}
    if not directory.exists():
        return schemes, errors, warnings

    for path in sorted(directory.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            errors.append(f"{path.name}: invalid JSON ({e})")
            continue
        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict):
                errors.append(f"{path.name}: expected a scheme object, got {type(item).__name__}")
                continue
            label = f"{path.name} [{item.get('id', '?')}]"
            errs, warns = validate(item, today)
            sid = item.get("id")
            if sid in seen:
                errs.append(f"duplicate id (already in {seen[sid]})")
            elif sid:
                seen[sid] = path.name
            errors += [f"{label}: {m}" for m in errs]
            warnings += [f"{label}: {m}" for m in warns]
            if not errs:
                schemes.append(item)
    return schemes, errors, warnings


def to_document(s: dict, doc_id: int) -> dict:
    """Convert a validated scheme into the section-based document the chunker expects."""
    sections = []
    for key, section_type, heading in CONTENT_FIELDS:
        text = field_text(s.get(key))
        if key == "summary" and str(s.get("ministry", "")).strip():
            text = f"Administered by: {str(s['ministry']).strip()}.\n{text}".strip()
        text = normalize(text)
        if text:
            sections.append({"type": section_type, "heading": heading, "text": text})
    return {
        "id": doc_id,
        "scheme_id": s["id"],
        "url": s["source_url"].strip(),
        "title": s["name"].strip(),
        "state": str(s["state"]).strip().lower(),
        "category": str(s.get("category", "")).strip().lower(),
        "verified_at": str(s["last_verified"]),
        "sections": sections,
    }


def new_scheme_file(slug: str, directory=None, today: date | None = None) -> Path:
    """Create a blank scheme file so you only fill in the facts. Required text fields start empty, so
    `python main.py validate` keeps flagging the file until you complete it."""
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", slug):
        raise ValueError("use lowercase letters, digits and hyphens, e.g. 'pm-kisan'")
    directory = Path(directory or config.SCHEMES_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{slug}.json"
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    skeleton = {
        "id": slug, "name": slug.replace("-", " ").title(), "level": "central", "state": "all-india",
        "category": "", "ministry": "", "summary": "", "eligibility": [], "benefits": "", "documents": [],
        "how_to_apply": "", "source_url": "", "last_verified": (today or date.today()).isoformat(),
    }
    path.write_text(json.dumps(skeleton, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
