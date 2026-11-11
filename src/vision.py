"""Image branch: read an uploaded document/photo and turn it into text context for retrieval."""
from src import llm

SYSTEM = (
    "You read photos of Indian government documents, notices, or farm/household scenes to help someone "
    "find welfare schemes. Extract ONLY scheme-relevant facts: document type, state, social category if "
    "printed, land size, occupation, age, and any scheme name mentioned. NEVER copy personal identifiers "
    "such as names, Aadhaar/PAN/ration card numbers, phone numbers or addresses. "
    'Reply as JSON: {"document_type": "...", "facts": ["..."], "summary": "one or two sentences"}.'
)


def describe_image(image: bytes, mime: str = "image/jpeg") -> dict:
    raw = llm.chat_vision(SYSTEM, "Extract the scheme-relevant facts from this image.", image, mime)
    return llm.parse_json(raw, {"document_type": "unknown", "facts": [], "summary": raw[:300]})


def image_context(image: bytes, mime: str = "image/jpeg") -> str:
    info = describe_image(image, mime)
    facts = "; ".join(str(f) for f in info.get("facts", []))
    return f"Uploaded {info.get('document_type', 'image')}: {info.get('summary', '')} Facts: {facts}".strip()
