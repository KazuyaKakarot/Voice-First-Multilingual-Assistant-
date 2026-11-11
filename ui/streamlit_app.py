"""Streamlit front end.  Run:  python main.py ui   (or: streamlit run ui/streamlit_app.py)

Runs the assistant in the same process (no separate API server needed): speak or type a question,
optionally attach a photo of a document, and see the answer, sources, agent steps and latency.
"""
import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # so `import src` works from any folder

import pandas as pd  # noqa: E402  (installed with streamlit)
import streamlit as st  # noqa: E402

LANGUAGES = {
    "Auto-detect": None, "English": "en", "Hindi": "hi", "Bengali": "bn", "Marathi": "mr", "Gujarati": "gu",
    "Tamil": "ta", "Telugu": "te", "Kannada": "kn", "Malayalam": "ml", "Punjabi": "pa", "Odia": "or",
}


@st.cache_resource(show_spinner="Loading models (first run only)...")
def get_assistant():
    from src.agent import build_graph
    from src.speech import get_speech
    from src.pipeline import Assistant
    from src.retriever import HybridRetriever

    return Assistant(build_graph(HybridRetriever()), get_speech())


def render(result: dict) -> None:
    st.subheader("Answer")
    st.markdown(result["answer"])
    if result.get("audio_b64"):
        audio = base64.b64decode(result["audio_b64"])
        from src.speech_free import audio_ext
        st.audio(audio, format="audio/mpeg" if audio_ext(audio) == "mp3" else "audio/wav")

    st.subheader("Sources")
    if result["sources"]:
        for s in result["sources"]:
            link = f"[{s['title']}]({s['url']})" if str(s["url"]).startswith("http") else s["title"]
            st.markdown(f"**[{s['n']}]** {link} ({s['section']}), verified {s['verified_at']}")
    else:
        st.caption("No sources: the assistant found nothing reliable to cite.")

    st.subheader("Latency")
    stages = {k: v for k, v in result["timings_ms"].items() if "." not in k}
    st.caption(f"Total {result['total_ms'] / 1000:.1f}s")
    st.bar_chart(pd.DataFrame({"ms": list(stages.values())}, index=list(stages)), horizontal=True)

    with st.expander("How the assistant understood the question"):
        st.write(f"Language: **{result['language']}**")
        st.write(f"Question in English: {result['question_en']}")
        if result.get("image_context"):
            st.write(f"From the image: {result['image_context']}")
        st.write(f"Search queries: {result['queries']}")
        st.write(f"State filter: {result['filter_state'] or 'none'}")
    with st.expander("Agent steps"):
        for step in result["trace"]:
            st.write("-", step)
    for note in result.get("notes", []):
        st.info(note)


def main(get_assistant_fn) -> None:
    st.set_page_config(page_title="Government Scheme Assistant", layout="wide")
    st.title("Government Scheme Assistant")
    st.caption("Ask about government schemes by voice, text, or a photo of a document. "
               "Answers come from the stored scheme data. Always confirm details on the official site.")

    with st.sidebar:
        language = st.selectbox("Your language", list(LANGUAGES), help="Auto-detect works for text; pick one for voice.")
        speak = st.checkbox("Speak the answer", value=True, help="Uses the speech provider set in .env (free by default)")

    left, right = st.columns(2)
    with left:
        text = st.text_area("Type your question", placeholder="Which schemes can a farmer in Bihar get?")
        audio = st.audio_input("...or record it", sample_rate=16000)
    with right:
        image = st.file_uploader("Optional: photo of a document or notice", type=["jpg", "jpeg", "png"])
        if image:
            st.image(image, width=240)

    if st.button("Ask", type="primary"):
        if not (text.strip() or audio):
            st.warning("Type or record a question first.")
        else:
            try:
                with st.spinner("Thinking..."):
                    st.session_state["result"] = get_assistant_fn().ask(
                        text=text.strip() or None,
                        audio=audio.getvalue() if audio else None,
                        image=image.getvalue() if image else None,
                        lang=LANGUAGES[language], speak=speak,
                        image_mime=(image.type if image else None) or "image/jpeg", audio_format="wav")
            except Exception as e:  # show any pipeline failure (missing key, Bhashini error) instead of a stack trace
                st.session_state.pop("result", None)
                st.error(f"{type(e).__name__}: {e}")

    if "result" in st.session_state:
        render(st.session_state["result"])


if __name__ == "__main__":
    main(get_assistant)
