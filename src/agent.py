"""Agentic RAG as a LangGraph state machine (English-in, English-out; the voice/translation layer
lives in pipeline.py).

route -> rewrite -> retrieve -> grade --(nothing relevant, retries left)--> rewrite
grade --(relevant found, or retries used up)--> generate -> check
check --(not grounded, one retry allowed)--> generate
check --(grounded)--> END

Every node's wall-clock time is recorded in state["timings"] (milliseconds).
"""
import time
from typing import TYPE_CHECKING, TypedDict

from langgraph.graph import END, StateGraph

from src import config, llm

if TYPE_CHECKING:
    from src.retriever import HybridRetriever


class State(TypedDict, total=False):
    question: str            # English question (already translated if the user spoke another language)
    image_context: str       # facts extracted from an uploaded image, if any
    queries: list[str]       # rewritten search queries
    filter_state: str | None # Indian state extracted from the question/image, used as a retrieval filter
    docs: list[dict]
    relevant: list[dict]
    answer: str
    needs_retrieval: bool
    retrieval_tries: int
    generation_tries: int
    grounded: bool
    trace: list[str]
    timings: dict[str, float]


def _log(state: State, msg: str) -> list[str]:
    return state.get("trace", []) + [msg]


def _format_docs(docs: list[dict]) -> str:
    return "\n\n".join(
        f"[{i + 1}] (source: {d.get('url', d.get('source', '?'))}, verified {d.get('verified_at', 'unknown')})\n{d['text']}"
        for i, d in enumerate(docs)
    )


def _full_question(state: State) -> str:
    ctx = state.get("image_context")
    return f"{state['question']}\n\nContext from the user's uploaded image: {ctx}" if ctx else state["question"]


def _timed(name: str, fn):
    def wrapper(state: State) -> State:
        t0 = time.perf_counter()
        out = fn(state)
        timings = dict(state.get("timings", {}))
        timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0) * 1000
        out["timings"] = timings
        return out
    return wrapper


def build_graph(retriever: "HybridRetriever"):
    # ---- nodes ---------------------------------------------------------
    def route(state: State) -> State:
        out = llm.chat_json(
            "Decide if answering needs a lookup in a database of Indian government schemes. "
            'Reply {"retrieve": true} for questions about schemes, eligibility, benefits, documents or '
            'how to apply; {"retrieve": false} for greetings, thanks, or unrelated chit-chat.',
            _full_question(state),
            default={"retrieve": True},
        )
        need = bool(out.get("retrieve", True))
        return {"needs_retrieval": need, "trace": _log(state, f"route: retrieve={need}")}

    def rewrite(state: State) -> State:
        tries = state.get("retrieval_tries", 0)
        hint = ""
        if tries > 0:
            hint = (
                f"\nThe previous queries {state.get('queries')} returned nothing relevant. "
                "Use different keywords or synonyms, or break the question into parts."
            )
        out = llm.chat_json(
            "Produce search queries for a database of Indian government scheme pages (sections: "
            "eligibility, benefits, documents, application). Write 1-3 short keyword-rich queries that "
            "together cover the question: a direct one, a synonym/expanded one, and (for compound "
            "questions) one per sub-question. Also extract the Indian state the user lives in, if stated "
            'or visible in the image context, else null. Reply {"queries": ["..."], "state": "Bihar" or null}.'
            + hint,
            _full_question(state),
            default={"queries": [state["question"]], "state": None},
        )
        queries = [q for q in out.get("queries", []) if isinstance(q, str) and q.strip()][:3] or [state["question"]]
        st = out.get("state")
        st = st.strip().lower() if isinstance(st, str) and st.strip() else state.get("filter_state")
        return {"queries": queries, "filter_state": st,
                "trace": _log(state, f"rewrite: {queries} state={st}")}

    def retrieve(state: State) -> State:
        docs = retriever.search(state["queries"], state["question"], k=config.TOP_K, state=state.get("filter_state"))
        return {"docs": docs, "retrieval_tries": state.get("retrieval_tries", 0) + 1,
                "trace": _log(state, f"retrieve: {len(docs)} chunks")}

    def grade(state: State) -> State:
        out = llm.chat_json(
            "You grade retrieved chunks. Given a question and numbered chunks, return the numbers "
            'of chunks that contain information useful for answering it: {"relevant": [1, 3]}. '
            'Return {"relevant": []} if none help.',
            f"Question: {_full_question(state)}\n\n{_format_docs(state['docs'])}",
            default={"relevant": list(range(1, len(state["docs"]) + 1))},
        )
        idx = {i for i in out.get("relevant", []) if isinstance(i, int)}
        relevant = [d for i, d in enumerate(state["docs"], start=1) if i in idx]
        return {"relevant": relevant,
                "trace": _log(state, f"grade: {len(relevant)}/{len(state['docs'])} relevant")}

    def generate(state: State) -> State:
        tries = state.get("generation_tries", 0)
        if not state.get("needs_retrieval", True):
            answer = llm.chat("You are a friendly assistant for an Indian welfare-scheme helpline. "
                              "Reply briefly and invite the user to ask about a scheme.", state["question"])
            return {"answer": answer, "grounded": True, "generation_tries": tries + 1,
                    "trace": _log(state, "generate: direct reply (no retrieval)")}
        if not state.get("relevant"):
            return {"answer": "I couldn't find reliable information about that in the scheme database. "
                              "Please check the official scheme website or your nearest common service centre.",
                    "grounded": True, "generation_tries": tries + 1,
                    "trace": _log(state, "generate: no relevant context, abstaining")}
        strict = " Use ONLY facts stated in the context. Do not add outside knowledge." if tries else ""
        answer = llm.chat(
            "You help Indian citizens find welfare schemes. Answer using ONLY the numbered context. "
            "Write in short plain sentences, at most 120 words, because the answer may be read aloud. "
            "Cite sources inline like [1]. When you state amounts or rules, say they should be confirmed on "
            "the official site. If the context is insufficient, say so." + strict,
            f"Context:\n{_format_docs(state['relevant'])}\n\nQuestion: {_full_question(state)}",
        )
        return {"answer": answer, "generation_tries": tries + 1,
                "trace": _log(state, f"generate: attempt {tries + 1}")}

    def check(state: State) -> State:
        if not state.get("needs_retrieval", True) or not state.get("relevant"):
            return {"grounded": True, "trace": _log(state, "check: skipped")}
        out = llm.chat_json(
            "Check whether EVERY claim in the answer is supported by the context. "
            'Reply {"grounded": true} or {"grounded": false, "reason": "..."}.',
            f"Context:\n{_format_docs(state['relevant'])}\n\nAnswer:\n{state['answer']}",
            default={"grounded": True},
        )
        grounded = bool(out.get("grounded", True))
        return {"grounded": grounded, "trace": _log(state, f"check: grounded={grounded}")}

    # ---- routing -------------------------------------------------------
    def after_route(state: State) -> str:
        return "rewrite" if state["needs_retrieval"] else "generate"

    def after_grade(state: State) -> str:
        if state["relevant"] or state["retrieval_tries"] > config.MAX_RETRIEVAL_RETRIES:
            return "generate"
        return "rewrite"

    def after_check(state: State) -> str:
        if state["grounded"] or state["generation_tries"] > config.MAX_GENERATION_RETRIES:
            return END
        return "generate"

    # ---- graph ---------------------------------------------------------
    g = StateGraph(State)
    for name, fn in [("route", route), ("rewrite", rewrite), ("retrieve", retrieve),
                     ("grade", grade), ("generate", generate), ("check", check)]:
        g.add_node(name, _timed(name, fn))
    g.set_entry_point("route")
    g.add_conditional_edges("route", after_route, {"rewrite": "rewrite", "generate": "generate"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade, {"rewrite": "rewrite", "generate": "generate"})
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", after_check, {END: END, "generate": "generate"})
    return g.compile()
