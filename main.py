"""CLI.

  python main.py new-scheme ID     create a blank scheme file in data/schemes/
  python main.py import-csv FILE   convert a schemes CSV export into data/schemes/ JSON
  python main.py make-golden FAQ   draft data/golden.jsonl from an FAQ CSV (review it afterwards)
  python main.py validate          check the scheme files in data/schemes/
  python main.py ingest            validate, chunk, embed and index them
  python main.py ask "question" [--lang hi] [--audio q.wav] [--image card.jpg] [--no-speak]
  python main.py chat              interactive (text)
  python main.py serve             FastAPI server on :8000
  python main.py ui                Streamlit front end on :8501
  python main.py bhashini-check    test the speech provider (free or Bhashini), one service at a time
  python main.py eval              golden-set evaluation: ablation + naive RAG vs agent
  python main.py bench FILE        latency p50/p95 over the questions in FILE
"""
import argparse
import json
from pathlib import Path


def build_assistant():
    from src.agent import build_graph
    from src.speech import get_speech
    from src.pipeline import Assistant
    from src.retriever import HybridRetriever

    return Assistant(build_graph(HybridRetriever()), get_speech())


def show(result: dict):
    print("\n--- understanding ---")
    print(f"  language : {result['language']}")
    print(f"  question : {result['question']}")
    if result["language"] != "en":
        print(f"  in English: {result['question_en']}")
    if result["image_context"]:
        print(f"  image    : {result['image_context']}")
    print(f"  queries  : {result['queries']}  state filter: {result['filter_state']}")
    print("\n--- agent steps ---")
    for step in result["trace"]:
        print("  ", step)
    print("\n--- answer ---")
    print(result["answer"])
    if result["language"] != "en":
        print(f"\n(English) {result['answer_en']}")
    print("\n--- sources ---")
    for s in result["sources"]:
        print(f"  [{s['n']}] {s['title']} ({s['section']}) {s['url']}  verified {s['verified_at']}")
    print("\n--- latency (ms) ---")
    for k, v in result["timings_ms"].items():
        print(f"  {k:<22}{v:>7}")
    print(f"  {'TOTAL':<22}{result['total_ms']:>7}")
    for n in result["notes"]:
        print("note:", n)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("new-scheme")
    new.add_argument("scheme_id", help="lowercase-with-hyphens, e.g. pm-kisan")
    imp = sub.add_parser("import-csv")
    imp.add_argument("csv_file")
    imp.add_argument("--verified", help="date the CSV was exported/checked, YYYY-MM-DD (default: today)")
    imp.add_argument("--limit", type=int, help="convert only the first N rows (good for a first trial)")
    mg = sub.add_parser("make-golden")
    mg.add_argument("faq_csv")
    mg.add_argument("--n", type=int, default=60)
    mg.add_argument("--seed", type=int, default=1)
    mg.add_argument("--out")
    sub.add_parser("validate")
    sub.add_parser("ingest")
    ask = sub.add_parser("ask")
    ask.add_argument("question", nargs="?")
    ask.add_argument("--lang")
    ask.add_argument("--audio")
    ask.add_argument("--image")
    ask.add_argument("--no-speak", action="store_true")
    ask.add_argument("--out", default="answer.wav", help="where to save the spoken answer")
    sub.add_parser("chat")
    sub.add_parser("serve")
    sub.add_parser("ui")
    check = sub.add_parser("bhashini-check")
    check.add_argument("--lang", default="hi")
    check.add_argument("--out", default="bhashini_check.wav")
    ev = sub.add_parser("eval")
    ev.add_argument("--golden", help="path to golden.jsonl (default data/golden.jsonl)")
    ev.add_argument("--limit", type=int, help="only the first N questions (cheap trial run)")
    ev.add_argument("--no-judge", action="store_true", help="retrieval metrics only, no LLM judging")
    ev.add_argument("--no-cache", action="store_true", help="ignore the LLM cache (honest latency, more API calls)")
    bench = sub.add_parser("bench")
    bench.add_argument("questions", help="text file with one question per line")
    bench.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.cmd == "new-scheme":
        from src import schemes
        try:
            path = schemes.new_scheme_file(args.scheme_id)
        except (ValueError, FileExistsError) as e:
            raise SystemExit(f"error: {e}")
        print(f"Created {path}. Fill it in from the official source page, then run: python main.py validate")
    elif args.cmd == "import-csv":
        from src import import_csv
        try:
            path, n = import_csv.import_csv(args.csv_file, verified=args.verified, limit=args.limit)
        except (OSError, ValueError) as e:
            raise SystemExit(f"error: {e}")
        print(f"Wrote {n} schemes to {path}. Next: python main.py validate, then python main.py ingest")
    elif args.cmd == "make-golden":
        from src import golden_gen
        try:
            golden = golden_gen.make_golden(args.faq_csv, n=args.n, seed=args.seed)
            path = golden_gen.write_golden(golden, args.out)
        except (OSError, ValueError) as e:
            raise SystemExit(f"error: {e}")
        print(f"Wrote {len(golden)} questions to {path}. Open it and review before running: python main.py eval")
    elif args.cmd == "validate":
        from src import schemes
        valid, errors, warnings = schemes.load_all()
        for w in warnings[:50]:
            print("warning:", w)
        if len(warnings) > 50:
            print(f"... and {len(warnings) - 50} more warnings (mostly short or missing sections in the source data)")
        for e in errors:
            print("ERROR:", e)
        print(f"\n{len(valid)} valid scheme(s), {len(errors)} error(s), {len(warnings)} warning(s)")
        raise SystemExit(1 if errors else 0)
    elif args.cmd == "ingest":
        from src import ingest
        ingest.main()
    elif args.cmd == "serve":
        import uvicorn
        uvicorn.run("src.app:app", host="0.0.0.0", port=8000)
    elif args.cmd == "ui":
        import subprocess
        import sys
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "ui" / "streamlit_app.py")])
    elif args.cmd == "bhashini-check":
        from src.bhashini_check import run_check
        from src.speech import get_speech
        client = get_speech()
        if not client.available:
            raise SystemExit("No speech provider is ready. Set LLM_API_KEY (free mode) or the Bhashini credentials in .env.")
        print(f"Testing speech provider: {type(client).__name__}")
        results = run_check(client, args.lang, args.out)
        for r in results:
            print(f"{'OK  ' if r['ok'] else 'FAIL'} {r['stage']:<22}{r['ms']:>6} ms  {r['detail']}")
        raise SystemExit(0 if results and all(r["ok"] for r in results) and len(results) == 4 else 1)
    elif args.cmd == "eval":
        from src import evaluation, llm
        from src.agent import build_graph
        from src.retriever import HybridRetriever
        if args.no_cache:
            llm.set_cache_enabled(False)
        golden = evaluation.load_golden(args.golden)
        retriever = HybridRetriever()
        result = evaluation.run_eval(golden, retriever, build_graph(retriever), limit=args.limit,
                                     judge=not args.no_judge)
        csv_path, md_path = evaluation.save_results(result)
        print("\n" + result["markdown"])
        print(f"\nPer-question results: {csv_path}\nSummary: {md_path}")
    elif args.cmd == "bench":
        from src import bench, llm
        if args.no_cache:
            llm.set_cache_enabled(False)
        questions = bench.load_questions(args.questions)
        rows = bench.run_bench(build_assistant(), questions)
        markdown = bench.format_summary(bench.summarize(rows))
        csv_path, md_path = bench.save(rows, markdown)
        print("\n" + markdown)
        print(f"\nPer-question timings: {csv_path}\nSummary: {md_path}")
    elif args.cmd == "ask":
        assistant = build_assistant()
        image = Path(args.image).read_bytes() if args.image else None
        mime = "image/png" if args.image and args.image.lower().endswith(".png") else "image/jpeg"
        audio = Path(args.audio).read_bytes() if args.audio else None
        fmt = Path(args.audio).suffix.lstrip(".") if args.audio else "wav"
        result = assistant.ask(text=args.question, audio=audio, image=image, lang=args.lang,
                               speak=not args.no_speak, image_mime=mime, audio_format=fmt)
        if result["audio_b64"]:
            import base64
            audio_bytes = base64.b64decode(result["audio_b64"])
            from src.speech_free import audio_ext
            out_path = Path(args.out).with_suffix("." + audio_ext(audio_bytes))
            out_path.write_bytes(audio_bytes)
            print(f"(spoken answer saved to {out_path})")
        show(result)
        Path("last_result.json").write_text(json.dumps({**result, "audio_b64": None}, ensure_ascii=False, indent=2))
    else:  # chat
        assistant = build_assistant()
        while True:
            q = input("\nask> ").strip()
            if q in {"", "exit", "quit"}:
                break
            show(assistant.ask(text=q, speak=False))


if __name__ == "__main__":
    main()
