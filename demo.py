"""
CLI demo for KazMorphCorpus-2026.
Quick test of the morphological analysis pipeline from terminal.
"""

import sys
from kazmorph.pipeline import Pipeline


def main():
    pipeline = Pipeline()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        print("KazMorphCorpus-2026 CLI Demo")
        print("Enter Kazakh text (or 'quit' to exit):")
        while True:
            try:
                text = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue

            _print_results(pipeline, text)
        return

    _print_results(pipeline, text)


def _print_results(pipeline: Pipeline, text: str):
    results = pipeline.process(text)

    print(f"\nInput: {text}")
    print(f"Tokens: {len(results)}")
    print("-" * 60)

    for r in results:
        suffixes = " + ".join(r["suffixes"]) if r["suffixes"] else "-"
        print(
            f"  {r['token']:15s} → lemma={r['lemma']:10s} "
            f"POS={r['pos']:5s} conf={r['confidence']:.2f} "
            f"[{r['difficulty_class']}] suffixes={suffixes}"
        )


if __name__ == "__main__":
    main()
