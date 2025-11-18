"""CLI wrapper to run a QA evaluation over a question set."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tecnoquimicas_kb.config.logging_config import setup_logging
from tecnoquimicas_kb.domain.eval_dataset import load_eval_questions
from tecnoquimicas_kb.domain.qa_service import answer_question


def _iter_questions(
    raw_questions: Iterable[Any],
) -> Iterable[Dict[str, Any]]:
    """Normalize a questions iterable into dicts with at least a 'q' key."""
    for item in raw_questions:
        if isinstance(item, str):
            yield {"q": item}
        elif isinstance(item, dict):
            qtext = (
                item.get("q") or item.get("question") or item.get("text") or str(item)
            )
            normalized = dict(item)
            normalized["q"] = qtext
            yield normalized
        else:
            yield {"q": str(item), "raw": item}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run a QA evaluation over a question set and write a JSONL "
            "file containing questions and answers."
        )
    )

    parser.add_argument(
        "--questions-file",
        "-qf",
        type=Path,
        help=(
            "Optional path to a questions file. If omitted, the default "
            "built-in evaluation set is used."
        ),
    )

    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=Path("qa_eval_results.jsonl"),
        help="Path to the JSONL file where evaluation results will be written.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of questions to evaluate.",
    )

    parser.add_argument(
        "--provider",
        help=(
            "Optional LLM provider identifier (e.g. 'gemini', 'ollama'). "
            "Forwarded to the QA service if supported."
        ),
    )

    parser.add_argument(
        "--model-id",
        help=(
            "Optional model identifier to be used by the QA service. "
            "Forwarded as a hint when supported."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run the evaluation loop and persist the results."""
    setup_logging()
    args = parse_args()

    # Load questions from file or use the default in-memory dataset
    try:
        questions_raw = load_eval_questions(args.questions_file)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        sys.exit(1)

    questions: List[Dict[str, Any]] = list(_iter_questions(questions_raw))

    if args.limit is not None and args.limit > 0:
        questions = questions[: args.limit]

    if not questions:
        logging.error("No questions were loaded for evaluation.")
        sys.exit(1)

    logging.info("Loaded %d questions for evaluation.", len(questions))
    logging.info("Output file: %s", args.output_file)

    results: List[Dict[str, Any]] = []

    qa_kwargs: Dict[str, Any] = {}
    if args.provider:
        qa_kwargs["provider"] = args.provider
    if args.model_id:
        qa_kwargs["model_id"] = args.model_id

    total = len(questions)
    for idx, item in enumerate(questions, start=1):
        qtext = item["q"]
        logging.info("(%d/%d) Answering: %s", idx, total, qtext)

        try:
            answer_text = answer_question(qtext, **qa_kwargs)
        except TypeError:
            # If QA service does not support kwargs, retry without them
            answer_text = answer_question(qtext)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Failed to answer question '%s': %s", qtext, exc)
            answer_text = None

        row: Dict[str, Any] = {
            "q": qtext,
            "a": answer_text,
            "ts": time.time(),
        }

        # Preserve original per-question metadata when present
        for key, value in item.items():
            if key not in {"q"}:
                row[key] = value

        results.append(row)

    output_path = args.output_file.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    logging.info(
        "Evaluation finished. %d questions processed. Results saved to %s",
        len(results),
        output_path,
    )


if __name__ == "__main__":
    # Standard CLI guard
    main()
