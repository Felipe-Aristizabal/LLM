"""CLI entrypoint to run a manual QA evaluation over a question set."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tecnoquimicas_kb.domain.eval_dataset import load_eval_questions
from tecnoquimicas_kb.domain.qa_service import answer_question


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script."""
    # Create an argument parser with a clear description
    parser = argparse.ArgumentParser(
        description=(
            "Run a QA evaluation over a predefined question set and "
            "store the answers in a JSONL file."
        )
    )

    # Optional path to a questions file; if omitted, use the package default
    parser.add_argument(
        "--questions-file",
        "-qf",
        type=Path,
        help=(
            "Path to a questions file used for evaluation. "
            "If omitted, the default dataset defined in eval_dataset will be used."
        ),
    )

    # Output file for the evaluation results
    parser.add_argument(
        "--output-file",
        "-o",
        type=Path,
        default=Path("qa_eval_results.jsonl"),
        help="Path to the JSONL file where evaluation results will be written.",
    )

    # Limit on the number of questions to evaluate (useful for quick smoke tests)
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional maximum number of questions to evaluate.",
    )

    # Optional model configuration (forwarded to the QA service if supported)
    parser.add_argument(
        "--provider",
        help="Optional LLM provider identifier (e.g. 'gemini', 'ollama').",
    )
    parser.add_argument(
        "--model-id",
        help="Optional model identifier to be used by the QA service.",
    )

    return parser.parse_args()


def _iter_questions(raw_questions: Iterable[Any]) -> Iterable[Dict[str, Any]]:
    """Normalize the questions iterable into a dict-based representation.

    This helper makes the script robust to different shapes, such as:
    - a list of plain strings
    - a list of dicts with 'q' or 'question' keys
    """
    for item in raw_questions:
        if isinstance(item, str):
            yield {"q": item}
        elif isinstance(item, dict):
            # Preserve the original dict but ensure a 'q' key is present
            question_text = (
                item.get("q") or item.get("question") or item.get("text") or str(item)
            )
            normalized = dict(item)
            normalized["q"] = question_text
            yield normalized
        else:
            # Fallback: convert unknown types to string
            yield {"q": str(item), "raw": item}


def main() -> None:
    """Run the evaluation loop and persist the results."""
    # Configure logging for the evaluation run
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    args = parse_args()

    # Load the evaluation questions from disk or package data
    try:
        if args.questions_file is not None:
            questions_raw = load_eval_questions(args.questions_file)
        else:
            # Assume the helper can also be called without arguments
            questions_raw = load_eval_questions()  # type: ignore[call-arg]
    except TypeError:
        # Fallback if load_eval_questions strictly requires a path
        if args.questions_file is None:
            logging.error(
                "load_eval_questions requires a questions file path but none was "
                "provided. Use --questions-file."
            )
            sys.exit(1)
        questions_raw = load_eval_questions(args.questions_file)

    normalized_questions: List[Dict[str, Any]] = list(_iter_questions(questions_raw))

    if args.limit is not None and args.limit > 0:
        normalized_questions = normalized_questions[: args.limit]

    if not normalized_questions:
        logging.error("No questions were loaded for evaluation.")
        sys.exit(1)

    logging.info("Loaded %d questions for evaluation.", len(normalized_questions))
    logging.info("Output file: %s", args.output_file)

    results: List[Dict[str, Any]] = []

    # Prepare kwargs to forward optional model configuration to the QA service
    qa_kwargs: Dict[str, Any] = {}
    if args.provider:
        qa_kwargs["provider"] = args.provider
    if args.model_id:
        qa_kwargs["model_id"] = args.model_id

    # For each question, call the QA service to obtain an answer
    for idx, item in enumerate(normalized_questions, start=1):
        question_text = item["q"]
        logging.info(
            "(%d/%d) Answering: %s", idx, len(normalized_questions), question_text
        )

        try:
            answer_text = answer_question(question_text, **qa_kwargs)
        except TypeError:
            # If the QA service does not support the forwarded kwargs, retry without them
            answer_text = answer_question(question_text)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Failed to answer question '%s': %s", question_text, exc)
            answer_text = None

        row: Dict[str, Any] = {
            "q": question_text,
            "a": answer_text,
            "ts": time.time(),
        }

        # Optionally preserve per-question metadata if present
        for key, value in item.items():
            if key not in {"q"}:
                row[key] = value

        results.append(row)

    # Persist the results as JSONL
    output_path = args.output_file.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Log a short summary (number of questions, output path)
    logging.info(
        "Evaluation finished. %d questions processed. Results written to %s",
        len(results),
        output_path,
    )


if __name__ == "__main__":
    # Standard entrypoint guard
    main()
