"""
Step 1 of the BookNLP -> BookCoref pipeline.

Rebuilds plain-text book files from BookCoref's gold `sentences` field,
to be fed into BookNLP, WHILE recording the exact byte-offset span of
every gold (flat) token in the resulting text. This offset map is the
critical piece that lets us later re-project BookNLP's own token spans
(from its own tokenizer) back onto BookCoref's gold token indices.

Why byte offsets (not character indices)? Because BookNLP's .tokens
file reports byte_onset / byte_offset as BYTE offsets into the UTF-8
encoded input file, not Python string (codepoint) indices. Books
contain non-ASCII characters (curly quotes, em-dashes, accented
letters...) where 1 character != 1 byte, so using character indices
would silently drift and misalign spans throughout the book.

Usage:
    python prepare_booknlp_input.py --output-dir data/booknlp_input/

This will create, for every book in the BookCoref test split:
    data/booknlp_input/<doc_key>.txt              (BookNLP input)
    data/booknlp_input/<doc_key>.offsets.json      (gold token -> character span)
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def build_text_and_offsets(sentences):
    """
    Flattens `sentences` (list[list[str]]) into a single text string and
    computes, for each flat token index (matching the indexing scheme used
    by BookCoref's `clusters` spans), its [start_char, end_char) span in
    the text, IN PYTHON CHARACTER (CODEPOINT) UNITS.

    IMPORTANT: BookNLP's .tokens file columns are named "byte_onset" /
    "byte_offset", but they are NOT UTF-8 byte offsets. Looking at BookNLP's
    source (booknlp/common/pipelines.py), these values are populated
    directly from spaCy's `tok.idx`, which is a character index into the
    Python string that was tokenized. Using actual UTF-8 byte lengths here
    (e.g. via `.encode("utf-8")`) will silently drift out of sync with
    BookNLP's offsets on every non-ASCII character (curly quotes, em-dashes,
    accented letters...), which is common in literary text and previously
    caused near-total misalignment.

    Tokens are joined with a single space. A literal "\\n" token (used by
    BookCoref to mark paragraph/line breaks) is emitted as "\\n\\n" so that
    BookNLP's own paragraph segmentation picks it up, but WITHOUT the
    per-sentence `.strip()` that can silently collapse edge tokens to zero
    width and desync offsets.
    """
    pieces = []       # text fragments to join with " "
    char_offsets = []  # [start_char, end_char) per flat token, aligned to `pieces` order

    cursor = 0
    first = True
    for sentence in sentences:
        for tok in sentence:
            piece = "\n\n" if tok == "\n" else tok

            if not first:
                cursor += 1  # account for the joining space (1 character)
            first = False

            start = cursor
            cursor += len(piece)  # character length, NOT byte length
            end = cursor
            char_offsets.append([start, end])
            pieces.append(piece)

    text = " ".join(pieces)
    return text, char_offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--configuration",
        type=str,
        default="default",
        choices=["default", "split"],
        help="Use 'default' for full-book test.jsonl (mode=full/gold_window), "
        "'split' for test_split.jsonl (mode=split)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.configuration == "default":
        dataset = load_dataset("/data/bookcoref")
    else:
        dataset = load_dataset("/data/bookcoref", "split")

    doc_order = []
    for doc in dataset[args.split]:
        doc_key = doc["doc_key"] if "doc_key" in doc else doc["doc_id"]
        text, char_offsets = build_text_and_offsets(doc["sentences"])

        (args.output_dir / f"{doc_key}.txt").write_text(text, encoding="utf-8")
        (args.output_dir / f"{doc_key}.offsets.json").write_text(
            json.dumps(char_offsets), encoding="utf-8"
        )
        doc_order.append(doc_key)

    (args.output_dir / "doc_order.txt").write_text("\n".join(doc_order), encoding="utf-8")
    print(f"Wrote {len(doc_order)} book(s) + offset maps to {args.output_dir}")
    print(f"doc_order.txt records the exact order to pass to booknlp_to_bookcoref.py --doc-order")


if __name__ == "__main__":
    main()