"""
Step 3 of the BookNLP -> BookCoref pipeline.

Converts raw BookNLP output (.tokens + .entities) into the JSONL format
expected by BookCoref's evaluate.py, by RE-PROJECTING BookNLP's own
token-index spans onto BookCoref's gold token indices via byte offsets.

This is the critical fix over a naive conversion: BookNLP tokenizes text
with its own tokenizer (spaCy), so BookNLP's start_token/end_token values
do NOT correspond to the same positions as BookCoref's gold token indices,
even if BookNLP was run on text reconstructed from gold tokens. Directly
copying BookNLP's indices into "clusters" (as if they were gold indices)
silently misaligns nearly every mention, which produces near-zero F1 in
evaluate.py.

Requires the character-offset map produced by prepare_booknlp_input.py
(<doc_key>.offsets.json): a list where element i = [start_char, end_char)
of gold flat-token index i, in the exact text that was fed to BookNLP.

NOTE: BookNLP's .tokens columns are named "byte_onset"/"byte_offset" but
are actually CHARACTER offsets (spaCy's tok.idx), not UTF-8 byte offsets.
This script treats them as character offsets accordingly; do not convert
them via .encode("utf-8") or you will silently misalign every mention on
books containing non-ASCII characters (curly quotes, em-dashes, etc.).

Pipeline recap:
    1. prepare_booknlp_input.py   -> data/booknlp_input/<doc_key>.txt
                                      data/booknlp_input/<doc_key>.offsets.json
    2. Run BookNLP on each .txt file -> <book_id>.tokens / <book_id>.entities
    3. booknlp_to_bookcoref.py (this script) -> predictions/booknlp_custom.jsonl
    4. python evaluate.py --predictions predictions/booknlp_custom.jsonl --mode full

Usage (single book):
    python booknlp_to_bookcoref.py \
        --tokens output_dir/siddhartha_2500/siddhartha_2500.tokens \
        --entities output_dir/siddhartha_2500/siddhartha_2500.entities \
        --offsets data/booknlp_input/siddhartha_2500.offsets.json \
        --doc-key siddhartha_2500 \
        --output predictions/booknlp_custom.jsonl

Usage (batch, one BookNLP output subfolder per book named after book_id):
    python booknlp_to_bookcoref.py \
        --booknlp-output-dir output_dir/ \
        --offsets-dir data/booknlp_input/ \
        --doc-order data/booknlp_input/doc_order.txt \
        --output predictions/booknlp_custom.jsonl
"""

import argparse
import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_booknlp_tokens(tokens_path: Path):
    """
    Returns a dict: booknlp_token_id -> (start_char, end_char)
    as reported natively by BookNLP for each of ITS OWN tokens.

    NOTE: BookNLP's own column names are "byte_onset"/"byte_offset", but
    they are populated from spaCy's tok.idx, i.e. CHARACTER offsets into
    the input text, not UTF-8 byte offsets. We read them as-is (integers)
    and treat them as character offsets throughout this script.
    """
    spans = {}
    with open(tokens_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            tok_id = int(row["token_ID_within_document"])
            spans[tok_id] = (int(row["byte_onset"]), int(row["byte_offset"]))
    return spans


class GoldCharIndex:
    """Maps a character offset (in the text fed to BookNLP) to the gold flat-token index."""

    def __init__(self, offsets):
        self.starts = [span[0] for span in offsets]
        self.ends = [span[1] for span in offsets]

    def token_at(self, char_pos: int) -> int:
        idx = bisect.bisect_right(self.starts, char_pos) - 1
        if idx < 0:
            idx = 0
        if idx >= len(self.starts):
            idx = len(self.starts) - 1
        return idx


def project_span(gold_index: GoldCharIndex, start_char: int, end_char: int):
    """Projects a [start_char, end_char) BookNLP mention span onto gold token indices."""
    gold_start = gold_index.token_at(start_char)
    # end_char is exclusive; look up the token covering the last included character
    gold_end = gold_index.token_at(max(end_char - 1, start_char))
    if gold_end < gold_start:
        gold_end = gold_start
    return [gold_start, gold_end]


def read_entities(entities_path: Path, booknlp_token_spans, gold_index: GoldCharIndex, keep_categories):
    clusters = defaultdict(list)
    with open(entities_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if keep_categories is not None and row["cat"] not in keep_categories:
                continue

            start_tok = int(row["start_token"])
            end_tok = int(row["end_token"])
            if start_tok not in booknlp_token_spans or end_tok not in booknlp_token_spans:
                continue

            start_char = booknlp_token_spans[start_tok][0]
            end_char = booknlp_token_spans[end_tok][1]

            gold_span = project_span(gold_index, start_char, end_char)
            clusters[row["COREF"]].append(gold_span)

    ordered_clusters = [
        sorted(mentions, key=lambda span: span[0]) for mentions in clusters.values()
    ]
    return ordered_clusters


def convert_book(tokens_path: Path, entities_path: Path, offsets_path: Path, doc_key: str, keep_categories):
    booknlp_token_spans = read_booknlp_tokens(tokens_path)
    gold_offsets = json.loads(offsets_path.read_text(encoding="utf-8"))
    gold_index = GoldCharIndex(gold_offsets)

    clusters = read_entities(entities_path, booknlp_token_spans, gold_index, keep_categories)
    return {"doc_key": doc_key, "clusters": clusters}


def find_book_files(booknlp_output_dir: Path):
    books = []
    for subdir in sorted(p for p in booknlp_output_dir.iterdir() if p.is_dir()):
        book_id = subdir.name
        tokens_path = subdir / f"{book_id}.tokens"
        entities_path = subdir / f"{book_id}.entities"
        if tokens_path.exists() and entities_path.exists():
            books.append((book_id, tokens_path, entities_path))
        else:
            print(f"[warn] Skipping '{book_id}': missing .tokens or .entities file")
    return books


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--entities", type=Path)
    parser.add_argument("--offsets", type=Path, help="Path to <doc_key>.offsets.json (single book mode)")
    parser.add_argument("--doc-key", type=str)

    parser.add_argument("--booknlp-output-dir", type=Path, help="Batch mode: one subfolder per book")
    parser.add_argument("--offsets-dir", type=Path, help="Batch mode: dir containing <doc_key>.offsets.json files")

    parser.add_argument("--doc-order", type=Path, default=None, help="File with one doc_key per line, in gold order")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--all-categories",
        action="store_true",
        help="Keep all BookNLP entity categories instead of PER-only (default: PER-only, matching BookCoref's character-only gold clusters)",
    )
    args = parser.parse_args()

    keep_categories = None if args.all_categories else {"PER"}

    docs = {}
    if args.tokens and args.entities and args.offsets:
        doc_key = args.doc_key or args.tokens.stem
        docs[doc_key] = convert_book(args.tokens, args.entities, args.offsets, doc_key, keep_categories)
    elif args.booknlp_output_dir and args.offsets_dir:
        for book_id, tokens_path, entities_path in find_book_files(args.booknlp_output_dir):
            offsets_path = args.offsets_dir / f"{book_id}.offsets.json"
            if not offsets_path.exists():
                print(f"[warn] Skipping '{book_id}': no offsets file found at {offsets_path}")
                continue
            docs[book_id] = convert_book(tokens_path, entities_path, offsets_path, book_id, keep_categories)
    else:
        parser.error("Provide either --tokens/--entities/--offsets or --booknlp-output-dir/--offsets-dir")

    if args.doc_order:
        order = [line.strip() for line in args.doc_order.read_text().splitlines() if line.strip()]
        missing = [d for d in order if d not in docs]
        if missing:
            print(f"[warn] {len(missing)} doc_key(s) from --doc-order not found: {missing}")
        ordered_docs = [docs[d] for d in order if d in docs]
    else:
        ordered_docs = list(docs.values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for doc in ordered_docs:
            fout.write(json.dumps(doc) + "\n")

    print(f"Wrote {len(ordered_docs)} document(s) to {args.output}")


if __name__ == "__main__":
    main()