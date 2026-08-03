"""
Step 1 of the BookNLP -> LitBank evaluation pipeline.

Parses LitBank's coreference gold annotations (standard CoNLL-2012 format,
one file per book, in litbank/coref/conll/*.conll) into:
    - a gold.jsonl file (same schema as BookCoref: {"doc_key", "sentences",
      "clusters"}, with `clusters` spans as FLAT document-level token
      indices, consistent with BookNLP's own "token_ID_within_document"
      indexing and with the character-offset projection approach used for
      BookCoref)
    - one <doc_key>.txt file per book (BookNLP input, reconstructed from
      the gold tokens themselves, since LitBank's coref layer only covers
      a ~2,000-token excerpt of each book, not the full text)
    - one <doc_key>.offsets.json file per book (gold flat-token index ->
      [start_char, end_char) in that .txt file), needed by
      booknlp_to_bookcoref.py to re-project BookNLP's own token spans onto
      LitBank's gold token indices

CoNLL coreference cell format (last column of each token row):
    "_"          -> no coreference tag on this token
    "(N"         -> opens a mention span belonging to cluster N
    "N)"         -> closes the most recently opened span for cluster N
    "(N)"        -> a single-token mention belonging to cluster N
    "(N|M)"      -> multiple tags on the same token, separated by "|"

Usage:
    python prepare_litbank_input.py \
        --conll-dir litbank/coref/conll/ \
        --output-dir data/litbank_input/
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict


def parse_conll_file(path: Path):
    doc_key = None
    sentences = []
    current_sentence = []
    clusters = defaultdict(list)
    open_spans = defaultdict(list)  # cluster_id -> stack of flat_idx starts
    flat_idx = 0

    with open(path, "r", encoding="utf-8") as fin:
        for raw_line in fin:
            line = raw_line.rstrip("\n")

            if line.startswith("#begin document"):
                match = re.search(r"\((.*?)\)", line)
                doc_key = match.group(1) if match else path.stem
                continue
            if line.startswith("#end document"):
                continue

            if line.strip() == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            cols = line.split()
            word = cols[3]
            coref_cell = cols[-1]

            current_sentence.append(word)

            if coref_cell != "_":
                for tag in coref_cell.split("|"):
                    if tag.startswith("(") and tag.endswith(")"):
                        cluster_id = tag[1:-1]
                        clusters[cluster_id].append([flat_idx, flat_idx])
                    elif tag.startswith("("):
                        cluster_id = tag[1:]
                        open_spans[cluster_id].append(flat_idx)
                    elif tag.endswith(")"):
                        cluster_id = tag[:-1]
                        if open_spans[cluster_id]:
                            start = open_spans[cluster_id].pop()
                            clusters[cluster_id].append([start, flat_idx])
                        else:
                            print(f"[warn] {path.name}: unmatched close tag '{tag}' at token {flat_idx}")

            flat_idx += 1

    if current_sentence:
        sentences.append(current_sentence)

    clusters_by_id = {
        cid: sorted(mentions, key=lambda span: span[0]) for cid, mentions in clusters.items()
    }
    ordered_clusters = list(clusters_by_id.values())

    return {
        "doc_key": doc_key,
        "sentences": sentences,
        "clusters": ordered_clusters,
        "clusters_by_id": clusters_by_id,  # e.g. clusters_by_id["28"] -> [[start,end], ...]
    }


def build_text_and_offsets(sentences):
    """Same character-offset-preserving construction used for BookCoref."""
    pieces = []
    char_offsets = []
    cursor = 0
    first = True
    for sentence in sentences:
        for tok in sentence:
            piece = tok
            if not first:
                cursor += 1
            first = False
            start = cursor
            cursor += len(piece)
            end = cursor
            char_offsets.append([start, end])
            pieces.append(piece)
    text = " ".join(pieces)
    return text, char_offsets

import os 

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conll-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gold_docs = []
    doc_order = []
    TARGET_FILES = [f.strip() for f in open('litbank_eval_files.txt').readlines()]
    for conll_path in sorted(args.conll_dir.glob("*.conll")):
        file = os.path.split(conll_path)[1].split('.')[0]
        if file not in TARGET_FILES : 
            continue
            
        doc = parse_conll_file(conll_path)
        gold_docs.append(doc)
        doc_order.append(doc["doc_key"])

        text, char_offsets = build_text_and_offsets(doc["sentences"])
        (args.output_dir / f"{doc['doc_key']}.txt").write_text(text, encoding="utf-8")
        (args.output_dir / f"{doc['doc_key']}.offsets.json").write_text(
            json.dumps(char_offsets), encoding="utf-8"
        )

    gold_path = args.output_dir / "gold.jsonl"
    with open(gold_path, "w", encoding="utf-8") as fout:
        for doc in gold_docs:
            fout.write(json.dumps(doc) + "\n")

    (args.output_dir / "doc_order.txt").write_text("\n".join(doc_order), encoding="utf-8")

    print(f"Parsed {len(gold_docs)} book(s) from {args.conll_dir}")
    print(f"Gold JSONL: {gold_path}")
    print(f"BookNLP input .txt + .offsets.json files written to {args.output_dir}")


if __name__ == "__main__":
    main()