"""
Step 1 of the BookNLP -> LitBank speaker attribution evaluation pipeline.

Parses LitBank's quotation + speaker attribution annotations
(litbank/quotations/tsv/*.ann + *.txt) and combines them with LitBank's
coreference annotations (litbank/coref/conll/*.conll) to produce, for each
book:
    - the gold quote span (flat document-level token indices)
    - the FULL set of mention spans belonging to the gold speaker's
      character (i.e. their entire coreference chain), since a speaker
      attribution is considered correct if the predicted speaker mention
      corefers with the gold character, not only if it is the exact same
      surface mention.

LitBank quotations .ann format (tab/whitespace separated):
    QUOTE   <quote_id>  <start_sent> <start_tok> <end_sent> <end_tok>  <quotation text...>
    ATTRIB  <quote_id>  <Speaker_Name-ClusterID>

The numeric suffix of the Speaker ID (e.g. "Turkey-28", "narrator-0") is the
coreference cluster ID from the CORRESPONDING coref/conll file for that
same book (verified: cluster "0" in bartleby's coref/conll is the
narrator's "I" chain, matching "narrator-0").

Sentence/token indices in .ann are relative to the paired .txt file
(one sentence per line, whitespace-tokenized), which uses the exact same
tokenization as the coref/conll files for the same book.

Usage:
    python prepare_litbank_speaker_input.py \
        --quotations-dir litbank/quotations/tsv/ \
        --coref-dir litbank/coref/conll/ \
        --output-dir data/litbank_speaker_input/
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from prepare_litbank_input import parse_conll_file


def tokenize_txt_with_offsets(text: str):
    """
    Tokenizes a LitBank quotations .txt file (one sentence per line,
    whitespace-separated tokens) and returns:
        - sentences: list[list[str]]
        - char_offsets: list[[start_char, end_char)] per flat token index,
          computed directly from the ACTUAL file content (not a
          reconstruction), so it is exact by construction.
    """
    sentences = []
    char_offsets = []
    cursor = 0

    for line in text.split("\n"):
        tokens = []
        for match in re.finditer(r"\S+", line):
            tokens.append(match.group())
            char_offsets.append([cursor + match.start(), cursor + match.end()])
        sentences.append(tokens)
        cursor += len(line) + 1  # +1 for the newline separating lines

    # Drop a possible trailing empty sentence caused by a final newline
    if sentences and len(sentences[-1]) == 0:
        sentences.pop()

    return sentences, char_offsets


def sentence_token_offsets(sentences):
    """Returns, for each sentence index, the flat-index of its first token."""
    offsets = []
    running = 0
    for sent in sentences:
        offsets.append(running)
        running += len(sent)
    return offsets


def parse_ann_file(ann_path: Path, sent_offsets):
    quotes = {}   # quote_id -> [start_flat, end_flat]
    attribs = {}  # quote_id -> speaker_label

    for raw_line in ann_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        parts = re.split(r"\s+", raw_line.strip())
        label = parts[0]

        if label == "QUOTE":
            quote_id = parts[1]
            start_sent, start_tok, end_sent, end_tok = (int(x) for x in parts[2:6])
            start_flat = sent_offsets[start_sent] + start_tok
            end_flat = sent_offsets[end_sent] + end_tok
            quotes[quote_id] = [start_flat, end_flat]
        elif label == "ATTRIB":
            quote_id = parts[1]
            speaker_label = parts[2]
            attribs[quote_id] = speaker_label

    return quotes, attribs
    
import os 

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotations-dir", type=Path, required=True)
    parser.add_argument("--coref-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gold_docs = []
    doc_order = []

    TARGET_FILES = [f.strip() for f in open('litbank_eval_files.txt').readlines()]
    
    for txt_path in sorted(args.quotations_dir.glob("*.txt")):
        file = os.path.split(txt_path)[1].split('.')[0]
        if file not in TARGET_FILES : 
            continue
        doc_key = txt_path.stem
        ann_path = args.quotations_dir / f"{doc_key}.ann"
        conll_path = args.coref_dir / f"{doc_key}.conll"

        if not ann_path.exists():
            print(f"[warn] Skipping '{doc_key}': no matching .ann file")
            continue
        if not conll_path.exists():
            print(f"[warn] Skipping '{doc_key}': no matching coref .conll file at {conll_path}")
            continue

        text = txt_path.read_text(encoding="utf-8")
        sentences, char_offsets = tokenize_txt_with_offsets(text)
        sent_offsets = sentence_token_offsets(sentences)

        quotes, attribs = parse_ann_file(ann_path, sent_offsets)
        coref_doc = parse_conll_file(conll_path)
        clusters_by_id = coref_doc["clusters_by_id"]

        gold_quotes = []
        for quote_id, quote_span in quotes.items():
            speaker_label = attribs.get(quote_id)
            if speaker_label is None:
                continue  # quote without an attribution annotation

            cluster_id = speaker_label.rsplit("-", 1)[-1]
            speaker_cluster_spans = clusters_by_id.get(cluster_id)
            if speaker_cluster_spans is None:
                print(
                    f"[warn] {doc_key}: speaker '{speaker_label}' references coref cluster "
                    f"'{cluster_id}' which was not found in {conll_path.name}"
                )
                continue

            gold_quotes.append(
                {
                    "quote_id": quote_id,
                    "quote_span": quote_span,
                    "speaker_label": speaker_label,
                    "speaker_cluster_spans": speaker_cluster_spans,
                }
            )

        gold_docs.append({"doc_key": doc_key, "quotes": gold_quotes})
        doc_order.append(doc_key)

        # Copy the exact .txt file used to compute offsets, as BookNLP input
        shutil.copy(txt_path, args.output_dir / f"{doc_key}.txt")
        (args.output_dir / f"{doc_key}.offsets.json").write_text(
            json.dumps(char_offsets), encoding="utf-8"
        )

    gold_path = args.output_dir / "gold_speaker_attribution.jsonl"
    with open(gold_path, "w", encoding="utf-8") as fout:
        for doc in gold_docs:
            fout.write(json.dumps(doc) + "\n")

    (args.output_dir / "doc_order.txt").write_text("\n".join(doc_order), encoding="utf-8")

    total_quotes = sum(len(d["quotes"]) for d in gold_docs)
    print(f"Parsed {len(gold_docs)} book(s), {total_quotes} attributed quotes total")
    print(f"Gold JSONL: {gold_path}")
    print(f"BookNLP input .txt + .offsets.json files written to {args.output_dir}")


if __name__ == "__main__":
    main()
