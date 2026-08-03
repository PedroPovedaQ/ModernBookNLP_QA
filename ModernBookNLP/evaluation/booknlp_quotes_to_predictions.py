"""
Step 3 of the BookNLP -> LitBank speaker attribution pipeline.

Converts BookNLP's own quote + entity output (.quotes + .entities + .tokens)
into a predictions JSONL: for each predicted quote, the quote span AND the
predicted speaker's full coreference cluster (all its mention spans),
projected onto LitBank's gold token indices via the same character-offset
alignment used for coreference evaluation.

BookNLP's .quotes columns (see english_booknlp.py process()):
    quote_start, quote_end, mention_start, mention_end, mention_phrase,
    char_id, quote
    -- quote_start/end and mention_start/end are BookNLP's own
       "token_ID_within_document" indices (same scheme as .tokens/.entities)
    -- char_id is BookNLP's internal predicted coreference cluster id for
       the speaker (same id space as the COREF column in .entities)
    -- char_id / mention_start / mention_end can be "None" if BookNLP could
       not attribute a speaker to that quote

Usage (single book):
    python booknlp_quotes_to_predictions.py \
        --tokens output_dir/<doc_key>/<doc_key>.tokens \
        --entities output_dir/<doc_key>/<doc_key>.entities \
        --quotes output_dir/<doc_key>/<doc_key>.quotes \
        --offsets data/litbank_speaker_input/<doc_key>.offsets.json \
        --doc-key <doc_key> \
        --output predictions/booknlp_speaker_attribution.jsonl

Usage (batch, one BookNLP output subfolder per book named after doc_key):
    python booknlp_quotes_to_predictions.py \
        --booknlp-output-dir output_dir/ \
        --offsets-dir data/litbank_speaker_input/ \
        --doc-order data/litbank_speaker_input/doc_order.txt \
        --output predictions/booknlp_speaker_attribution.jsonl
"""

import argparse
import bisect
import csv
import json
from collections import defaultdict
from pathlib import Path


def read_booknlp_tokens(tokens_path: Path):
    spans = {}
    with open(tokens_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            tok_id = int(row["token_ID_within_document"])
            spans[tok_id] = (int(row["byte_onset"]), int(row["byte_offset"]))
    return spans


class GoldCharIndex:
    def __init__(self, offsets):
        self.starts = [span[0] for span in offsets]

    def token_at(self, char_pos: int) -> int:
        idx = bisect.bisect_right(self.starts, char_pos) - 1
        return max(0, min(idx, len(self.starts) - 1))


def project_span(gold_index: GoldCharIndex, start_char: int, end_char: int):
    gold_start = gold_index.token_at(start_char)
    gold_end = gold_index.token_at(max(end_char - 1, start_char))
    return [gold_start, max(gold_end, gold_start)]


def read_entity_clusters(entities_path: Path, booknlp_token_spans, gold_index: GoldCharIndex):
    """Returns dict: booknlp COREF id (str) -> list of gold-projected [start,end] spans."""
    clusters = defaultdict(list)
    with open(entities_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            start_tok = int(row["start_token"])
            end_tok = int(row["end_token"])
            if start_tok not in booknlp_token_spans or end_tok not in booknlp_token_spans:
                continue
            start_char = booknlp_token_spans[start_tok][0]
            end_char = booknlp_token_spans[end_tok][1]
            clusters[row["COREF"]].append(project_span(gold_index, start_char, end_char))
    return clusters


def read_quotes(quotes_path: Path, booknlp_token_spans, gold_index: GoldCharIndex, entity_clusters):
    predicted_quotes = []
    with open(quotes_path, "r", encoding="utf-8") as fin:
        reader = csv.DictReader(fin, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            q_start, q_end = int(row["quote_start"]), int(row["quote_end"])
            if q_start not in booknlp_token_spans or q_end not in booknlp_token_spans:
                continue
            quote_span = project_span(
                gold_index,
                booknlp_token_spans[q_start][0],
                booknlp_token_spans[q_end][1],
            )

            char_id = row["char_id"]
            speaker_cluster_spans = []
            speaker_mention_span = None
            if char_id and char_id != "None":
                speaker_cluster_spans = entity_clusters.get(char_id, [])
                m_start, m_end = row.get("mention_start"), row.get("mention_end")
                if m_start not in (None, "None") and m_end not in (None, "None"):
                    m_start, m_end = int(m_start), int(m_end)
                    if m_start in booknlp_token_spans and m_end in booknlp_token_spans:
                        speaker_mention_span = project_span(
                            gold_index,
                            booknlp_token_spans[m_start][0],
                            booknlp_token_spans[m_end][1],
                        )

            predicted_quotes.append(
                {
                    "quote_span": quote_span,
                    "speaker_char_id": char_id if char_id != "None" else None,
                    "speaker_mention_span": speaker_mention_span,
                    "speaker_cluster_spans": speaker_cluster_spans,
                }
            )
    return predicted_quotes


def convert_book(tokens_path, entities_path, quotes_path, offsets_path, doc_key):
    booknlp_token_spans = read_booknlp_tokens(tokens_path)
    gold_offsets = json.loads(offsets_path.read_text(encoding="utf-8"))
    gold_index = GoldCharIndex(gold_offsets)

    entity_clusters = read_entity_clusters(entities_path, booknlp_token_spans, gold_index)
    quotes = read_quotes(quotes_path, booknlp_token_spans, gold_index, entity_clusters)

    return {"doc_key": doc_key, "quotes": quotes}


def find_book_files(booknlp_output_dir: Path):
    books = []
    for subdir in sorted(p for p in booknlp_output_dir.iterdir() if p.is_dir()):
        book_id = subdir.name
        tokens_path = subdir / f"{book_id}.tokens"
        entities_path = subdir / f"{book_id}.entities"
        quotes_path = subdir / f"{book_id}.quotes"
        if tokens_path.exists() and entities_path.exists() and quotes_path.exists():
            books.append((book_id, tokens_path, entities_path, quotes_path))
        else:
            print(f"[warn] Skipping '{book_id}': missing .tokens/.entities/.quotes file")
    return books


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--entities", type=Path)
    parser.add_argument("--quotes", type=Path)
    parser.add_argument("--offsets", type=Path)
    parser.add_argument("--doc-key", type=str)

    parser.add_argument("--booknlp-output-dir", type=Path)
    parser.add_argument("--offsets-dir", type=Path)

    parser.add_argument("--doc-order", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    docs = {}
    if args.tokens and args.entities and args.quotes and args.offsets:
        doc_key = args.doc_key or args.tokens.stem
        docs[doc_key] = convert_book(args.tokens, args.entities, args.quotes, args.offsets, doc_key)
    elif args.booknlp_output_dir and args.offsets_dir:
        for book_id, tokens_path, entities_path, quotes_path in find_book_files(args.booknlp_output_dir):
            offsets_path = args.offsets_dir / f"{book_id}.offsets.json"
            if not offsets_path.exists():
                print(f"[warn] Skipping '{book_id}': no offsets file at {offsets_path}")
                continue
            docs[book_id] = convert_book(tokens_path, entities_path, quotes_path, offsets_path, book_id)
    else:
        parser.error("Provide either --tokens/--entities/--quotes/--offsets or --booknlp-output-dir/--offsets-dir")

    if args.doc_order:
        order = [line.strip() for line in args.doc_order.read_text().splitlines() if line.strip()]
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
