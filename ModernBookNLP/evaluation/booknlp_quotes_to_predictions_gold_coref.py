"""
Step 3 (gold-coreference variant) of the BookNLP -> LitBank speaker
attribution pipeline.

Unlike booknlp_quotes_to_predictions.py, this script does NOT rely on
BookNLP's own predicted coreference clusters (the .entities file /
COREF column) to expand a predicted speaker into its full set of mention
spans. Instead, it uses LitBank's GOLD coreference clusters (produced by
prepare_litbank_input.py as gold.jsonl, field "clusters_by_id") as the
source of truth for "which spans belong to the same entity".

Concretely, for each BookNLP-predicted quote:
    1. The quote span itself is projected onto LitBank's gold token
       indices, exactly as in booknlp_quotes_to_predictions.py.
    2. BookNLP's predicted speaker MENTION (quotes.mention_start /
       quotes.mention_end -- i.e. WHERE BookNLP thinks the speaker is
       named/mentioned) is projected onto LitBank's gold token indices.
    3. That projected mention span is looked up against LitBank's GOLD
       coreference clusters for the same book: whichever gold cluster has
       a mention overlapping the projected span the most is considered
       the speaker's gold entity, and ALL of that gold cluster's mention
       spans are used as `speaker_cluster_spans` for evaluation.
    4. If the mention span cannot be matched to any gold cluster (no
       predicted mention, or no overlapping gold mention), the speaker is
       left unresolved (empty `speaker_cluster_spans`), exactly like the
       "char_id is None" case in the original script.

This removes any dependency on BookNLP's .entities file / predicted
coreference resolution: only .tokens, .quotes, the gold.jsonl produced by
prepare_litbank_input.py, and the matching .offsets.json are required.

BookNLP's .quotes columns (see english_booknlp.py process()):
    quote_start, quote_end, mention_start, mention_end, mention_phrase,
    char_id, quote
    -- quote_start/end and mention_start/end are BookNLP's own
       "token_ID_within_document" indices (same scheme as .tokens)
    -- char_id is BookNLP's internal predicted coreference cluster id;
       kept in the output purely for reference/debugging, it plays no
       role in building speaker_cluster_spans here
    -- char_id / mention_start / mention_end can be "None" if BookNLP
       could not attribute a speaker to that quote

Usage (single book):
    python booknlp_quotes_to_predictions_gold_coref.py \
        --tokens output_dir/<doc_key>/<doc_key>.tokens \
        --quotes output_dir/<doc_key>/<doc_key>.quotes \
        --offsets data/litbank_speaker_input/<doc_key>.offsets.json \
        --gold data/litbank_speaker_input/gold.jsonl \
        --doc-key <doc_key> \
        --output predictions/booknlp_speaker_attribution_gold_coref.jsonl

Usage (batch, one BookNLP output subfolder per book named after doc_key):
    python booknlp_quotes_to_predictions_gold_coref.py \
        --booknlp-output-dir output_dir/ \
        --offsets-dir data/litbank_speaker_input/ \
        --gold data/litbank_speaker_input/gold.jsonl \
        --doc-order data/litbank_speaker_input/doc_order.txt \
        --output predictions/booknlp_speaker_attribution_gold_coref.jsonl
"""

import argparse
import bisect
import csv
import json
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


def load_gold_clusters(gold_path: Path):
    """Returns dict: doc_key -> {gold_cluster_id (str) -> [[start,end], ...]}."""
    gold_clusters_by_doc = {}
    with open(gold_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            clusters_by_id = doc.get("clusters_by_id")
            if clusters_by_id is None:
                # Fall back to the unlabeled "clusters" list if clusters_by_id
                # isn't present, assigning positional ids.
                clusters_by_id = {
                    str(i): spans for i, spans in enumerate(doc.get("clusters", []))
                }
            gold_clusters_by_doc[doc["doc_key"]] = clusters_by_id
    return gold_clusters_by_doc


def find_matching_gold_cluster(mention_span, clusters_by_id):
    """Finds the gold cluster with the mention span that overlaps `mention_span`
    the most (by number of overlapping gold token indices).

    Returns (gold_cluster_id, cluster_spans) or (None, []) if no gold cluster
    has a mention overlapping `mention_span`.
    """
    if mention_span is None or not clusters_by_id:
        return None, []

    m_start, m_end = mention_span
    best_id, best_overlap = None, 0
    for cluster_id, spans in clusters_by_id.items():
        for start, end in spans:
            if end < m_start or start > m_end:
                continue
            overlap = min(end, m_end) - max(start, m_start) + 1
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = cluster_id

    if best_id is None:
        return None, []
    return best_id, clusters_by_id[best_id]


def read_quotes(quotes_path: Path, booknlp_token_spans, gold_index: GoldCharIndex, gold_clusters_by_id):
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
            speaker_mention_span = None
            m_start, m_end = row.get("mention_start"), row.get("mention_end")
            if m_start not in (None, "None") and m_end not in (None, "None"):
                m_start, m_end = int(m_start), int(m_end)
                if m_start in booknlp_token_spans and m_end in booknlp_token_spans:
                    speaker_mention_span = project_span(
                        gold_index,
                        booknlp_token_spans[m_start][0],
                        booknlp_token_spans[m_end][1],
                    )

            gold_cluster_id, speaker_cluster_spans = find_matching_gold_cluster(
                speaker_mention_span, gold_clusters_by_id
            )

            predicted_quotes.append(
                {
                    "quote_span": quote_span,
                    "speaker_booknlp_char_id": char_id if char_id != "None" else None,
                    "speaker_mention_span": speaker_mention_span,
                    "speaker_char_id": gold_cluster_id,
                    "speaker_cluster_spans": speaker_cluster_spans,
                }
            )
    return predicted_quotes


def convert_book(tokens_path, quotes_path, offsets_path, gold_clusters_by_id, doc_key):
    booknlp_token_spans = read_booknlp_tokens(tokens_path)
    gold_offsets = json.loads(offsets_path.read_text(encoding="utf-8"))
    gold_index = GoldCharIndex(gold_offsets)

    quotes = read_quotes(quotes_path, booknlp_token_spans, gold_index, gold_clusters_by_id)

    return {"doc_key": doc_key, "quotes": quotes}


def find_book_files(booknlp_output_dir: Path):
    books = []
    for subdir in sorted(p for p in booknlp_output_dir.iterdir() if p.is_dir()):
        book_id = subdir.name
        tokens_path = subdir / f"{book_id}.tokens"
        quotes_path = subdir / f"{book_id}.quotes"
        if tokens_path.exists() and quotes_path.exists():
            books.append((book_id, tokens_path, quotes_path))
        else:
            print(f"[warn] Skipping '{book_id}': missing .tokens/.quotes file")
    return books


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--quotes", type=Path)
    parser.add_argument("--offsets", type=Path)
    parser.add_argument("--doc-key", type=str)

    parser.add_argument("--booknlp-output-dir", type=Path)
    parser.add_argument("--offsets-dir", type=Path)

    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
        help="gold.jsonl produced by prepare_litbank_input.py (needs 'clusters_by_id' per doc)",
    )
    parser.add_argument("--doc-order", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gold_clusters_by_doc = load_gold_clusters(args.gold)

    docs = {}
    if args.tokens and args.quotes and args.offsets:
        doc_key = args.doc_key or args.tokens.stem
        gold_clusters_by_id = gold_clusters_by_doc.get(doc_key, {})
        if not gold_clusters_by_id:
            print(f"[warn] No gold coreference clusters found for '{doc_key}' in {args.gold}")
        docs[doc_key] = convert_book(args.tokens, args.quotes, args.offsets, gold_clusters_by_id, doc_key)
    elif args.booknlp_output_dir and args.offsets_dir:
        for book_id, tokens_path, quotes_path in find_book_files(args.booknlp_output_dir):
            offsets_path = args.offsets_dir / f"{book_id}.offsets.json"
            if not offsets_path.exists():
                print(f"[warn] Skipping '{book_id}': no offsets file at {offsets_path}")
                continue
            gold_clusters_by_id = gold_clusters_by_doc.get(book_id, {})
            if not gold_clusters_by_id:
                print(f"[warn] No gold coreference clusters found for '{book_id}' in {args.gold}")
            docs[book_id] = convert_book(tokens_path, quotes_path, offsets_path, gold_clusters_by_id, book_id)
    else:
        parser.error("Provide either --tokens/--quotes/--offsets or --booknlp-output-dir/--offsets-dir")

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
