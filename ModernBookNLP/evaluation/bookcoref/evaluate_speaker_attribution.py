"""
Step 4: evaluates BookNLP's speaker attribution against LitBank gold using
the full CoNLL-2012 metric suite (MUC, B3, CEAFe, and their average
"CoNLL F1"), to allow comparison against Sims & Bamman (2020), "Measuring
Information Propagation in Literary Social Networks".

Reports both:
    - the POOLED score (all books' quotes merged together into one
      clustering evaluation)
    - the PER-BOOK mean +/- standard deviation (macro-averaged across
      books)

This reuses bookcoref's OfficialCoNLL2012CorefEvaluator (see src/metrics.py
in the bookcoref repo), which is metric-agnostic to what a "mention" is:
it just compares clusters of hashable items. Here, each ITEM is a gold
quote_id, instead of a text span:
    - a GOLD cluster groups all quote_ids that share the same true speaker
      (i.e. the same LitBank coreference cluster ID)
    - a SYSTEM cluster groups all quote_ids that BookNLP attributed to the
      same predicted character (char_id), with quotes that were missed
      entirely (no overlapping predicted quote) or left unattributed (no
      predicted speaker) each placed in their own singleton cluster.

Run this from inside your cloned bookcoref/ repo (or copy src/metrics.py
alongside this script).

Usage:
    python evaluate_speaker_attribution.py \
        --gold data/litbank_speaker_input/gold_speaker_attribution.jsonl \
        --predictions predictions/booknlp_speaker_attribution.jsonl \
        --doc-order data/litbank_speaker_input/doc_order.txt \
        --save-json results/litbank_speaker_attribution_results.json
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from src.metrics import OfficialCoNLL2012CorefEvaluator

METRICS = ["muc", "b_cubed", "ceafe", "conll2012"]


def spans_overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


def overlap_amount(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def best_matching_quote(gold_quote_span, predicted_quotes):
    best, best_overlap = None, 0
    for pred in predicted_quotes:
        ov = overlap_amount(gold_quote_span, pred["quote_span"])
        if ov > best_overlap:
            best_overlap, best = ov, pred
    return best


def extract_mentions_to_clusters(clusters):
    mention_to_cluster = {}
    for cluster in clusters:
        for item in cluster:
            mention_to_cluster[item] = cluster
    return mention_to_cluster


def build_doc_clusters(gold_doc, pred_doc):
    """Returns (gold_clusters, system_clusters, n_no_quote, n_no_speaker) for one document."""
    predicted_quotes = pred_doc["quotes"] if pred_doc else []

    gold_clusters_by_label = defaultdict(list)
    system_clusters_by_label = defaultdict(list)

    n_no_quote = 0
    n_no_speaker = 0

    for gq in gold_doc["quotes"]:
        qid = gq["quote_id"]

        cluster_id = gq["speaker_label"].rsplit("-", 1)[-1]
        gold_clusters_by_label[cluster_id].append(qid)

        match = best_matching_quote(gq["quote_span"], predicted_quotes)
        if match is None:
            system_clusters_by_label[f"__no_quote__{qid}"].append(qid)
            n_no_quote += 1
            continue

        char_id = match.get("speaker_char_id")
        if char_id is None:
            system_clusters_by_label[f"__no_speaker__{qid}"].append(qid)
            n_no_speaker += 1
        else:
            system_clusters_by_label[f"char::{char_id}"].append(qid)

    gold_clusters = [tuple(v) for v in gold_clusters_by_label.values()]
    system_clusters = [tuple(v) for v in system_clusters_by_label.values()]

    return gold_clusters, system_clusters, n_no_quote, n_no_speaker


def score_doc(gold_clusters, system_clusters):
    evaluator = OfficialCoNLL2012CorefEvaluator()
    mention_to_gold = extract_mentions_to_clusters(gold_clusters)
    mention_to_system = extract_mentions_to_clusters(system_clusters)
    evaluator.update(system_clusters, gold_clusters, mention_to_system, mention_to_gold)
    return {metric: dict(zip(["precision", "recall", "f1"], evaluator.get_prf(metric))) for metric in METRICS}


def macro_mean_std(per_book_results):
    summary = {}
    book_keys = list(per_book_results.keys())
    for metric in METRICS:
        for score_type in ["precision", "recall", "f1"]:
            values = [per_book_results[b][metric][score_type] for b in book_keys]
            mean = statistics.mean(values) if values else 0.0
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            summary.setdefault(metric, {})[score_type] = {"mean": mean, "std": std}
    return summary


def print_results(pooled_results, macro_summary, n_books, total_quotes, total_no_quote, total_no_speaker):
    print(f"Total gold attributed quotes: {total_quotes}")
    print(f"  Quotes with no overlapping predicted quote: {total_no_quote}")
    print(f"  Quotes matched but with no predicted speaker: {total_no_speaker}")
    print()

    print("[Pooled results — all books' quotes merged into one clustering evaluation]")
    for metric in METRICS:
        r = pooled_results[metric]
        label = "CoNLL F1 (avg)" if metric == "conll2012" else metric
        print(f"  {label:16s} P: {100*r['precision']:.2f}  R: {100*r['recall']:.2f}  F1: {100*r['f1']:.2f}")

    print()
    print(f"[Per-book mean +/- std across {n_books} books]")
    for metric in METRICS:
        label = "CoNLL F1 (avg)" if metric == "conll2012" else metric
        f1 = macro_summary[metric]["f1"]
        p = macro_summary[metric]["precision"]
        r = macro_summary[metric]["recall"]
        print(
            f"  {label:16s} "
            f"P: {100*p['mean']:.2f} (+/-{100*p['std']:.2f})  "
            f"R: {100*r['mean']:.2f} (+/-{100*r['std']:.2f})  "
            f"F1: {100*f1['mean']:.2f} (+/-{100*f1['std']:.2f})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--doc-order", type=Path, default=None)
    parser.add_argument("--save-json", type=Path, default=None, help="Optional path to save full results as JSON")
    args = parser.parse_args()

    gold_docs_list = [json.loads(line) for line in args.gold.read_text().splitlines() if line.strip()]
    pred_docs_list = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]

    gold_docs = {d["doc_key"]: d for d in gold_docs_list}
    pred_by_key = {d["doc_key"]: d for d in pred_docs_list}

    if args.doc_order:
        doc_order = [line.strip() for line in args.doc_order.read_text().splitlines() if line.strip()]
    else:
        doc_order = list(gold_docs.keys())

    pooled_evaluator = OfficialCoNLL2012CorefEvaluator()
    per_book_results = {}

    total_quotes = 0
    total_no_quote = 0
    total_no_speaker = 0

    for doc_key in doc_order:
        gold_doc = gold_docs.get(doc_key)
        if gold_doc is None or not gold_doc["quotes"]:
            continue
        pred_doc = pred_by_key.get(doc_key)
        if pred_doc is None:
            print(f"[warn] No predictions found for '{doc_key}', all its quotes will be treated as missed")

        gold_clusters, system_clusters, n_no_quote, n_no_speaker = build_doc_clusters(gold_doc, pred_doc)

        mention_to_gold = extract_mentions_to_clusters(gold_clusters)
        mention_to_system = extract_mentions_to_clusters(system_clusters)
        pooled_evaluator.update(system_clusters, gold_clusters, mention_to_system, mention_to_gold)

        per_book_results[doc_key] = score_doc(gold_clusters, system_clusters)

        total_quotes += len(gold_doc["quotes"])
        total_no_quote += n_no_quote
        total_no_speaker += n_no_speaker

    pooled_results = {
        metric: dict(zip(["precision", "recall", "f1"], pooled_evaluator.get_prf(metric))) for metric in METRICS
    }
    macro_summary = macro_mean_std(per_book_results)

    print_results(pooled_results, macro_summary, len(per_book_results), total_quotes, total_no_quote, total_no_speaker)

    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "pooled": pooled_results,
            "per_book_mean_std": macro_summary,
            "per_book": per_book_results,
            "n_books": len(per_book_results),
            "total_quotes": total_quotes,
            "total_no_quote": total_no_quote,
            "total_no_speaker": total_no_speaker,
        }
        args.save_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print()
        print(f"Full results (pooled + per-book + mean/std) saved to: {args.save_json}")


if __name__ == "__main__":
    main()