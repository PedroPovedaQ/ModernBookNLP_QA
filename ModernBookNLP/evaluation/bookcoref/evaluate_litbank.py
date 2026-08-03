"""
Evaluates coreference predictions against LitBank gold annotations, using
the same CoNLL-2012 metric implementation (MUC, B-cubed, CEAFe, average)
as BookCoref's evaluate.py.

Run this from inside your cloned bookcoref/ repo (or copy src/metrics.py
alongside this script), since it reuses bookcoref's
OfficialCoNLL2012CorefEvaluator.

Usage:
    python evaluate_litbank.py \
        --gold data/litbank_input/gold.jsonl \
        --predictions predictions/booknlp_litbank.jsonl
"""

import argparse
import json
from pathlib import Path

from rich import print

from src.metrics import OfficialCoNLL2012CorefEvaluator


def print_results(results):
    print("[bold green]Evaluation Results:[/bold green]")
    for metric, values in results.items():
        print(f"[bold blue]{metric}[/bold blue]:")
        for k, v in values.items():
            print(f"  {k}: {(100 * v):.2f}")


def extract_mentions_to_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[mention] = gc
    return mention_to_gold


def evaluate(gold_elements_list, predicted_elements_list):
    gold = []
    predictions = []
    evaluator = OfficialCoNLL2012CorefEvaluator()

    for gold_doc_elem, pred_doc_elem in zip(gold_elements_list, predicted_elements_list):
        gold_doc_clusters = []
        for cluster in gold_doc_elem["clusters"]:
            elem = tuple([(span[0], span[1]) for span in cluster])
            gold_doc_clusters.append(elem)
        gold.append(gold_doc_clusters)

        pred_doc_clusters = []
        for cluster in pred_doc_elem["clusters"]:
            elem = tuple([(span[0], span[1]) for span in cluster])
            pred_doc_clusters.append(elem)
        predictions.append(pred_doc_clusters)

    mention_to_gold = [extract_mentions_to_clusters([tuple(g) for g in gg]) for gg in gold]
    mention_to_predicted = [extract_mentions_to_clusters([tuple(p) for p in pp]) for pp in predictions]

    results = {}
    for p, g, m2p, m2g in zip(predictions, gold, mention_to_predicted, mention_to_gold):
        evaluator.update(p, g, m2p, m2g)
    for metric in ["muc", "b_cubed", "ceafe", "conll2012"]:
        results[metric] = dict(zip(["precision", "recall", "f1"], evaluator.get_prf(metric)))

    print_results(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Path to gold.jsonl from prepare_litbank_input.py")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to predictions JSONL")
    parser.add_argument(
        "--doc-order",
        type=Path,
        default=None,
        help="Optional file with one doc_key per line to reorder/subselect predictions to match gold order "
        "(recommended: use the doc_order.txt produced by prepare_litbank_input.py)",
    )
    args = parser.parse_args()

    gold = [json.loads(line) for line in args.gold.read_text().splitlines() if line.strip()]
    predicted = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]

    if args.doc_order:
        order = [line.strip() for line in args.doc_order.read_text().splitlines() if line.strip()]
        gold_by_key = {d["doc_key"]: d for d in gold}
        pred_by_key = {d["doc_key"]: d for d in predicted}
        gold = [gold_by_key[k] for k in order if k in gold_by_key]
        predicted = [pred_by_key[k] for k in order if k in pred_by_key]

    if len(gold) != len(predicted):
        print(
            f"[bold red]Warning:[/bold red] gold has {len(gold)} docs but predictions has {len(predicted)}. "
            "evaluate() zips them in order, so a mismatch here means you are silently comparing the wrong "
            "book pairs. Use --doc-order to align them explicitly."
        )

    evaluate(gold, predicted)


if __name__ == "__main__":
    main()