"""Build-time only: bake every model artifact into the image so boot never downloads.

Fetches the checksum-pinned BookNLP weights plus configs and tokenizers from the exact
Hugging Face base-model revisions the runtime resolves (base weights are skipped: see
``booknlp/english/base_models.py``), then points each repo's ``refs/main`` at the pinned
commit so ``HF_HUB_OFFLINE=1`` lookups by name resolve to the baked snapshot.
"""

import os
from pathlib import Path

from huggingface_hub import hf_hub_download

from service.model import verified_weights

# Revisions match the snapshots the 2026-10-15 pilot resolved at runtime.
HF_FILES = {
    "answerdotai/ModernBERT-large": (
        "45bb4654a4d5aaff24dd11d4781fa46d39bf8c13",
        [
            "config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    ),
    "google/bert_uncased_L-12_H-768_A-12": (
        "cc478efa3482fefe275eb2733363db9713d499ef",
        ["config.json", "vocab.txt"],
    ),
    "google/bert_uncased_L-6_H-768_A-12": (
        "c132ecc85d3d73b460b741cc50aa9ed18446c335",
        ["config.json", "vocab.txt"],
    ),
}


def main():
    verified_weights(Path(os.environ["BOOKNLP_MODEL_DIR"]))
    hub = Path(os.environ["HF_HOME"]) / "hub"
    for repo, (revision, files) in HF_FILES.items():
        for name in files:
            hf_hub_download(repo, name, revision=revision)
        refs = hub / ("models--" + repo.replace("/", "--")) / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(revision)


if __name__ == "__main__":
    main()
