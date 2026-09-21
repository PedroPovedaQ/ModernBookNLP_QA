"""Translate upstream files into exact, explicitly indexed source spans."""

import csv
import hashlib
import os
import sys
import tempfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from service.config import MODEL_VERSION, Settings

WEIGHTS = {
    "entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model": "c67654ac10b3b49371544ccaf8ed76d224f0d274d23f7f7ea09e74f173ddbe06",
    "coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model": "9b301a05eff4573d34ad2672f310110b716801759470044b07d1979a0dbb50d8",
    "ModernBERT_T2000.safetensors": "a32c85b99e6491f559e0c46c6257b18be0a39a06c15c281389ca0934c229cfb2",
}


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE))


def normalize_output(text: str, directory: Path, name: str = "document") -> dict:
    tokens = read_rows(directory / f"{name}.tokens")
    utf16 = [0]
    for char in text:
        utf16.append(utf16[-1] + (2 if ord(char) > 0xFFFF else 1))
    for token in tokens:
        start, end = int(token["byte_onset"]), int(token["byte_offset"])
        if not 0 <= start < end <= len(text) or text[start:end] != token["word"]:
            raise ValueError("Model token offsets do not match source")

    def span(a, b):
        start_token, end_token = int(a), int(b)
        if not 0 <= start_token <= end_token < len(tokens):
            raise ValueError("Invalid token span")
        start = int(tokens[start_token]["byte_onset"])
        end = int(tokens[end_token]["byte_offset"])
        return {
            "start": start,
            "end": end,
            "start_utf16": utf16[start],
            "end_utf16": utf16[end],
        }

    names = defaultdict(Counter)
    aliases = defaultdict(set)
    for entity in read_rows(directory / f"{name}.entities"):
        if entity["cat"] == "PER" and entity["COREF"] not in ("None", "-1"):
            entity_span = span(entity["start_token"], entity["end_token"])
            label = text[entity_span["start"] : entity_span["end"]]
            char_id = "character-" + entity["COREF"]
            if entity["prop"] == "PROP":
                names[char_id][label] += 1
                aliases[char_id].add(label)

    quotes, character_ids = [], set()
    for row in read_rows(directory / f"{name}.quotes"):
        quote_span = span(row["quote_start"], row["quote_end"])
        mention = (
            None
            if row["mention_start"] == "None"
            else span(row["mention_start"], row["mention_end"])
        )
        char_id = (
            "character-" + row["char_id"]
            if mention and row["char_id"] not in ("None", "-1")
            else None
        )
        if char_id:
            character_ids.add(char_id)
        quotes.append(
            {
                **quote_span,
                "text": text[quote_span["start"] : quote_span["end"]],
                "character_id": char_id,
                "speaker_mention": mention,
            }
        )
    characters = [
        {
            "id": char_id,
            "display_name": names[char_id].most_common(1)[0][0]
            if names[char_id]
            else char_id,
            "aliases": sorted(aliases[char_id]),
        }
        for char_id in sorted(character_ids)
    ]
    return {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "offsets": "half-open Unicode code points; *_utf16 are UTF-16 code units",
        "character_id_scope": "this analysis only",
        "characters": characters,
        "quotes": quotes,
    }


def verified_weights(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in WEIGHTS.items():
        path = directory / name
        if not path.exists():
            url = (
                "https://huggingface.co/gasmichel/ModernBookNLP/resolve/main/" + name
                if name.endswith(".safetensors")
                else "https://people.ischool.berkeley.edu/~dbamman/booknlp_models/"
                + name
            )
            temporary = path.with_suffix(".download")
            try:
                with (
                    urllib.request.urlopen(url, timeout=60) as source,
                    temporary.open("wb") as target,
                ):
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
                with temporary.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise ValueError("Model download checksum mismatch")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError("Model cache checksum mismatch")


class Analyzer:
    def __init__(self, settings: Settings):
        self.settings = settings
        import torch

        torch.set_num_threads(settings.threads)
        path = Path(
            os.environ.get("BOOKNLP_MODEL_DIR", str(Path.home() / "booknlp_models"))
        )
        verified_weights(path)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ModernBookNLP"))
        from booknlp.english import modern_qa

        modern_qa.MODEL_PATH = str(path)
        from booknlp.english.english_booknlp import EnglishBookNLP

        self.model = EnglishBookNLP(
            {
                "pipeline": "entity,quote,coref",
                "model": "big",
                "modern_qa": True,
                "model_path": str(path),
            }
        )
        self.model.quote_attrib.batch_size = settings.batch_size

    def analyze(self, text: str) -> dict:
        # A job owns all outputs. Upstream debug pickle writes are removed in this fork.
        with tempfile.TemporaryDirectory(
            prefix="analysis-", dir=self.settings.data_dir
        ) as temp:
            directory = Path(temp)
            source = directory / "source.txt"
            source.write_text(text, encoding="utf-8", newline="")
            self.model.process(str(source), str(directory), "document")
            result = normalize_output(text, directory)
            result["model_version"] = self.settings.analysis_version
            return result
