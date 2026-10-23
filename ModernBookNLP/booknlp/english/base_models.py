"""Construct Hugging Face base encoders, optionally without reading their weights.

Every BookNLP component overwrites its encoder with fine-tuned weights right after
construction (ModernQA strictly; the BERT taggers cover every encoder key). When
``BOOKNLP_SKIP_BASE_WEIGHTS=1`` the base checkpoint is therefore skipped and only its
config is read, which removes ~2.3 GB of disk reads from a cold start without changing
the loaded parameters. Unset, this is exactly ``from_pretrained``.
"""

import os

from transformers import AutoConfig


def load_base(model_cls, name, **kwargs):
    if os.environ.get("BOOKNLP_SKIP_BASE_WEIGHTS") != "1":
        return model_cls.from_pretrained(name, **kwargs)
    config = AutoConfig.from_pretrained(name)
    dtype = kwargs.pop("dtype", None)
    if dtype is not None:
        kwargs["torch_dtype"] = dtype
    return model_cls._from_config(config, **kwargs).eval()
