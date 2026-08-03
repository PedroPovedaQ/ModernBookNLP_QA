This repository contains the code to reproduce our experiments from the paper ***Fast and Accurate Quotation Attribution in Literary Texts***.

It is divided in two components:

- [`ModernBookNLP`](ModernBookNLP/README.md) is our modified fork of BookNLP, which allows to replace the vanilla quotation attribution model with our models (both *direct* and *joint* trained with $T=2000$ and $S=512$)
- [`PDNC_experiments`](PDNC_experiments/README.md) contains all necesarry data and scripts to reproduce our training and analysis on PDNC.

# Installation

Please start by creating a fresh python environment and install the dependencies:

```bash
python3 -m venv /tmp/fast_qa
. /tmp/fast_qa/bin/activate
pip3 install -U pip
pip3 install -r requirements.txt
```

And install spacy:

```bash
python -m spacy download en_core_web_sm
```

Then, you can browse subfolders to reproduce our experiments.