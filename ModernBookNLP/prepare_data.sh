#/usr/bin/sh
cd data/
git clone https://github.com/dbamman/litbank.git

python prepare_litbank_input.py \
  --conll-dir litbank/coref/conll/ \
  --output-dir litbank_coref_input/

python prepare_litbank_speaker_input.py \
  --quotations-dir litbank/quotations/tsv/ \
  --coref-dir litbank/coref/conll/ \
  --output-dir litbank_speaker_input/