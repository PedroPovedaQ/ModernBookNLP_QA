#/usr/bin/sh
set -e

MODEL=$1

if [ -z "$MODEL" ]; then
  echo "Usage: $0 {booknlp|joint|direct}"
  exit 1
fi


if [ "$MODEL" = "booknlp" ]; then
    OUTPUT_DIR="data/booknlp_litbank_out/"
    OUTPUT_COREFS="evaluation/predictions/corefs/booknlp.jsonl"
    OUTPUT_SPK="evaluation/predictions/speakers/booknlp.jsonl"
    OUTPUT_SPK_GOLD="evaluation/predictions/speakers_gold_coref/booknlp.jsonl"
elif [ "$MODEL" = "joint" ]; then
    OUTPUT_DIR="data/joint_litbank_out/"
    OUTPUT_COREFS="evaluation/predictions/corefs/joint.jsonl"
    OUTPUT_SPK="evaluation/predictions/speakers/joint.jsonl"
    OUTPUT_SPK_GOLD="evaluation/predictions/speakers_gold_coref/joint.jsonl"
elif [ "$MODEL" = "direct" ]; then      
    OUTPUT_DIR="data/direct_litbank_out/"
    OUTPUT_COREFS="evaluation/predictions/corefs/direct.jsonl"
    OUTPUT_SPK="evaluation/predictions/speakers/direct.jsonl"
    OUTPUT_SPK_GOLD="evaluation/predictions/speakers_gold_coref/direct.jsonl"
else
    echo "Invalid model: $MODEL"
    exit 1
fi



### COREFS ###
python evaluation/booknlp_to_corefs.py \
  --booknlp-output-dir $OUTPUT_DIR \
  --offsets-dir data/litbank_coref_input/ \
  --doc-order data/litbank_coref_input/doc_order.txt \
  --output $OUTPUT_COREFS\
  --all-categories

### QUOTES ###
python evaluation/booknlp_quotes_to_predictions.py \
    --booknlp-output-dir $OUTPUT_DIR \
    --offsets-dir data/litbank_speaker_input/ \
    --doc-order data/litbank_speaker_input/doc_order.txt \
    --output $OUTPUT_SPK

### QUOTES GOLD COREFS ###
python evaluation/booknlp_quotes_to_predictions_gold_coref.py \
  --booknlp-output-dir $OUTPUT_DIR \
  --offsets-dir data/litbank_speaker_input/ \
  --doc-order data/litbank_speaker_input/doc_order.txt \
  --output $OUTPUT_SPK_GOLD \
  --gold data/litbank_coref_input/gold.jsonl

### EVALUATIONS 


### COREFS ###
echo "\n\n\n---------------"
echo "Coreference Evaluation"
echo "---------------\n\n\n"


python evaluation/bookcoref/evaluate_litbank.py \
  --gold data/litbank_coref_input/gold.jsonl \
  --predictions $OUTPUT_COREFS \
  --doc-order data/litbank_coref_input/doc_order.txt

echo "\n\n\n---------------"
echo "Done"
echo "---------------\n\n\n"

### QUOTES ###
echo "\n\n\n---------------"
echo "Quotation Attribution Evaluation"
echo "---------------\n\n\n"

python evaluation/bookcoref/evaluate_speaker_attribution.py \
  --gold data/litbank_speaker_input/gold_speaker_attribution.jsonl \
  --predictions $OUTPUT_SPK \
  --doc-order data/litbank_speaker_input/doc_order.txt
echo "\n\n\n---------------"
echo "Done"
echo "---------------\n\n\n"


### QUOTES ###
echo "\n\n\n---------------"
echo "Quotation Attribution Evaluation (Gold Corefs)"
echo "---------------\n\n\n"

python evaluation/bookcoref/evaluate_speaker_attribution.py \
  --gold data/litbank_speaker_input/gold_speaker_attribution.jsonl \
  --predictions $OUTPUT_SPK_GOLD \
  --doc-order data/litbank_speaker_input/doc_order.txt
echo "\n\n\n---------------"
echo "Done"
echo "---------------\n\n\n"
