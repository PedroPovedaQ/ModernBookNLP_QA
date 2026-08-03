This is a modified fork of the [original BookNLP](https://github.com/booknlp/booknlp). Please refer to the original repository for further information.

## What's different

We replaced the vanilla Quotation Attribution model with our best scoring systems, for both *joint* and *direct* scoring, trained with $T=2000$ and $S=512$.
We observed largely improved quotation attribution results on LitBank with our models compared to vanilla BookNLP:

|                        | QA       | Coref | Time (s)        |
|------------------------|----------|-------|-----------------|
| %                      | CoNLL    | CoNLL | (s)             |
| BookNLP                | 68.1     | 71.3  | **5.2** +/- 0.4 |
| ModernBookNLP (Direct) | 72.6     | 71.4  | 10.8 +/- 4.8    |
| ModernBookNLP (Joint)  | **77.5** | 71.5  | 5.5 +/- 0.4     |


## How to use 

To use as a standalone library, start by installing it:

```bash
python setup.py install
```

We provide seamless integration voa a user-defined model parameter:

```python
from booknlp.english.english_booknlp import EnglishBookNLP

model_params={
		"pipeline":"entity,quote,coref", 
		"model":"big",
        "modern_qa": True, # To use our model with Joint scoring (better and faster)
		"direct_qa": False # Set to True if you want to try our Direct scoring system
	}

booknlp = EnglishBookNLP(model_params)

input_file="examples/158_emma.txt"

# Output directory to store resulting files in
output_directory="examples/158_emma/"

# File within this directory will be named ${book_id}.entities, ${book_id}.tokens, etc.
book_id="158_emma"

booknlp.process(input_file, output_directory, book_id)
```


## Paper Experiments

To reproduce our main External Validation experiments, please follow the [standard installation](../README.md#Installation). Then you can start by downloading and pre-processing [LitBank](https://github.com/dbamman/litbank):

```bash
sh prepate_data.sh
```
This will download LitBank data and build gold evaluation files.

Then, (Modern) BookNLP can be run with:

```bash
MODEL='joint' # can be one of (booknlp, joint, direct)
python run_booknlp.py $MODEL
```

This will process all necessary LitBank books and create the associated files (tokens, entities and quotes) within the `data/` folder.

Once ran, the evaluation can be done with:
```bash
MODEL='joint' # can be one of (booknlp, joint, direct)
sh eval.sh $MODEL
```

Note that we use [BookCoref](https://github.com/sapienzanlp/bookcoref) official evaluation scripts to evaluate quotation attribution an coreference resolution performance.
