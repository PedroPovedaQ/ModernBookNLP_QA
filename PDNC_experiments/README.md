This folder contains all necessary data and scripts to reproduce our PDNC experiments.
It contains PDNC books preprocessed by BookNLP within the [`data/`](data/pdnc_source/) folder.

## Installation

Please refer to [the installation setup](../README.md#installation).

## Build Data Inputs

To build data inputs from BookNLP preprocessed files, you can run the following:

```bash
python preprocess/preprocess.py --T 2000 --S 512 
```

This will build all inputs with a context size of $T=2000$ and stride $S=512$. You can modify these parameters to match our experiments with varying $T$ and $S$.
By default, it will create two files in each PDNC book directories: `Graph_*.pkl` and `Components_*.pkl` files.
The `Graph` file contains a graph-based representation of a tokenized book, while the `Components` file contains a list of contextual windows, as defined in Section 3 of the paper.

By default, it builds data for `coreference-based` models. To build data which only use aliases as candidates, you can use:

```bash
python preprocess/preprocess.py --T 2000 --S 512 --restrict
```
This will create a `Restricted_components*.pkl` file, which contains contextual windows with only proper-named mentions as candidate mentions.

To build with `Longformer`, use 

```bash
python preprocess/preprocess.py --T 2000 --S 512 --model_id allenai/longformer-base-4096
```

## Training with PDNC

To train models and evaluate with PNC 5-fold splits, you can use:

```bash
accelerate launch train/train_pdnc.py \
    --batch_size 16 \
    --weight_decay 0 \
    --lr 7e-5 \
    --drop_p 0 \
    --save_path results/pdnc/baseline_N2000_S512/ \
    --name_key Components_N2000_S512_K200_ModernBERT_Large.pkl
```

where the `name_key` parameter varies depending on the configuration (varying $T$ and $S$, restricted to aliases or using Longformer).
This will launch the training script and will use all available GPUs by default to run training in parallel. Note that if you have $N$ GPUs available, then the model will use an effective batch size of $N \times 16$ with the script above.

## Representation Similarity Analysis

The representation similarity analysis is ran in [the following notebook](train/representation_similarity.ipynb), and uses analysis functions from [this module](train/quote_speaker_analysis.py).

## Model Profiling

We estimate GPU runtimes and peak inference with the following script:

```bash
python train/profile_models.py
```

By default, it uses one warmup run per model per novel, and then proceed to processing already preprocessed data inputs 5 times.
Note that by default this script expects all tested configuration to have been pre-processed with the preprocessing scripts.


## Availability of Model Checkpoints

The current model checkpoints are hosted on Huggingface. For anonymity reasons, we tried to remove all explicit references to the huggingface repository.
They can be downloaded easily with 

```bash
cd train
python download_ckpts.py
```

