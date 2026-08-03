"""
profile_pdnc.py

Profile inference running time and peak GPU memory of one or several trained
PDNC models, PER NOVEL, across multiple repeated runs.

It reuses the exact data-conversion logic from `pdnc_dataset.py` (PDNCDataset /
IndividualPDNCDataset) but builds one dataset per novel instead of merging all
novels of a split together, so timing/memory can be reported novel-by-novel.

------------------------------------------------------------------------------
Example usage
------------------------------------------------------------------------------
python profile_pdnc.py \
    --model_paths results/pdnc/modelA.safetensors results/pdnc/modelB.safetensors \
    --model_names modelA modelB \
    --config pdnc_config.yaml \
    --graph_path ../data/pdnc_source \
    --split test --split_num 0 \
    --batch_size 16 \
    --num_runs 5 --warmup_runs 1 \
    --out_prefix profiling_pdnc

This produces:
    profiling_pdnc_raw.csv          (one row per model x novel x run)
    profiling_pdnc_summary.csv      (mean/std per model x novel, over runs)
------------------------------------------------------------------------------
"""

import os
import time
import argparse
from functools import partial

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from pdnc_dataset import PDNCDataset, IndividualPDNCDataset, collate_fn, _load_split
from model import Baseline

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

from pathlib import Path

# Directory containing this script (e.g. root/preprocess)
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root is one level up from preprocess/
ROOT_DIR = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Filename pattern used by each dataset class to locate a novel's graph file
# (mirrors what's hard-coded in pdnc_dataset.py). Adjust here if your data
# layout differs.
# PKL_FILENAME = {
#     PDNCDataset: 'NEW_components_N512.pkl',
#     IndividualPDNCDataset: 'Restricted_components_N1000_S256_K200.pkl',
# }


class NovelDataset(Dataset):
    """Wraps the graphs of a single novel so it can be timed independently."""

    def __init__(self, graphs):
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


def build_per_novel_datasets(dataset_cls, filename, graph_path, split, split_num, filter_flag):
    """
    Re-uses dataset_cls.convert_to_data (defined in pdnc_dataset.py) but calls
    it once per novel, so we get a dict {novel_id: NovelDataset} instead of one
    dataset merging every novel of the split.
    """
    # split_info = _load_split(split, split_num)
    novels = [f for f in os.listdir(graph_path) if os.path.isdir(os.path.join(graph_path,f))]

    # Build a "shell" instance without running __init__ (which would eagerly
    # load + merge every novel). We only need convert_to_data(), which reads
    # self.split_info and self.filter.
    shell = dataset_cls.__new__(dataset_cls)
    shell.split_info = pd.read_csv(f'{ROOT_DIR}/data/splits/all.csv',header=None)
    shell.filter = False

    # filename = PKL_FILENAME[dataset_cls]

    per_novel = {}
    for novel in novels:
        path = f'{graph_path}/{novel}/{filename}'
        if not os.path.exists(path):
            print(f'[WARN] Missing file for novel {novel}: {path} -- skipping')
            continue
        data = shell.convert_to_data(path)
        if len(data) == 0:
            print(f'[WARN] No usable graphs for novel {novel} -- skipping')
            continue
        per_novel[novel] = NovelDataset(data)

    return per_novel


def to_device(data: dict, device: torch.device) -> dict:
    """Move all tensors in a dict (possibly nested / in lists) to `device`."""
    out = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, (list, tuple)):
            out[k] = type(v)(
                item.to(device) if isinstance(item, torch.Tensor) else item
                for item in v
            )
        elif isinstance(v, dict):
            out[k] = to_device(v, device)
        else:
            out[k] = v
    return out


def load_weights(model, path, strict=True):
    if path.endswith('.safetensors'):
        if load_safetensors is None:
            raise RuntimeError('safetensors is not installed but a .safetensors path was given')
        state_dict = load_safetensors(path)
    else:
        state_dict = torch.load(path, map_location='cpu')
        # in case the checkpoint was saved as {'model': state_dict, ...}
        if isinstance(state_dict, dict) and 'state_dict' in state_dict and not any(
            k.startswith(('bert.', 'proj.', 'gnn_backbone.')) for k in state_dict
        ):
            state_dict = state_dict['state_dict']
    result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(f'  Missing keys: {result.missing_keys}')
        print(f'  Unexpected keys: {result.unexpected_keys}')
    return model


# ---------------------------------------------------------------------------
# Profiling core
# ---------------------------------------------------------------------------

def profile_model_on_novel(model, loader, device, use_bf16_autocast=True):
    """
    Runs a full pass over `loader` (one novel) and returns:
        elapsed_seconds, peak_alloc_MB, peak_reserved_MB, n_graphs, n_batches
    Assumes CUDA. Resets peak-memory stats before timing.
    """
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    n_graphs, n_batches = 0, 0
    t0 = time.perf_counter()
    with torch.no_grad():
        ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if use_bf16_autocast else torch.no_grad()
        with ctx:
            for batch in loader:
                batch.pop('candidate_labels', None)
                batch = to_device(batch, device)
                _ = model(**batch)
                n_batches += 1
                n_graphs += batch['g'].batch_q.max().item() + 1
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    return (t1 - t0), peak_alloc, peak_reserved, n_graphs, n_batches




MODELS = {
    'baseline_N2000_S512': {
        'name_key': 'Components_N2000_S512_K200_ModernBERT_Large.pkl',
        'dataset_cls': PDNCDataset,
    },
    'baseline_N1000_S256': {
        'name_key': 'Components_N1000_S256_K200_ModernBERT_Large.pkl',
        'dataset_cls': PDNCDataset,

    },
    'baseline_N500_S256': {
        'name_key': 'Components_N500_S256_K200_ModernBERT_Large.pkl',
        'dataset_cls': PDNCDataset,

    },
    'individual_N2000_S512': {
        'name_key': 'Components_N2000_S512_K200_ModernBERT_Large.pkl',
        'dataset_cls': IndividualPDNCDataset,
    },
    'individual_N1000_S256': {
        'name_key': 'Components_N1000_S256_K200_ModernBERT_Large.pkl',
        'dataset_cls': IndividualPDNCDataset,

    },
    'individual_N500_S256': {
        'name_key': 'Components_N500_S256_K200_ModernBERT_Large.pkl',
        'dataset_cls': IndividualPDNCDataset,
    },
    'baseline_N1000_S256_restricted' :{
        'name_key': 'Restricted_components_N1000_S256_K200_ModernBERT_Large.pkl',
        'dataset_cls': PDNCDataset,
    },
    'individual_N1000_restricted': {
        'name_key': 'Restricted_components_N1000_S256_K200_ModernBERT_Large.pkl',
        'dataset_cls': IndividualPDNCDataset,
    },
    'longformer_baseline_N2000_S512': {
        'name_key': 'Components_N2000_S512_K200_Longformer.pkl',
        'dataset_cls': PDNCDataset,
    },
    'longformer_direct_N2000_S512': {
        'name_key': 'Components_N2000_S512_K200_Longformer.pkl',
        'dataset_cls': IndividualPDNCDataset,
}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--split_num', default=0, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_runs', default=5, type=int, help='Number of *measured* runs per novel')
    parser.add_argument('--warmup_runs', default=1, type=int, help='Number of extra untimed warmup runs per model')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out_prefix', default='profiling_pdnc')
    parser.add_argument('--strict_load', action='store_true', default=True)
    parser.add_argument('--no_strict_load', dest='strict_load', action='store_false')
    parser.add_argument('--filter', action='store_true', default=False,
                         help='Match train_pdnc.py: val/test use filter=False, train uses filter=True')
    args = parser.parse_args()

    device = torch.device(args.device)
    config = yaml.safe_load(open(f'{SCRIPT_DIR}/pdnc_config.yaml'))

    records = []
    os.makedirs(f'{SCRIPT_DIR}/profiling/', exist_ok=True)
    for model_name in MODELS:
        print(f'\n=== Model: {model_name} ===')
        per_novel_data = build_per_novel_datasets(
            MODELS[model_name]['dataset_cls'], MODELS[model_name]['name_key'],
            f'{ROOT_DIR}/data/pdnc_source', args.split, args.split_num, filter_flag=args.filter
        )
        print(f'Found {len(per_novel_data)} novels: {list(per_novel_data.keys())}')

        if 'longformer' in model_name: 
            pad_token_id = 1
        else : 
            pad_token_id= 50283
        collate_fn_ = partial(collate_fn, data_args={'pad_token_id' :pad_token_id })

        per_novel_loaders = {
            novel: DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn_, num_workers=0)
            for novel, ds in per_novel_data.items()
        }

        if 'longformer' in model_name : 
            config['model']['model_id'] = 'allenai/longformer-base-4096'
        
        model = Baseline(config['model'])
        print(model)
        model.to(device)
        model.eval()

        for run in range(args.warmup_runs + args.num_runs):
            is_warmup = run < args.warmup_runs
            for novel, loader in per_novel_loaders.items():
                elapsed, peak_alloc, peak_reserved, n_graphs, n_batches = profile_model_on_novel(
                    model, loader, device
                )
                if is_warmup:
                    continue  # discard warmup timings entirely
                run_idx = run - args.warmup_runs
                records.append({
                    'model': model_name,
                    'novel': novel,
                    'run': run_idx,
                    'n_graphs': n_graphs,
                    'n_batches': n_batches,
                    'time_s': elapsed,
                    'time_s_per_graph': elapsed / max(n_graphs, 1),
                    'peak_alloc_MB': peak_alloc,
                    'peak_reserved_MB': peak_reserved,
                })
                print(f'  [{model_name}] novel={novel} run={run_idx} '
                      f'time={elapsed:.3f}s ({n_graphs} graphs) '
                      f'peak_alloc={peak_alloc:.1f}MB peak_reserved={peak_reserved:.1f}MB')

        # free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

        raw_df = pd.DataFrame.from_records(records)
        raw_df.to_csv(f'{SCRIPT_DIR}/profiling/{model_name}_raw.csv')

    raw_df = pd.DataFrame.from_records(records)
    raw_path = f'{SCRIPT_DIR}/profiling/{args.out_prefix}_raw.csv'
    raw_df.to_csv(raw_path, index=False)
    print(f'\nSaved raw per-run results to {raw_path}')

    summary_df = (
        raw_df
        .groupby(['model', 'novel'])
        .agg(
            n_graphs=('n_graphs', 'first'),
            time_s_mean=('time_s', 'mean'),
            time_s_std=('time_s', 'std'),
            time_s_per_graph_mean=('time_s_per_graph', 'mean'),
            time_s_per_graph_std=('time_s_per_graph', 'std'),
            peak_alloc_MB_mean=('peak_alloc_MB', 'mean'),
            peak_alloc_MB_max=('peak_alloc_MB', 'max'),
            peak_reserved_MB_mean=('peak_reserved_MB', 'mean'),
            peak_reserved_MB_max=('peak_reserved_MB', 'max'),
        )
        .reset_index()
    )
    summary_path = f'{SCRIPT_DIR}/profiling/{args.out_prefix}_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f'Saved per-novel summary (mean/std over {args.num_runs} runs) to {summary_path}')

    # Also a compact per-model summary (averaged across novels)
    per_model_df = (
        raw_df
        .groupby('model')
        .agg(
            total_time_s_mean=('time_s', lambda x: x.groupby(raw_df.loc[x.index, 'run']).sum().mean()),
            peak_alloc_MB_max=('peak_alloc_MB', 'max'),
            peak_reserved_MB_max=('peak_reserved_MB', 'max'),
        )
        .reset_index()
    )
    per_model_path = f'{SCRIPT_DIR}/profiling/{args.out_prefix}_per_model.csv'
    per_model_df.to_csv(per_model_path, index=False)
    print(f'Saved per-model summary to {per_model_path}')


if __name__ == '__main__':
    main()