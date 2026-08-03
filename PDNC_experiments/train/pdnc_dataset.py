import glob, os
import hashlib
import random
from torch.utils.data import Dataset
import numpy as np 
import torch
from functools import lru_cache
import torch.distributed as dist
import pandas as pd
from torch_geometric.data.hetero_data import * 
from torch_geometric.data.data import Data
from typing import *
import pickle 
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.data.collate import collate
from torch_geometric.data.hetero_data import * 
from torch_geometric.data.data import Data
from typing import *
# from torch_geometric.utils import to_undirected 
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected
from pathlib import Path

# Directory containing this script (e.g. root/preprocess)
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root is one level up from preprocess/
ROOT_DIR = SCRIPT_DIR.parent


class MyData(Data):
    def __inc__(self, key, value, store=None, *args, **kwargs):

        if 'input_ids' in key :
            return torch.tensor(0)
        if 'spans' in key : 
            return torch.tensor(0)
        # try : 
        if 'type' in key :
            return torch.tensor(0)
        if 'index' in key:
            try :
                out = torch.tensor(store.size()).view(2, 1)    
            except Exception as e : 
                out = torch.tensor([[0], [0]])
            
            return out
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any,
                    store: Optional[NodeOrEdgeStorage] = None, *args,
                    **kwargs) -> Any:
        if is_sparse(value) and 'adj' in key:
            return (0, 1)
        elif isinstance(store, EdgeStorage) and 'index' in key:
            return -1
        elif key == 'edge_index_for_pred' : 
            return -1
        elif 'spans' in key : 
            return -1
        return 0


def _view_seed(item_seed: int, view_idx: int) -> int:
    return _seed_from(item_seed, view_idx)   # reuse the hashing helper from before
        
def _seed_from(*parts: int) -> int:
    """Deterministically derive a 32-bit seed from a tuple of ints."""
    h = hashlib.sha256(",".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:4], "big")


def _load_split(split_type, split) : 
    if split_type =='test' : 
        return pd.read_csv(f'{ROOT_DIR}/data/splits/split_{split}/test.csv', header=None)
    else : 
        return pd.read_csv(f'{ROOT_DIR}/data/splits/split_{split}/{split_type}.csv', header=None)


class PDNCDataset(Dataset) :
    
    def __init__(self, graph_path, split='train', split_num=0, filter=True, name_key='NEW_components.pkl'):
        super(PDNCDataset).__init__()

        self.split_info = _load_split(split, split_num)
        novels = self.split_info[0].unique()
        self.graphs = []
        self.filter = filter

        for n in novels : 
            D = f'{graph_path}/{n}/{name_key}'

            print(f'Using data at {D}')
            self.graphs.extend(self.convert_to_data(D ))

    
    def convert_to_data(self, path) : 
        with open(path, 'rb') as f : 
            Gs = pickle.load(f)
        gid = path.split('/')[-2]
        sub = self.split_info[self.split_info[0] == gid]
        qids = sub[1].unique()

        all_data = []
        for G in Gs : 
            if len(G.speakers) > 0 : 
                if all([
                    (G.speakers[0] != -1),
                    (len([q for q in G.quote_ids if q in qids]) >0),
                ]):
                    
                    is_pred = [i for i in range(len(G.speakers)) if (G.speakers[i] != -1) and G.quote_ids[i] in qids]

                    candidates_labels = []
                    final_is_preds = []
                    for i in is_pred : 
                        sid = G.speakers[i]
                        is_valid_candidates = torch.zeros_like(G.corefs)
                        is_valid_candidates[G.corefs==sid] = 1
                        if self.filter : 
                            if is_valid_candidates.sum() > 0 :
                                candidates_labels.append(is_valid_candidates)
                                final_is_preds.append(i)
                        else :
                            candidates_labels.append(is_valid_candidates)
                            final_is_preds.append(i)

                    if len(candidates_labels) > 0 :
                        G.is_valid_candidates = torch.stack(candidates_labels)
                        G.is_pred = final_is_preds

                        
                        G.quote_ids = [f'{gid}_{i[0]}' if i is not None else f'{gid}_none'for i in G.quote_ids]
                        all_data.append(G)
        return all_data

    def __len__(self) : 
        return len(self.graphs)
    
    def num_quotes(self) : 
        return sum([len(g.is_pred) for g in self.graphs])
    
    def num_unvalid(self) : 
        return sum([sum(g.is_valid_candidates.sum(-1) ==0 ) for g in self.graphs])
    
    def __getitem__(self, idx) : 
        return self.graphs[idx]
    


class IndividualPDNCDataset(Dataset) :
    
    def __init__(self, graph_path, split='train', split_num=0, filter=True, name_key='NEW_components.pkl'):
        super(IndividualPDNCDataset).__init__()

        self.split_info = _load_split(split, split_num)
        novels = self.split_info[0].unique()
        self.graphs = []
        self.filter = filter
        for n in novels : 
            D = f'{graph_path}/{n}/{name_key}'

            print(f'Using data at {D}')
            self.graphs.extend(self.convert_to_data(D))

    def convert_to_data(self, path) : 
        with open(path, 'rb') as f : 
            Gs = pickle.load(f)
        gid = path.split('/')[-2]
        sub = self.split_info[self.split_info[0] == gid]
        qids = sub[1].unique()
        all_data = {}
        
        for G in Gs : 
            if len(G.speakers) > 0 : 
                if all([
                    (G.speakers[0] != -1),
                    (len([q for q in G.quote_ids if q in qids]) >0),
                ]):

                    candidates_labels = []
                    final_is_preds = []
                    for cnt in range(len(G.speakers)) : 
                        if (G.speakers[cnt] != -1) and (G.quote_ids[cnt] in qids) : 
                            sid = G.speakers[cnt]
                            is_valid_candidates = torch.zeros_like(G.corefs)
                            is_valid_candidates[G.corefs==sid] = 1
                            if self.filter : 
                                if is_valid_candidates.sum() > 0 :
                                    g = G.clone()
                                    to_keep = torch.ones(g.num_nodes,).bool()
                                    c = [j for j in range(len(G.speakers)) if j!=cnt]
                                    to_remove = torch.where(g.node_types==1)[0][c]
                                    to_keep[to_remove] = False
                                    g.node_types = g.node_types[to_keep]
                                    g.spans = g.spans[:, to_keep]
                                    g.speakers = torch.LongTensor([g.speakers[cnt]])
                                    g.quote_ids = [f'{gid}_{g.quote_ids[cnt][0]}' if g.quote_ids[cnt] is not None else f'{gid}_none' ]
                                    g.is_valid_candidates = is_valid_candidates.unsqueeze(0)
                                    g.is_pred = torch.LongTensor([[0]])
                                    if g.quote_ids[0] not in all_data : 
                                        all_data[g.quote_ids[0]] = g
                            else :
                                
                                g = G.clone()
                                to_keep = torch.ones(g.num_nodes,).bool()
                                c = [j for j in range(len(G.speakers)) if j!=cnt]
                                to_remove = torch.where(g.node_types==1)[0][c] #[j for j in is_pred if j!=i]
                                to_keep[to_remove] = False
                                g.node_types = g.node_types[to_keep]
                                g.spans = g.spans[:, to_keep]
                                g.speakers = torch.LongTensor([g.speakers[cnt]])
                                g.quote_ids = [f'{gid}_{g.quote_ids[cnt][0]}' if g.quote_ids[cnt] is not None else f'{gid}_none' ]
                                g.is_valid_candidates = is_valid_candidates.unsqueeze(0)
                                g.is_pred = torch.LongTensor([[0]])
                                if g.quote_ids[0] not in all_data : 
                                    all_data[g.quote_ids[0]] = g
        return list(all_data.values())

    def __len__(self) : 
        return len(self.graphs)
    
    def num_quotes(self) : 
        return sum([len(g.is_pred) for g in self.graphs])
    
    def num_unvalid(self) : 
        return sum([sum(g.is_valid_candidates.sum(-1) ==0 ) for g in self.graphs])
    
    def __getitem__(self, idx) : 
        return self.graphs[idx]

    
    
from collections import defaultdict
def collate_fn(datalist, data_args) : 
    gs, input_ids, st, et, att_mask = [], [], [], [], []
    batch_q = []
    batch_m = []
    candidate_labels = []
    anaphora_data =[]# defaultdict(list)
    for c, G in enumerate(datalist) : 
        g = G.clone()
        g.edge_index = remove_self_loops(g.edge_index)[0]
        
        input_ids.append(g.pop('input_ids').squeeze())
        att_mask.append(torch.ones_like(input_ids[-1]))


        anaphora_data.append([torch.LongTensor(vv) for vv in build_coref_samples(g)])

        cst, cet = g.pop('spans').chunk(2)

        st.append(torch.stack((
            torch.full((cst.size(1), ), fill_value=c),
            cst[0])))
        et.append(torch.stack((
            torch.full((cet.size(1), ), fill_value=c),
            cet[0])))
        
        g.edge_index = to_undirected(g.edge_index, num_nodes=g.num_nodes)
        g.edge_index = add_self_loops(g.edge_index, num_nodes=g.num_nodes)[0]
        batch_q.append(torch.full((len(g.speakers),), c))
        batch_m.append(torch.full((len(g.is_valid_candidates[0]),), c))

        candidate_labels.append(g.pop('is_valid_candidates'))
        gs.append(g)
        
        # print(g)

    gs = collate(gs[0].__class__, gs)[0]
    # virtual nodes
    st = torch.cat(st, dim=1)
    et = torch.cat(et, dim=1)
    
    virtual_nodes = torch.where((gs.node_types == 1) & (st[1] == -1))[0]
    gs.is_virtual = torch.zeros_like(gs.node_types).bool()
    if len(virtual_nodes) > 0 : 
        gs.is_virtual[virtual_nodes] = 1
    # print

    gs.batch_q= torch.cat(batch_q)
    gs.batch_m = torch.cat(batch_m)
    return {
        'g' : gs,
        'st' : st,
        'et' : et,
        'candidate_labels' : candidate_labels,
        'input_ids' : pad_sequence(input_ids, batch_first=True,padding_value=data_args['pad_token_id']),
        'att_mask' : pad_sequence(att_mask, batch_first=True,padding_value=0),
        'anaphora_data' : anaphora_data
    }


import torch
import random


def build_coref_samples(
    data,
    window: int = 10,
    hard_negative_prob: float = 1.0,
    fallback_to_random: bool = True,
    seed=None,
):
    """
    Build (anchor, positive, negative) mention triplets for coreference training.

    - anchor: a mention node
    - positive: the closest preceding mention with the SAME character id (antecedent)
    - negative: a mention with a DIFFERENT character id, preferentially sampled
      close to the anchor in text order ("hard negative")

    Args:
        data: PyG Data object with node_types, spans, corefs.
        window: max distance (in mention-index units) around the anchor to look
            for a hard negative. E.g. window=10 -> negative is chosen among the
            10 mentions before and 10 mentions after the anchor.
        hard_negative_prob: probability of trying to sample a hard negative
            (in-window) vs a plain random negative. 1.0 = always try hard first.
        fallback_to_random: if no valid candidate exists inside the window,
            fall back to a uniformly random negative from the whole mention list
            (if False, the anchor is skipped instead).
        seed: optional random seed for reproducibility.

    Returns a dict of tensors indexed both in "mention space" and "global node
    space", plus their spans in the context, plus a boolean mask telling you
    which negatives ended up being "hard" (in-window) vs "random" (fallback).
    """
    if seed is not None:
        random.seed(seed)

    node_types = data.node_types
    mention_mask = node_types == 2
    mention_node_idx = mention_mask.nonzero(as_tuple=True)[0]  # global node ids, in text order

    mention_spans = data.spans[:, mention_mask]  # [2, num_mentions]
    corefs = data.corefs
    num_mentions = mention_node_idx.size(0)
    assert corefs.size(0) == num_mentions, (
        f"corefs ({corefs.size(0)}) and number of mention nodes ({num_mentions}) mismatch"
    )
    corefs_list = corefs.tolist()

    anchors, positives, negatives, is_hard = [], [], [], []
    
    for i in range(num_mentions):
        char_id = corefs_list[i]

        # --- positive: closest preceding mention with same char id ---
        antecedent = None
        for j in range(i - 1, -1, -1):
            if corefs_list[j] == char_id:
                antecedent = j
                break
        if antecedent is None:
            continue  # first mention of its character -> skip as anchor

        # --- negative: hard (in-window, different char id) first, else fallback ---
        negative = None
        hard = False

        if random.random() <= hard_negative_prob:
            lo = max(0, i - window)
            hi = min(num_mentions - 1, i + window)
            window_candidates = [
                k for k in range(lo, hi + 1)
                if k != i and corefs_list[k] != char_id
            ]
            if window_candidates:
                negative = random.choice(window_candidates)
                hard = True

        if negative is None:
            if not fallback_to_random:
                continue
            global_candidates = [k for k in range(num_mentions) if corefs_list[k] != char_id]
            if not global_candidates:
                continue
            negative = random.choice(global_candidates)
            hard = False

        anchors.append(i)
        positives.append(antecedent)
        negatives.append(negative)
        is_hard.append(hard)

    anchors = torch.tensor(anchors, dtype=torch.long)
    positives = torch.tensor(positives, dtype=torch.long)
    negatives = torch.tensor(negatives, dtype=torch.long)
    is_hard = torch.tensor(is_hard, dtype=torch.bool)

    return anchors, positives, negatives
