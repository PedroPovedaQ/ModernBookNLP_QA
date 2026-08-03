#!/usr/bin/env python3
"""
graph_quotation_attribution.py
================================

Upper-level wrapper around the trained graph-based `Baseline` speaker
attribution model, meant to play the exact same role `QuotationAttribution`
(booknlp/english/bert_qa.py) plays in stock BookNLP, i.e. be assignable to
`self.quote_attrib` and expose a `.tag(...)`-style call, except this one
operates on a *list* of per-context `torch_geometric.data.Data` graphs (one
per CTX_* context window), built on top of the networkx graph produced by
`booknlp_graph_adapter.build_inference_graph`, rather than on raw
`(quotes, entities, tokens)`.

Contract expected from every input `Data` graph
-------------------------------------------------
Everything your training-time graphs already carry (since we reuse
`collate_fn` unmodified): `input_ids`, `spans`, `speakers`,
`is_valid_candidates`, `is_pred`, `node_types`, `edge_index`, `corefs`, ...

PLUS the two id fields you already track on every graph:

  - `quote_ids`:   one entry per quote-type node in that graph, each entry
    itself a 1-element list holding a string `"Q_<i>"`, where `<i>` is the
    node's position in the book-level `quotes` list (same order/length as
    `is_pred` for that graph, since both are indexed over quote-type
    nodes).
  - `mention_ids`: one entry per mention-type node in that graph, each
    entry a 1-element list holding a string `"M_<j>"`, where `<j>` is the
    node's position in the book-level `entities` list (same order/length
    as the mention-node slice `batch_m == bs` selects for that graph).

Because these are plain Python lists (not node-level tensors sized to
`num_nodes`), PyG's generic `collate()` keeps them as one list-per-graph,
exactly like it already does for `is_pred`, so `g.quote_ids[bs]` /
`g.mention_ids[bs]` give graph `bs`'s own id list, which is what `tag()`
relies on below.

Why this is needed: `Baseline.forward` returns, per context-graph in the
batch, a `[n_pred_quotes, n_mentions]` score matrix, but nothing in
`forward`/`collate_fn` tracks which *original* quote/entity each row/column
corresponds to, since that mapping only exists in the graph you build. Note
that a given quote can be duplicated across several overlapping context
windows (per the sliding-window design), so scores for the same original
quote may show up in several different `Data` graphs; `tag()` resolves
this by keeping, per original quote, only the highest-confidence
prediction across all of its duplicates.
"""

from functools import partial
from typing import Dict, List, Optional

from huggingface_hub import hf_hub_download
import torch
from torch.utils.data import DataLoader

# Import your own modules here:
# from baseline_model import Baseline
# from your_training_module import collate_fn
from booknlp.english.inp_to_graph import build_inference_graph, WordAligner, to_direct_input
import torch.nn as nn 
from transformers import ModernBertModel
import torch_geometric.nn as conv_nn
import torch.nn.functional as F 
import torch 
from safetensors.torch import load_file
from transformers.utils import is_flash_attn_2_available
from torch_geometric.data.collate import collate
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm

if is_flash_attn_2_available() : 
    ATTN_IMP = 'flash_attention_2'
else :
    ATTN_IMP = 'sdpa'

if torch.cuda.is_bf16_supported(including_emulation=False) :
    DTYPE = torch.bfloat16
else : 
    DTYPE = torch.float32 


########### DO NOT READ IF A REVIEWER ############
REPO_ID = 'gasmichel/ModernBookNLP' #####################
########### DO NOT READ IF A REVIEWER ############


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    token_embeddings: [B, S, H] - transformer output
    attention_mask:   [B, S]    - 1 for real tokens, 0 for padding
    Returns: [B, H] - mean-pooled sentence embeddings
    """
    # 1. Expand mask to match embedding dimensions: [B, S] -> [B, S, H]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # 2. Zero out embeddings at padding positions, then sum over sequence dim
    sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)  # [B, H]

    # 3. Count real (non-padded) tokens per sequence, avoid division by zero
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # [B, H]

    # 4. Divide sum by count -> mean over valid tokens only
    mean_pooled = sum_embeddings / sum_mask  # [B, H]

    return mean_pooled
    
class ModernQA(nn.Module) : 
    def __init__(self, config) : 
        super().__init__()
        self.config = config

        
        self.bert = ModernBertModel.from_pretrained(
            config['model_id'],
            attn_implementation=ATTN_IMP,
            dtype=DTYPE
)
        num_bert_layers = self.bert.config.num_hidden_layers

        if not config['bert_start_train_layers'] == 'all' : 

            if  config['bert_start_train_layers'] == 'none'  : 
                for n,p in self.bert.named_parameters() : 
                    p.requires_grad = False
            else : 
                trained_layers = [
                    i for i in range(num_bert_layers + config['bert_start_train_layers'], num_bert_layers)
                ]
                for n,p in self.bert.named_parameters() : 
                    if not any([f'layers.{i}' in n for i in trained_layers]) : 
                        p.requires_grad = False
        
        self.BH = self.bert.config.hidden_size

        # this is deprecated but the model was trained with it, so we keep it for compatibility
        self.virtualH = nn.Embedding(1,self.BH*2).weight

        self.proj = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size*4, self.bert.config.hidden_size*2),
            nn.BatchNorm1d(self.bert.config.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.bert.config.hidden_size*2,1),
        )

        self.special_ids = torch.LongTensor([50280, 50282, 50283, 50281, 50284])

    def _get_input_embeddings(self, g, input_ids, att_mask,  st, et) : 
        
        H = self.bert(input_ids=input_ids, attention_mask=att_mask).last_hidden_state
 
        x = torch.cat((H[st[0], st[1]], H[et[0], et[1]]), dim=-1)

        # if self.config['mean_pool'] : 
        #     is_special = torch.isin(input_ids, self.special_ids.to(input_ids.device))  # [B, S]
        #     combined_mask = att_mask * (~is_special).long()
        #     cH = mean_pooling(H, combined_mask)
        #     x[g.node_types == 0] = torch.cat((cH, cH), dim=-1)

        # # x = x.to(torch.float32)
        # x[g.is_virtual == 1] = self.virtualH

        # x = self.bert2hidden(x)
        return x 
        

    def forward(self, g, input_ids, att_mask, st, et) : 
        x  = self._get_input_embeddings(g, input_ids, att_mask, st, et)

        quote_x = x[g.node_types==1]#[g.is_pred]
        m_x = x[g.node_types==2]#[g.is_pred]
        
        
        embs = []
        for bs in range(g.batch_q.max()+1) : 
            qx = quote_x[g.batch_q==bs]#[g.is_pred[bs]] #[q, H]
            mx = m_x[g.batch_m==bs]#[g.is_pred[bs]] # [m,H]]

            qx_rep = qx.repeat_interleave(mx.size(0), dim=0)   # [q*m, H]

            mx_rep = mx.repeat(qx.size(0), 1)                  # [q*m, H]

            # concatenate along feature dim
            out = torch.cat([qx_rep, mx_rep], dim=-1) 

            embs.append(out)

        scores = self.proj(torch.cat(embs)).squeeze(-1)

        out = self._rearange(g, scores)
        return out
        
    def _loss(self, g, quote_x, m_x) : 
        pass
    
    def _rearange(self, g, scores) : 
        cnt = 0 
        outs = [] 
        # list of [n_pred_quotes, n_mentions] per context in the batch
        for bs in range(g.batch_q.max()+1) :
            sm = (g.batch_m==bs).sum()
            sq = (g.batch_q==bs).sum()
            size = sq * sm
            outs.append(scores[cnt:cnt+size].view(sq, sm))
            cnt+=size

        return outs
    


def collate_fn(datalist) : 
    gs, input_ids, st, et, att_mask = [], [], [], [], []
    batch_q = []
    batch_m = []
    candidate_labels = []
    for c, G in enumerate(datalist) : 
        g = G.clone()
        
        input_ids.append(g.pop('input_ids').squeeze())
        att_mask.append(torch.ones_like(input_ids[-1]))

        cst, cet = g.pop('spans').chunk(2)

        st.append(torch.stack((
            torch.full((cst.size(1), ), fill_value=c),
            cst[0])))
        et.append(torch.stack((
            torch.full((cet.size(1), ), fill_value=c),
            cet[0])))


        batch_q.append(torch.full(((g.node_types==1).sum(),), c))
        batch_m.append(torch.full(((g.node_types==2).sum(),), c))

        gs.append(g)


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
        'input_ids' : pad_sequence(input_ids, batch_first=True,padding_value=50283),
        'att_mask' : pad_sequence(att_mask, batch_first=True,padding_value=0),
    }




MODEL_CONFIG={
    'model_id': 'answerdotai/ModernBERT-large',
    'bert_start_train_layers': 'all',
    'mean_pool': True,
}


from pathlib import Path
import os 

home = str(Path.home())
MODEL_PATH=os.path.join(home, "booknlp_models")
os.makedirs(MODEL_PATH,exist_ok=True) 

class GraphQuotationAttribution:
    """
    Drop-in analogue of `QuotationAttribution` (bert_qa.py), for the
    graph-based `Baseline` model. Construct with an already-loaded, eval'd
    `Baseline` instance (or use `from_checkpoint`), then call `.tag(graphs)`
    with the list of per-context `Data` graphs for a book.
    """

    def __init__(
        self,
        batch_size: int = 32,
        device: Optional[str] = None,
        is_direct=False
    ):
        
        self.model = ModernQA(MODEL_CONFIG)
        if not is_direct : 
            model_path = os.path.join(MODEL_PATH, "ModernBERT_T2000.safetensors")
            if not os.path.exists(model_path) : 
                print(f'Downloading ModernBERT_T2000.safetensors to {model_path}...')
                # import urllib
                # urllib.request.urlretrieve(JOINT_URL, model_path)
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id=REPO_ID, filename="ModernBERT_T2000.safetensors", local_dir=MODEL_PATH)
        else :
            model_path = os.path.join(MODEL_PATH, "Direct_ModernBERT_T2000.safetensors")
            if not os.path.exists(model_path) : 
                from huggingface_hub import hf_hub_download
                print(f'Downloading Direct_ModernBERT_T2000.safetensors to {model_path}...')
                hf_hub_download(repo_id=REPO_ID, filename="Direct_ModernBERT_T2000.safetensors", local_dir=MODEL_PATH)

        state_dict = load_file(model_path)
        self.model.load_state_dict(state_dict)
        
        # state = torch.load(checkpoint_path, map_location="cpu")
        # model.load_state_dict(state)
        
        self.batch_size = batch_size

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.aligner = WordAligner('answerdotai/ModernBERT-large')
        self.is_direct = is_direct

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_cls,
        model_config: Dict,
        collate_fn,
        data_args: Dict,
        **kwargs,
    ) -> "GraphQuotationAttribution":
        """
        Convenience constructor mirroring `QuotationAttribution.__init__(self,
        modelFile)`: builds the `Baseline` model from `model_config`, loads
        weights from `checkpoint_path`, wraps it.
        """
        model = model_cls(model_config)
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state)
        return cls(model, collate_fn, data_args, **kwargs)

    @staticmethod
    def _parse_id(tag: str) -> int:
        """'Q_12' -> 12, 'M_7' -> 7 (also tolerates a longer node-id-style
        string such as 'Q_12_CTX_3', only the piece right after the first
        '_' is used)."""
        return int(tag.split("_")[1])

    def tag(self, quotes, entities, tokens) -> List[Optional[int]]:
        """
        Run the model over every per-context graph for one book and return
        `attributed_quotations`, aligned with the book-level `quotes` list,
        matching the *exact* contract `english_booknlp.py` expects from
        `self.quote_attrib.tag(quotes, entities, tokens)`:

            attributed_quotations[i] is either
              - None                      (no speaker resolved), or
              - int                       an index into `entities`
                                           (NOT a (start, end) tuple -- see
                                           bert_qa.py: `entity_by_position`
                                           maps (start, end) -> index).

        `n_quotes`: pass `len(quotes)` from the original book-level list if
        you want the output padded to the right length even when trailing
        quotes had no candidate graph at all; otherwise it's inferred from
        the largest quote index actually observed in `quote_ids`.
        """

        # K does not matter here, we use it for compatibility with the training-time graph builder
        graphs = build_inference_graph(quotes, entities, tokens, self.aligner, N=2000, S=512, K=200)
        if self.is_direct : 
            graphs = to_direct_input(graphs)
        print(f'Processing {len(graphs)} inputs')
        
        # if not graphs:
        #     return [None] * (n_quotes or 0)

        loader = DataLoader(
            graphs,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn)#, data_args=self.data_args),
        )

        best_score: Dict[int, float] = {}
        best_mention: Dict[int, int] = {}
        max_quote_idx = -1
        device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        
        with torch.no_grad():
            for batch in tqdm(loader):
                with torch.autocast(device_type='cuda', dtype=DTYPE):
                    g = batch["g"].to(self.device)
                    input_ids = batch["input_ids"].to(self.device)
                    att_mask = batch["att_mask"].to(self.device)
                    st = batch["st"].to(self.device)
                    et = batch["et"].to(self.device)
    
                    out = self.model(g, input_ids, att_mask, st, et)
    
                    for bs, scores in enumerate(out):
                        if scores.numel() == 0:
                            continue
    
                        # `is_pred[bs]` / `quote_ids[bs]` are both indexed over
                        # this graph's quote-type nodes, in the same order (see
                        # forward(): qx = quote_x[batch_q==bs][is_pred[bs]]) --
                        # keep only the entries forward() actually scored.
                        # is_pred_bs = g.is_pred[bs]
                        # is_pred_bs = is_pred_bs.tolist() if torch.is_tensor(is_pred_bs) else list(is_pred_bs)
                        quote_ids_bs = g.quote_ids[bs]
                        # print(quote_ids_bs)
                        q_gidx = [self._parse_id(mid[0]) for mid in quote_ids_bs]
                        # [
                            # self._parse_id(quote_ids_bs[k][0])
                            # for k, keep in enumerate(is_pred_bs)
                            # if keep
                        # ]
    
                        # All mention-type nodes in this graph are candidates
                        # (see forward(): mx = m_x[batch_m==bs], no is_pred mask).
                        mention_ids_bs = g.mention_ids[bs]
                        m_gidx = [self._parse_id(mid[0]) for mid in mention_ids_bs]
    
                        probs = torch.softmax(scores, dim=-1)
                        top_vals, top_local = probs.max(dim=-1)
                        
                        for qg, val, local in zip(q_gidx, top_vals.tolist(), top_local.tolist()):
                            if qg is None or qg < 0:
                                continue  # virtual / non-predicted quote node
                            max_quote_idx = max(max_quote_idx, qg)
    
                            mg = m_gidx[local]
                            if mg is None or mg < 0:
                                continue  # top choice was a virtual/placeholder mention -> no speaker
    
                            if qg not in best_score or val > best_score[qg]:
                                best_score[qg] = val
                                best_mention[qg] = mg

        size = max_quote_idx +1 #n_quotes if n_quotes is not None else max_quote_idx + 1
        attributed_quotations: List[Optional[int]] = [None] * size
        for qg, mg in best_mention.items():
            if qg < size:
                attributed_quotations[qg] = mg
        return attributed_quotations