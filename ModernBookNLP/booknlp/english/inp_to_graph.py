#!/usr/bin/env python3
"""
booknlp_graph_adapter.py
=========================

Step 1 of replacing BookNLP's bundled `QuotationAttribution` (bert_qa.py) by
a graph-based model: convert the *in-memory* objects available at the exact
call site in `english_booknlp.py`

    attributed_quotations = self.quote_attrib.tag(quotes, entities, tokens)   # line ~391

into the same node/edge structure produced by `booknlp_to_graph.py`, without
touching disk and without requiring anything that isn't actually available
yet at that point in the pipeline (no COREF ids, no gold speaker spans).

Design notes / deltas vs. booknlp_to_graph.py
----------------------------------------------
1. Inputs, at this call site, are Python objects, not `.tokens`/`.entities`/
   `.quotes` TSV files:
     - quotes:   List[Tuple[int, int]]                    (quote_start, quote_end)
     - entities: List[Tuple[int, int, str, str]]          (start, end, cat, text)
                 cat is a compound tag, e.g. "PROP_PER", "NOM_LOC", "PRON_PER"
     - tokens:   List[Token]  (booknlp/common/pipelines.py Token objects)

   `tokens_to_df` / `entities_to_df` / `quotes_to_df` below build DataFrames
   with the *same column names and dtypes* load_tokens/load_entities/
   load_quotes normally produce by reading files, so `build_context_nodes`
   can be reused unchanged.

2. `COREF` is NOT available at this point in the real pipeline (BookNLP's
   own neural coref, litbank_coref.tag(...), actually *consumes*
   attributed_quotations, so it necessarily runs *after* quote attribution;
   even the cheaper rule-based NameCoref also runs after, in the current
   code). Per your confirmation, COREF / Global Character nodes /
   mention-mention edges are only needed to build *silver training labels*,
   not as runtime model input. So this adapter builds only:
       Context --contains--> Quote-in-context
       Context --contains--> Mention-in-context
       Quote-in-context --candidate (K tokens)--> Mention-in-context
   and skips Global Character nodes / mention-mention edges entirely.

3. `mention_idx` on Mention nodes and `quote_idx` on Quote nodes are literal
   positions in the `entities` / `quotes` lists you passed in, so a
   downstream model's prediction per quote node can be mapped straight back
   to `entities[mention_idx]` -- which is exactly the `(int)` index contract
   the real `attributed_quotations` list expects (see bert_qa.py: it stores
   `entity_by_position[(start, end)]`, an index into `entities`, not a
   (start, end) tuple).
"""

from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

# Reuse everything that doesn't depend on gold/COREF columns directly from
# your existing script.
from booknlp.english.booknlp_to_graph import (
    WordAligner,
    build_context_nodes,
    _containing_or_overlapping_contexts,
    _resolve_span,
    add_context_quote_edges,
    add_context_mention_edges,
)


# --------------------------------------------------------------------------- #
# quotes / entities / tokens (in-memory) -> DataFrames
# --------------------------------------------------------------------------- #

def tokens_to_df(tokens: List) -> pd.DataFrame:
    """
    Convert BookNLP's in-memory `List[Token]` (booknlp/common/pipelines.py)
    into the DataFrame shape `build_context_nodes` expects (same column
    names/semantics as `load_tokens` reading a `.tokens` file):
      - token_ID_within_document (int, 0-indexed, contiguous)
      - word
      - paragraph_ID
      - sentence_ID
      - POS_tag / dependency_relation are carried over too (unused for
        inference-time graph building, but harmless/cheap to keep for any
        downstream debugging or future feature use).
    """
    rows = [{
        "token_ID_within_document": tok.token_id,
        "word": tok.text,
        "paragraph_ID": tok.paragraph_id,
        "sentence_ID": tok.sentence_id,
        "POS_tag": tok.pos,
        "dependency_relation": tok.deprel,
    } for tok in tokens]

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["word"])
    df["token_ID_within_document"] = df["token_ID_within_document"].astype(int)
    return df


def entities_to_df(entities: List[Tuple[int, int, str, str]]) -> pd.DataFrame:
    """
    Convert BookNLP's in-memory `entities` (List[(start, end, cat, text)],
    e.g. cat="PROP_PER") into the DataFrame shape `build_mention_instances`
    expects (same column semantics as `load_entities`), EXCEPT no `COREF`
    column: it isn't available yet at this point in the pipeline, and per
    your confirmation it's only needed to build silver training labels, not
    as model input.

    - "prop": the mention-form prefix (PROP / NOM / PRON)
    - "cat":  the NER type suffix (PER / LOC / ORG / FAC / GPE / VEH ...)
      -> matches booknlp_to_graph.build_mention_instances' `entities["cat"]
      == "PER"` filter.

    The DataFrame's row index equals the position of each entity in the
    original `entities` list (i.e. `df.index[i] == i`), so `mention_idx`
    built downstream is directly usable as an index back into `entities`.
    """
    rows = []
    for start, end, cat, text in entities:
        parts = cat.split("_", 1)
        prop = parts[0] if len(parts) > 1 else ""
        ner_type = parts[1] if len(parts) > 1 else parts[0]
        rows.append({
            "start_token": int(start),
            "end_token": int(end),
            "prop": prop,
            "cat": ner_type,
            "text": text,
        })
    df = pd.DataFrame(rows, columns=["start_token", "end_token", "prop", "cat", "text"])
    return df


def quotes_to_df(quotes: List[Tuple[int, int]], tokens: List) -> pd.DataFrame:
    """
    Convert BookNLP's in-memory `quotes` (List[(quote_start, quote_end)])
    into the DataFrame shape `build_quote_instances` expects, EXCEPT no
    `mention_start`/`mention_end`/`mention_phrase`/`char_id` columns: those
    are the *target* of speaker attribution, not something we have yet.

    Row index equals position in the original `quotes` list, so `quote_idx`
    built downstream maps straight back to `quotes[quote_idx]`.
    """
    rows = []
    for q_start, q_end in quotes:
        text = " ".join(tok.text for tok in tokens[q_start:q_end + 1])
        rows.append({
            "quote_start": int(q_start),
            "quote_end": int(q_end),
            "quote": text,
        })
    return pd.DataFrame(rows, columns=["quote_start", "quote_end", "quote"])


# --------------------------------------------------------------------------- #
# Instance builders (gold/COREF-free variants of booknlp_to_graph's)
# --------------------------------------------------------------------------- #

def build_quote_instances_infer(quotes: pd.DataFrame, contexts: List[Dict]) -> List[Dict]:
    """
    Same node ids/spans as booknlp_to_graph.build_quote_instances
    (`Q_{i}_{ctx_id}`, absolute + local (st, et) span), but without any
    gold-speaker fields (mention_start/end/phrase, char_id, is_explicit),
    since we don't have (and don't need) them at inference time.
    """
    instances = []
    for i, row in quotes.iterrows():
        q_start, q_end = row["quote_start"], row["quote_end"]
        ctxs, clipped = _containing_or_overlapping_contexts(contexts, q_start, q_end)
        if not ctxs:
            print(f"    [!] quote {i} ({q_start}-{q_end}) matches no context window at all, skipping")
            continue
        for ctx in ctxs:
            span = _resolve_span(ctx, q_start, q_end, clipped)
            if span is None:
                continue
            st, et = span
            instances.append({
                "id": f"Q_{i}_{ctx['id']}",
                "quote_idx": i,          # index into the original `quotes` list
                "ctx_id": ctx["id"],
                "start": q_start,
                "end": q_end,
                "st": st,
                "et": et,
                "text": row.get("quote", ""),
            })
    return instances


def build_mention_instances_infer(entities: pd.DataFrame, contexts: List[Dict]) -> List[Dict]:
    """
    Same node ids/spans as booknlp_to_graph.build_mention_instances
    (`M_{i}_{ctx_id}`, absolute + local (st, et) span, `prop`/`text` entity
    type info), restricted to PER mentions, but without `coref` (not
    available yet -- see module docstring).
    """
    per = entities[entities["cat"] == "PER"]
    instances = []
    for i, row in per.iterrows():
        m_start, m_end = row["start_token"], row["end_token"]
        ctxs, clipped = _containing_or_overlapping_contexts(contexts, m_start, m_end)
        if not ctxs:
            print(f"    [!] mention {i} ({m_start}-{m_end}) matches no context window at all, skipping")
            continue
        for ctx in ctxs:
            span = _resolve_span(ctx, m_start, m_end, clipped)
            if span is None:
                continue
            st, et = span
            instances.append({
                "id": f"M_{i}_{ctx['id']}",
                "mention_idx": i,        # index into the original `entities` list
                "ctx_id": ctx["id"],
                "start": m_start,
                "end": m_end,
                "st": st,
                "et": et,
                "prop": row["prop"],
                "text": row["text"],
            })
    return instances


def add_quote_mention_candidate_edges(
    G: nx.Graph, quote_instances: List[Dict], mention_instances: List[Dict], K: int
):
    """
    Same candidate-generation logic as booknlp_to_graph.add_quote_mention_edges:
    same-context, within a +/-K token window of the quote, and not itself
    inside the quote's own span.
    """
    mentions_by_ctx: Dict[str, List[Dict]] = {}
    for m in mention_instances:
        mentions_by_ctx.setdefault(m["ctx_id"], []).append(m)

    for q in quote_instances:
        win_start, win_end = q["start"] - K, q["end"] + K
        for m in mentions_by_ctx.get(q["ctx_id"], []):
            in_window = m["start"] <= win_end and m["end"] >= win_start
            inside_quote = m["start"] <= q["end"] and m["end"] >= q["start"]
            if in_window and not inside_quote:
                G.add_edge(q["id"], m["id"], type="quote-mention")


# --------------------------------------------------------------------------- #
# Entry point: quotes/entities/tokens (in-memory) -> inference-time graph
# --------------------------------------------------------------------------- #

def build_inference_graph(
    quotes: List[Tuple[int, int]],
    entities: List[Tuple[int, int, str, str]],
    tokens: List,
    aligner: WordAligner,
    N: int = 200,
    S: int = 100,
    K: int = 50,
) -> nx.Graph:
    """
    Drop-in replacement for `QuotationAttribution.tag`'s inputs -> graph
    conversion. Call this where `english_booknlp.py` currently does:

        attributed_quotations = self.quote_attrib.tag(quotes, entities, tokens)

    i.e. as:

        G = build_inference_graph(quotes, entities, tokens, aligner)
        # ... run your graph model on G to score quote<->mention edges ...
        # ... map predictions back to entities[mention_idx] per quote_idx ...
    """
    tokens_df = tokens_to_df(tokens)
    entities_df = entities_to_df(entities)
    quotes_df = quotes_to_df(quotes, tokens)

    contexts = build_context_nodes(tokens_df, N, S, aligner)
    quote_instances = build_quote_instances_infer(quotes_df, contexts)
    mention_instances = build_mention_instances_infer(entities_df, contexts)

    G = nx.Graph()
    for c in contexts:
        G.add_node(
            c["id"], node_type="context",
            start=c["start"], end=c["end"],
            input_ids=c["input_ids"], text=c["text"],
        )
    for q in quote_instances:
        G.add_node(q["id"], node_type="quote", **{k: v for k, v in q.items() if k != "id"})
    for m in mention_instances:
        G.add_node(m["id"], node_type="mention", **{k: v for k, v in m.items() if k != "id"})

    add_context_quote_edges(G, quote_instances)
    add_context_mention_edges(G, mention_instances)
    add_quote_mention_candidate_edges(G, quote_instances, mention_instances, K)

    G.graph["params"] = {"N": N, "S": S, "K": K, "tokenizer": aligner.tokenizer.name_or_path}
    G.graph["n_quotes"] = len(quotes)
    G.graph["n_entities"] = len(entities)
    return convert(G)


import re 
import torch_geometric
import torch
def convert(G) : 

    # path = os.path.split(ff)[0]
    # os.makedirs(f'{path}/components/', exist_ok=True)
    # if not os.path.exists(f'{path}/components.pkl') : 
    list_of_graphs = []
    # try : 
    # with open(ff,'rb') as f:
        # G = pickle.load(f)
    connecteds = list(nx.connected_components(G) )

    for idx, component in (enumerate(connecteds)):#, total=len(connecteds)) : 

        if len([i for i in component if 'Q_' in i]) == 0 :
            continue
            
        subG = G.subgraph(component)
        mapp = {v:k for k,v in enumerate(component)}
        inv_mapp = {k:v for k,v in enumerate(component)}
        subG = nx.relabel_nodes(subG, mapp)
        node_type = []
        input_ids = []
        # spans_q = []
        # spans_m = []
        spans = []
        # coref = []
        # speakers = []
        quote_ids = []
        to_keep = []
        mention_ids = []
        for v in range(len(subG)) : 
           
            if 'node_type' not in subG.nodes[v] : 
                subG.nodes[v]['node_type'] = 'quote'
            if subG.nodes[v]['node_type'] == 'mention' : 

                # if RESTRICT : 
                    # if subG.nodes[v]['mention_type'] != 'PROPN' : 
                        # continue
                        
                node_type.append(2)
                spans.append((subG.nodes[v]['st'], subG.nodes[v]['et'] ))
                mention_ids.append(re.findall('M_[\d]+', inv_mapp[v]))
                
                # coref.append(int(subG.nodes[v]['char_id']))
                
            elif subG.nodes[v]['node_type'] == 'quote' : 
                node_type.append(1)
                
                if 'Q_' in inv_mapp[v] : 
                    quote_ids.append(re.findall('Q_[\d]+', inv_mapp[v]))
                else : 
                    quote_ids.append(None)
                if 'st' in subG.nodes[v] : 
                    spans.append((subG.nodes[v]['st'], subG.nodes[v]['et'] ))
                else : 
                    spans.append((-1,-1))

                # else its a virtual node
                # if 'char_id' in subG.nodes[v] :
                #     spk_id = subG.nodes[v]['char_id']
                #     if spk_id is not None : 
                #         speakers.append(int(spk_id))
                #     else :
                #         speakers.append(-1)
                # else :
                #     speakers.append(-1)
                    
            else : 
                node_type.append(0)
                spans.append((0,0))
                
            if 'input_ids' in subG.nodes[v]:
                input_ids.append(subG.nodes[v]['input_ids'])

            to_keep.append(v)

        subG = subG.subgraph(to_keep)
        
        data = torch_geometric.data.Data(edge_index=torch.LongTensor(list(subG.edges)).t(),num_nodes=len(subG) )
        # data.edge_index = torch_geometric.utils.add_self_loops(data.edge_index)[0]
        data.node_types = torch.LongTensor(node_type)
        data.input_ids = torch.LongTensor(input_ids)
        data.spans = torch.LongTensor(spans).t()
        # data.corefs = torch.LongTensor(coref)
        # data.speakers = torch.LongTensor(speakers)
        data.quote_ids = quote_ids
        data.mention_ids = mention_ids
        # data.spans_m = torch.LongTensor(spans_m).t()

        list_of_graphs.append(data)

    return list_of_graphs


def to_direct_input(Gs) : 
    all_data = []
    
    for G in Gs : 
        # if len(G.speakers) > 0 : 
        #     if all([
        #         (G.speakers[0] != -1),
        #         (len([q for q in G.quote_ids if q in qids]) >0),
        #     ]):
        for cnt in range((G.node_types==1).sum()) : 

            g = G.clone()
            to_keep = torch.ones(g.num_nodes,).bool()
            c = [j for j in range((G.node_types==1).sum()) if j!=cnt]
            to_remove = torch.where(g.node_types==1)[0][c] #[j for j in is_pred if j!=i]
            to_keep[to_remove] = False
            g.node_types = g.node_types[to_keep]
            g.spans = g.spans[:, to_keep]
            g.quote_ids = [g.quote_ids[cnt]]
            # if g.quote_ids[0] not in all_data : 
            #     all_data[g.quote_ids[0]] = g
            all_data.append(g)
    return all_data