#!/usr/bin/env python3
"""
booknlp_to_graph.py
====================

Convert BookNLP's raw output (`*.tokens`, `*.entities`, `*.quotes`) into a
hierarchical graph, ready to seed node representations from a ModernBERT
model for self-supervised learning:

    Context  --contains-->  Quote-in-context
    Context  --contains-->  Mention-in-context
    Quote-in-context  --near (K tokens)-->  Mention-in-context   (same context only)
    Mention-in-context  --coref-->  Global Character

Because context windows overlap (stride S < N), a quote/mention that falls
inside several context windows is DUPLICATED, once per containing context
window. Each duplicate gets its own scalar (st, et): the start/end subword
token position of that quote/mention *within that particular context's own
ModernBERT tokenization*. Global Character nodes are NOT duplicated, so they
are what connects the different (locally-tokenized) context subgraphs back
together across the whole book.

Usage
-----
    python booknlp_to_graph.py \
        --inp_dir /path/to/booknlp_output \
        --out_dir /path/to/graphs \
        --N 200 --S 100 --K 50 \
        --model_name answerdotai/ModernBERT-large

`inp_dir` can hold the output for ONE book or MANY books (BookNLP names files
`<book_id>.tokens`, `<book_id>.entities`, `<book_id>.quotes`); the script
auto-discovers every `book_id` present, unless you pass --book_id explicitly.

Output
------
For every book: `<out_dir>/<book_id>.graph.pkl` (a pickled networkx.Graph)
and `<out_dir>/<book_id>.graph.json` (node-link JSON, human-inspectable).
"""

import argparse
import glob
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
from transformers import AutoTokenizer


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #

def find_books(inp_dir: str) -> List[str]:
    """Discover every book_id in inp_dir by looking for `<book_id>.tokens` files."""
    token_files = glob.glob(os.path.join(inp_dir, "*.tokens"))
    return sorted(os.path.basename(f)[: -len(".tokens")] for f in token_files)


def _read_tsv(path: str) -> pd.DataFrame:
    # quoting=3 (QUOTE_NONE) avoids pandas choking on stray quote chars inside
    # quote/entity text; keep_default_na=False avoids turning tokens like
    # "NA" or empty strings into NaN.
    return pd.read_csv(path, sep="\t", quoting=3, keep_default_na=True, dtype=str)


def load_tokens(inp_dir: str, book_id: str) -> pd.DataFrame:
    df = _read_tsv(os.path.join(inp_dir, f"{book_id}.tokens"))

    df = df.dropna(subset=['word'])
    df["token_ID_within_document"] = df["token_ID_within_document"].apply(_fuzzy_int)
    return df

def load_entities(inp_dir: str, book_id: str) -> pd.DataFrame:
    df = _read_tsv(os.path.join(inp_dir, f"{book_id}.entities"))
    # df["start_token"] = df["start_token"].apply(_fuzzy_int)
    # df["end_token"] = df["end_token"].apply(_fuzzy_int)
    # df["COREF"] = df["COREF"].apply(_fuzzy_int)
    df["start_token"] = df["start_token"].astype(int)
    df["end_token"] = df["end_token"].astype(int)
    df["COREF"] = df["COREF"].astype(int)

    return df



def _fuzzy_int(i) : 
    try :
        return int(i) 
    except : 
        return None
    
def load_quotes(inp_dir: str, book_id: str) -> pd.DataFrame:
    df = _read_tsv(os.path.join(inp_dir, f"{book_id}.quotes"))
    df = df.dropna(subset=['quote_start','quote_end', 'mention_start', 'mention_end'])

    df["quote_start"] = df["quote_start"].apply(_fuzzy_int)
    df["quote_end"] = df["quote_end"].apply(_fuzzy_int)
    df["mention_start"] = df["mention_start"].apply(_fuzzy_int)
    df["mention_end"] = df["mention_end"].apply(_fuzzy_int)
    return df


# --------------------------------------------------------------------------- #
# Tokenizer wrapper
# --------------------------------------------------------------------------- #

class WordAligner:
    """
    Wraps a fast HF tokenizer. Context text is assembled token by token (see
    `build_context_nodes`, which tracks the exact character span of every
    token as it appends it), and tokenized as a *plain string* (not via
    `is_split_into_words`), using `return_offsets_mapping=True` to get the
    character span covered by every subword token.

    Word -> subword alignment for a given quote/mention is then a two-step,
    exact lookup: (1) the token's own recorded character span (no
    re-derivation needed, see `_resolve_span`), mapped to (2) the subword
    token(s) whose offsets overlap that character span (`char_span_to_token_span`).
    """

    def __init__(self, model_name: str, add_special_tokens: bool = True):
        self.add_special_tokens = add_special_tokens
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Could not load tokenizer '{model_name}': {e}")
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                f"Tokenizer '{model_name}' is not a fast tokenizer; "
                "character-offset alignment (return_offsets_mapping) requires a fast tokenizer."
            )

    def encode_text(self, text: str):
        """
        Tokenize `text` and return (input_ids, offset_mapping, special_tokens_mask).
        offset_mapping[k] = (char_start, char_end) of subword token k in `text`
        (both (0, 0) for special tokens). special_tokens_mask[k] == 1 for
        special tokens (e.g. [CLS]/[SEP]), 0 otherwise.
        """
        enc = self.tokenizer(
            text,
            add_special_tokens=self.add_special_tokens,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        return enc["input_ids"], enc["offset_mapping"], enc["special_tokens_mask"]

    @staticmethod
    def char_span_to_token_span(
        offset_mapping: List[Tuple[int, int]],
        special_tokens_mask: List[int],
        char_start: int,
        char_end: int,
    ) -> Optional[Tuple[int, int]]:
        """
        Given the tokenizer's offset_mapping/special_tokens_mask for a text,
        and a [char_start, char_end) character range, return the inclusive
        [st, et] subword token-index range whose spans overlap that range
        (special tokens excluded), or None if nothing overlaps.
        """
        positions = [
            pos for pos, (a, b) in enumerate(offset_mapping)
            if not special_tokens_mask[pos] and a < char_end and b > char_start
        ]
        if not positions:
            return None
        return min(positions), max(positions)


# --------------------------------------------------------------------------- #
# Char-span resolution
# --------------------------------------------------------------------------- #

def _resolve_span(
    ctx: Dict, start: int, end: int, clipped: bool
) -> Optional[Tuple[int, int]]:
    """
    Resolve the (st, et) inclusive subword-token span, within `ctx`'s own
    tokenization, for the absolute (document-level) token range [start, end].

    This is exact by construction: `ctx["_token_char_spans"]` records, for
    every absolute token id present in this context, the precise character
    span that token occupies within `ctx["text"]` (tracked while the text
    was being assembled in `build_context_nodes`), so no re-derivation
    (regex or arithmetic) is needed here.
    """
    t_start = max(start, ctx["start"])
    t_end = min(end, ctx["end"])
    char_start = ctx["_token_char_spans"][t_start][0]
    char_end = ctx["_token_char_spans"][t_end][1]

    tok_span = WordAligner.char_span_to_token_span(
        ctx["_offset_mapping"], ctx["_special_tokens_mask"], char_start, char_end
    )
    if tok_span is None:
        return None

    st, et = tok_span
    if clipped:
        st = max(st, 0)
        et = min(et, len(ctx["input_ids"]) - 1)
    return st, et


# --------------------------------------------------------------------------- #
# Node builders
# --------------------------------------------------------------------------- #

def _to_bool(x) -> bool:
    """Parse a BookNLP-style boolean column value (may already be a real
    bool, or a "True"/"False"/"1"/"0" string, since tables are loaded as
    str)."""
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in ("true", "1", "yes")


# def _extend_to_sentence_boundary(
#     end: int,
#     total_tokens: int,
#     sentence_by_id: Dict[int, str],
#     in_quote_by_id: Dict[int, bool],
# ) -> int:
#     """
#     Extend `end` forward (never shrink it) to the nearest token that is both
#     (a) the last token of its sentence, and (b) not inside a quotation, so a
#     context window never splits a sentence in the middle, and specifically
#     never splits a quotation in half either (a quotation can itself contain
#     several sentences; we only stop once we're back outside of it). Falls
#     back to the last token of the document if no such boundary is found.
#     """
#     t = end
#     while t < total_tokens - 1:
#         is_sentence_end = sentence_by_id.get(t) != sentence_by_id.get(t + 1)
#         is_in_quote = in_quote_by_id.get(t, False)
#         if is_sentence_end and not is_in_quote:
#             return t
#         t += 1
#     return total_tokens - 1


def _extend_to_sentence_boundary(
    end: int,
    total_tokens: int,
    sentence_by_id: Dict[int, str],
) -> int:
    """
    Extend `end` forward (never shrink it) to the nearest token that is both
    (a) the last token of its sentence, and (b) not inside a quotation, so a
    context window never splits a sentence in the middle, and specifically
    never splits a quotation in half either (a quotation can itself contain
    several sentences; we only stop once we're back outside of it). Falls
    back to the last token of the document if no such boundary is found.
    """
    t = end
    while t < total_tokens - 1:
        is_sentence_end = sentence_by_id.get(t) != sentence_by_id.get(t + 1)
        if is_sentence_end:
            return t
        t += 1
    return total_tokens - 1

def build_context_nodes(tokens: pd.DataFrame, N: int, S: int, aligner: WordAligner) -> List[Dict]:
    """
    Chunk the whole document (all tokens, in `token_ID_within_document` order)
    into consecutive, overlapping windows of (up to) `N` tokens with stride
    `S`. `start`/`end` are inclusive, absolute (document-level) token ids.

    The target end (`start + N - 1`) is extended forward, never shrunk, to
    the nearest sentence boundary that is not inside a quotation (see
    `_extend_to_sentence_boundary`), using the tokens table's `sentence_ID`
    and `inQuote` columns. So a window's actual length can exceed `N`
    whenever the natural cut point falls mid-sentence or inside a quote.

    Context text is assembled word by word, inserting a paragraph break
    ("\\n\\n") whenever `paragraph_ID` changes and a single space after every
    token, exactly mirroring how a raw book's text would read. While doing
    so, we track a running character offset and record, for every token id
    in the window, the exact (char_start, char_end) span its word occupies
    in the resulting `text` -- this is what lets us resolve any quote/mention
    span to character positions with zero ambiguity, whatever punctuation or
    paragraph breaks happen to surround it.

    Each context stores:
      - text: the assembled, human-readable string fed to the tokenizer
      - input_ids: the list of ModernBERT subword token ids for `text`
      - _offset_mapping / _special_tokens_mask: needed internally to map a
        character span to its subword token span; not written to disk
      - _token_char_spans: {absolute_token_id: (char_start, char_end)}; not
        written to disk
    """
    total_tokens = int(tokens["token_ID_within_document"].max()) + 1
    word_by_id = dict(zip(tokens["token_ID_within_document"], tokens["word"]))
    par_by_id = dict(zip(tokens["token_ID_within_document"], tokens["paragraph_ID"]))
    sentence_by_id = dict(zip(tokens["token_ID_within_document"], tokens["sentence_ID"]))
    # in_quote_by_id = dict(zip(
    #     tokens["token_ID_within_document"], tokens["inQuote"].apply(_to_bool)
    # ))

    contexts = []
    start, idx = 0, 0
    par_id = 0  # persists across context windows, like the paragraph id truly does in the book
    while start < total_tokens:
        end = min(start + N - 1, total_tokens - 1)
        end = _extend_to_sentence_boundary(end, total_tokens, sentence_by_id)

        text_parts: List[str] = []
        token_char_spans: Dict[int, Tuple[int, int]] = {}
        char_offset = 0

        for t in range(start, end + 1):
            curr_pid = par_by_id.get(t, "")
            if curr_pid != "" and int(curr_pid) != par_id:
                text_parts.append("\n\n")
                char_offset += len("\n\n")
                par_id = int(curr_pid)

            w = word_by_id.get(t, "")
            tok_start = char_offset
            text_parts.append(w)
            try : 
                char_offset += len(w)
            except :
                print(w)
            token_char_spans[t] = (tok_start, char_offset)  # end exclusive

            text_parts.append(" ")
            char_offset += 1

        text = "".join(text_parts)
        input_ids, offset_mapping, special_tokens_mask = aligner.encode_text(text)

        contexts.append({
            "id": f"CTX_{idx}",
            "start": start,
            "end": end,
            "text": text,
            "input_ids": input_ids,
            "_offset_mapping": offset_mapping,
            "_special_tokens_mask": special_tokens_mask,
            "_token_char_spans": token_char_spans,
        })
        idx += 1
        if end == total_tokens - 1:
            break
        start += N-S
    return contexts


def _containing_or_overlapping_contexts(
    contexts: List[Dict], start: int, end: int
) -> Tuple[List[Dict], bool]:
    """
    Return the list of contexts that fully CONTAIN [start, end]. If none do
    (span longer than any single context window), fall back to contexts that
    at least OVERLAP [start, end], and flag that a clipping fallback is used.
    """
    containing = [c for c in contexts if start >= c["start"] and end <= c["end"]]
    if containing:
        return containing, False
    overlapping = [c for c in contexts if start <= c["end"] and end >= c["start"]]
    return overlapping, True


def validate_speaker_is_noun(tok_row) : 
    if tok_row['POS_tag'] == 'PROPN' :
        return True
    elif (tok_row['POS_tag'] == 'ADJ') & (tok_row['dependency_relation'] == 'nsubj') : 
        return True
    else : 
        return False
    
def build_quote_instances(quotes: pd.DataFrame, contexts: List[Dict], tokens) -> List[Dict]:
    """
    One instance per (quote, containing-context) pair. Node id: Q_{i}_{ctx_id}.
    """
    instances = []
    n_fallback = 0
    n_not_found = 0
    for i, row in quotes.iterrows():
        q_start, q_end = row["quote_start"], row["quote_end"]
        ctxs, clipped = _containing_or_overlapping_contexts(contexts, q_start, q_end)
        if not ctxs:
            print(f"    [!] quote {i} ({q_start}-{q_end}) matches no context window at all, skipping")
            continue
        if clipped:
            n_fallback += 1
        for ctx in ctxs:
            span = _resolve_span(ctx, q_start, q_end, clipped)
            if span is None:
                n_not_found += 1
                continue
            st, et = span
            ms, me = row["mention_start"], row["mention_end"]


            instances.append({
                "id": f"Q_{i}_{ctx['id']}",
                "quote_idx": i,
                "ctx_id": ctx["id"],
                "start": q_start,
                "end": q_end,
                "st": st,
                "et": et,
                "speaker_mention_start": ms if ms >= 0 else None,
                "speaker_mention_end": me if me >= 0 else None,
                "speaker_mention_phrase": row.get("mention_phrase", ""),
                "char_id": row.get("char_id", None),
                "text": row.get("quote", ""),
                "is_explicit" : validate_speaker_is_noun(tokens.iloc[ms])
            })

    return instances


def build_mention_instances(entities: pd.DataFrame, contexts: List[Dict]) -> List[Dict]:
    """
    One instance per (PER mention, containing-context) pair. Node id: M_{i}_{ctx_id}.
    """
    per = entities[entities["cat"] == "PER"]
    instances = []
    n_fallback = 0
    n_not_found = 0
    for i, row in per.iterrows():
        m_start, m_end = row["start_token"], row["end_token"]
        ctxs, clipped = _containing_or_overlapping_contexts(contexts, m_start, m_end)
        if not ctxs:
            print(f"    [!] mention {i} ({m_start}-{m_end}) matches no context window at all, skipping")
            continue
        if clipped:
            n_fallback += 1
        for ctx in ctxs:
            span = _resolve_span(ctx, m_start, m_end, clipped)
            if span is None:
                n_not_found += 1
                continue
            st, et = span
            instances.append({
                "id": f"M_{i}_{ctx['id']}",
                "mention_idx": i,
                "ctx_id": ctx["id"],
                "start": m_start,
                "end": m_end,
                "st": st,
                "et": et,
                "prop": row["prop"],
                "text": row["text"],
                "coref": row["COREF"],
            })
    # if n_fallback:
    #     print(f"    [!] {n_fallback} mention(s) longer than a context window; used clipped overlap fallback")
    # if n_not_found:
    #     print(f"    [!] {n_not_found} mention-context pair(s) had no overlapping subword tokens, skipped")
    return instances


def build_global_character_nodes(mention_instances: List[Dict]) -> List[Dict]:
    corefs = sorted({m["coref"] for m in mention_instances})
    return [{"id": f"C_{c}", "coref": c} for c in corefs]


# --------------------------------------------------------------------------- #
# Edge builders
# --------------------------------------------------------------------------- #

def add_context_quote_edges(G: nx.Graph, quote_instances: List[Dict]):
    for q in quote_instances:
        G.add_edge(q["ctx_id"], q["id"], type="context-quote")


def add_context_mention_edges(G: nx.Graph, mention_instances: List[Dict]):
    for m in mention_instances:
        G.add_edge(m["ctx_id"], m["id"], type="context-mention")


def add_mention_mention_edges(G, mention_instances, ctxt_to_keep) : 
    mentions_by_ctx = {}
    quotes_by_ctx = {}

    for m in mention_instances:
        mentions_by_ctx.setdefault(m["ctx_id"], []).append(m)
    # for m in quote_instances:
    #     quotes_by_ctx.setdefault(m["ctx_id"], []).append(m)

    edges = []
    for ctx_id, m_within in mentions_by_ctx.items() : 
        if ctx_id in ctxt_to_keep : 
            prev_m = m_within[0]
            for m in m_within[1:] : 
                edge = (prev_m['id'], m['id'])
                G.add_edge(prev_m['id'], m['id'], type="mention-mention")
                prev_m = m
    cids = set(mentions_by_ctx) - set(ctxt_to_keep)
    
    for j, ms in enumerate(cids) : 
        G.add_node(f'VIRTUAL_{j}')
        G.add_edge(f'VIRTUAL_{j}', ms, type='quote-mention')
        ms = mentions_by_ctx[ms]

        for m in ms : 
            G.add_edge(f'VIRTUAL_{j}', m['id'], type='quote-mention')

# def add_quote_mention_edges(G: nx.Graph, quote_instances: List[Dict], mention_instances: List[Dict], K: int):
#     """
#     Connect a quote-instance to a mention-instance if:
#       1) they belong to the SAME context window (same ctx_id), and
#       2) the mention's absolute span overlaps the quote's absolute span
#          padded by K tokens on each side.
#     """
#     mentions_by_ctx: Dict[str, List[Dict]] = {}
#     for m in mention_instances:
#         mentions_by_ctx.setdefault(m["ctx_id"], []).append(m)

#     for q in quote_instances:
#         win_start, win_end = q["start"] - K, q["end"] + K
#         for m in mentions_by_ctx.get(q["ctx_id"], []):
#             if m["start"] <= win_end and m["end"] >= win_start:
#                 G.add_edge(q["id"], m["id"], type="quote-mention")
def add_quote_mention_edges(G: nx.Graph, quote_instances: List[Dict], mention_instances: List[Dict], K: int):
    """
    Connect a quote-instance to a mention-instance if:
      1) they belong to the SAME context window (same ctx_id),
      2) the mention's absolute span overlaps the quote's absolute span
         padded by K tokens on each side, AND
      3) the mention does NOT itself fall inside the quote's own span (i.e.
         it's a surrounding-context mention, not a mention that is part of
         the quoted text itself, e.g. "The darling" inside "The darling!").
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


def add_mention_character_edges(G: nx.Graph, mention_instances: List[Dict]):
    for m in mention_instances:
        G.add_edge(m["id"], f"C_{m['coref']}", type="mention-character")

import re 
def post_process(G) : 
    connecteds = list(nx.connected_components(G) )
    it =0 
    for subG in connecteds : 
        num_c = len([i for i in subG if i.startswith('CTX')])
        num_q = len([i for i in subG if any([i.startswith('Q'), i.startswith('VIRT')])])
        num_m = len([i for i in subG if i.startswith('M')])
        if not all([num_c>0, num_q>0, num_m>0]) : 
            if num_m > 0 : 
                m = list(subG)[0]
                ctxt_id = re.findall('CTX_[\d]+', m )[0]
                cids = [i for i in subG if i.startswith('M')]
                G.add_node(f'VIRTUAL_{1000+it}')
                G.add_edge(f'VIRTUAL_{1000+it}', ctxt_id, type='quote-mention')

                for j, ms in enumerate(cids) : 
                    # ms = mentions_by_ctx[ms]
                    # for m in ms : 
                    G.add_edge(f'VIRTUAL_{1000+it}', ms, type='quote-mention')
                it += 1
            # G.remove_nodes_from(subG)
        #     for m in subG.nodes : 
        #         ctxt_id = re.findall('CTX_[\d]+', m )
        #         break
# --------------------------------------------------------------------------- #
# Pipeline for a single book
# --------------------------------------------------------------------------- #

def build_book_graph(
    inp_dir: str, book_id: str, N: int, S: int, K: int, aligner: WordAligner
) -> nx.Graph:
    tokens = load_tokens(inp_dir, book_id)
    entities = load_entities(inp_dir, book_id)
    quotes_df = load_quotes(inp_dir, book_id)

    contexts = build_context_nodes(tokens, N, S, aligner)
    quote_instances = build_quote_instances(quotes_df, contexts, tokens)
    mention_instances = build_mention_instances(entities, contexts)
    characters = build_global_character_nodes(mention_instances)

    G = nx.Graph()
    for c in contexts:
        G.add_node(
            c["id"], node_type="context",
            start=c["start"], end=c["end"],
            input_ids=c["input_ids"], text=c["text"],
        )
    for q in quote_instances:
        G.add_node(q["id"], node_type="quote", **{k: v for k, v in q.items() if k != "id"})
    quotes_by_ctx = {}
    for m in quote_instances:
        quotes_by_ctx.setdefault(m["ctx_id"], []).append(m)
    ctxt_to_keep = list(quotes_by_ctx.keys())

    num_virtual = 0 
    for m in mention_instances:
        if m['ctx_id'] in ctxt_to_keep : 
            G.add_node(m["id"], node_type="mention", **{k: v for k, v in m.items() if k != "id"})
        else : 
            G.add_node(m["id"], node_type="mention", **{k: v for k, v in m.items() if k != "id"})
            # if 'VIRTUAL_{num_virtual}' not in G.nodes : 
            #     G.add_node(f'VIRTUAL_{num_virtual}', node_type='virtual')
            #     num_virtual += 1
            
    # for ch in characters:
    #     G.add_node(ch["id"], node_type="character", **{k: v for k, v in ch.items() if k != "id"})

    add_context_quote_edges(G, quote_instances)
    # add_context_mention_edges(G, mention_instances)
    add_quote_mention_edges(G, quote_instances, mention_instances, K)
    add_mention_mention_edges(G, mention_instances, ctxt_to_keep)
    # remove_isolated_chunks(G, quote_instances)
    # all_ctxt = [v for v in G.nodes if v.startswith('CTX')]
    # to_remove = set(all_ctxt) - set(ctxt_to_keep) #[i for i in a]
    # G.remove_nodes_from(to_remove)
    
    post_process(G)

    # add_mention_character_edges(G, mention_instances)
    G.graph["book_id"] = book_id
    G.graph["params"] = {"N": N, "S": S, "K": K, "tokenizer": aligner.tokenizer.name_or_path}
    return G


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Convert BookNLP output (.tokens/.entities/.quotes) into a hierarchical, "
                     "ModernBERT-tokenized graph for self-supervised learning."
    )
    parser.add_argument("--inp_dir", required=True, help="Directory with BookNLP output files")
    parser.add_argument("--out_dir", required=True, help="Directory to write the resulting graph(s) to")
    parser.add_argument("--N", type=int, default=200, help="Context node length, in BookNLP tokens")
    parser.add_argument("--S", type=int, default=100, help="Stride between consecutive context nodes")
    parser.add_argument("--K", type=int, default=50, help="Quote-Mention window size, in tokens (fwd/bwd)")
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-large",
                         help="HF model name/path used to tokenize context nodes")
    parser.add_argument("--book_id", default=None, help="Process only this book_id (default: all found)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    book_ids = [args.book_id] if args.book_id else find_books(args.inp_dir)
    if not book_ids:
        raise RuntimeError(f"No *.tokens files found in {args.inp_dir}")

    print(f"[+] Loading tokenizer '{args.model_name}' ...")
    aligner = WordAligner(args.model_name)

    for book_id in book_ids:
        print(f"[+] Building graph for '{book_id}' ...")
        G = build_book_graph(args.inp_dir, book_id, args.N, args.S, args.K, aligner)
        n_ctx = sum(1 for _, d in G.nodes(data=True) if d["node_type"] == "context")
        n_q = sum(1 for _, d in G.nodes(data=True) if d["node_type"] == "quote")
        n_m = sum(1 for _, d in G.nodes(data=True) if d["node_type"] == "mention")
        n_c = sum(1 for _, d in G.nodes(data=True) if d["node_type"] == "character")
        print(f"    -> {G.number_of_nodes()} nodes "
              f"(ctx={n_ctx}, quote_inst={n_q}, mention_inst={n_m}, char={n_c}) "
              f"/ {G.number_of_edges()} edges")

        pkl_path = os.path.join(args.out_dir, f"{book_id}.graph.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(G, f)

        json_path = os.path.join(args.out_dir, f"{book_id}.graph.json")
        with open(json_path, "w") as f:
            json.dump(nx.node_link_data(G), f)

        print(f"    -> saved to {pkl_path} and {json_path}")


if __name__ == "__main__":
    main()
