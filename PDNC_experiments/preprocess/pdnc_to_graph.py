#!/usr/bin/env python3
"""
pdnc_to_graph.py
====================

Convert PDNC (Project Dialogism Novel Corpus) data, layered on top of
BookNLP's own tokenization, into a hierarchical graph, ready to seed node
representations from a ModernBERT model for self-supervised learning:

    Context  --contains-->  Quote-in-context
    Context  --contains-->  Mention-in-context
    Quote-in-context  --near (K tokens)-->  Mention-in-context   (same context
                                              only, excluding mentions that
                                              fall INSIDE the quote itself)

Data sources per book:
  - `<book_id>.tokens`: BookNLP's own tokenization (word/paragraph ids, and
    crucially byte_onset/byte_offset), used to build Context nodes exactly
    as before. PDNC's own files don't provide their own tokenization, only
    character-level (byte) offsets over the same underlying raw text.
  - `<book_id>.quotes.csv`: PDNC's gold quote table (qID, qText, qSpan,
    speaker, startByte, endByte, ...), replacing BookNLP's own `.quotes`.
  - `<book_id>.mentions_used.csv`: PDNC's deduplicated, resolved mention set
    (source, iden, text, startByte, endByte, pdncID, mID, ...), replacing
    BookNLP's own `.entities`.

Both PDNC tables locate things via character (byte) offsets into the raw
text, not BookNLP token ids; `ByteIndex` maps those offsets to BookNLP token
ids using the tokens table's own byte_onset/byte_offset columns (with a
nearest-token fallback for slightly fuzzy annotation boundaries).

Because context windows overlap (stride S < N), a quote/mention that falls
inside several context windows is DUPLICATED, once per containing context
window. Each duplicate gets its own scalar (st, et): the start/end subword
token position of that quote/mention *within that particular context's own
ModernBERT tokenization*.

Note: Global Character nodes / Mention-Character edges are intentionally
NOT built here; `pdncID` is kept as a plain mention node attribute only.

Usage
-----
    python pdnc_to_graph.py \
        --inp_dir /path/to/data \
        --out_dir /path/to/graphs \
        --N 200 --S 100 --K 50 \
        --model_name answerdotai/ModernBERT-large

`inp_dir` can hold the data for ONE book or MANY books; the script
auto-discovers every `book_id` present (via `*.tokens` files), unless you
pass --book_id explicitly.

Output
------
For every book: `<out_dir>/<book_id>.graph.pkl` (a pickled networkx.Graph)
and `<out_dir>/<book_id>.graph.json` (node-link JSON, human-inspectable).
"""

import argparse
import ast
import glob
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
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
    return pd.read_csv(path, sep="\t", quoting=3, keep_default_na=False, dtype=str)


def load_tokens(inp_dir: str, book_id: str) -> pd.DataFrame:
    df = _read_tsv(os.path.join(inp_dir, f"{book_id}.tokens"))
    df["token_ID_within_document"] = df["token_ID_within_document"].astype(int)
    # Needed by ByteIndex to map the PDNC quote table's character-level
    # startByte/endByte offsets back to BookNLP token ids.
    df["byte_onset"] = df["byte_onset"].astype(int)
    df["byte_offset"] = df["byte_offset"].astype(int)
    return df


def load_mentions_used(inp_dir: str, book_id: str) -> pd.DataFrame:
    """
    Load the PDNC "mentions_used" table for `book_id`: `<book_id>.mentions_used.csv`,
    the deduplicated, resolved mention set PDNC uses downstream (combining
    BookNLP-coref-based mentions and explicit name-matched mentions).
    Columns: source (exp/booknlp), idColName, iden, text, startByte,
    endByte, pdncID, paraID, chapID, mID.

    Like the PDNC quote table, mentions are located via character-level
    offsets into the raw book text (startByte/endByte), not BookNLP token
    ids -- mapped to BookNLP token ids downstream via `ByteIndex`.
    """
    path = os.path.join(inp_dir, f"mentions_used.csv")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["startByte"] = df["startByte"].astype(int)
    df["endByte"] = df["endByte"].astype(int)
    return df


_PDNC_LIST_COLUMNS = ["qSpan", "addressee", "menTexts", "menSpans", "menEnts"]


def _parse_list_literal(x):
    """
    Parse a column value that may be a Python-literal string (e.g.
    "[6693, 6705]" or "['May Wellend']") into an actual Python object.
    Leaves already-parsed values (or genuinely unparsable strings)
    untouched.
    """
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


def load_quotes(inp_dir: str, book_id: str) -> pd.DataFrame:
    """
    Load the PDNC-style quote table for `book_id`: `<book_id>.quotes.csv`,
    one row per quote, with columns such as qID, qText, qSpan, speaker,
    addressee, qType, refExp, menTexts, menSpans, menEnts, speakerGender,
    dialogueTurn, speakerType, novel, startParaID, endParaID, startChapID,
    endChapID, speakerID, startByte, endByte.

    Unlike BookNLP's own `.quotes` file, this table locates quotes via
    character-level offsets into the raw book text (`startByte`/`endByte`,
    redundantly mirrored in `qSpan`), not BookNLP token ids -- those offsets
    are mapped to BookNLP token ids downstream via `ByteIndex`.
    """
    path = os.path.join(inp_dir, f"quote_info.csv")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    for col in _PDNC_LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_list_literal)

    df["startByte"] = df["startByte"].astype(int)
    df["endByte"] = df["endByte"].astype(int)
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
# Byte-offset -> BookNLP token-id resolution (for the PDNC quote table)
# --------------------------------------------------------------------------- #

class ByteIndex:
    """
    Maps raw-text character offsets (as used by the PDNC quote table's
    startByte/endByte, a.k.a. qSpan) to BookNLP token ids, using the tokens
    table's own byte_onset/byte_offset columns.

    Primary strategy: the resolved BookNLP token range spans every token
    whose own [byte_onset, byte_offset) span overlaps the queried
    [start_byte, end_byte) span. PDNC's annotation boundaries are sometimes
    slightly off relative to BookNLP's own tokenization (e.g. punctuation or
    quotation marks included/excluded differently), so if no token overlaps
    at all, each boundary is instead resolved to its nearest token
    independently.
    """

    def __init__(self, tokens: pd.DataFrame):
        df = tokens.sort_values("token_ID_within_document")
        self.token_ids = df["token_ID_within_document"].to_numpy()
        self.onsets = df["byte_onset"].to_numpy()
        self.offsets = df["byte_offset"].to_numpy()

    def span_to_tokens(self, start_byte: int, end_byte: int) -> Optional[Tuple[int, int]]:
        if len(self.token_ids) == 0:
            return None

        overlap = np.where((self.onsets < end_byte) & (self.offsets > start_byte))[0]
        if len(overlap) > 0:
            return int(self.token_ids[overlap.min()]), int(self.token_ids[overlap.max()])

        # Fuzzy fallback: no token exactly overlaps; resolve each boundary to
        # its nearest token independently, then take the enclosing range.
        start_idx = self._nearest_token_idx(start_byte)
        end_idx = self._nearest_token_idx(end_byte)
        lo, hi = min(start_idx, end_idx), max(start_idx, end_idx)
        return int(self.token_ids[lo]), int(self.token_ids[hi])

    def _nearest_token_idx(self, pos: int) -> int:
        i = int(np.searchsorted(self.onsets, pos, side="right")) - 1
        i = max(0, min(i, len(self.onsets) - 1))
        candidates = {i, min(i + 1, len(self.onsets) - 1), max(i - 1, 0)}
        return min(
            candidates,
            key=lambda k: min(abs(int(self.onsets[k]) - pos), abs(int(self.offsets[k]) - pos)),
        )


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


def _extend_to_sentence_boundary(
    end: int,
    total_tokens: int,
    sentence_by_id: Dict[int, str],
    in_quote_by_id: Dict[int, bool],
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
        is_in_quote = in_quote_by_id.get(t, False)
        if is_sentence_end and not is_in_quote:
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
    in_quote_by_id = dict(zip(
        tokens["token_ID_within_document"], tokens["inQuote"].apply(_to_bool)
    ))

    contexts = []
    start, idx = 0, 0
    par_id = 0  # persists across context windows, like the paragraph id truly does in the book
    while start < total_tokens:
        end = min(start + N - 1, total_tokens - 1)
        end = _extend_to_sentence_boundary(end, total_tokens, sentence_by_id, in_quote_by_id)

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


def build_quote_instances(quotes: pd.DataFrame, contexts: List[Dict], byte_index: "ByteIndex") -> List[Dict]:
    """
    One instance per (quote, containing-context) pair. Node id: Q_{qID}_{ctx_id}.

    Each PDNC quote row is first localized in BookNLP's own token stream via
    `byte_index.span_to_tokens(startByte, endByte)` (character offsets ->
    BookNLP token id range), then handled exactly as before: attached to
    every context window that contains (or, as a fallback, overlaps) that
    token range.
    """
    instances = []
    n_unmapped = 0
    n_fallback = 0
    n_not_found = 0
    for i, row in quotes.iterrows():
        qid = row["qID"]
        token_span = byte_index.span_to_tokens(row["startByte"], row["endByte"])
        if token_span is None:
            n_unmapped += 1
            print(f"    [!] quote {qid} ({row['startByte']}-{row['endByte']}) could not be mapped to any BookNLP token, skipping")
            continue
        q_start, q_end = token_span

        ctxs, clipped = _containing_or_overlapping_contexts(contexts, q_start, q_end)
        if not ctxs:
            print(f"    [!] quote {qid} (tokens {q_start}-{q_end}) matches no context window at all, skipping")
            continue
        if clipped:
            n_fallback += 1
        for ctx in ctxs:
            span = _resolve_span(ctx, q_start, q_end, clipped)
            if span is None:
                n_not_found += 1
                continue
            st, et = span
            instances.append({
                "id": f"Q_{qid}_{ctx['id']}",
                "quote_idx": qid,
                "ctx_id": ctx["id"],
                "start": q_start,
                "end": q_end,
                "st": st,
                "et": et,
                "text": row.get("qText", ""),
                "speaker": row.get("speaker", ""),
                "char_id": row.get("speakerID", None),
                "speaker_gender": row.get("speakerGender", ""),
                "speaker_type": row.get("speakerType", ""),
                "addressee": row.get("addressee", []),
                "quote_type": row.get("qType", ""),
                "ref_exp": row.get("refExp", ""),
                "dialogue_turn": row.get("dialogueTurn", None),
                "mention_texts": row.get("menTexts", []),
                "mention_spans": row.get("menSpans", []),
                "mention_entities": row.get("menEnts", []),
                "start_para_id": row.get("startParaID", None),
                "end_para_id": row.get("endParaID", None),
                "start_chap_id": row.get("startChapID", None),
                "end_chap_id": row.get("endChapID", None),
            })
    if n_unmapped:
        print(f"    [!] {n_unmapped} quote(s) could not be mapped to any BookNLP token at all, skipped")
    if n_fallback:
        print(f"    [!] {n_fallback} quote(s) longer than a context window; used clipped overlap fallback")
    if n_not_found:
        print(f"    [!] {n_not_found} quote-context pair(s) had no overlapping subword tokens, skipped")
    return instances


def build_mention_instances(
    mentions: pd.DataFrame, contexts: List[Dict], byte_index: "ByteIndex"
) -> List[Dict]:
    """
    One instance per (PDNC mention, containing-context) pair. Node id: M_{mID}_{ctx_id}.

    Mentions now come from PDNC's own `mentions_used.csv` (deduplicated,
    combining BookNLP-coref-based and explicit name-matched mentions) rather
    than BookNLP's raw `.entities` file. Each row is first localized in
    BookNLP's own token stream via `byte_index.span_to_tokens(startByte,
    endByte)` (character offsets -> BookNLP token id range), then handled
    exactly like before: attached to every context window that contains
    (or, as a fallback, overlaps) that token range.

    Note: we deliberately do NOT build Global Character nodes / Mention-
    Character edges from `pdncID` here; it is kept only as a node attribute.
    """
    instances = []
    n_unmapped = 0
    n_fallback = 0
    n_not_found = 0
    for i, row in mentions.iterrows():
        mid = row["mID"]
        token_span = byte_index.span_to_tokens(row["startByte"], row["endByte"])
        if token_span is None:
            n_unmapped += 1
            print(f"    [!] mention {mid} ({row['startByte']}-{row['endByte']}) could not be mapped to any BookNLP token, skipping")
            continue
        m_start, m_end = token_span

        ctxs, clipped = _containing_or_overlapping_contexts(contexts, m_start, m_end)
        if not ctxs:
            print(f"    [!] mention {mid} (tokens {m_start}-{m_end}) matches no context window at all, skipping")
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
                "id": f"M_{mid}_{ctx['id']}",
                "mention_idx": mid,
                "ctx_id": ctx["id"],
                "start": m_start,
                "end": m_end,
                "st": st,
                "et": et,
                "text": row.get("text", ""),
                "source": row.get("source", ""),
                "id_col_name": row.get("idColName", ""),
                "iden": row.get("iden", ""),
                "char_id": row.get("pdncID", None),
                "para_id": row.get("paraID", None),
                "chap_id": row.get("chapID", None),
                "mention_type": row.get("mention_type", None),
            })
    if n_unmapped:
        print(f"    [!] {n_unmapped} mention(s) could not be mapped to any BookNLP token at all, skipped")
    if n_fallback:
        print(f"    [!] {n_fallback} mention(s) longer than a context window; used clipped overlap fallback")
    if n_not_found:
        print(f"    [!] {n_not_found} mention-context pair(s) had no overlapping subword tokens, skipped")
    return instances


# --------------------------------------------------------------------------- #
# Edge builders
# --------------------------------------------------------------------------- #
#
# NOTE: Global Character nodes / Mention-Character edges are intentionally
# NOT built in this PDNC-data pipeline. `pdncID` is kept as a plain mention
# node attribute (see build_mention_instances) in case you want to add that
# back later, but no character nodes/edges are constructed from it here.

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
                G.add_node(f'VIRTUAL_{10000+it}')
                G.add_edge(f'VIRTUAL_{10000+it}', ctxt_id, type='quote-mention')

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
    mentions_df = load_mentions_used(inp_dir, book_id)
    quotes_df = load_quotes(inp_dir, book_id)

    contexts = build_context_nodes(tokens, N, S, aligner)
    byte_index = ByteIndex(tokens)
    quote_instances = build_quote_instances(quotes_df, contexts, byte_index)
    mention_instances = build_mention_instances(mentions_df, contexts, byte_index)

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
            
    add_context_quote_edges(G, quote_instances)
    # add_context_mention_edges(G, mention_instances)
    add_quote_mention_edges(G, quote_instances, mention_instances, K)
    add_mention_mention_edges(G, mention_instances, ctxt_to_keep)
    # remove_isolated_chunks(G, quote_instances)
    # all_ctxt = [v for v in G.nodes if v.startswith('CTX')]
    # to_remove = set(all_ctxt) - set(ctxt_to_keep) #[i for i in a]
    # G.remove_nodes_from(to_remove)
    
    post_process(G)

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
    parser.add_argument("--inp_dir", required=True,
                         help="Directory with <book_id>.tokens (BookNLP), <book_id>.quotes.csv "
                              "and <book_id>.mentions_used.csv (PDNC)")
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