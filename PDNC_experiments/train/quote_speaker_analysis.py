"""
quote_speaker_analysis.py
==========================

Implements the two evidence-gathering analyses described for testing:

  H1 - joint-scoring quote representations encode speaker identity better
       than direct-scoring representations
                                        -> Section 2: RSA-style comparison
  H2 - attention is (part of) the mechanism carrying that information
                                        -> Section 3: attention-based analysis

Both sections are written against the actual `Baseline` class in
`model.py` (the joint-scoring model, trained with the graph-wide `SALoss`,
i.e. L_joint), which represents a quotation q_i by the concatenation of its
start/end token embeddings:

    x = torch.cat((H[st[0], st[1]], H[et[0], et[1]]), dim=-1)     # [num_nodes, 2H]
    h_q = x[g.node_types == 1]                                    # quote nodes

and represents a mention m_j the same way, filtered with `node_types == 2`.
This module reuses exactly that logic instead of re-deriving it, so the
"only difference is the loss/formalization, not the context available" when
we compare against a direct-scoring model trained on the same window size T.

--------------------------------------------------------------------------
Interface assumed for BOTH the joint model (`Baseline`) and the
direct-scoring model you compare it to
--------------------------------------------------------------------------
`model_adapter.get_quote_and_mention_reps(g, input_ids, att_mask, st, et)`
    -> per-window list of dicts, each with:
         "h_q"            : FloatTensor [k_i, 2H]   quote representations
         "h_m"            : FloatTensor [n_i, 2H]   mention representations
         "quote_spans"    : list[(batch_idx, start_pos, end_pos)] len k_i
         "mention_spans"  : list[(batch_idx, start_pos, end_pos)] len n_i

`Baseline` already exposes everything needed for this through
`_get_input_embeddings` + `g.node_types/is_pred/batch_q/batch_m`, so
`BaselineAdapter` below is a thin wrapper. A direct-scoring model that
predicts speakers independently per (quote, mention) pair, but still keeps
`st`/`et` token indices for a quote's start/end tokens and mention's
start/end tokens, can reuse `generic_quote_mention_reps` directly.

You must additionally supply, for every window, the *gold* speaker id for
each quote and each mention (not defined in model.py). Pass these in as a
callback `speaker_id_fn(g, window_idx) -> (quote_speaker_ids, mention_speaker_ids)`
so this module never guesses your graph's attribute names.

--------------------------------------------------------------------------
Direct-scoring model: one quote per forward vs. Baseline's one window per
forward
--------------------------------------------------------------------------
Unlike `Baseline`, a direct-scoring model may forward a single quote at a
time (one `h_q` and one speaker id per call), with only `st`/`et` (shape
[1, 2]) changing across the k calls for a given window C, while input_ids
stay identical. Since `st`/`et` never feed back into the encoder (`h_q` is
just `torch.cat((H[start], H[end]))`, computed from H *after* the encoder
forward), H - and hence attention - is identical across those k calls.
Practically this means:
  - You do NOT need to call the direct model k times and stitch results
    back together. Use `extract_window_reps_shared_encoder` to run its
    `.bert` submodule ONCE per window and pool every quote's/mention's own
    span directly; this reproduces exactly what k separate calls would
    give you, at 1/k the cost, and yields a single attention matrix per
    window valid for every quote-to-quote / quote-to-mention comparison
    (instead of k identical, redundant ones).
  - If some future direct-scoring architecture *does* condition the
    encoder itself on which quote is the current target (so the above
    equivalence breaks), fall back to `merge_single_quote_windows`, which
    groups genuinely separate single-quote extraction results sharing the
    same window id back into one multi-quote `WindowReps`.

--------------------------------------------------------------------------
IMPORTANT: attention extraction and FlashAttention-2
--------------------------------------------------------------------------
`Baseline.bert` is built with `attn_implementation='flash_attention_2'`
(model.py, ~line 292-295 and the Longformer branch does not set this but
Longformer's own local+global attention only exposes attention for the
"global" tokens unless `output_attentions=True` AND the model falls back to
its eager path). FlashAttention-2 never materializes attention weights, so
`output_attentions=True` will just return `None`s. Before running section
3, re-instantiate / reload the backbone with `attn_implementation='eager'`
(same checkpoint, only this kwarg changes) -- see `to_eager_attention`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - torch not available in this sandbox
    torch = None
    F = None


def _no_grad_decorator(fn):
    """`@torch.no_grad()` that degrades to a no-op when torch isn't installed,
    so this module can still be imported (e.g. to run the pure-numpy
    self-test / statistics layer) in environments without torch."""
    if torch is None:
        return fn
    return torch.no_grad()(fn)

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None


# ==========================================================================
# 0. Small shared containers
# ==========================================================================

@dataclass
class WindowReps:
    """Everything needed from one context window C for both analyses."""
    h_q: "torch.Tensor"                       # [k, 2H] quote reps (already pooled)
    h_m: "torch.Tensor"                       # [n, 2H] mention reps
    quote_spans: List[Tuple[int, int, int]]    # (batch_idx, start_pos, end_pos) per quote
    mention_spans: List[Tuple[int, int, int]]  # (batch_idx, start_pos, end_pos) per mention
    quote_speaker_ids: Sequence[int]           # gold speaker id per quote, len k
    mention_speaker_ids: Sequence[int]         # speaker id each mention refers to, len n
    batch_idx: int                             # which element of the padded batch this window is
    quote_ids: Optional[List[List[str]]]
    quote_types: Optional[Sequence[str]] = None  # e.g. 'Explicit'/'Anaphoric'/'Implicit' per quote, len k
    predicted_m: Optional[Sequence[Optional[int]]] = None  # index into h_m/mention_spans of the
                                                             # mention predicted for each quote, len k
                                                             # (an entry may be None if that quote had
                                                             # no predicted mention, e.g. no candidates)
    acc: Optional[Sequence[bool]] = None  # whether predicted_m[i]'s speaker matches quote i's gold
                                           # speaker, len k - i.e. this model's own per-quote
                                           # correctness for the predicted-mention analysis


@dataclass
class SeparationResult:
    """Output of comparing a same-speaker vs different-speaker distribution."""
    n_same: int
    n_diff: int
    mean_same: float
    mean_diff: float
    std_same: float
    std_diff: float
    cohens_d: float          # (mean_same - mean_diff) / pooled_std ; >0 means same > diff, as predicted
    mwu_stat: float
    mwu_pvalue: float
    same_values: np.ndarray = field(repr=False)
    diff_values: np.ndarray = field(repr=False)


# ==========================================================================
# 1. Extracting h_q / h_m for a batch, using Baseline's own pooling logic
# ==========================================================================

class BaselineAdapter:
    """
    Wraps a `Baseline` (see model.py) instance so it exposes the shared
    `get_window_reps` interface used by both analyses. Works unchanged for
    any other model that keeps the same `_get_input_embeddings` contract
    (concatenated start/end token embeddings, `node_types` in {0,1,2} for
    context/quote/mention, `is_pred` boolean mask per quote, `batch_q` /
    `batch_m` window ids per node).
    """

    def __init__(self, model, score=False):
        self.model = model
        self.score = score
    @_no_grad_decorator
    def get_window_reps(
        self,
        g,
        input_ids: "torch.Tensor",
        att_mask: "torch.Tensor",
        st: Tuple["torch.Tensor", "torch.Tensor"],
        et: Tuple["torch.Tensor", "torch.Tensor"],
        speaker_id_fn: Callable[[object, int], Tuple[Sequence[int], Sequence[int]]],
        quote_type_fn: Optional[Callable[[object, int], Sequence[str]]] = None,
    ) -> List[WindowReps]:
        self.model.eval()
        # x: [num_nodes, 2H] -- identical tensor Baseline.forward scores from
        x = self.model._get_input_embeddings(g, input_ids, att_mask, st, et)

        quote_mask = g.node_types == 1
        mention_mask = g.node_types == 2
        quote_x_all = x[quote_mask]
        mention_x_all = x[mention_mask]

        # st/et are aligned with node order (see model.py _get_input_embeddings),
        # so we can pull start/end (batch_idx, pos) per node the same way.
        q_start_b, q_start_p = st[0][quote_mask], st[1][quote_mask]
        q_end_b, q_end_p = et[0][quote_mask], et[1][quote_mask]
        m_start_b, m_start_p = st[0][mention_mask], st[1][mention_mask]
        m_end_b, m_end_p = et[0][mention_mask], et[1][mention_mask]

        windows = []
        embs = []
        n_windows = int(g.batch_q.max().item()) + 1
        for bs in range(n_windows):
            q_win_mask = g.batch_q == bs
            pred_mask = g.is_pred[bs]  # keep exactly Baseline's own filtering
            m_win_mask = g.batch_m == bs

            h_q = quote_x_all[q_win_mask][pred_mask]
            h_m = mention_x_all[m_win_mask]

            quote_spans = list(zip(
                q_start_b[q_win_mask][pred_mask].tolist(),
                q_start_p[q_win_mask][pred_mask].tolist(),
                q_end_p[q_win_mask][pred_mask].tolist(),
            ))
            mention_spans = list(zip(
                m_start_b[m_win_mask].tolist(),
                m_start_p[m_win_mask].tolist(),
                m_end_p[m_win_mask].tolist(),
            ))

            book_ids = g.quote_ids[0][0].split('_')[0]
            q_speakers = [f'{book_ids}_{sid}' for sid in g.speakers[g.batch_q ==bs][g.is_pred[bs]]]
            m_speakers =  [f'{book_ids}_{sid}' for sid in g.corefs[g.batch_m == bs]]
            assert len(q_speakers) == h_q.size(0), (
                f"speaker_id_fn returned {len(q_speakers)} quote labels but "
                f"window {bs} has {h_q.size(0)} predicted quotes"
            )

            quote_types = None
            if quote_type_fn is not None:
                quote_types = quote_type_fn(g, bs)
                assert len(quote_types) == h_q.size(0), (
                    f"quote_type_fn returned {len(quote_types)} quote types but "
                    f"window {bs} has {h_q.size(0)} predicted quotes"
                )

            if self.score : 
                qx_rep = h_q.repeat_interleave(h_m.size(0), dim=0)   # [q*m, H]
                # qx_rep = qx.repeat_interleave(pooled_mx.size(0), dim=0)   # [q*m, H]
    
                # tile mx q times: [m0,m1,...,mM, m0,m1,...,mM, ...]
                # pooled_mx_rep = pooled_mx.repeat(qx.size(0), 1)                  # [q*m, H]
                mx_rep = h_m.repeat(h_q.size(0), 1)                  # [q*m, H]
                out = torch.cat([qx_rep, mx_rep], dim=-1) 
                # ground_truth = 
                embs.append(out)
                
            batch_idx = quote_spans[0][0] if quote_spans else bs
            windows.append(WindowReps(
                h_q=h_q.cpu(), h_m=h_m.cpu(),
                quote_spans=quote_spans, mention_spans=mention_spans,
                quote_speaker_ids=q_speakers, mention_speaker_ids=m_speakers,
                batch_idx=batch_idx,
                quote_ids=g.quote_ids[bs]
            ))
        if self.score : 
            embs = torch.cat(embs)
            scores = self.model.proj(embs).squeeze(-1)
    
            # size_per_batch.append(len(pooled_mx))
            out = self.model._rearange(g, scores)
            for cnt, o in enumerate(out) : 
                windows[cnt].predicted_m = o.argmax(1)
        return windows


def extract_window_reps_shared_encoder(
    bert_module,
    input_ids: "torch.Tensor",      # [S] token ids for ONE window C (no batch dim needed)
    att_mask: "torch.Tensor",       # [S]
    quote_spans: Sequence[Tuple[int, int]],     # [(start_pos, end_pos), ...] len k, in this window's tokenization
    mention_spans: Sequence[Tuple[int, int]],   # [(start_pos, end_pos), ...] len n
    quote_speaker_ids: Sequence[int],
    mention_speaker_ids: Sequence[int],
    window_id: int = 0,
    quote_types: Optional[Sequence[str]] = None,
) -> WindowReps:
    """
    Model-agnostic extraction, usable for BOTH the joint (`Baseline`) and a
    direct-scoring model that only ever forwards one quote at a time.

    Why this is valid for the direct model too: within a window C its
    input_ids are identical regardless of which single quote is currently
    being scored (confirmed - only `st`/`et` change per forward, shape
    [1, 2] each time, selecting which quote's tokens get pooled). Since
    `st`/`et` never feed back into the encoder itself (`Baseline`'s own
    `h_q = torch.cat((H[st], H[et]))` computes H first, independently of
    which node is selected), H - and therefore attention - is identical
    across all k of the direct model's per-quote forwards for that window.

    So rather than calling the direct model k times and stitching results
    back together, just run its `.bert` submodule ONCE per window and pool
    every quote's/mention's own (start, end) span yourself, exactly the way
    `Baseline._get_input_embeddings` already does it
    (`torch.cat((H[start], H[end]), dim=-1)`, no extra projection). This is
    ~k times cheaper, gives you a single attention matrix per window valid
    for every quote-to-quote/quote-to-mention comparison (instead of k
    redundant, identical ones), and produces a `WindowReps` with all k
    quotes at once - directly comparable to what `BaselineAdapter` builds
    for the joint model, with no merging step required.

    If your direct model's actual `forward()` does something
    quote-conditioned upstream of the encoder (so this equivalence would
    NOT hold), don't use this function - use `merge_single_quote_windows`
    below instead, and run the real forward k times.
    """
    ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
    mask = att_mask.unsqueeze(0) if att_mask.dim() == 1 else att_mask
    with _inference_ctx():
        H = bert_module(input_ids=ids, attention_mask=mask).last_hidden_state[0]  # [S, hidden]

    def pool(span: Tuple[int, int]) -> "torch.Tensor":
        s, e = span
        return torch.cat((H[s], H[e]), dim=-1)

    h_q = torch.stack([pool(sp) for sp in quote_spans]) if quote_spans else torch.empty(0, H.size(-1) * 2)
    h_m = torch.stack([pool(sp) for sp in mention_spans]) if mention_spans else torch.empty(0, H.size(-1) * 2)

    return WindowReps(
        h_q=h_q, h_m=h_m,
        quote_spans=[(window_id, s, e) for s, e in quote_spans],
        mention_spans=[(window_id, s, e) for s, e in mention_spans],
        quote_speaker_ids=quote_speaker_ids, mention_speaker_ids=mention_speaker_ids,
        batch_idx=window_id, quote_types=quote_types,
    )


def _inference_ctx():
    return torch.inference_mode() if torch is not None else _NullCtx()


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def merge_single_quote_windows(
    single_quote_records: Iterable[dict],
    window_id_fn: Callable[[dict], int],
) -> List[WindowReps]:
    """
    Fallback for when you genuinely must call the direct model once per
    quote (e.g. its forward is quote-conditioned upstream of the encoder,
    so `extract_window_reps_shared_encoder` would not be valid). Groups
    per-quote extraction results that belong to the same window C back
    into a single multi-quote `WindowReps`, so the rest of the pipeline
    (`pairwise_same_diff_similarities`, attention-mass functions, etc.)
    works unchanged.

    Each item of `single_quote_records` is a dict with the single quote's
    own data:
        {"h_q": Tensor[1, 2H] or [2H], "h_m": Tensor[n, 2H],
         "quote_span": (batch_idx, start, end), "mention_spans": [...],
         "quote_speaker_id": int, "mention_speaker_ids": [...]}
    `window_id_fn(record) -> int` tells us which window C a record
    belongs to (e.g. a window/chunk id you already track in your data
    pipeline).

    NOTE: mention reps/spans are expected to be identical across every
    record of the same window (same window C => same mentions); this
    function takes them from the first record seen per window and asserts
    they line up, rather than concatenating duplicates.
    """
    by_window: dict = {}
    for rec in single_quote_records:
        wid = window_id_fn(rec)
        by_window.setdefault(wid, []).append(rec)

    windows = []
    for wid, recs in by_window.items():
        h_q = torch.cat([
            (r["h_q"] if r["h_q"].dim() == 2 else r["h_q"].unsqueeze(0)) for r in recs
        ], dim=0)
        quote_spans = [r["quote_span"] for r in recs]
        quote_speaker_ids = [r["quote_speaker_id"] for r in recs]

        first = recs[0]
        h_m = first["h_m"]
        mention_spans = first["mention_spans"]
        mention_speaker_ids = first["mention_speaker_ids"]
        for r in recs[1:]:
            assert r["mention_spans"] == mention_spans, (
                f"window {wid}: mention spans differ across per-quote records; "
                "these should be identical if they truly share window C"
            )

        windows.append(WindowReps(
            h_q=h_q, h_m=h_m,
            quote_spans=quote_spans, mention_spans=mention_spans,
            quote_speaker_ids=quote_speaker_ids, mention_speaker_ids=mention_speaker_ids,
            batch_idx=wid,
        ))
    return windows

def generic_quote_mention_reps(model, g, input_ids, att_mask, st, et, speaker_id_fn, score=False):
    """
    Same extraction, expressed without assuming the exact `Baseline` class,
    for a direct-scoring model as long as it (a) has a `_get_input_embeddings`
    method with the same signature/semantics, or (b) directly exposes
    `h_q`/`h_m` some other way -- in that case just build `WindowReps`
    objects yourself instead of calling this function.
    """
    return BaselineAdapter(model, score=score).get_window_reps(g, input_ids, att_mask, st, et, speaker_id_fn)



def filter_windows_by_quote_type(
    windows: Iterable[WindowReps],
    quote_type: str,
) -> List[WindowReps]:
    """
    Returns a new list of `WindowReps`, one per input window that contains
    at least one quote of `quote_type`, keeping only the quotes matching
    that type (h_q, quote_spans, quote_speaker_ids, quote_types all
    subsetted accordingly) while leaving mentions (h_m, mention_spans,
    mention_speaker_ids) untouched.

    This is the building block for running the whole analysis "per type of
    quote" (e.g. 'Explicit'/'Anaphoric'/'Implicit'): for quote-quote RSA
    (`run_rsa_comparison`), filtering both the joint and direct window
    lists to one type means every remaining pair is between two quotes of
    that same type, so the resulting Cohen's d tells you the separation
    specifically *among* e.g. Anaphoric quotes. For quote-mention RSA
    (`run_rsa_mention_comparison`), only the quote side is filtered, so you
    get "how well do Explicit quotes' representations line up with their
    speaker's mentions", with the full mention set still available to
    pair against.

    Windows require `quote_types` to have been populated at extraction
    time (see `quote_type_fn` on `BaselineAdapter.get_window_reps` /
    `extract_window_reps_shared_encoder`). A window with no quotes of the
    requested type is dropped entirely (nothing to pair within/against).
    """
    filtered = []
    for w in windows:
        if w.quote_types is None:
            raise ValueError(
                "WindowReps.quote_types is not set; pass quote_type_fn (or "
                "quote_types=...) when building windows before filtering "
                "by quote type."
            )
        keep_idx = [i for i, t in enumerate(w.quote_types) if t == quote_type]
        if not keep_idx:
            continue
        filtered.append(WindowReps(
            h_q=w.h_q[keep_idx],
            h_m=w.h_m,
            quote_spans=[w.quote_spans[i] for i in keep_idx],
            mention_spans=w.mention_spans,
            quote_speaker_ids=[w.quote_speaker_ids[i] for i in keep_idx],
            mention_speaker_ids=w.mention_speaker_ids,
            batch_idx=w.batch_idx,
            quote_types=[w.quote_types[i] for i in keep_idx],
            quote_ids=[w.quote_ids[i] for i in keep_idx],
            # predicted_m/acc are per-quote, same as quote_speaker_ids/quote_types,
            # so they need the same keep_idx subsetting to stay aligned with h_q -
            # without this, run_predicted_mention_rsa_by_bin(..., quote_type=...)
            # would silently pair quotes with the wrong predicted_m/acc entries.
            predicted_m=[w.predicted_m[i] for i in keep_idx] if w.predicted_m is not None else None,
            acc=[w.acc[i] for i in keep_idx] if w.acc is not None else None,
        ))
    return filtered


# ==========================================================================
# 2. Representation Similarity Analysis (Section 2 / H1)
# ==========================================================================

def pairwise_same_diff_similarities(
    window: WindowReps,
    max_token_distance: Optional[int] = None,
    min_token_distance: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a single window, compute cosine similarity for every unordered pair
    of quotations, split into same-speaker vs different-speaker using gold
    labels. Returns (same_sims, diff_sims) as 1-D numpy arrays.

    `max_token_distance`/`min_token_distance`: if set, only keep quote
    pairs whose spans are within `[min_token_distance, max_token_distance]`
    tokens of each other (see `_span_token_distance`, 0 = overlapping;
    either bound may be `None` for "no constraint on that side"). Pass
    both `None` (default) to keep every pair in the window, matching the
    original unfiltered behaviour. Pass only `max_token_distance` to
    reproduce the original single cutoff. Pass both to isolate a binned
    range, e.g. `min_token_distance=50, max_token_distance=100` keeps only
    quote pairs 50-100 tokens apart. Useful for the same reason as on the
    quote-mention side: two quotations at opposite ends of a long window
    are a weak test of local same-speaker binding, and dilute the
    separation with mostly irrelevant far-away pairs.
    """
    h_q = window.h_q
    k = h_q.size(0)
    if k < 2:
        return np.array([]), np.array([])

    h_norm = F.normalize(h_q, dim=-1)
    sim_matrix = (h_norm @ h_norm.T).detach().cpu().numpy()  # [k, k]

    speakers = np.asarray(window.quote_speaker_ids)
    same_vals, diff_vals = [], []
    for i in range(k):
        for j in range(i + 1, k):
            if min_token_distance is not None or max_token_distance is not None:
                dist = _span_token_distance(window.quote_spans[i], window.quote_spans[j])
                if not _distance_in_range(dist, min_token_distance, max_token_distance):
                    continue
            sim = sim_matrix[i, j]
            if speakers[i] == speakers[j]:
                same_vals.append(sim)
            else:
                diff_vals.append(sim)
    return np.asarray(same_vals), np.asarray(diff_vals)


def collect_similarities(
    windows: Iterable[WindowReps],
    max_token_distance: Optional[int] = None,
    min_token_distance: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pool same-/different-speaker quote-quote similarities across an entire
    test set, optionally restricted to quote pairs within
    `[min_token_distance, max_token_distance]` tokens of each other (see
    `pairwise_same_diff_similarities`).
    """
    same_all, diff_all = [], []
    for w in windows:
        same, diff = pairwise_same_diff_similarities(
            w, max_token_distance=max_token_distance, min_token_distance=min_token_distance
        )
        same_all.append(same)
        diff_all.append(diff)
    return np.concatenate(same_all) if same_all else np.array([]), \
        np.concatenate(diff_all) if diff_all else np.array([])


def _span_token_distance(span_a: Tuple[int, int, int], span_b: Tuple[int, int, int]) -> float:
    """
    Token distance between two spans, both given as
    (batch_idx, start_pos, end_pos) - the same format as
    `WindowReps.quote_spans` / `mention_spans`.

    - Returns `math.inf` if the spans belong to different windows/batch
      indices (so cross-window pairs are automatically excluded by any
      `max_token_distance` filter, which only makes sense within one
      window anyway).
    - Returns 0 if the spans overlap.
    - Otherwise returns the number of tokens between their closest edges.
    """
    batch_a, start_a, end_a = span_a
    batch_b, start_b, end_b = span_b
    if batch_a != batch_b:
        return math.inf
    if end_a < start_b:
        return start_b - end_a
    if end_b < start_a:
        return start_a - end_b
    return 0.0  # overlapping spans


def _distance_in_range(
    dist: float,
    min_token_distance: Optional[int],
    max_token_distance: Optional[int],
) -> bool:
    """
    Shared range check used by both the quote-quote and quote-mention
    pairwise functions. `min_token_distance`/`max_token_distance` are each
    optional and inclusive, so passing only `max_token_distance` reproduces
    the original "within N tokens" cutoff (equivalent to a range of
    [0, N]), while passing both lets you isolate a binned range like
    [50, 100] (tokens 50 to 100 away, excluding closer or farther pairs).
    `None` on either bound means "no constraint on that side".
    """
    if min_token_distance is not None and dist < min_token_distance:
        return False
    if max_token_distance is not None and dist > max_token_distance:
        return False
    return True


def pairwise_same_diff_mention_similarities(
    window: WindowReps,
    max_token_distance: Optional[int] = None,
    min_token_distance: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a single window, compute cosine similarity between every
    (quote, mention) pair, split into same-speaker vs. different-speaker
    using gold labels. Mirrors `pairwise_same_diff_similarities` but for
    quote-mention pairs.

    `max_token_distance`/`min_token_distance`: if set, only keep pairs
    whose quote/mention spans are within `[min_token_distance,
    max_token_distance]` tokens of each other (using
    `_span_token_distance`, 0 = overlapping; either bound may be `None`
    for "no constraint on that side"). Pass both `None` (default) to keep
    every pair in the window, matching the original unfiltered behaviour.
    Pass only `max_token_distance` to reproduce the original single cutoff.
    Pass both to isolate a binned range, e.g. `min_token_distance=50,
    max_token_distance=100` keeps only pairs 50-100 tokens apart.
    """
    h_q = window.h_q
    h_m = window.h_m
    k = h_q.size(0)
    m = h_m.size(0)
    if k < 1 or m < 1:
        return np.array([]), np.array([])

    h_norm = F.normalize(h_q, dim=-1)
    m_norm = F.normalize(h_m, dim=-1)
    sim_matrix = (h_norm @ m_norm.T).detach().cpu().numpy()  # [k, m]

    speakers = np.asarray(window.quote_speaker_ids)
    mention_speakers = np.asarray(window.mention_speaker_ids)

    same_vals, diff_vals = [], []
    for i in range(k):
        for j in range(m):
            if min_token_distance is not None or max_token_distance is not None:
                dist = _span_token_distance(window.quote_spans[i], window.mention_spans[j])
                if not _distance_in_range(dist, min_token_distance, max_token_distance):
                    continue
            sim = sim_matrix[i, j]
            if speakers[i] == mention_speakers[j]:
                same_vals.append(sim)
            else:
                diff_vals.append(sim)
    return np.asarray(same_vals), np.asarray(diff_vals)


def collect_mention_similarities(
    windows: Iterable[WindowReps],
    max_token_distance: Optional[int] = None,
    min_token_distance: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pool same-/different-speaker quote-mention similarities across an
    entire test set, optionally restricted to mentions within
    `[min_token_distance, max_token_distance]` tokens of the quote (see
    `pairwise_same_diff_mention_similarities`).
    """
    same_all, diff_all = [], []
    for w in windows:
        same, diff = pairwise_same_diff_mention_similarities(
            w, max_token_distance=max_token_distance, min_token_distance=min_token_distance
        )
        same_all.append(same)
        diff_all.append(diff)
    return np.concatenate(same_all) if same_all else np.array([]), \
        np.concatenate(diff_all) if diff_all else np.array([])


def effect_size_and_test(same_values: np.ndarray, diff_values: np.ndarray) -> SeparationResult:
    """
    Cohen's d (pooled-SD, same - diff, so positive & large == good separation
    in the predicted direction) and a one-sided Mann-Whitney U test for
    same > diff.
    """
    if len(same_values) < 2 or len(diff_values) < 2:
        raise ValueError(
            f"Need at least 2 values per group to compute Cohen's d / Mann-Whitney U "
            f"(got n_same={len(same_values)}, n_diff={len(diff_values)}). "
            f"If you're using max_token_distance, it may be filtering out too much - "
            f"try a larger distance or pool across more windows."
        )
    n1, n2 = len(same_values), len(diff_values)
    m1, m2 = same_values.mean(), diff_values.mean()
    s1, s2 = same_values.std(ddof=1), diff_values.std(ddof=1)
    pooled_std = math.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    cohens_d = (m1 - m2) / pooled_std if pooled_std > 0 else float("nan")

    if scipy_stats is None:
        raise ImportError("scipy is required for the Mann-Whitney U test")
    mwu = scipy_stats.mannwhitneyu(same_values, diff_values, alternative="greater")

    return SeparationResult(
        n_same=n1, n_diff=n2,
        mean_same=float(m1), mean_diff=float(m2),
        std_same=float(s1), std_diff=float(s2),
        cohens_d=float(cohens_d),
        mwu_stat=float(mwu.statistic), mwu_pvalue=float(mwu.pvalue),
        same_values=same_values, diff_values=diff_values,
    )


def run_rsa_comparison(
    joint_windows: Iterable[WindowReps],
    direct_windows: Iterable[WindowReps],
    control_windows: Optional[Iterable[WindowReps]] = None,
    max_token_distance: Optional[int] = None,
    min_token_distance: Optional[int] = None,
) -> Tuple[SeparationResult, SeparationResult, Optional[SeparationResult]]:
    """
    High-level entry point for Section 2. Pass in the `WindowReps` extracted
    from the joint model (Baseline / L_joint) and from a direct-scoring
    model trained on the same window size T, over the *same* test-set
    windows (so pairing / gold labels line up).

    `control_windows`: optional third model to compare against (e.g. a
    control/ablation variant), extracted over the same test-set windows.
    Leave `None` (default) to compare only joint vs. direct, as before.

    `max_token_distance`/`min_token_distance`: restrict to quote pairs
    within `[min_token_distance, max_token_distance]` tokens of each other
    (see `_span_token_distance` / `pairwise_same_diff_similarities`). Pass
    only `max_token_distance` for the original single cutoff, or both to
    isolate a binned range (e.g. `min_token_distance=50,
    max_token_distance=100`). Leave both `None` for the original
    unfiltered, whole-window behaviour.

    Returns `(joint_result, direct_result, control_result)`. `control_result`
    is `None` when `control_windows` isn't passed. The prediction to report
    is `joint_result.cohens_d > direct_result.cohens_d` (and, if used,
    `> control_result.cohens_d`) and the `mwu_pvalue`s (ideally the joint
    one much smaller / more significant).
    """
    same_j, diff_j = collect_similarities(
        joint_windows, max_token_distance=max_token_distance, min_token_distance=min_token_distance
    )
    same_d, diff_d = collect_similarities(
        direct_windows, max_token_distance=max_token_distance, min_token_distance=min_token_distance
    )
    control_res = None
    if control_windows is not None:
        same_c, diff_c = collect_similarities(
            control_windows, max_token_distance=max_token_distance, min_token_distance=min_token_distance
        )
        control_res = effect_size_and_test(same_c, diff_c)
    return effect_size_and_test(same_j, diff_j), effect_size_and_test(same_d, diff_d), control_res


def run_rsa_mention_comparison(
    joint_windows: Iterable[WindowReps],
    direct_windows: Iterable[WindowReps],
    control_windows: Optional[Iterable[WindowReps]] = None,
    max_token_distance: Optional[int] = None,
    min_token_distance: Optional[int] = None,
) -> Tuple[SeparationResult, SeparationResult, Optional[SeparationResult]]:
    """
    Quote-mention counterpart of `run_rsa_comparison`. Pass in the
    `WindowReps` extracted from the joint model and from the direct-scoring
    model, over the *same* test-set windows.

    `control_windows`: optional third model to compare against, same
    semantics as in `run_rsa_comparison`. Leave `None` (default) to compare
    only joint vs. direct, as before.

    `max_token_distance`/`min_token_distance`: restrict to (quote, mention)
    pairs within `[min_token_distance, max_token_distance]` tokens of each
    other (see `_span_token_distance` /
    `pairwise_same_diff_mention_similarities`). Useful because a mention
    arbitrarily far away in a long window is a weak test of local
    quote-speaker binding, and dilutes the same-/different-speaker
    separation with mostly-irrelevant far-away pairs. Pass only
    `max_token_distance` for the original single cutoff, or both bounds to
    isolate a binned range. Leave both `None` to keep the original
    unfiltered, whole-window behaviour.

    Returns `(joint_result, direct_result, control_result)`, same semantics
    as `run_rsa_comparison`.
    """
    same_j, diff_j = collect_mention_similarities(
        joint_windows, max_token_distance=max_token_distance, min_token_distance=min_token_distance
    )
    same_d, diff_d = collect_mention_similarities(
        direct_windows, max_token_distance=max_token_distance, min_token_distance=min_token_distance
    )
    control_res = None
    if control_windows is not None:
        same_c, diff_c = collect_mention_similarities(
            control_windows, max_token_distance=max_token_distance, min_token_distance=min_token_distance
        )
        control_res = effect_size_and_test(same_c, diff_c)
    return effect_size_and_test(same_j, diff_j), effect_size_and_test(same_d, diff_d), control_res


def _quote_key(window: WindowReps, i: int) -> str:
    """
    Stable, dict-friendly key for quote `i` of `window`, used to key the
    per-quote separability dictionaries below. `WindowReps.quote_ids[i]`
    may itself be a list of raw ids (see the field's docstring on
    `WindowReps`), so a list/tuple is joined into a single `"id1|id2"`
    string rather than used as-is (lists aren't hashable). Falls back to
    a synthetic `"<batch_idx>_<quote_index>"` key when `quote_ids` wasn't
    populated at extraction time, so this never raises even if you built
    `WindowReps` by hand without ids.
    """
    if window.quote_ids is None:
        return f"{window.batch_idx}_{i}"
    raw = window.quote_ids[i]
    if isinstance(raw, (list, tuple)):
        return "|".join(str(x) for x in raw)
    return str(raw)


def per_quote_qq_separability(
    window: WindowReps,
    max_token_distance: Optional[int] = None,
    min_token_distance: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """
    Per-quote refinement of `pairwise_same_diff_similarities`: instead of
    pooling every same-/different-speaker quote pair in the window into
    one flat pair of arrays, this keeps one score PER QUOTE.

    For each quote `i` in `window`, the score is:

        mean(sim(i, j) for j same speaker as i) -
        mean(sim(i, j) for j different speaker from i)

    i.e. how much closer (in cosine similarity, on `h_q`) quote `i` sits
    to the OTHER quotes sharing its gold speaker than to quotes from
    different speakers, on average. Positive => good separation for that
    quote; near zero or negative => that quote's representation doesn't
    distinguish its speaker from others.

    A quote's score is `None` if it has no eligible same-speaker partner,
    no eligible different-speaker partner (after the optional distance
    filter), or if it's the only quote in the window - there's nothing to
    compare it against in one of the two groups.

    `max_token_distance`/`min_token_distance`: same semantics as
    `pairwise_same_diff_similarities` - restrict comparisons to quote `j`s
    within `[min_token_distance, max_token_distance]` tokens of quote `i`
    (see `_span_token_distance`/`_distance_in_range`). Leave both `None`
    (default) to compare against every other quote in the window.

    Returns `{quote_id: score_or_None}`, one entry per quote in `window`,
    keyed via `_quote_key`.
    """
    h_q = window.h_q
    k = h_q.size(0)
    scores: Dict[str, Optional[float]] = {}
    if k == 0:
        return scores
    if k < 2:
        return {_quote_key(window, 0): None}

    h_norm = F.normalize(h_q, dim=-1)
    sim_matrix = (h_norm @ h_norm.T).detach().cpu().numpy()  # [k, k]
    speakers = np.asarray(window.quote_speaker_ids)

    for i in range(k):
        same_vals, diff_vals = [], []
        for j in range(k):
            if i == j:
                continue
            if min_token_distance is not None or max_token_distance is not None:
                dist = _span_token_distance(window.quote_spans[i], window.quote_spans[j])
                if not _distance_in_range(dist, min_token_distance, max_token_distance):
                    continue
            sim = sim_matrix[i, j]
            if speakers[j] == speakers[i]:
                same_vals.append(sim)
            else:
                diff_vals.append(sim)
        key = _quote_key(window, i)
        if not same_vals or not diff_vals:
            scores[key] = None
        else:
            scores[key] = float(np.mean(same_vals) - np.mean(diff_vals))
    return scores


def per_quote_qm_separability(
    window: WindowReps,
    max_token_distance: Optional[int] = None,
    min_token_distance: Optional[int] = None,
) -> Dict[str, Optional[float]]:
    """
    Quote-mention counterpart of `per_quote_qq_separability`: mirrors
    `pairwise_same_diff_mention_similarities`, but keeps one score PER
    QUOTE instead of pooling into flat same-/different-speaker arrays.

    For each quote `i` in `window`, the score is:

        mean(sim(quote_i, mention_j) for mentions j referring to i's speaker) -
        mean(sim(quote_i, mention_j) for mentions j referring to a different speaker)

    computed on `h_q` vs. `h_m`. Positive => quote `i`'s representation
    sits closer to mentions of its own (gold) speaker than to mentions of
    other characters, on average.

    A quote's score is `None` if there are no mentions of its own speaker
    anywhere in the window, no mentions of any other speaker (after the
    optional distance filter), or the window has no mentions at all.

    `max_token_distance`/`min_token_distance`: same semantics as
    `pairwise_same_diff_mention_similarities` - restrict to mentions
    within `[min_token_distance, max_token_distance]` tokens of quote `i`
    (see `_span_token_distance`/`_distance_in_range`). Leave both `None`
    (default) to compare against every mention in the window.

    Returns `{quote_id: score_or_None}`, one entry per quote in `window`,
    keyed via `_quote_key` (the same keys `per_quote_qq_separability`
    produces for the same window, so the two can be zipped/merged
    directly - see `per_quote_separability`).
    """
    h_q, h_m = window.h_q, window.h_m
    k, m = h_q.size(0), h_m.size(0)
    scores: Dict[str, Optional[float]] = {_quote_key(window, i): None for i in range(k)}
    if k == 0 or m == 0:
        return scores

    h_norm = F.normalize(h_q, dim=-1)
    m_norm = F.normalize(h_m, dim=-1)
    sim_matrix = (h_norm @ m_norm.T).detach().cpu().numpy()  # [k, m]

    speakers = np.asarray(window.quote_speaker_ids)
    mention_speakers = np.asarray(window.mention_speaker_ids)

    for i in range(k):
        same_vals, diff_vals = [], []
        for j in range(m):
            if min_token_distance is not None or max_token_distance is not None:
                dist = _span_token_distance(window.quote_spans[i], window.mention_spans[j])
                if not _distance_in_range(dist, min_token_distance, max_token_distance):
                    continue
            sim = sim_matrix[i, j]
            if mention_speakers[j] == speakers[i]:
                same_vals.append(sim)
            else:
                diff_vals.append(sim)
        key = _quote_key(window, i)
        if not same_vals or not diff_vals:
            scores[key] = None
        else:
            scores[key] = float(np.mean(same_vals) - np.mean(diff_vals))
    return scores


def per_quote_separability(
    windows: Iterable[WindowReps],
    max_token_distance: Optional[int] = None,
    min_token_distance: Optional[int] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    High-level, per-quote entry point mirroring `run_rsa_comparison`/
    `run_rsa_mention_comparison`, but WITHOUT pooling across quotes into a
    single Cohen's d: this keeps one quote-quote and one quote-mention
    separability score per individual quote, pooled across every window
    in `windows`, so you can inspect, sort, or plot the distribution of
    per-quote separability rather than only a single aggregate statistic
    per model/condition.

    For every quote across `windows`, computes:
      - `"quote_quote"`: `per_quote_qq_separability`'s score for that
        quote - its average similarity to other quotes sharing its gold
        speaker minus its average similarity to quotes with a different
        gold speaker.
      - `"quote_mention"`: `per_quote_qm_separability`'s score for that
        quote - the same idea, against mentions instead of other quotes.

    Either value is `None` when there weren't enough same-/different-
    speaker partners of that type to compare against for that quote (see
    the two per-window functions for the exact conditions); it is left as
    `None` rather than dropped, so every quote you passed in still gets an
    entry.

    `max_token_distance`/`min_token_distance` are forwarded unchanged to
    both `per_quote_qq_separability` and `per_quote_qm_separability` (see
    `_span_token_distance`/`_distance_in_range`); pass both `None`
    (default) to compare each quote against every other quote/mention in
    its window regardless of distance.

    Returns `{quote_id: {"quote_quote": float or None, "quote_mention": float or None}}`.
    Quote ids come from `WindowReps.quote_ids` via `_quote_key`; if the
    same id appears in more than one window in `windows`, the later
    window's entry overwrites the earlier one in the returned dict, so
    make sure ids are unique across the windows you pass in if that
    matters for your use case (e.g. don't mix train/dev/test windows that
    happen to reuse ids).
    """
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for w in windows:
        qq_scores = per_quote_qq_separability(
            w, max_token_distance=max_token_distance, min_token_distance=min_token_distance
        )
        qm_scores = per_quote_qm_separability(
            w, max_token_distance=max_token_distance, min_token_distance=min_token_distance
        )
        for key, qq in qq_scores.items():
            result[key] = {
                "quote_quote": qq,
                "quote_mention": qm_scores.get(key),
            }
    return result


def pairwise_same_diff_mention_mention_similarities(
    window: WindowReps,
    max_token_distance: Optional[int] = None,
    min_token_distance: Optional[int] = None,
    negative_k: Optional[int] = None,
    _rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a single window, compute cosine similarity for every unordered pair
    of MENTIONS (not quotes), split into same-speaker vs different-speaker
    using gold labels (i.e. whether the two mentions refer to the same
    underlying speaker). Mirrors `pairwise_same_diff_similarities`
    (quote-quote) exactly, but on `h_m`/`mention_spans`/
    `mention_speaker_ids` instead of the quote-side fields.

    `max_token_distance`/`min_token_distance`: if set, only keep mention
    pairs whose spans are within `[min_token_distance, max_token_distance]`
    tokens of each other (see `_span_token_distance`, 0 = overlapping;
    either bound may be `None` for "no constraint on that side"). Pass
    both `None` (default) to keep every pair in the window. Pass only
    `max_token_distance` for a single cutoff, or both to isolate a binned
    range, e.g. `min_token_distance=50, max_token_distance=100`.

    `negative_k`: if set, instead of keeping every eligible different-
    speaker pair as an unordered pair, negatives are collected per ANCHOR
    mention (every mention in the window takes a turn as anchor against
    every other mention passing the distance filter), grouped by the
    character the candidate mention refers to, and randomly sampled down
    to at most `negative_k` per unique different character (see
    `_sample_k_per_character`; without replacement, a character with fewer
    than `negative_k` eligible candidates for that anchor keeps all of
    them). This prevents one frequently-mentioned background character
    from dominating the negative pool just because it has many mentions in
    the window. Because every mention takes a turn as anchor, an unordered
    different-speaker pair `(i, j)` can be sampled from both directions
    (once with `i` as anchor, once with `j` as anchor) - this mirrors the
    intentional double-counting semantics used elsewhere in this module
    (e.g. `collect_nearest_mention_mention_similarities_by_bin`) rather
    than deduplicating into a single unordered-pair count. Same-speaker
    pairs are unaffected and still counted once per unordered pair, as
    without `negative_k`. Pass `None` (default) to keep the original
    unsampled, unordered-pair behaviour for negatives too.

    `_rng`: internal - a `np.random.Generator` to draw samples from when
    `negative_k` is set. Callers normally don't need to pass this
    directly; `collect_mention_mention_similarities` creates one `rng` per
    call (from its own `random_seed` argument) and reuses it across all
    windows so the sampling stream isn't reset per window. If left `None`
    while `negative_k` is set, a fresh default-seeded generator is created
    (not reproducible across separate calls).
    """
    h_m = window.h_m
    m = h_m.size(0)
    if m < 2:
        return np.array([]), np.array([])

    m_norm = F.normalize(h_m, dim=-1)
    sim_matrix = (m_norm @ m_norm.T).detach().cpu().numpy()  # [m, m]

    mention_speakers = np.asarray(window.mention_speaker_ids)
    same_vals: List[float] = []
    diff_vals: List[float] = []

    if negative_k is None:
        for i in range(m):
            for j in range(i + 1, m):
                if min_token_distance is not None or max_token_distance is not None:
                    dist = _span_token_distance(window.mention_spans[i], window.mention_spans[j])
                    if not _distance_in_range(dist, min_token_distance, max_token_distance):
                        continue
                sim = sim_matrix[i, j]
                if mention_speakers[i] == mention_speakers[j]:
                    same_vals.append(sim)
                else:
                    diff_vals.append(sim)
        return np.asarray(same_vals), np.asarray(diff_vals)

    rng = _rng if _rng is not None else np.random.default_rng()

    # Same-speaker pairs: unchanged, one unordered pair each.
    for i in range(m):
        for j in range(i + 1, m):
            if mention_speakers[i] != mention_speakers[j]:
                continue
            if min_token_distance is not None or max_token_distance is not None:
                dist = _span_token_distance(window.mention_spans[i], window.mention_spans[j])
                if not _distance_in_range(dist, min_token_distance, max_token_distance):
                    continue
            same_vals.append(sim_matrix[i, j])

    # Different-speaker pairs: sampled per anchor, per distractor character.
    for i in range(m):
        eligible_j = []
        for j in range(m):
            if j == i or mention_speakers[j] == mention_speakers[i]:
                continue
            if min_token_distance is not None or max_token_distance is not None:
                dist = _span_token_distance(window.mention_spans[i], window.mention_spans[j])
                if not _distance_in_range(dist, min_token_distance, max_token_distance):
                    continue
            eligible_j.append(j)
        if not eligible_j:
            continue
        sampled_j = _sample_k_per_character(eligible_j, mention_speakers, negative_k, rng)
        for j in sampled_j:
            diff_vals.append(sim_matrix[i, j])

    return np.asarray(same_vals), np.asarray(diff_vals)


def collect_mention_mention_similarities(
    windows: Iterable[WindowReps],
    max_token_distance: Optional[int] = None,
    min_token_distance: Optional[int] = None,
    negative_k: Optional[int] = None,
    random_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pool same-/different-speaker mention-mention similarities across an
    entire test set, optionally restricted to mention pairs within
    `[min_token_distance, max_token_distance]` tokens of each other (see
    `pairwise_same_diff_mention_mention_similarities`).

    `negative_k`/`random_seed`: forwarded to
    `pairwise_same_diff_mention_mention_similarities` - if `negative_k` is
    set, different-speaker pairs are collected per anchor mention and
    randomly sampled down to at most `negative_k` per unique different
    character instead of keeping every eligible unordered pair (see that
    function's docstring for exact semantics, including the intentional
    double-counting of unordered pairs from both anchors' perspectives).
    A single `np.random.Generator` is created from `random_seed` here and
    reused across every window in `windows`, so the sampling stream isn't
    reset per window; pass `None` (default) to keep the original
    unsampled behaviour.
    """
    rng = np.random.default_rng(random_seed) if negative_k is not None else None
    same_all, diff_all = [], []
    for w in windows:
        same, diff = pairwise_same_diff_mention_mention_similarities(
            w,
            max_token_distance=max_token_distance,
            min_token_distance=min_token_distance,
            negative_k=negative_k,
            _rng=rng,
        )
        same_all.append(same)
        diff_all.append(diff)
    return np.concatenate(same_all) if same_all else np.array([]), \
        np.concatenate(diff_all) if diff_all else np.array([])


def run_rsa_mention_mention_comparison(
    joint_windows: Iterable[WindowReps],
    direct_windows: Iterable[WindowReps],
    control_windows: Optional[Iterable[WindowReps]] = None,
    max_token_distance: Optional[int] = None,
    min_token_distance: Optional[int] = None,
    negative_k: Optional[int] = None,
    random_seed: int = 0,
) -> Tuple[SeparationResult, SeparationResult, Optional[SeparationResult]]:
    """
    Mention-mention counterpart of `run_rsa_comparison`/
    `run_rsa_mention_comparison`: instead of quote-quote or quote-mention
    pairs, this compares pairs of MENTIONS to each other, testing whether
    mention representations (same `h_m` used elsewhere) separate
    same-speaker from different-speaker mentions, and whether that
    separation differs between the joint and direct-scoring models.

    Pass in the `WindowReps` extracted from the joint model and from the
    direct-scoring model, over the *same* test-set windows.

    `control_windows`: optional third model to compare against, same
    semantics as in `run_rsa_comparison`. Leave `None` (default) to compare
    only joint vs. direct, as before.

    `max_token_distance`/`min_token_distance`: restrict to mention pairs
    within `[min_token_distance, max_token_distance]` tokens of each other
    (see `_span_token_distance` /
    `pairwise_same_diff_mention_mention_similarities`). Pass only
    `max_token_distance` for a single cutoff, or both bounds to isolate a
    binned range (compatible with `sweep_cohens_d_over_distance_bins` /
    `sweep_cohens_d_over_distance_bins_by_quote_type` by passing this
    function as `comparison_fn` - note quote-type filtering via
    `filter_windows_by_quote_type` only touches the quote side of a
    window, so it has no effect here; mention-mention pairs aren't
    associated with a quote type). Leave both `None` to keep the original
    unfiltered, whole-window behaviour.

    `negative_k`/`random_seed`: forwarded to
    `collect_mention_mention_similarities` - if `negative_k` is set,
    different-speaker pairs are sampled per anchor mention, up to
    `negative_k` per unique distractor character, instead of keeping every
    eligible unordered different-speaker pair (see
    `pairwise_same_diff_mention_mention_similarities` for exact
    semantics). This guards against a single frequent background
    character dominating the negative distribution, the same concern the
    nearest-neighbour mention-mention analysis
    (`run_nearest_mention_mention_rsa_by_bin`) already addresses. Joint,
    direct, and control windows are all sampled independently (each gets
    its own generator seeded from `random_seed`); pass `None` (default) to
    keep every eligible pair, as before.

    Returns `(joint_result, direct_result, control_result)`, same
    semantics as `run_rsa_comparison`.
    """
    same_j, diff_j = collect_mention_mention_similarities(
        joint_windows,
        max_token_distance=max_token_distance,
        min_token_distance=min_token_distance,
        negative_k=negative_k,
        random_seed=random_seed,
    )
    same_d, diff_d = collect_mention_mention_similarities(
        direct_windows,
        max_token_distance=max_token_distance,
        min_token_distance=min_token_distance,
        negative_k=negative_k,
        random_seed=random_seed,
    )
    control_res = None
    if control_windows is not None:
        same_c, diff_c = collect_mention_mention_similarities(
            control_windows,
            max_token_distance=max_token_distance,
            min_token_distance=min_token_distance,
            negative_k=negative_k,
            random_seed=random_seed,
        )
        control_res = effect_size_and_test(same_c, diff_c)
    return effect_size_and_test(same_j, diff_j), effect_size_and_test(same_d, diff_d), control_res


def collect_nearest_mention_similarities_by_bin(
    windows: Iterable[WindowReps],
    bins: Sequence[Tuple[float, float]],
) -> Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Alternative pairing scheme for the quote-mention RSA analysis, designed
    to be run on a single quote type at a time (e.g. pre-filter with
    `filter_windows_by_quote_type(windows, 'Implicit')` before calling
    this, since Implicit quotes are the intended use case - there's no
    explicit local cue, so the interesting question is whether the
    representation still "finds" the right, possibly distant, speaker
    mention better than distractors).

    For each quote in each window:
      1. Find its NEAREST mention that refers to its gold speaker (same
         speaker id) - this is the one positive pair for that quote. If it
         has no same-speaker mention anywhere in the window, the quote
         contributes nothing (skipped).
      2. Assign that (quote, nearest same-speaker mention) pair to the
         first bin `(lo, hi)` in `bins` (checked in the given order) whose
         range contains the distance between them, inclusive of both ends
         (see `_span_token_distance` / `_distance_in_range`) - e.g. a
         distance of 180 tokens lands in bin `(0, 200)` if that's the
         first bin covering it. If the distance exceeds every bin's upper
         bound, the quote is skipped (no bin can hold it).
      3. Within that same bin, EVERY different-speaker mention strictly
         closer than the bin's upper bound (`distance < hi`) becomes a
         negative pair for that quote. This is deliberately asymmetric
         with the positive side: the negative pool is cumulative up to the
         bin's ceiling `hi` (not restricted to `>= lo`), so the question
         answered is "at the distance where the true speaker's mention was
         actually found, how does its similarity compare against every
         distractor at least as close", rather than "distractors at the
         same [lo, hi] range as the true mention".

    Returns `{bin: (positive_sims, negative_sims)}` for every bin in
    `bins` (as numpy arrays, possibly empty), the same shape
    `effect_size_and_test` expects. Feed straight into
    `run_nearest_mention_rsa_by_bin` rather than calling this directly in
    most cases.
    """
    same_by_bin: Dict[Tuple[float, float], List[float]] = {b: [] for b in bins}
    diff_by_bin: Dict[Tuple[float, float], List[float]] = {b: [] for b in bins}

    for w in windows:
        h_q, h_m = w.h_q, w.h_m
        k, m = h_q.size(0), h_m.size(0)
        if k < 1 or m < 1:
            continue

        h_norm = F.normalize(h_q, dim=-1)
        m_norm = F.normalize(h_m, dim=-1)
        sim_matrix = (h_norm @ m_norm.T).detach().cpu().numpy()  # [k, m]

        speakers = np.asarray(w.quote_speaker_ids)
        mention_speakers = np.asarray(w.mention_speaker_ids)

        for i in range(k):
            distances = np.array([
                _span_token_distance(w.quote_spans[i], w.mention_spans[j]) for j in range(m)
            ])
            same_mask = mention_speakers == speakers[i]
            if not same_mask.any():
                continue  # no same-speaker mention anywhere in this window

            same_idx = np.where(same_mask)[0]
            nearest_j = same_idx[np.argmin(distances[same_idx])]
            d_pos = distances[nearest_j]

            target_bin = None
            for lo, hi in bins:
                if _distance_in_range(d_pos, lo, hi):
                    target_bin = (lo, hi)
                    break
            if target_bin is None:
                continue  # nearest same-speaker mention is farther than every bin covers

            same_by_bin[target_bin].append(float(sim_matrix[i, nearest_j]))

            hi_bin = target_bin[1]
            for j in np.where(~same_mask)[0]:
                if distances[j] < hi_bin:
                    diff_by_bin[target_bin].append(float(sim_matrix[i, j]))

    return {b: (np.asarray(same_by_bin[b]), np.asarray(diff_by_bin[b])) for b in bins}


def run_nearest_mention_rsa_by_bin(
    joint_windows: Iterable[WindowReps],
    direct_windows: Iterable[WindowReps],
    bins: Sequence[Tuple[float, float]],
    control_windows: Optional[Iterable[WindowReps]] = None,
) -> BinnedSweepResult:
    """
    High-level entry point for the nearest-mention pairing scheme (see
    `collect_nearest_mention_similarities_by_bin`). Runs it independently
    on `joint_windows`, `direct_windows`, and (if given) `control_windows`
    (the positive-pair selection and bin assignment only depend on token
    spans/gold labels, which should be identical across all of them if
    they cover the same underlying quotes/mentions - only the resulting
    cosine similarities differ, since those come from each model's own
    `h_q`/`h_m`), then computes Cohen's d and Mann-Whitney U per bin for
    each.

    `control_windows`: optional third model's windows, run through the
    same nearest-mention pairing scheme. Leave `None` (default) to compare
    only joint vs. direct, as before - `control_cohens_d`/
    `control_pvalues`/`control_n` are then left empty on the returned
    result.

    Returns a `BinnedSweepResult`, the same structure produced by
    `sweep_cohens_d_over_distance_bins`, so it's a drop-in for
    `plot_cohens_d_by_bin` (pair-count bars included) - just note that
    `joint_n`/`direct_n`/`control_n` here count (1 positive, several
    negatives) per quote rather than "all pairs in range", so the bars
    mean "how many quotes had their nearest same-speaker mention land in
    this bin" plus "how many distractors that implies", not a plain
    pairwise count.

    Typical usage: pre-filter to one quote type first, e.g.
    `run_nearest_mention_rsa_by_bin(filter_windows_by_quote_type(joint_windows, "Implicit"),
    filter_windows_by_quote_type(direct_windows, "Implicit"), bins)`.
    """
    joint_by_bin = collect_nearest_mention_similarities_by_bin(joint_windows, bins)
    direct_by_bin = collect_nearest_mention_similarities_by_bin(direct_windows, bins)
    control_by_bin = (
        collect_nearest_mention_similarities_by_bin(control_windows, bins)
        if control_windows is not None else None
    )

    result = BinnedSweepResult()
    for lo, hi in bins:
        same_j, diff_j = joint_by_bin[(lo, hi)]
        same_d, diff_d = direct_by_bin[(lo, hi)]
        try:
            joint_res = effect_size_and_test(same_j, diff_j)
            direct_res = effect_size_and_test(same_d, diff_d)
            control_res = None
            if control_by_bin is not None:
                same_c, diff_c = control_by_bin[(lo, hi)]
                control_res = effect_size_and_test(same_c, diff_c)
        except ValueError as e:
            # Note: if control_windows is given, a bin is skipped entirely
            # (for all three models) if ANY of joint/direct/control lacks
            # enough data there, so the three resulting lists always stay
            # aligned to the same set of bins for plotting.
            print(f"[run_nearest_mention_rsa_by_bin] skipping bin [{lo}, {hi}]: {e}")
            continue
        result.bins.append((lo, hi))
        result.bin_labels.append(f"[{lo:g}, {hi:g}]")
        result.joint_cohens_d.append(joint_res.cohens_d)
        result.direct_cohens_d.append(direct_res.cohens_d)
        result.joint_pvalues.append(joint_res.mwu_pvalue)
        result.direct_pvalues.append(direct_res.mwu_pvalue)
        result.joint_n.append((joint_res.n_same, joint_res.n_diff))
        result.direct_n.append((direct_res.n_same, direct_res.n_diff))
        if control_res is not None:
            result.control_cohens_d.append(control_res.cohens_d)
            result.control_pvalues.append(control_res.mwu_pvalue)
            result.control_n.append((control_res.n_same, control_res.n_diff))
    return result


def _predicted_mention_distance(
    quote_span: Tuple[int, int, int],
    mention_span: Tuple[int, int, int],
) -> float:
    """
    Token distance between a quote and its PREDICTED mention, per the
    exact convention requested for the predicted-mention analysis: the
    minimum of `abs(quote_start - mention_end)` and
    `abs(mention_start - quote_end)`. This is deliberately NOT
    `_span_token_distance`'s "0 if overlapping, otherwise the gap on
    whichever side is valid" convention - here overlapping/out-of-order
    spans get a small (possibly nonzero) distance instead of collapsing
    to 0, and taking `abs(...)` on both sides means the result doesn't
    depend on which span comes first in the sequence.

    Returns `math.inf` if the two spans belong to different
    windows/batch indices (same convention as `_span_token_distance`).
    """
    batch_q, q_start, q_end = quote_span
    batch_m, m_start, m_end = mention_span
    if batch_q != batch_m:
        return math.inf
    return float(min(abs(q_start - m_end), abs(m_start - q_end)))


from tqdm import tqdm

def collect_predicted_mention_similarities_by_bin(
    windows: Iterable[WindowReps],
    bins: Sequence[Tuple[float, float]],
) -> Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    RSA-style pairing scheme built on the MODEL'S OWN predicted mention
    for each quote (`WindowReps.predicted_m`/`WindowReps.acc`), rather
    than the gold nearest same-speaker mention used by
    `collect_nearest_mention_similarities_by_bin`.

    For each quote `i` in each window (with `predicted_m[i]` set):
      1. Bin assignment: take `j_pred = predicted_m[i]`, the mention this
         model actually chose to attribute the quote to. Compute the
         distance between quote `i` and mention `j_pred` via
         `_predicted_mention_distance` (NOT `_span_token_distance` - see
         that function's docstring for how the convention differs), and
         assign quote `i` to the first bin `(lo, hi)` in `bins` (checked
         in the given order) whose range contains that distance,
         inclusive of both ends (see `_distance_in_range`). If the
         distance exceeds every bin's upper bound, the quote contributes
         nothing (skipped for both groups below).
      2. Positive group: `sim(quote_i, mention_{j_pred})` is added to that
         bin's "same" (positive) group UNCONDITIONALLY - i.e. regardless
         of whether the prediction is actually correct (`acc[i]`). This
         tests "how similar is a quote to whatever mention the model
         actually points to, at this distance", not "... to its correct
         mention" - a wrong prediction's similarity still counts as a
         positive here.
      3. Negative group: independently of step 2, for EVERY OTHER mention
         `j` in the same window (`j != j_pred`) that does NOT refer to
         quote `i`'s gold speaker (i.e. `mention_speaker_ids[j] !=
         quote_speaker_ids[i]`), compute its own distance to quote `i`
         (same `_predicted_mention_distance` formula) and, if that
         distance falls in the SAME bin `(lo, hi)` quote `i` was assigned
         to in step 1 (exact `[lo, hi]` range check via
         `_distance_in_range`, not a cumulative "closer than `hi`"
         threshold), add `sim(quote_i, mention_j)` to that bin's "diff"
         (negative) group. The predicted mention itself is excluded from
         this pool even if it turns out to be wrong-speaker (`acc[i]` is
         `False`) - it only ever contributes to the positive group, never
         double-counted as a negative too.

    This asks the same question `run_nearest_mention_rsa_by_bin` asks
    ("is quote-mention similarity part of the mechanism"), but restricted
    to whichever mention the model actually predicted (positives), tested
    against every plausible wrong-speaker distractor available at the same
    distance (negatives) - i.e. "does the mention the model actually
    picked tend to be more similar to the quote than the wrong-speaker
    alternatives it could have picked instead, at this distance", rather
    than "is a correct prediction more similar than an incorrect one".

    Quotes with `predicted_m[i] is None` (no predicted mention, e.g. no
    candidates in the window) are skipped entirely. Raises `ValueError`
    eagerly if a window is missing `predicted_m`/`acc` altogether, rather
    than silently skipping it, since that usually means an upstream data
    bug (see also `run_predicted_mention_rsa_by_bin`, which typically
    pre-filters windows with `filter_windows_by_quote_type` before this
    runs - that function has been updated to keep `predicted_m`/`acc`
    aligned with the filtered quotes).

    Returns `{bin: (same_sims, diff_sims)}` for every bin in `bins`, the
    same shape `effect_size_and_test` expects. Feed straight into
    `run_predicted_mention_rsa_by_bin` rather than calling this directly
    in most cases.
    """
    same_by_bin: Dict[Tuple[float, float], List[float]] = {b: [] for b in bins}
    diff_by_bin: Dict[Tuple[float, float], List[float]] = {b: [] for b in bins}

    for w in tqdm(windows):
        k = w.h_q.size(0)
        m = w.h_m.size(0)
        if k == 0:
            continue
        if w.predicted_m is None or w.acc is None:
            raise ValueError(
                "collect_predicted_mention_similarities_by_bin requires "
                "WindowReps.predicted_m and WindowReps.acc to be populated "
                "for every window; got a window with one or both unset."
            )
        assert len(w.predicted_m) == k and len(w.acc) == k, (
            f"predicted_m/acc must have one entry per quote (k={k}), got "
            f"len(predicted_m)={len(w.predicted_m)}, len(acc)={len(w.acc)}"
        )
        if m == 0:
            continue

        h_norm = F.normalize(w.h_q, dim=-1)
        m_norm = F.normalize(w.h_m, dim=-1)
        sim_matrix = (h_norm @ m_norm.T).detach().cpu().numpy()  # [k, m]

        quote_speakers = np.asarray(w.quote_speaker_ids)
        mention_speakers = np.asarray(w.mention_speaker_ids)

        for i in range(k):
            j_pred = w.predicted_m[i]
            if j_pred is None:
                continue  # this quote had no predicted mention

            dist_pred = _predicted_mention_distance(w.quote_spans[i], w.mention_spans[j_pred])
            target_bin = None
            for lo, hi in bins:
                if _distance_in_range(dist_pred, lo, hi):
                    target_bin = (lo, hi)
                    break
            if target_bin is None:
                continue  # farther than every bin covers

            # Positive: whatever mention the model predicted, regardless
            # of whether that prediction is actually correct.
            same_by_bin[target_bin].append(float(sim_matrix[i, j_pred]))

            # Negative: every OTHER, wrong-speaker mention that independently
            # falls in this same [lo, hi] bin relative to quote i.
            lo, hi = target_bin
            for j in range(m):
                if j == j_pred:
                    continue  # never double-count the predicted mention as a negative
                if mention_speakers[j] == quote_speakers[i]:
                    continue  # refers to the gold speaker, not a negative
                dist_j = _predicted_mention_distance(w.quote_spans[i], w.mention_spans[j])
                if _distance_in_range(dist_j, lo, hi):
                    diff_by_bin[target_bin].append(float(sim_matrix[i, j]))

    return {b: (np.asarray(same_by_bin[b]), np.asarray(diff_by_bin[b])) for b in bins}


def collect_predicted_mention_accuracy_by_bin(
    windows: Iterable[WindowReps],
    bins: Sequence[Tuple[float, float]],
) -> Dict[Tuple[float, float], Tuple[float, int]]:
    """
    Companion to `collect_predicted_mention_similarities_by_bin`: instead
    of pooling cosine similarities, pools raw prediction accuracy
    (`WindowReps.acc`) into the same distance bins, using the exact same
    `_predicted_mention_distance` binning rule, so the two line up
    bin-for-bin.

    Returns `{bin: (accuracy, n)}` where `accuracy` is `mean(acc)` over
    every quote whose predicted-mention distance fell in that bin (or
    `nan` if no quote fell in that bin) and `n` is the number of
    contributing quotes. Feed straight into
    `run_predicted_mention_rsa_by_bin` rather than calling this directly
    in most cases.
    """
    vals_by_bin: Dict[Tuple[float, float], List[bool]] = {b: [] for b in bins}
    for w in windows:
        k = w.h_q.size(0)
        if k == 0:
            continue
        if w.predicted_m is None or w.acc is None:
            raise ValueError(
                "collect_predicted_mention_accuracy_by_bin requires "
                "WindowReps.predicted_m and WindowReps.acc to be populated "
                "for every window; got a window with one or both unset."
            )
        for i in range(k):
            j = w.predicted_m[i]
            if j is None:
                continue
            dist = _predicted_mention_distance(w.quote_spans[i], w.mention_spans[j])
            for lo, hi in bins:
                if _distance_in_range(dist, lo, hi):
                    vals_by_bin[(lo, hi)].append(bool(w.acc[i]))
                    break
    return {
        b: (float(np.mean(v)) if v else float("nan"), len(v))
        for b, v in vals_by_bin.items()
    }


@dataclass
class PredictedMentionBinnedResult:
    """
    Per-distance-bin Cohen's d (correct vs. incorrect predicted-mention
    attributions) AND raw accuracy, for the joint model, the direct model,
    and optionally a control model - output of
    `run_predicted_mention_rsa_by_bin`. `quote_type` records which quote
    type (if any) this particular result was filtered to, so a caller
    building `{quote_type: PredictedMentionBinnedResult}` for
    `plot_predicted_mention_accuracy_and_cohens_d` doesn't need to track
    it separately.

    All `*_n`/`*_acc_n`/`*_cohens_d`/`*_accuracy`/`*_pvalues` lists stay
    aligned with `bins`/`bin_labels` - a bin is only appended to any of
    them if `effect_size_and_test` succeeded for every model that has
    data (see `run_predicted_mention_rsa_by_bin`), so index `i` always
    refers to the same bin across every field.
    """
    bins: List[Tuple[float, float]] = field(default_factory=list)
    bin_labels: List[str] = field(default_factory=list)
    quote_type: Optional[str] = None

    joint_cohens_d: List[float] = field(default_factory=list)
    direct_cohens_d: List[float] = field(default_factory=list)
    joint_pvalues: List[float] = field(default_factory=list)
    direct_pvalues: List[float] = field(default_factory=list)
    joint_n: List[Tuple[int, int]] = field(default_factory=list)   # (n_correct, n_incorrect) per bin
    direct_n: List[Tuple[int, int]] = field(default_factory=list)

    joint_accuracy: List[float] = field(default_factory=list)
    direct_accuracy: List[float] = field(default_factory=list)
    joint_acc_n: List[int] = field(default_factory=list)
    direct_acc_n: List[int] = field(default_factory=list)

    control_cohens_d: List[float] = field(default_factory=list)
    control_pvalues: List[float] = field(default_factory=list)
    control_n: List[Tuple[int, int]] = field(default_factory=list)
    control_accuracy: List[float] = field(default_factory=list)
    control_acc_n: List[int] = field(default_factory=list)


def run_predicted_mention_rsa_by_bin(
    joint_windows: Iterable[WindowReps],
    direct_windows: Iterable[WindowReps],
    bins: Sequence[Tuple[float, float]],
    control_windows: Optional[Iterable[WindowReps]] = None,
    quote_type: Optional[str] = None,
) -> PredictedMentionBinnedResult:
    """
    High-level entry point for Task 1: per-distance-bin Cohen's d
    (`collect_predicted_mention_similarities_by_bin`) AND raw accuracy
    (`collect_predicted_mention_accuracy_by_bin`), for the joint model,
    the direct-scoring model, and optionally a third control model, all
    binned by the token distance between each quote and the mention that
    model itself predicted for it (`_predicted_mention_distance`).

    `quote_type`: if given, every window list is first filtered down to
    quotes of that type via `filter_windows_by_quote_type` (which now
    also keeps `predicted_m`/`acc` aligned with the filtered quotes) - this
    is what makes the function "runnable per quote type": call it once per
    quote type (e.g. once each for `'Explicit'`, `'Anaphoric'`,
    `'Implicit'`) to get separate per-type curves, or leave it `None`
    (default) to pool every quote type together. `WindowReps.quote_types`
    must be populated if you pass this. The resulting `quote_type` is
    recorded on the returned `PredictedMentionBinnedResult`, so a caller
    can build `{quote_type: result}` for
    `plot_predicted_mention_accuracy_and_cohens_d` just by calling this
    once per type and keying the dict off `quote_type`.

    `control_windows`: optional third model to include, same semantics as
    elsewhere in this module (e.g. `run_nearest_mention_rsa_by_bin`).
    Leave `None` (default) to compare only joint vs. direct -
    `control_*` fields are then left empty on the returned result.

    A bin is skipped entirely (for every model) if `effect_size_and_test`
    raises `ValueError` for ANY model with data in that bin (e.g. fewer
    than 2 correct or 2 incorrect predicted-mention pairs there) - this
    keeps every list on `PredictedMentionBinnedResult` aligned to the same
    bins, at the cost of also dropping that bin's accuracy even though
    accuracy itself might have had enough samples; a warning is printed to
    stdout when this happens.

    Returns a `PredictedMentionBinnedResult`. Feed straight into
    `plot_predicted_mention_accuracy_and_cohens_d` (typically after
    collecting one result per quote type into a dict).
    """
    if quote_type is not None:
        joint_windows = filter_windows_by_quote_type(list(joint_windows), quote_type)
        direct_windows = filter_windows_by_quote_type(list(direct_windows), quote_type)
        if control_windows is not None:
            control_windows = filter_windows_by_quote_type(list(control_windows), quote_type)

    joint_sim_by_bin = collect_predicted_mention_similarities_by_bin(joint_windows, bins)
    direct_sim_by_bin = collect_predicted_mention_similarities_by_bin(direct_windows, bins)
    control_sim_by_bin = (
        collect_predicted_mention_similarities_by_bin(control_windows, bins)
        if control_windows is not None else None
    )

    joint_acc_by_bin = collect_predicted_mention_accuracy_by_bin(joint_windows, bins)
    direct_acc_by_bin = collect_predicted_mention_accuracy_by_bin(direct_windows, bins)
    control_acc_by_bin = (
        collect_predicted_mention_accuracy_by_bin(control_windows, bins)
        if control_windows is not None else None
    )

    result = PredictedMentionBinnedResult(quote_type=quote_type)
    for lo, hi in bins:
        same_j, diff_j = joint_sim_by_bin[(lo, hi)]
        same_d, diff_d = direct_sim_by_bin[(lo, hi)]
        try:
            joint_res = effect_size_and_test(same_j, diff_j)
            direct_res = effect_size_and_test(same_d, diff_d)
            control_res = None
            if control_sim_by_bin is not None:
                same_c, diff_c = control_sim_by_bin[(lo, hi)]
                control_res = effect_size_and_test(same_c, diff_c)
        except ValueError as e:
            qt_suffix = f" (quote_type={quote_type!r})" if quote_type else ""
            print(f"[run_predicted_mention_rsa_by_bin] skipping bin [{lo}, {hi}]{qt_suffix}: {e}")
            continue

        result.bins.append((lo, hi))
        result.bin_labels.append(f"[{lo:g}, {hi:g}]")

        result.joint_cohens_d.append(joint_res.cohens_d)
        result.direct_cohens_d.append(direct_res.cohens_d)
        result.joint_pvalues.append(joint_res.mwu_pvalue)
        result.direct_pvalues.append(direct_res.mwu_pvalue)
        result.joint_n.append((joint_res.n_same, joint_res.n_diff))
        result.direct_n.append((direct_res.n_same, direct_res.n_diff))

        joint_acc, joint_n = joint_acc_by_bin[(lo, hi)]
        direct_acc, direct_n = direct_acc_by_bin[(lo, hi)]
        result.joint_accuracy.append(joint_acc)
        result.direct_accuracy.append(direct_acc)
        result.joint_acc_n.append(joint_n)
        result.direct_acc_n.append(direct_n)

        if control_res is not None:
            result.control_cohens_d.append(control_res.cohens_d)
            result.control_pvalues.append(control_res.mwu_pvalue)
            result.control_n.append((control_res.n_same, control_res.n_diff))
            control_acc, control_n = control_acc_by_bin[(lo, hi)]
            result.control_accuracy.append(control_acc)
            result.control_acc_n.append(control_n)
    return result


def _lighten_color(color, amount: float = 0.5):
    """
    Blends `color` toward white by `amount` (0 = unchanged, 1 = white).
    Used by `plot_predicted_mention_accuracy_and_cohens_d` to derive a
    per-quote-type shade of a model's base hue: same hue per model,
    progressively lighter per quote type, instead of unrelated colors for
    every (model, quote_type) line. `color` may be any matplotlib color
    spec (name, hex string, or RGB(A) tuple).
    """
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(color))
    white = np.array([1.0, 1.0, 1.0])
    blended = rgb + (white - rgb) * amount
    return tuple(np.clip(blended, 0.0, 1.0))


def plot_predicted_mention_accuracy_and_cohens_d(
    results_by_quote_type: Dict[str, PredictedMentionBinnedResult],
    out_path: str,
    title: str = "",
    models: Sequence[str] = ("joint", "direct"),
    base_colors: Optional[Dict[str, str]] = None,
) -> None:
    """
    Two-panel figure for Task 2. Left panel: predicted-mention attribution
    accuracy vs. distance bin. Right panel: Cohen's d (correct vs.
    incorrect attribution) vs. distance bin. One line per
    (model, quote_type) combination.

    `results_by_quote_type`: `{quote_type: PredictedMentionBinnedResult}`
    - run `run_predicted_mention_rsa_by_bin(joint_windows, direct_windows,
    bins, quote_type=qt)` once per `qt` you care about (e.g. `'Explicit'`,
    `'Anaphoric'`, `'Implicit'`) and collect the results into this dict
    before calling.

    Different quote types (and, within a quote type, joint vs. direct)
    routinely end up with DIFFERENT sets of surviving bins - a bin gets
    dropped by `run_predicted_mention_rsa_by_bin` whenever any model
    lacked 2+ correct/incorrect predicted-mention pairs there, and that's
    quote-type-specific (e.g. 'Explicit' quotes might only have enough
    data in 3 of 15 possible bins, while 'Anaphoric' survives in 13).
    Naively plotting each line against its own `bin_labels` by position
    would put e.g. 'Explicit's second surviving bin (`[60, 80]`) at the
    same x position as 'Anaphoric's second bin (`[20, 40]`) - silently
    misaligned. To avoid that, this function first builds ONE shared,
    numerically-sorted x-axis pooling every `(lo, hi)` bin that appears in
    ANY result passed in, then places each line's points at the x
    position matching its own bin identity, leaving a genuine gap
    (no interpolation across the missing bin) wherever a given
    (model, quote_type) has no surviving data for a bin that other lines
    do have. The x-axis tick labels reflect this full, pooled set of bins,
    not any single result's `bin_labels`.

    `models`: which of `"joint"`/`"direct"`/`"control"` to draw a line
    for (matched against the `{model}_accuracy`/`{model}_cohens_d`/
    `{model}_bins` attributes of each `PredictedMentionBinnedResult`);
    defaults to just joint + direct. Pass `("joint", "direct", "control")`
    if you also ran with `control_windows`. A (model, quote_type)
    combination with no data (e.g. `control` when `control_windows`
    wasn't used) is silently skipped.

    Color scheme (as requested, since this plot can have many lines -
    `len(models) * len(results_by_quote_type)`): each model gets its own
    base hue (default `tab:blue` for `"joint"`, `tab:orange` for
    `"direct"`, `tab:green` for `"control"` - override via
    `base_colors={"joint": ..., "direct": ..., "control": ...}`), and each
    quote type within a model is a progressively LIGHTER shade of that
    same hue (blended toward white via `_lighten_color`) rather than an
    unrelated color - e.g. with 2 models x 3 quote types = 6 lines, models
    are told apart by hue at a glance, quote types by shade within each
    hue. Quote types are lightened in the order they appear in
    `results_by_quote_type` (insertion order), from the model's base color
    (first quote type) to noticeably lighter (last quote type) - pass an
    already-ordered dict (e.g. `{'Explicit': ..., 'Anaphoric': ...,
    'Implicit': ...}`) if you want a specific darkest-to-lightest order.

    Legend entries are labelled `"{model} - {quote_type}"`, shared across
    both panels and placed below the figure (there can be many lines).
    Saves the figure to `out_path` (dpi=150, `bbox_inches='tight'`) and
    closes it; doesn't return anything.
    """
    import matplotlib.pyplot as plt

    if base_colors is None:
        base_colors = {"joint": "tab:blue", "direct": "tab:orange", "control": "tab:green"}
    model_labels = {"joint": "Joint (L_joint)", "direct": "Direct", "control": "Control"}

    quote_types = list(results_by_quote_type.keys())
    n_types = max(len(quote_types), 1)
    # First quote type keeps the model's base (darkest) color; later quote
    # types get progressively lighter, capped well short of pure white
    # (1.0) so even the lightest line stays visible.
    lighten_amounts = np.linspace(0.0, 0.65, n_types)

    # Pool every bin appearing in ANY result, sorted by (lo, hi), so every
    # line is placed on the same, consistently-ordered x-axis regardless
    # of which subset of bins that particular (model, quote_type)
    # actually survived with. `bins` is the same list a `[lo, hi]` pair
    # regardless of model within a result (both joint/direct/control share
    # `result.bins`), so pooling across `results_by_quote_type.values()`
    # is enough - no need to also pool per-model.
    all_bins = set()
    for result in results_by_quote_type.values():
        all_bins.update(result.bins)
    sorted_bins = sorted(all_bins, key=lambda b: (b[0], b[1]))
    bin_to_x = {b: i for i, b in enumerate(sorted_bins)}
    x_positions = np.arange(len(sorted_bins))
    x_tick_labels = [f"[{lo:g}, {hi:g}]" for lo, hi in sorted_bins]

    fig, (ax_acc, ax_d) = plt.subplots(1, 2, figsize=(14, 5.5))

    def _to_dense(values, bins):
        """Scatter `values` (aligned with `bins`) onto the shared,
        pooled x-axis, leaving `nan` (a real gap, not interpolated) at
        every pooled bin this particular line has no data for."""
        dense = np.full(len(sorted_bins), np.nan)
        
        for b, v in zip(bins, values):
            dense[bin_to_x[b]] = v
        return dense

    for model in models:
        base = base_colors.get(model, "gray")
        for qt, amount in zip(quote_types, lighten_amounts):
            result = results_by_quote_type[qt]
            acc = getattr(result, f"{model}_accuracy", [])
            d = getattr(result, f"{model}_cohens_d", [])
            if not acc and not d:
                continue  # this model/quote_type combination has no data
            color = _lighten_color(base, amount)
            label = f"{model_labels.get(model, model)} - {qt}"
            if acc:
                # ax_acc.plot(x_positions, _to_dense(acc, result.bins), marker="o", color=color, label=label)
                ax_acc.plot([bin_to_x[b] for b in result.bins], acc, marker="o", color=color, label=label)

            if d:
                ax_d.plot([bin_to_x[b] for b in result.bins], d, marker="o", color=color, label=label)

                # ax_d.plot(x_positions, _to_dense(d, result.bins), marker="o", color=color, label=label)

    ax_acc.set_xticks(x_positions)
    ax_acc.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    ax_acc.set_xlabel("Predicted-mention distance bin")
    ax_acc.set_ylabel("Accuracy (predicted mention refers to gold speaker)")
    ax_acc.set_title("Accuracy by predicted-mention distance")

    ax_d.set_xticks(x_positions)
    ax_d.set_xticklabels(x_tick_labels, rotation=45, ha="right")
    ax_d.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax_d.set_xlabel("Predicted-mention distance bin")
    ax_d.set_ylabel("Cohen's d (correct - incorrect attribution)")
    ax_d.set_title("Separation by predicted-mention distance")

    handles, legend_labels = ax_d.get_legend_handles_labels()
    if not handles:
        handles, legend_labels = ax_acc.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(handles), 4) if handles else 1, fontsize=8,
    )

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sample_k_per_character(
    candidate_idx: Sequence[int],
    speaker_ids: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Groups `candidate_idx` by the character (speaker id) they refer to
    (via `speaker_ids[j]` for each `j` in `candidate_idx`), then randomly
    samples up to `k` indices per unique character, without replacement.
    A character with fewer than `k` eligible candidates keeps all of them.
    Used to cap how much a single high-frequency background character can
    dominate a negative pool, rather than including every eligible mention
    of every different character.
    """
    by_char: Dict[object, List[int]] = {}
    for j in candidate_idx:
        by_char.setdefault(speaker_ids[j], []).append(j)

    sampled: List[int] = []
    for _, idxs in by_char.items():
        if len(idxs) <= k:
            sampled.extend(idxs)
        else:
            sampled.extend(rng.choice(idxs, size=k, replace=False).tolist())
    return sampled


def _directional_span_distance(
    anchor_span: Tuple[int, int, int],
    candidate_span: Tuple[int, int, int],
    direction: str = "both",
) -> float:
    """
    Like `_span_token_distance`, but optionally restricted to only count
    a candidate span if it falls on a specific side of the anchor span in
    sequence order. Used by the nearest-neighbour pairing schemes to
    optionally search only "into the past" (or only "into the future")
    relative to an anchor, instead of the nearest match in either
    direction.

    `direction`:
      - `"both"` (default): identical to `_span_token_distance` - order-
        agnostic, symmetric distance (0 if overlapping).
      - `"backward"`: only counts `candidate_span` if it occurs strictly
        before `anchor_span` in the sequence (`candidate`'s end token is
        at or before `anchor`'s start token); returns `math.inf`
        otherwise, including for overlapping spans (ambiguous "before"),
        spans in a different window/batch, or a candidate that starts at
        or after the anchor.
      - `"forward"`: mirror image of `"backward"` - only counts
        `candidate_span` if it occurs at or after `anchor_span` ends.

    Overlapping spans are never counted as "before"/"after" in the
    directional modes (treated as `math.inf`, i.e. not eligible), unlike
    `"both"` where they collapse to a distance of 0.
    """
    batch_a, start_a, end_a = anchor_span
    batch_b, start_b, end_b = candidate_span
    if batch_a != batch_b:
        return math.inf
    if direction == "both":
        return _span_token_distance(anchor_span, candidate_span)
    if direction == "backward":
        if end_b <= start_a:
            return start_a - end_b
        return math.inf
    if direction == "forward":
        if end_a <= start_b:
            return start_b - end_a
        return math.inf
    raise ValueError(f"Unknown direction: {direction!r} (expected 'both', 'backward', or 'forward')")



def collect_nearest_mention_mention_similarities_by_bin(
    windows: Iterable[WindowReps],
    bins: Sequence[Tuple[float, float]],
    negative_k: Optional[int] = None,
    random_seed: int = 0,
    direction: str = "both",
) -> Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
    """
    Mention-mention counterpart of `collect_nearest_mention_similarities_by_bin`:
    same nearest-neighbour pairing scheme, but with a MENTION as the anchor
    on both sides instead of a quote anchored against mentions.

    For each mention `i` in each window:
      1. Find its NEAREST *other* mention that refers to the same speaker
         (excluding itself) - the positive pair. By default (`direction=
         "both"`) this searches both directions in the sequence (the
         nearest same-speaker mention may occur before or after mention
         `i`). If it has no eligible same-speaker partner anywhere in the
         window, mention `i` is skipped.
      2. Assign that (mention_i, nearest same-speaker mention) pair to the
         first bin `(lo, hi)` in `bins` (in the given order) whose range
         contains the distance between them, inclusive of both ends. If the
         distance exceeds every bin's upper bound, mention `i` is skipped.
      3. Within that bin, every different-speaker mention strictly closer
         than the bin's upper bound (`distance < hi`) is *eligible* as a
         negative pair for mention `i`, same cumulative-threshold rule as
         the quote-anchored version.

    `direction`: controls which candidates (for both the positive nearest-
    neighbour search and the negative pool) are eligible relative to
    anchor mention `i`, via `_directional_span_distance`:
      - `"both"` (default): original behaviour - the nearest same-speaker
        mention may be earlier or later in the sequence than `i`, and
        negatives are drawn from either direction too.
      - `"backward"`: only mentions whose span occurs strictly *before*
        `i`'s span in the sequence are eligible, for both the positive
        search and the negative pool - i.e. "look only into the past
        relative to `i`". A mention with no eligible backward same-speaker
        partner is skipped, even if a same-speaker mention exists later in
        the window.
      - `"forward"`: mirror image of `"backward"` - only mentions occurring
        at or after `i`'s span are eligible.
    Overlapping spans are never eligible under `"backward"`/`"forward"`
    (ambiguous ordering), unlike `"both"` where they count as distance 0.

    `negative_k`: if set, instead of keeping EVERY eligible different-
    speaker mention as a negative, group the eligible mentions by the
    character they refer to (there may be several distinct "other"
    characters within the distance cutoff) and randomly sample up to
    `negative_k` mentions per unique character (see
    `_sample_k_per_character`; without replacement, a character with fewer
    than `negative_k` eligible mentions just keeps all of them). This
    prevents one frequently-mentioned background character from dominating
    the negative distribution simply because they have many nearby
    mentions, at the cost of fewer, sampled negatives instead of the full
    eligible set. Pass `None` (default) to keep every eligible mention, as
    before. `random_seed` controls the sampling for reproducibility - the
    same seed reproduces the same sampled negatives across runs.

    Note: because both the anchor and the candidate pool are mentions here
    (unlike the quote-anchored version, where quotes and mentions are
    disjoint), a mutually-nearest pair of same-speaker mentions (i's
    nearest is j, and j's nearest is also i) will contribute its
    similarity value twice, once with each of the two as anchor - this can
    still happen under `direction="backward"`/`"forward"` too, just from
    the specific anchor for which the relationship is directionally valid
    (e.g. under `"backward"`, only the later mention would ever pick the
    earlier one as its nearest backward partner, so this becomes rarer,
    but a chain of 3+ same-speaker mentions can still produce it). This is
    intentional and mirrors the quote-anchored version's semantics exactly
    ("for each anchor, what is true of its own nearest same-speaker
    partner"), just applied per-mention instead of per-quote; it is not
    deduplicated into a single unordered-pair count.

    Returns `{bin: (positive_sims, negative_sims)}` for every bin in
    `bins`. Feed straight into `run_nearest_mention_mention_rsa_by_bin`.
    """
    rng = np.random.default_rng(random_seed)
    same_by_bin: Dict[Tuple[float, float], List[float]] = {b: [] for b in bins}
    diff_by_bin: Dict[Tuple[float, float], List[float]] = {b: [] for b in bins}

    for w in windows:
        h_m = w.h_m
        m = h_m.size(0)
        if m < 2:
            continue

        m_norm = F.normalize(h_m, dim=-1)
        sim_matrix = (m_norm @ m_norm.T).detach().cpu().numpy()  # [m, m]

        mention_speakers = np.asarray(w.mention_speaker_ids)

        for i in range(m):
            distances = np.array([
                _directional_span_distance(w.mention_spans[i], w.mention_spans[j], direction)
                for j in range(m)
            ])
            same_mask = mention_speakers == mention_speakers[i]
            same_mask[i] = False  # a mention is never its own partner
            if not same_mask.any():
                continue  # no other same-speaker mention anywhere in this window

            same_idx = np.where(same_mask)[0]
            nearest_j = same_idx[np.argmin(distances[same_idx])]
            d_pos = distances[nearest_j]
            if math.isinf(d_pos):
                continue  # no eligible same-speaker mention in the requested direction

            target_bin = None
            for lo, hi in bins:
                if _distance_in_range(d_pos, lo, hi):
                    target_bin = (lo, hi)
                    break
            if target_bin is None:
                continue  # nearest same-speaker mention is farther than every bin covers

            same_by_bin[target_bin].append(float(sim_matrix[i, nearest_j]))

            hi_bin = target_bin[1]
            diff_mask = (mention_speakers != mention_speakers[i])
            eligible_diff_idx = [j for j in np.where(diff_mask)[0] if distances[j] < hi_bin]
            if negative_k is not None:
                eligible_diff_idx = _sample_k_per_character(
                    eligible_diff_idx, mention_speakers, negative_k, rng
                )
            for j in eligible_diff_idx:
                diff_by_bin[target_bin].append(float(sim_matrix[i, j]))

    return {b: (np.asarray(same_by_bin[b]), np.asarray(diff_by_bin[b])) for b in bins}


def run_nearest_mention_mention_rsa_by_bin(
    joint_windows: Iterable[WindowReps],
    direct_windows: Iterable[WindowReps],
    bins: Sequence[Tuple[float, float]],
    control_windows: Optional[Iterable[WindowReps]] = None,
    negative_k: Optional[int] = None,
    random_seed: int = 0,
    direction: str = "both",
) -> BinnedSweepResult:
    """
    Mention-mention counterpart of `run_nearest_mention_rsa_by_bin`: same
    high-level entry point (runs the nearest-neighbour pairing
    independently on `joint_windows`/`direct_windows`/`control_windows`,
    then computes Cohen's d + Mann-Whitney U per bin for each), but built
    on `collect_nearest_mention_mention_similarities_by_bin` instead, so
    every anchor/candidate is a mention rather than a quote.

    `control_windows`: optional third model's windows, run through the
    same nearest-mention-mention pairing scheme (with its own `negative_k`
    sampling, seeded independently from `random_seed` like the joint/
    direct calls). Leave `None` (default) to compare only joint vs.
    direct, as before - `control_cohens_d`/`control_pvalues`/`control_n`
    are then left empty on the returned result.

    `negative_k`/`random_seed`: forwarded to
    `collect_nearest_mention_mention_similarities_by_bin` - if `negative_k`
    is set, negatives are randomly sampled up to `negative_k` per unique
    different character instead of keeping every eligible different-
    speaker mention. Joint, direct, and control windows are all sampled
    with the same `random_seed` (but independently, since they're separate
    calls), so re-running with the same seed reproduces the same
    comparison.

    `direction`: forwarded to `collect_nearest_mention_mention_similarities_by_bin`
    - `"both"` (default) searches for the nearest same-speaker mention in
    either direction, `"backward"` restricts the nearest-neighbour search
    (and the negative pool) to mentions occurring strictly before the
    anchor in the sequence, `"forward"` restricts to mentions occurring at
    or after the anchor. Applied identically to joint, direct, and control.

    Returns a `BinnedSweepResult`, a drop-in for `plot_cohens_d_by_bin`
    (pair-count bars included). Note `filter_windows_by_quote_type` has no
    effect here since mentions aren't associated with a quote type; if you
    want to restrict this to, say, mentions co-occurring with a particular
    quote type's window, filter the windows yourself before calling this.
    """
    joint_by_bin = collect_nearest_mention_mention_similarities_by_bin(
        joint_windows, bins, negative_k=negative_k, random_seed=random_seed, direction=direction
    )
    direct_by_bin = collect_nearest_mention_mention_similarities_by_bin(
        direct_windows, bins, negative_k=negative_k, random_seed=random_seed, direction=direction
    )
    control_by_bin = (
        collect_nearest_mention_mention_similarities_by_bin(
            control_windows, bins, negative_k=negative_k, random_seed=random_seed, direction=direction
        )
        if control_windows is not None else None
    )

    result = BinnedSweepResult()
    for lo, hi in bins:
        same_j, diff_j = joint_by_bin[(lo, hi)]
        same_d, diff_d = direct_by_bin[(lo, hi)]
        try:
            joint_res = effect_size_and_test(same_j, diff_j)
            direct_res = effect_size_and_test(same_d, diff_d)
            control_res = None
            if control_by_bin is not None:
                same_c, diff_c = control_by_bin[(lo, hi)]
                control_res = effect_size_and_test(same_c, diff_c)
        except ValueError as e:
            # Note: if control_windows is given, a bin is skipped entirely
            # (for all three models) if ANY of joint/direct/control lacks
            # enough data there, keeping the three lists aligned for
            # plotting.
            print(f"[run_nearest_mention_mention_rsa_by_bin] skipping bin [{lo}, {hi}]: {e}")
            continue
        result.bins.append((lo, hi))
        result.bin_labels.append(f"[{lo:g}, {hi:g}]")
        result.joint_cohens_d.append(joint_res.cohens_d)
        result.direct_cohens_d.append(direct_res.cohens_d)
        result.joint_pvalues.append(joint_res.mwu_pvalue)
        result.direct_pvalues.append(direct_res.mwu_pvalue)
        result.joint_n.append((joint_res.n_same, joint_res.n_diff))
        result.direct_n.append((direct_res.n_same, direct_res.n_diff))
        result.joint_mean_same.append(joint_res.mean_same)
        result.joint_mean_diff.append(joint_res.mean_diff)
        result.direct_mean_same.append(direct_res.mean_same)
        result.direct_mean_diff.append(direct_res.mean_diff)
        if control_res is not None:
            result.control_cohens_d.append(control_res.cohens_d)
            result.control_pvalues.append(control_res.mwu_pvalue)
            result.control_n.append((control_res.n_same, control_res.n_diff))
            result.control_mean_same.append(control_res.mean_same)
            result.control_mean_diff.append(control_res.mean_diff)
    return result



@dataclass
class DistanceSweepResult:
    """Cohen's d (and supporting stats) for joint vs. direct (vs. optional
    control) at each `max_token_distance` cutoff tried by
    `sweep_cohens_d_over_distance`."""
    distances: List[int] = field(default_factory=list)
    joint_cohens_d: List[float] = field(default_factory=list)
    direct_cohens_d: List[float] = field(default_factory=list)
    joint_pvalues: List[float] = field(default_factory=list)
    direct_pvalues: List[float] = field(default_factory=list)
    joint_n: List[Tuple[int, int]] = field(default_factory=list)   # (n_same, n_diff) per distance
    direct_n: List[Tuple[int, int]] = field(default_factory=list)
    control_cohens_d: List[float] = field(default_factory=list)   # empty unless control_windows was used
    control_pvalues: List[float] = field(default_factory=list)
    control_n: List[Tuple[int, int]] = field(default_factory=list)


def sweep_cohens_d_over_distance(
    joint_windows: Iterable[WindowReps],
    direct_windows: Iterable[WindowReps],
    distances: Sequence[int],
    comparison_fn: Callable[..., Tuple[SeparationResult, SeparationResult, Optional[SeparationResult]]] = run_rsa_comparison,
    control_windows: Optional[Iterable[WindowReps]] = None,
) -> DistanceSweepResult:
    """
    Runs `comparison_fn` once per value in `distances`, collecting Cohen's d
    (plus p-value and sample sizes) for the joint and direct model (and,
    if `control_windows` is given, a third control model) at each
    `max_token_distance` cutoff. `comparison_fn` is `run_rsa_comparison`
    (quote-quote), `run_rsa_mention_comparison` (quote-mention), or
    `run_rsa_mention_mention_comparison` (mention-mention); all three
    already accept a `max_token_distance` kwarg and an optional
    `control_windows` kwarg, so any can be passed here directly.

    `joint_windows`/`direct_windows`/`control_windows` are materialized
    into lists internally since each distance value needs to re-scan them
    - pass a list (or any re-iterable), not a one-shot generator. Leave
    `control_windows` `None` (default) to sweep only joint vs. direct, as
    before - `control_cohens_d`/`control_pvalues`/`control_n` are then left
    empty on the returned result.

    If a given cutoff leaves fewer than 2 same- or different-speaker pairs
    on either side (raised as `ValueError` by `effect_size_and_test`,
    typically only at very small distances), that point is skipped with a
    warning printed to stdout rather than aborting the whole sweep.
    """
    joint_windows = list(joint_windows)
    direct_windows = list(direct_windows)
    if control_windows is not None:
        control_windows = list(control_windows)

    result = DistanceSweepResult()
    for d in distances:
        try:
            joint_res, direct_res, control_res = comparison_fn(
                joint_windows, direct_windows, control_windows=control_windows, max_token_distance=d
            )
        except ValueError as e:
            print(f"[sweep_cohens_d_over_distance] skipping max_token_distance={d}: {e}")
            continue
        result.distances.append(d)
        result.joint_cohens_d.append(joint_res.cohens_d)
        result.direct_cohens_d.append(direct_res.cohens_d)
        result.joint_pvalues.append(joint_res.mwu_pvalue)
        result.direct_pvalues.append(direct_res.mwu_pvalue)
        result.joint_n.append((joint_res.n_same, joint_res.n_diff))
        result.direct_n.append((direct_res.n_same, direct_res.n_diff))
        if control_res is not None:
            result.control_cohens_d.append(control_res.cohens_d)
            result.control_pvalues.append(control_res.mwu_pvalue)
            result.control_n.append((control_res.n_same, control_res.n_diff))
    return result


def plot_cohens_d_vs_distance(sweep: DistanceSweepResult, title: str, out_path: str):
    """
    Line plot: x = max_token_distance, y = Cohen's d, one line for the
    joint model, one for the direct model, and (if present) a third line
    for the control model. Feed it the output of
    `sweep_cohens_d_over_distance` (called once for quote-quote via
    `run_rsa_comparison`, and/or once for quote-mention via
    `run_rsa_mention_comparison` - call this twice, once per
    `DistanceSweepResult`, for the two separate figures).
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 5))
    plt.plot(sweep.distances, sweep.joint_cohens_d, marker="o", label="Joint (L_joint)")
    plt.plot(sweep.distances, sweep.direct_cohens_d, marker="o", label="Direct")
    if sweep.control_cohens_d:
        plt.plot(sweep.distances, sweep.control_cohens_d, marker="o", label="Control")
    plt.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.xlabel("max_token_distance")
    plt.ylabel("Cohen's d (same-speaker - different-speaker)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



@dataclass
class BinnedSweepResult:
    """Cohen's d (and supporting stats) for joint vs. direct (vs. optional
    control) within each disjoint `[lo, hi]` token-distance bin tried by
    `sweep_cohens_d_over_distance_bins`."""
    bins: List[Tuple[float, float]] = field(default_factory=list)
    bin_labels: List[str] = field(default_factory=list)
    joint_cohens_d: List[float] = field(default_factory=list)
    direct_cohens_d: List[float] = field(default_factory=list)
    joint_pvalues: List[float] = field(default_factory=list)
    direct_pvalues: List[float] = field(default_factory=list)
    joint_n: List[Tuple[int, int]] = field(default_factory=list)   # (n_same, n_diff) per bin
    direct_n: List[Tuple[int, int]] = field(default_factory=list)
    control_cohens_d: List[float] = field(default_factory=list)   # empty unless control_windows was used
    control_pvalues: List[float] = field(default_factory=list)
    control_n: List[Tuple[int, int]] = field(default_factory=list)
    # Raw average cosine similarity per bin (SeparationResult.mean_same/
    # mean_diff), alongside the Cohen's d summary above - lets you see the
    # actual similarity magnitudes/spread behind a given d, not just the
    # standardized effect size. control_mean_* stays empty unless
    # control_windows was used, same as the control_* fields above.
    joint_mean_same: List[float] = field(default_factory=list)
    joint_mean_diff: List[float] = field(default_factory=list)
    direct_mean_same: List[float] = field(default_factory=list)
    direct_mean_diff: List[float] = field(default_factory=list)
    control_mean_same: List[float] = field(default_factory=list)
    control_mean_diff: List[float] = field(default_factory=list)


def sweep_cohens_d_over_distance_bins(
    joint_windows: Iterable[WindowReps],
    direct_windows: Iterable[WindowReps],
    bins: Sequence[Tuple[float, float]],
    comparison_fn: Callable[..., Tuple[SeparationResult, SeparationResult, Optional[SeparationResult]]] = run_rsa_comparison,
    control_windows: Optional[Iterable[WindowReps]] = None,
) -> BinnedSweepResult:
    """
    Binned-range counterpart of `sweep_cohens_d_over_distance`. Instead of
    a cumulative "within N tokens" cutoff, this restricts each point to a
    disjoint `[lo, hi]` token-distance range, e.g.
    `bins=[(0, 50), (50, 100), (100, 200), (200, 400), (400, 800)]`
    isolates same-/different-speaker separation specifically among pairs
    50-100 tokens apart, 100-200 apart, etc., rather than "0 to 100" and
    "0 to 200" (which share most of their pairs and so can't tell you
    whether separation is concentrated nearby or spread out evenly).

    `comparison_fn` is `run_rsa_comparison` (quote-quote),
    `run_rsa_mention_comparison` (quote-mention), or
    `run_rsa_mention_mention_comparison` (mention-mention); all three
    already accept `min_token_distance`/`max_token_distance` and an
    optional `control_windows` kwarg, so any can be passed here directly.

    `joint_windows`/`direct_windows`/`control_windows` are materialized
    into lists internally since each bin needs to re-scan them - pass a
    list (or any re-iterable), not a one-shot generator. Leave
    `control_windows` `None` (default) to sweep only joint vs. direct, as
    before - `control_cohens_d`/`control_pvalues`/`control_n` are then left
    empty on the returned result.

    Bins that leave fewer than 2 same- or different-speaker pairs on
    either side are skipped with a warning printed to stdout, same as
    `sweep_cohens_d_over_distance`.
    """
    joint_windows = list(joint_windows)
    direct_windows = list(direct_windows)
    if control_windows is not None:
        control_windows = list(control_windows)

    result = BinnedSweepResult()
    for lo, hi in bins:
        try:
            joint_res, direct_res, control_res = comparison_fn(
                joint_windows, direct_windows, control_windows=control_windows,
                min_token_distance=lo, max_token_distance=hi,
            )
        except ValueError as e:
            print(f"[sweep_cohens_d_over_distance_bins] skipping bin [{lo}, {hi}]: {e}")
            continue
        result.bins.append((lo, hi))
        result.bin_labels.append(f"[{lo:g}, {hi:g}]")
        result.joint_cohens_d.append(joint_res.cohens_d)
        result.direct_cohens_d.append(direct_res.cohens_d)
        result.joint_pvalues.append(joint_res.mwu_pvalue)
        result.direct_pvalues.append(direct_res.mwu_pvalue)
        result.joint_n.append((joint_res.n_same, joint_res.n_diff))
        result.direct_n.append((direct_res.n_same, direct_res.n_diff))
        if control_res is not None:
            result.control_cohens_d.append(control_res.cohens_d)
            result.control_pvalues.append(control_res.mwu_pvalue)
            result.control_n.append((control_res.n_same, control_res.n_diff))
    return result


def _add_pair_count_bars(ax, x, sweep_n: List[Tuple[int, int]], width: float = 0.6):
    """
    Shared helper: draws a light, shadowed bar behind a Cohen's d line plot
    showing the total number of pairs (n_same + n_diff) supporting each x
    position, on a secondary y-axis to the right. Used by both
    `plot_cohens_d_by_bin` and `plot_cohens_d_by_bin_per_quote_type`.

    Bars sit behind the line plot (`ax`) via zorder + a transparent `ax`
    background, so the Cohen's d lines/markers remain fully visible on top.

    The secondary axis tick labels are forced into matplotlib's usual
    scientific notation (e.g. `1.5` with a shared `1e7` offset label above
    the axis) via `ScalarFormatter(useMathText=True)` + `scilimits=(0, 0)`,
    rather than the plain integer labels matplotlib would otherwise pick
    for the range of counts seen here.
    """
    import matplotlib.ticker as mticker

    ax2 = ax.twinx()
    pair_counts = [n_same + n_diff for n_same, n_diff in sweep_n]
    ax2.bar(list(x), pair_counts, width=width, color="grey", alpha=0.18, zorder=1, label="# pairs (joint)")
    ax2.set_ylabel("# pairs", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    ax2.set_ylim(0, max(pair_counts) * 1.4 if pair_counts and max(pair_counts) > 0 else 1)

    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # always scientific, not just outside matplotlib's default range
    ax2.yaxis.set_major_formatter(formatter)
    ax2.yaxis.get_offset_text().set_color("grey")

    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)  # let the bars on ax2 show through ax's background
    return ax2

def plot_cohens_d_by_bin(sweep: BinnedSweepResult, title: str, out_path: str):
    """
    Line plot: one categorical x position per token-distance bin (labelled
    with the bin's `[lo, hi]` range), one line for the joint model, one
    for the direct model, and (if present) a third for the control model,
    y = Cohen's d. Same visual style as `plot_cohens_d_vs_distance`, just
    with bin-range labels on the x-axis instead of a numeric cutoff.

    Each x position also gets a light, shadowed bar (secondary y-axis, on
    the right) showing the total number of pairs (n_same + n_diff, joint
    model) supporting that bin, so a reader can see at a glance which
    points are backed by plenty of data vs. just a handful of pairs.

    Feed it the output of `sweep_cohens_d_over_distance_bins`.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
    })
    x = range(len(sweep.bin_labels))

    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    
    ax2 = _add_pair_count_bars(ax, x, sweep.joint_n)

    ax.plot(x, sweep.joint_cohens_d, marker="o", label="Joint", zorder=3)
    ax.plot(x, sweep.direct_cohens_d, marker="o", label="Direct", zorder=3)
    if sweep.control_cohens_d:
        ax.plot(x, sweep.control_cohens_d, marker="o", label="ModernBERT", zorder=3)
    # ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(sweep.bin_labels, rotation=30, ha='right')
    ax.set_xlabel("Nearest Antecedent Token Distance")
    ax.set_ylabel("Effect Size (Cohen's $d$)")
    # ax.set_title(title)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    # ax.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


@dataclass
class QuoteTypeSweepResult:
    """Cohen's d (and supporting stats) for joint vs. direct (vs. optional
    control) within each quote type tried by
    `sweep_cohens_d_over_quote_type`."""
    quote_types: List[str] = field(default_factory=list)
    joint_cohens_d: List[float] = field(default_factory=list)
    direct_cohens_d: List[float] = field(default_factory=list)
    joint_pvalues: List[float] = field(default_factory=list)
    direct_pvalues: List[float] = field(default_factory=list)
    joint_n: List[Tuple[int, int]] = field(default_factory=list)   # (n_same, n_diff) per quote type
    direct_n: List[Tuple[int, int]] = field(default_factory=list)
    control_cohens_d: List[float] = field(default_factory=list)   # empty unless control_windows was used
    control_pvalues: List[float] = field(default_factory=list)
    control_n: List[Tuple[int, int]] = field(default_factory=list)


def sweep_cohens_d_over_quote_type(
    joint_windows: Iterable[WindowReps],
    direct_windows: Iterable[WindowReps],
    quote_types: Sequence[str],
    comparison_fn: Callable[..., Tuple[SeparationResult, SeparationResult, Optional[SeparationResult]]] = run_rsa_comparison,
    control_windows: Optional[Iterable[WindowReps]] = None,
    **comparison_kwargs,
) -> QuoteTypeSweepResult:
    """
    Runs `comparison_fn` once per value in `quote_types` (e.g.
    `['Explicit', 'Anaphoric', 'Implicit']`), after restricting the quote
    side of each window to that type via `filter_windows_by_quote_type`.
    `comparison_fn` is `run_rsa_comparison` (quote-quote),
    `run_rsa_mention_comparison` (quote-mention), or
    `run_rsa_mention_mention_comparison` (mention-mention).

    `control_windows`: optional third model's windows, filtered by quote
    type the same way as `joint_windows`/`direct_windows` and forwarded to
    `comparison_fn`. Leave `None` (default) to sweep only joint vs.
    direct, as before - `control_cohens_d`/`control_pvalues`/`control_n`
    are then left empty on the returned result.

    Any extra keyword arguments (e.g. `max_token_distance`,
    `min_token_distance`) are forwarded to `comparison_fn`, so you can
    combine the quote-type breakdown with a token-distance restriction in
    the same call, e.g.
    `sweep_cohens_d_over_quote_type(..., max_token_distance=100)`.

    `joint_windows`/`direct_windows`/`control_windows` are materialized
    into lists internally since each quote type needs to re-scan and
    re-filter them - pass a list (or any re-iterable), not a one-shot
    generator.

    Quote types that leave fewer than 2 same- or different-speaker pairs
    on either side are skipped with a warning printed to stdout, same as
    the distance sweeps.
    """
    joint_windows = list(joint_windows)
    direct_windows = list(direct_windows)
    if control_windows is not None:
        control_windows = list(control_windows)

    result = QuoteTypeSweepResult()
    for qt in quote_types:
        joint_sub = filter_windows_by_quote_type(joint_windows, qt)
        direct_sub = filter_windows_by_quote_type(direct_windows, qt)
        control_sub = filter_windows_by_quote_type(control_windows, qt) if control_windows is not None else None
        try:
            joint_res, direct_res, control_res = comparison_fn(
                joint_sub, direct_sub, control_windows=control_sub, **comparison_kwargs
            )
        except ValueError as e:
            print(f"[sweep_cohens_d_over_quote_type] skipping quote_type={qt!r}: {e}")
            continue
        result.quote_types.append(qt)
        result.joint_cohens_d.append(joint_res.cohens_d)
        result.direct_cohens_d.append(direct_res.cohens_d)
        result.joint_pvalues.append(joint_res.mwu_pvalue)
        result.direct_pvalues.append(direct_res.mwu_pvalue)
        result.joint_n.append((joint_res.n_same, joint_res.n_diff))
        result.direct_n.append((direct_res.n_same, direct_res.n_diff))
        if control_res is not None:
            result.control_cohens_d.append(control_res.cohens_d)
            result.control_pvalues.append(control_res.mwu_pvalue)
            result.control_n.append((control_res.n_same, control_res.n_diff))
    return result


def plot_cohens_d_by_quote_type(sweep: QuoteTypeSweepResult, title: str, out_path: str):
    """
    Line plot: one categorical x position per quote type (e.g.
    'Explicit'/'Anaphoric'/'Implicit'), one line for the joint model, one
    for the direct model, and (if present) a third for the control model,
    y = Cohen's d. Same visual style as `plot_cohens_d_by_bin`, just with
    quote-type labels on the x-axis.

    Feed it the output of `sweep_cohens_d_over_quote_type`.
    """
    import matplotlib.pyplot as plt

    x = range(len(sweep.quote_types))

    plt.figure(figsize=(7, 5))
    plt.plot(x, sweep.joint_cohens_d, marker="o", label="Joint (L_joint)")
    plt.plot(x, sweep.direct_cohens_d, marker="o", label="Direct")
    if sweep.control_cohens_d:
        plt.plot(x, sweep.control_cohens_d, marker="o", label="Control")
    plt.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    plt.xticks(list(x), sweep.quote_types, rotation=0)
    plt.xlabel("quote type")
    plt.ylabel("Cohen's d (same-speaker - different-speaker)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def sweep_cohens_d_over_distance_bins_by_quote_type(
    joint_windows: Iterable[WindowReps],
    direct_windows: Iterable[WindowReps],
    quote_types: Sequence[str],
    bins: Sequence[Tuple[float, float]],
    comparison_fn: Callable[..., Tuple[SeparationResult, SeparationResult, Optional[SeparationResult]]] = run_rsa_comparison,
    control_windows: Optional[Iterable[WindowReps]] = None,
) -> Dict[str, BinnedSweepResult]:
    """
    Combines `filter_windows_by_quote_type` with
    `sweep_cohens_d_over_distance_bins`: for each quote type, restricts the
    quote side of every window to that type, then runs the full
    token-distance-bin sweep within that subset. This is the "binned
    analysis, but separately per quote type" version - e.g. does the
    joint model's advantage over the direct model, as a function of
    distance to the nearest mention, look different for Explicit quotes
    (which likely have strong local cues) than for Implicit quotes (which
    don't)?

    `comparison_fn` is `run_rsa_comparison` (quote-quote),
    `run_rsa_mention_comparison` (quote-mention), or
    `run_rsa_mention_mention_comparison` (mention-mention).

    `control_windows`: optional third model's windows, filtered by quote
    type the same way as `joint_windows`/`direct_windows` and forwarded
    through to `comparison_fn` for every quote type. Leave `None`
    (default) to sweep only joint vs. direct, as before.

    `joint_windows`/`direct_windows`/`control_windows` are materialized
    into lists internally since each quote type needs to re-scan and
    re-filter them - pass a list (or any re-iterable), not a one-shot
    generator.

    Returns a dict `{quote_type: BinnedSweepResult}`, one entry per quote
    type in `quote_types` (types that end up with no usable bins are still
    included, just with empty lists in their `BinnedSweepResult`). Feed the
    result to `plot_cohens_d_by_bin_per_quote_type` for a single figure
    with one panel per quote type, or to `plot_cohens_d_by_bin` per entry
    for separate figures.
    """
    joint_windows = list(joint_windows)
    direct_windows = list(direct_windows)
    if control_windows is not None:
        control_windows = list(control_windows)

    results: Dict[str, BinnedSweepResult] = {}
    for qt in quote_types:
        joint_sub = filter_windows_by_quote_type(joint_windows, qt)
        direct_sub = filter_windows_by_quote_type(direct_windows, qt)
        control_sub = filter_windows_by_quote_type(control_windows, qt) if control_windows is not None else None
        results[qt] = sweep_cohens_d_over_distance_bins(
            joint_sub, direct_sub, bins, comparison_fn=comparison_fn, control_windows=control_sub
        )
    return results


def plot_cohens_d_by_bin_per_quote_type(
    results: Dict[str, BinnedSweepResult],
    title: str,
    out_path: str,
):
    """
    Multi-panel figure: one subplot per quote type (sharing the y-axis for
    easy visual comparison across types), x = token-distance bin, y =
    Cohen's d, one line for the joint model, one for the direct model, and
    (if present) a third for the control model, per panel. Feed it the
    output of `sweep_cohens_d_over_distance_bins_by_quote_type`.

    Each x position also gets a light, shadowed bar (secondary y-axis, on
    the right of each panel) showing the total number of pairs
    (n_same + n_diff, joint model) supporting that bin, so a reader can
    see at a glance which points are backed by plenty of data vs. just a
    handful of pairs. The secondary axis scale is shared across panels so
    bar heights are directly comparable across quote types.
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 11,
    })

    quote_types = list(results.keys())
    n = len(quote_types)
    fig, axes = plt.subplots(1, n,figsize=(13.0, 3.15), sharey=True)
    if n == 1:
        axes = [axes]

    all_pair_counts = [
        n_same + n_diff for sweep in results.values() for n_same, n_diff in sweep.joint_n
    ]
    shared_count_ylim = (0, max(all_pair_counts) * 1.4) if all_pair_counts else (0, 1)

    secondary_axes = []
    for ax, qt in zip(axes, quote_types):
        sweep = results[qt]
        x = range(len(sweep.bin_labels))

        ax2 = _add_pair_count_bars(ax, x, sweep.joint_n)
        ax2.set_ylim(*shared_count_ylim)
        secondary_axes.append(ax2)

        ax.plot(x, sweep.joint_cohens_d, marker="o", label="Joint", zorder=3)
        ax.plot(x, sweep.direct_cohens_d, marker="o", label="Direct", zorder=3)
        if sweep.control_cohens_d:
            ax.plot(x, sweep.control_cohens_d, marker="o", label="ModernBERT", zorder=3)
        # ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", zorder=2)
        ax.set_xticks(list(x))
        ax.set_xticklabels(sweep.bin_labels, rotation=30, ha='right')
        ax.set_xlabel("Token Distance Bin")
        ax.set_title(qt)

    # Only label/show the secondary axis on the rightmost panel to avoid
    # cluttering every panel with a redundant "# pairs" label.
    for ax2 in secondary_axes[:-1]:
        ax2.set_ylabel("")
        ax2.set_yticklabels([])
    secondary_axes[-1].set_ylabel("# pairs", color="grey")

    axes[0].set_ylabel("Effect Size (Cohen's $d$)")
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = secondary_axes[0].get_legend_handles_labels()
    # axes[0].legend(lines1 + lines2, labels1 + labels2, ncols=4, loc='center', bbox_to_anchor=(1.75, 1.2), frameon=False)
    # fig.suptitle(title)
    # plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_similarity_violin(
    joint_res: SeparationResult,
    direct_res: SeparationResult,
    out_path: str,
    control_res: Optional[SeparationResult] = None,
):
    """Convenience plot: violin of same/diff cosine similarity, joint vs
    direct (vs. optional control). Pass `control_res` (e.g. the third
    element returned by `run_rsa_comparison`/`run_rsa_mention_comparison`/
    `run_rsa_mention_mention_comparison` when `control_windows` was given)
    to add a third violin pair; leave `None` (default) for the original
    two-model plot."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    models = [("Joint (L_joint)", joint_res), ("Direct", direct_res)]
    if control_res is not None:
        models.append(("Control", control_res))

    rows = []
    for label, res in models:
        rows += [{"model": label, "pair_type": "same-speaker", "similarity": v} for v in res.same_values]
        rows += [{"model": label, "pair_type": "different-speaker", "similarity": v} for v in res.diff_values]
    df = pd.DataFrame(rows)

    plt.figure(figsize=(7, 5))
    sns.violinplot(data=df, x="model", y="similarity", hue="pair_type", split=True)
    plt.title("Quote representation cosine similarity: same- vs different-speaker pairs")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()



def _lighten_color(color, amount: float = 0.5):
    """
    Returns an RGB tuple for `color` blended toward white by `amount`
    (0 = unchanged, 1 = white), preserving hue/saturation. Used to derive
    a "different-speaker" line color from a model's base ("same-speaker")
    color in `plot_mean_similarity_by_bin_per_quote_type`, so the two
    lines for one model are visually related (same hue) but distinguishable
    at a glance (lighter = the different-speaker line).
    """
    import colorsys
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = l + (1.0 - l) * amount
    return colorsys.hls_to_rgb(h, min(1.0, l), s)
    
def plot_mean_similarity_by_bin_per_quote_type(
    sweep: BinnedSweepResult,
    title: str,
    out_path: str,
    lighten_amount: float = 0.55,
):
    """
    Multi-panel figure: one subplot per quote type (sharing the y-axis),
    x = token-distance bin, y = raw average cosine similarity - the
    `*_mean_same`/`*_mean_diff` fields on `BinnedSweepResult` (populated by
    `sweep_cohens_d_over_distance_bins`/`sweep_cohens_d_over_distance_bins_by_quote_type`,
    alongside Cohen's d), rather than the standardized Cohen's d itself.
    Feed it the output of `sweep_cohens_d_over_distance_bins_by_quote_type`.

    Draws up to 6 lines per panel: one same-speaker/different-speaker pair
    for each of the joint, direct, and (if present) control models. Each
    model gets its own base color (matplotlib's default color cycle, kept
    identical across panels and across the same-/different-speaker lines
    for that model); within a model, the same-speaker line is solid and
    full-color while the different-speaker line is a lighter shade of the
    exact same color (via `_lighten_color`, controlled by `lighten_amount`)
    and dashed - so at a glance, hue identifies the model and
    solid-vs-dashed/light-vs-dark identifies same- vs. different-speaker.
    The control lines are only drawn where `control_mean_same`/
    `control_mean_diff` are non-empty (i.e. `control_windows` was passed
    upstream).
    """
    import matplotlib.pyplot as plt

    # quote_types = list(results.keys())
    # n = len(quote_types)
    n = 1
    
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
    })
    x = range(len(sweep.bin_labels))

    # fig, ax = plt.subplots(figsize=(4.2, 3.0))
    
    fig, axes = plt.subplots(1, 1, figsize=(5.2, 2.9), sharey=True)
    if n == 1:
        axes = [axes]

    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_specs = [
        ("Joint", "joint_mean_same", "joint_mean_diff", prop_cycle[0]),
        ("Direct", "direct_mean_same", "direct_mean_diff", prop_cycle[1]),
        ("ModernBERT", "control_mean_same", "control_mean_diff", prop_cycle[2]),
    ]

    ax = axes[0]
    # for ax, qt in zip(axes, quote_types):
        # sweep = results[qt]
    x = range(len(sweep.bin_labels))
    
    for label, same_attr, diff_attr, color in model_specs:
        same_vals = getattr(sweep, same_attr)
        diff_vals = getattr(sweep, diff_attr)
        if not same_vals:
            continue  # e.g. control_mean_same/diff empty when no control_windows was used
        ax.plot(
            x, same_vals, marker="o", linestyle="-", color=color,
            label=f"{label} (same)", zorder=3,
        )
        ax.plot(
            x, diff_vals, marker="o", linestyle="--", color=_lighten_color(color, lighten_amount),
            label=f"{label} (different)", zorder=3,
        )

        ax.set_xticks(list(x))
        ax.set_xticklabels(sweep.bin_labels, rotation=30, ha='right')
        ax.set_xlabel("Nearest Antecedent Token Distance")
        # ax.set_ylabel("Effect Size (Cohen's $d$)")        # ax.set_title(qt)

    axes[0].set_ylabel("Mean cosine similarity")
    axes[0].legend(loc="center", fontsize="small", bbox_to_anchor=(0.45, 1.1), ncols=3, frameon=False)
    # fig.suptitle(title)
    # plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


import matplotlib.patches as patches



def plot_cohens_d_by_bin_per_quote_type_with_mean_similarity(
    results: Dict[str, BinnedSweepResult],
    overall_sweep: BinnedSweepResult,
    title: str,
    out_path: str,
    lighten_amount: float = 0.55,
):
    """
    4-panel figure combining `plot_cohens_d_by_bin_per_quote_type` and
    `plot_mean_similarity_by_bin_per_quote_type` into one row: the first
    `len(results)` panels are exactly `plot_cohens_d_by_bin_per_quote_type`
    (one per quote type, sharing a Cohen's d y-axis and the pair-count bars
    on a secondary axis), and a final column on the right is
    `plot_mean_similarity_by_bin_per_quote_type`'s single panel (raw mean
    cosine similarity vs. token-distance bin, same/different-speaker lines
    per model), driven by `overall_sweep` rather than by quote type -
    typically the un-filtered `BinnedSweepResult` from
    `sweep_cohens_d_over_distance_bins`, since similarity isn't itself
    broken out by quote type here.

    Feed `results` the output of
    `sweep_cohens_d_over_distance_bins_by_quote_type`, and `overall_sweep`
    the output of `sweep_cohens_d_over_distance_bins` (or any other single
    `BinnedSweepResult` you want as the rightmost column).

    The similarity panel is a different quantity (raw cosine similarity,
    not a standardized effect size) so it does NOT share the Cohen's d
    y-axis of the first `len(results)` panels, and it gets its own legend
    (up to 6 lines: joint/direct/control x same-/different-speaker).
    """
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 17,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "legend.fontsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 15,
    })

    quote_types = list(results.keys())
    n = len(quote_types)
    fig, axes = plt.subplots(1, n + 1, figsize=(20.0, 3.15))
    cohens_axes = list(axes[1:n+1])
    sim_ax = axes[0]
    for ax in cohens_axes[1:]:
        ax.sharey(cohens_axes[0])

    all_pair_counts = [
        n_same + n_diff for sweep in results.values() for n_same, n_diff in sweep.joint_n
    ]
    shared_count_ylim = (0, max(all_pair_counts) * 1.4) if all_pair_counts else (0, 1)

    secondary_axes = []
    for ax, qt in zip(cohens_axes, quote_types):
        sweep = results[qt]
        x = range(len(sweep.bin_labels))

        ax2 = _add_pair_count_bars(ax, x, sweep.joint_n)
        ax2.set_ylim(*shared_count_ylim)
        secondary_axes.append(ax2)

        ax.plot(x, sweep.joint_cohens_d, marker="o", label="Joint", zorder=3,  markersize=3.5)
        ax.plot(x, sweep.direct_cohens_d, marker="o", label="Direct", zorder=3,  markersize=3.5)
        if sweep.control_cohens_d:
            ax.plot(x, sweep.control_cohens_d, marker="o", label="ModernBERT", zorder=3,  markersize=3.5)
        ax.set_xticks(list(x))

        labs = []
        for cnt, i in enumerate(sweep.bin_labels) : 
            if cnt % 2 == 0 :
                labs.append(i)
            else : 
                labs.append('')
                
        ax.set_xticklabels(labs, rotation=-30, ha='left')
        ax.set_xlabel("Token Distance Bin")
        ax.set_title(f"Q-Q - {qt}")

    # cohens_axes[0].legend(loc="lower center", fontsize="small", bbox_to_anchor=(1, 1.15), ncols=3, frameon=False)
    # Only label/show the secondary axis on the rightmost Cohen's d panel
    # to avoid cluttering every panel with a redundant "# pairs" label.
    for ax2 in secondary_axes[:-1]:
        ax2.set_ylabel("")
        ax2.set_yticklabels([])
    secondary_axes[-1].set_ylabel("# pairs", color="grey")

    cohens_axes[0].set_ylabel("Effect Size (Cohen's $d$)")

    # Rightmost column: mean-similarity panel, same drawing logic as
    # `plot_mean_similarity_by_bin_per_quote_type`, fed by `overall_sweep`.
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    model_specs = [
        ("Joint", "joint_mean_same", "joint_mean_diff", prop_cycle[0]),
        ("Direct", "direct_mean_same", "direct_mean_diff", prop_cycle[1]),
        ("ModernBERT", "control_mean_same", "control_mean_diff", prop_cycle[2]),
    ]
    x_sim = range(len(overall_sweep.bin_labels))
    for label, same_attr, diff_attr, color in model_specs:
        same_vals = getattr(overall_sweep, same_attr)
        diff_vals = getattr(overall_sweep, diff_attr)
        if not same_vals:
            continue  # e.g. control_mean_same/diff empty when no control_windows was used
        sim_ax.plot(
            x_sim, same_vals, marker="s", linestyle="-", color=_lighten_color(color, 0.3),
            label=f"{label} (same)", zorder=3, markersize=5
        )
        sim_ax.plot(
            x_sim, diff_vals, marker="s", linestyle="--", color=_lighten_color(color, 0.6),
            label=f"{label} (different)", zorder=3,  markersize=5
        )
        sim_ax.set_title('Mention-Mention Cosine Similarities')
        
    sim_ax.set_xticks(list(x_sim))
    labs = []
    for cnt, i in enumerate(overall_sweep.bin_labels) : 
        if cnt % 2 == 0 :
            labs.append(i)
        else : 
            labs.append('')
            
    sim_ax.set_xticklabels(labs, rotation=30, ha='right')

    # sim_ax.set_xticklabels(overall_sweep.bin_labels, rotation=60, ha='right' )
    sim_ax.set_xlabel("Nearest Antecedent Token Distance")
    sim_ax.set_ylabel("Mean cosine similarity")
    # sim_ax.yaxis.set_label_position("right")
    # sim_ax.yaxis.tick_right()
    # sim_ax.set_facecolor('#f0f0f0') 
    fig.patches.append(patches.Rectangle(
        (0.085, -0.25), 0.25, 1.25,   # (x, y, width, height) in figure fraction coords
        transform=fig.transFigure,
        facecolor='#d6e4f0',
        zorder=-1
    ))

    sim_ax.legend(loc="lower center", fontsize="small", bbox_to_anchor=(0.55, 1.125), ncols=2, frameon=False)
    
    extra_gap = 0.05  # fraction of figure width
    
    for ax in axes[1:]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + extra_gap, pos.y0, pos.width, pos.height])

        
    # fig.suptitle(title)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def plot_cohens_d_by_bin_per_quote_type_with_overall(
    results: Dict[str, BinnedSweepResult],
    overall_sweep: BinnedSweepResult,
    title: str,
    out_path: str,
    overall_label: str = "All",
):
    """
    4-panel figure combining `plot_cohens_d_by_bin_per_quote_type` with an
    extra column reusing `plot_cohens_d_by_bin`: the first `len(results)`
    panels are exactly `plot_cohens_d_by_bin_per_quote_type` (one per quote
    type), and a final column on the right is `plot_cohens_d_by_bin`'s
    content (Cohen's d vs. token-distance bin, not restricted to any single
    quote type), driven by `overall_sweep`. All panels share the same
    Cohen's d y-axis and the same pair-count secondary-axis scale, since
    every column is the same quantity, the last one just isn't sliced by
    quote type.

    Feed `results` the output of
    `sweep_cohens_d_over_distance_bins_by_quote_type`, and `overall_sweep`
    the output of `sweep_cohens_d_over_distance_bins` run on the
    unfiltered (all quote types) windows. `overall_label` sets the title
    of the rightmost panel (default `"All"`).
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 17,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "legend.fontsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 15,
    })
    quote_types = list(results.keys())
    n = len(quote_types)
    fig, axes = plt.subplots(1, n + 1, figsize=(20, 3.15), sharey=True)

    all_sweeps = list(results.values()) + [overall_sweep]
    all_pair_counts = [
        n_same + n_diff for sweep in all_sweeps for n_same, n_diff in sweep.joint_n
    ]
    shared_count_ylim = (0, max(all_pair_counts) * 1.4) if all_pair_counts else (0, 1)

    panels = [(overall_label, overall_sweep)] + list(zip(quote_types, results.values())) 
    
    secondary_axes = []
    third_axis = []
    for idx, (ax, (label, sweep)) in enumerate(zip(axes, panels)):
        x = range(len(sweep.bin_labels))

        if idx >= 1 : 
            ax2 = _add_pair_count_bars(ax, x, sweep.joint_n)
            ax2.set_ylim(*shared_count_ylim)
            secondary_axes.append(ax2)
        else : 
            ax2 = _add_pair_count_bars(ax, x, sweep.joint_n)
            # ax2.set_ylim(*shared_count_ylim)
            third_axis.append(ax2)
            
        ax.plot(x, sweep.joint_cohens_d, marker="o", label="Joint", zorder=3,  markersize=3.5)
        ax.plot(x, sweep.direct_cohens_d, marker="o", label="Direct", zorder=3,  markersize=3.5)
        if sweep.control_cohens_d:
            ax.plot(x, sweep.control_cohens_d, marker="o", label="ModernBERT", zorder=3,  markersize=3.5)
        ax.set_xticks(list(x))
        if idx == 0: 
            # ax.set_xticks(list(range(0,len(sweep.bin_labels),2)))
            labs = []
            for cnt, i in enumerate(sweep.bin_labels) : 
                if cnt % 2 == 0 :
                    labs.append(i)
                else : 
                    labs.append('')
            ax.set_xticklabels(labs, rotation=30, ha='right')
            ax.set_xlabel("Nearest Antecedent Token Distance")
            ax.set_title("Mention-Mention Anaphora")
        else : 
            labs = []
            for cnt, i in enumerate(sweep.bin_labels) : 
                if cnt % 2 == 0 :
                    labs.append(i)
                else : 
                    labs.append('')
            # ax.set_xticks(list(range(0,len(sweep.bin_labels),2)))
            ax.set_xticklabels(labs, rotation=-30, ha='left')
            ax.set_xlabel("Token Distance Bin")
            ax.set_title(f"Q-M - {label}")

    # Only label/show the secondary axis on the rightmost panel to avoid
    # cluttering every panel with a redundant "# pairs" label.
    for ax2 in secondary_axes[:-1]:
        ax2.set_ylabel("")
        ax2.set_yticklabels([])
    secondary_axes[-1].set_ylabel("# pairs", color="grey")
    
    for ax2 in third_axis[:-1]:
        ax2.set_ylabel("")
        ax2.set_yticklabels([])
    third_axis[-1].set_ylabel("# pairs", color="grey")

    axes[0].set_ylabel("Effect Size (Cohen's $d$)")
    lines1, labels1 = axes[0].get_legend_handles_labels()
    lines2, labels2 = secondary_axes[0].get_legend_handles_labels()
    axes[0].legend(lines1 + lines2, labels1 + labels2, ncols=4, loc='center', bbox_to_anchor=(2.75, 1.25), frameon=False)
    # fig.suptitle(title)

    fig.patches.append(patches.Rectangle(
        (0.085, -0.25), 0.25, 1.35,   # (x, y, width, height) in figure fraction coords
        transform=fig.transFigure,
        facecolor='#d6e4f0',
        zorder=-1
    ))

    extra_gap = 0.05  # fraction of figure width
    
    for ax in axes[1:]:
        pos = ax.get_position()
        ax.set_position([pos.x0 + extra_gap, pos.y0, pos.width, pos.height])
        
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()



@dataclass
class AccuracySeparabilityPoint:
    """One (model, quote_type, bin) scatter point: downstream attribution
    accuracy for that bin, paired with the TRUE representation-level
    separability (Cohen's d) for that same bin -- exactly the H2 metric
    behind Figure 2's Q-M panels (`run_rsa_mention_comparison`), NOT
    `run_predicted_mention_rsa_by_bin`'s own Cohen's d (which separates the
    model's own predicted mention from distractors -- a prediction-
    conditioned diagnostic, not the raw gold-speaker-vs-wrong-speaker
    separability Figure 2 reports). See
    `collect_accuracy_and_qm_separability_by_bin`'s docstring for the full
    distinction.
    """
    model: str
    quote_type: str
    bin: Tuple[float, float]
    bin_label: str
    accuracy: float
    acc_n: int
    cohens_d: float
    sep_n_same: int
    sep_n_diff: int


def collect_accuracy_and_qm_separability_by_bin(
    joint_windows: Iterable[WindowReps],
    direct_windows: Iterable[WindowReps],
    quote_types: Sequence[str],
    bins: Sequence[Tuple[float, float]],
    control_windows: Optional[Iterable[WindowReps]] = None,
) -> List[AccuracySeparabilityPoint]:
    """
    Builds one `AccuracySeparabilityPoint` per (model, quote_type, bin):

      - accuracy: `collect_predicted_mention_accuracy_by_bin`'s per-bin
        accuracy (via `run_predicted_mention_rsa_by_bin`) -- among quotes
        whose *predicted* mention lands in this token-distance bin, the
        fraction correctly attributed. Exactly what your existing
        accuracy-vs-distance plot uses.

      - cohens_d: the genuine Q-M representation-separability effect size
        for that SAME bin, from `run_rsa_mention_comparison` (gold-speaker
        mention similarity vs. wrong-speaker mention similarity, over
        EVERY quote-mention pair whose token distance falls in the bin --
        independent of what the model predicted). This is exactly the
        metric behind Figure 2's Q-M panels.

    IMPORTANT: this deliberately does NOT reuse
    `run_predicted_mention_rsa_by_bin`'s own `*_cohens_d` output for the
    x-axis -- that Cohen's d is conditioned on the model's own prediction
    (see the module-level comparison table in the accompanying writeup),
    which is a different, prediction-conditioned quantity from the raw
    separability Figure 2 reports. Mixing the two would misrepresent what
    Figure 2 measured. Here, accuracy and separability are computed by
    two independently-correct code paths and joined only by sharing the
    same (quote_type, bin) key.

    `joint_windows`/`direct_windows`/`control_windows` are materialized
    into lists once up front since both machineries re-scan/re-filter
    them per quote type.
    """
    joint_windows = list(joint_windows)
    direct_windows = list(direct_windows)
    if control_windows is not None:
        control_windows = list(control_windows)

    points: List[AccuracySeparabilityPoint] = []

    for qt in quote_types:
        # --- accuracy side: reuse the existing predicted-mention machinery
        acc_result = run_predicted_mention_rsa_by_bin(
            joint_windows, direct_windows, bins,
            control_windows=control_windows, quote_type=qt,
        )
        acc_by_bin = {
            "joint": dict(zip(acc_result.bins, zip(acc_result.joint_accuracy, acc_result.joint_acc_n))),
            "direct": dict(zip(acc_result.bins, zip(acc_result.direct_accuracy, acc_result.direct_acc_n))),
        }
        if acc_result.control_accuracy:
            acc_by_bin["control"] = dict(
                zip(acc_result.bins, zip(acc_result.control_accuracy, acc_result.control_acc_n))
            )

        # --- separability side: true Q-M Cohen's d, Figure-2 style, same bins
        joint_qt = filter_windows_by_quote_type(joint_windows, qt)
        direct_qt = filter_windows_by_quote_type(direct_windows, qt)
        control_qt = filter_windows_by_quote_type(control_windows, qt) if control_windows is not None else None

        for lo, hi in bins:
            try:
                joint_sep, direct_sep, control_sep = run_rsa_mention_comparison(
                    joint_qt, direct_qt, control_windows=control_qt,
                    min_token_distance=lo, max_token_distance=hi,
                )
            except ValueError as e:
                print(f"[collect_accuracy_and_qm_separability_by_bin] "
                      f"skipping quote_type={qt!r} bin=[{lo}, {hi}] (separability): {e}")
                continue

            sep_by_model = {"joint": joint_sep, "direct": direct_sep}
            if control_sep is not None:
                sep_by_model["control"] = control_sep

            for model, sep_res in sep_by_model.items():
                acc_entry = acc_by_bin.get(model, {}).get((lo, hi))
                if acc_entry is None:
                    continue  # this bin didn't survive on the accuracy side for this model
                accuracy, acc_n = acc_entry
                if math.isnan(accuracy):
                    continue
                points.append(AccuracySeparabilityPoint(
                    model=model, quote_type=qt, bin=(lo, hi),
                    bin_label=f"[{lo:g}, {hi:g}]",
                    accuracy=accuracy, acc_n=acc_n,
                    cohens_d=sep_res.cohens_d,
                    sep_n_same=sep_res.n_same, sep_n_diff=sep_res.n_diff,
                ))
    return points




def plot_accuracy_vs_separability_by_quote_type(
    points: List["AccuracySeparabilityPoint"],
    out_path: str,
    quote_type_colors: Optional[Dict[str, str]] = None,
    models: Sequence[str] = ("joint", "direct"),
    markers_by_model: Optional[Dict[str, str]] = None,
    size_scale: float = 4.0,
    fit_per_model_within_type: bool = False,
    annotate_correlation: bool = True,
) -> None:
    """
    Scatter plot: x = Q-M Cohen's d, y = attribution accuracy, one point
    per (model, quote_type, bin), same data as
    `plot_accuracy_vs_separability_scatter` but regrouped so that:

      - color encodes quote_type (checking the within-type relationship,
        i.e. controlling for quote-type difficulty),
      - marker shape encodes model,
      - the regression line + Pearson r / Spearman rho are fit PER QUOTE
        TYPE, pooling every model's points of that type together (unless
        `fit_per_model_within_type=True`, see below).

    `fit_per_model_within_type`: if True, additionally draws one dashed fit
    line PER (quote_type, model) pair instead of one per quote_type alone --
    use this if you specifically want to see whether joint and direct trace
    out the SAME within-type relationship (lines roughly overlapping) or
    genuinely different slopes even after controlling for quote type (lines
    diverging). Default False keeps the plot readable with one line per
    quote_type only.

    `quote_type_colors`: override the default color per quote_type (falls
    back to matplotlib's tab10 cycle over the sorted quote types found in
    `points`).

    `markers_by_model`: override the default marker per model (falls back
    to `{"joint": "o", "direct": "s", "control": "^"}`).

    Feed it the output of `collect_accuracy_and_qm_separability_by_bin`.
    """
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "legend.fontsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 15,
    })
    
    if scipy_stats is None:
        raise ImportError("scipy is required for this plot (regression + correlation)")

    quote_types = sorted({p.quote_type for p in points})
    if quote_type_colors is None:
        cmap = plt.get_cmap("tab10")
        quote_type_colors = {qt: cmap(i % 10) for i, qt in enumerate(quote_types)}

    if markers_by_model is None:
        markers_by_model = {"joint": "o", "direct": "s", "control": "^"}

    model_labels = {"joint": "Joint", "direct": "Direct", "control": "Control"}

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    all_x, all_y = [], []
    for qt in quote_types:
        qt_points = [p for p in points if p.quote_type == qt]
        color = quote_type_colors[qt]

        # Scatter: split by model only for marker shape, same color per qt.
        for model in models:
            model_qt_points = [p for p in qt_points if p.model == model]
            if not model_qt_points:
                continue
            xs = np.array([p.cohens_d for p in model_qt_points])
            ys = np.array([p.accuracy for p in model_qt_points])
            sizes = size_scale * np.sqrt(np.array([max(p.acc_n, 1) for p in model_qt_points]))
            ax.scatter(
                xs, ys, s=sizes, color=color, marker=markers_by_model.get(model, "o"),
                alpha=0.8, edgecolor="white", linewidth=0.4,
                label=f"{qt} - {model_labels.get(model, model)}",
            )

        # Pooled-across-models fit line for this quote type.
        xs_all = np.array([p.cohens_d for p in qt_points])
        ys_all = np.array([p.accuracy for p in qt_points])
        if len(xs_all) >= 3:
            slope, intercept, r, p_val, _ = scipy_stats.linregress(xs_all, ys_all)
            rho, rho_p = scipy_stats.spearmanr(xs_all, ys_all)
            x_line = np.linspace(xs_all.min(), xs_all.max(), 50)
            ax.plot(
                x_line, slope * x_line + intercept, color=color, linestyle="--", linewidth=1.8,
                label=f"{qt} fit (r={r:.2f}, p={p_val:.3f}; $\\rho$={rho:.2f}, n={len(xs_all)})",
            )
        all_x.extend(xs_all.tolist())
        all_y.extend(ys_all.tolist())

        # Optional: separate fit per (quote_type, model), lighter/dotted.
        if fit_per_model_within_type:
            for model in models:
                model_qt_points = [p for p in qt_points if p.model == model]
                xs_m = np.array([p.cohens_d for p in model_qt_points])
                ys_m = np.array([p.accuracy for p in model_qt_points])
                if len(xs_m) < 3:
                    continue
                slope_m, intercept_m, r_m, p_m, _ = scipy_stats.linregress(xs_m, ys_m)
                x_line_m = np.linspace(xs_m.min(), xs_m.max(), 50)
                ax.plot(
                    x_line_m, slope_m * x_line_m + intercept_m,
                    color=color, linestyle=":", linewidth=1.0, alpha=0.7,
                    label=f"{qt} - {model_labels.get(model, model)} fit (r={r_m:.2f})",
                )

    if annotate_correlation and len(all_x) >= 3:
        r_all, p_all = scipy_stats.pearsonr(all_x, all_y)
        rho_all, rho_p_all = scipy_stats.spearmanr(all_x, all_y)
        ax.text(
            0.02, 1.1,
            f"Overall: Pearson r={r_all:.2f} (p={p_all:.3g})\n"
            f"Spearman $\\rho$={rho_all:.2f} (p={rho_p_all:.3g})\nn={len(all_x)} points",
            transform=ax.transAxes, va="top", ha="left", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("Q-M - Cohen's $d$")
    ax.set_ylabel("Attribution accuracy")
    # ax.set_title("Accuracy vs. separability, grouped by quote type")
    ax.legend(loc="lower right", fontsize=9, ncol=1, framealpha=.3, markerfirst=False)#, bbox_to_anchor=(1.3,0))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def within_quote_type_correlation_table(
    points: List["AccuracySeparabilityPoint"],
    models: Sequence[str] = ("joint", "direct"),
    print_table: bool = True,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """
    Companion diagnostic to `plot_accuracy_vs_separability_by_quote_type`:
    computes Pearson r / Spearman rho separately for every
    (quote_type, model) pair (i.e. the finest-grained breakdown, without
    pooling models together), so you can directly check whether the
    accuracy~separability coupling is consistent across models WITHIN each
    quote type, rather than only inspecting it visually.

    Returns `{(quote_type, model): {"r": ..., "r_p": ..., "rho": ...,
    "rho_p": ..., "n": ...}}`. Also prints a formatted table to stdout by
    default (`print_table=True`) since this is primarily meant as a quick
    robustness check to report alongside the plot, not as an intermediate
    value you always need to consume programmatically.
    """
    if scipy_stats is None:
        raise ImportError("scipy is required for this diagnostic")

    quote_types = sorted({p.quote_type for p in points})
    results: Dict[Tuple[str, str], Dict[str, float]] = {}

    rows = []
    for qt in quote_types:
        for model in models:
            subset = [p for p in points if p.quote_type == qt and p.model == model]
            if len(subset) < 3:
                continue
            xs = np.array([p.cohens_d for p in subset])
            ys = np.array([p.accuracy for p in subset])
            r, r_p = scipy_stats.pearsonr(xs, ys)
            rho, rho_p = scipy_stats.spearmanr(xs, ys)
            results[(qt, model)] = {"r": r, "r_p": r_p, "rho": rho, "rho_p": rho_p, "n": len(subset)}
            rows.append((qt, model, r, r_p, rho, rho_p, len(subset)))

    if print_table and rows:
        header = f"{'quote_type':<12}{'model':<10}{'r':>8}{'p(r)':>12}{'rho':>8}{'p(rho)':>12}{'n':>5}"
        print(header)
        print("-" * len(header))
        for qt, model, r, r_p, rho, rho_p, n in rows:
            print(f"{qt:<12}{model:<10}{r:>8.2f}{r_p:>12.3g}{rho:>8.2f}{rho_p:>12.3g}{n:>5}")

    return results