"""
Low-dimensional random-projection embeddings of EVERY training example's chosen
gradient (PiSSA-LoRA), so the chosen gradients can be clustered.

WHY
---
train_weighted_dpo.py scores each rejected token against two chosen directions --
this example's own chosen gradient ("single_*") and the dataset-mean chosen
gradient ("aggregated_*") -- and takes the elementwise max. The aggregated
direction is ONE mean over 210k heterogeneous examples. If the chosen gradients
form modes, K cluster means are a strictly richer reference set, and
    cos_max_cluster[t] = max_k cos(g_token[t], c_k)
lets every token follow whichever chosen MODE it is closest to, instead of an
average that may sit between all of them.

Clustering needs a distance between examples, but the gradient dimension here is
    D = sum of LoRA numels ~ 1.6e8   (r=64, all-linear, 32 layers)
so the gradients can never be materialized side by side (210k x D x 4B ~ 130 TB).
This script produces the surrogate: one EMB_DIM-dimensional vector per example
whose inner products (hence cosines) match the full-space ones up to a
relative error of O(1/sqrt(EMB_DIM)) ~ 3% at EMB_DIM=1000. Cluster on those,
then re-materialize the K cluster means in FULL space (see the pipeline note at
the bottom of this docstring).

PROJECTION BACKENDS
-------------------
Both are exact Johnson-Lindenstrauss-style sketches with variance O(1/EMB_DIM);
they differ only in cost. Neither ever materializes the full D x EMB_DIM matrix:
the matrix is defined chunk by chunk from a fixed seed, so any process/shard
reproduces the identical projection.

  "hashed" (default, CountSketch / feature hashing, Weinberger et al. 2009)
      Each of the D input coordinates is hashed to exactly ONE output bucket with
      a random +-1 sign:   emb[b] = sum_{d: bucket[d]==b} sign[d] * g[d].
      The "matrix" is therefore D bucket ids + D signs (~0.8 GB, built once from
      per-chunk seeds and kept resident), and projecting one example is a single
      fused scatter-add over D elements -- milliseconds.
      Cost per example: O(D)  =  1.6e8 ops.

  "dense" (Rademacher/Gaussian, the literal "chunked random matrix" approach)
      Regenerates a [PROJ_CHUNK, EMB_DIM] random block per chunk per example and
      accumulates g_chunk @ R_chunk.
      Cost per example: O(D * EMB_DIM) = 1.6e11 RNG draws + MACs, i.e. minutes
      per example (the sibling Compare-Different-Gradients-Projections project
      measured exactly this and switched to a sparse sketch to reach 9B). Over
      210k examples that is months of GPU time -- NOT viable for the full pass.
      It is kept here as the validation reference: run it on a few hundred
      examples with --validate_dense to confirm the hashed sketch preserves the
      cosine structure, then use "hashed" for the full 210k.

Both backends are seeded per chunk (seed = PROJ_SEED * 1_000_003 + chunk_idx) and
generated on the CPU generator, so the projection is identical across machines,
shards, and GPUs. PROJ_SEED, PROJ_CHUNK and EMB_DIM are recorded in meta.json --
changing any of them changes the embedding space and invalidates old shards.

OUTPUT (per shard, in EMB_DIR)
------------------------------
  emb_{start:06d}_{end:06d}.npz
      idx      int64 [n]              original train-split indices
      emb_<w>  fp32  [n, EMB_DIM]     projected chosen gradient, per weighting
      gnorm_<w> fp32 [n]              EXACT full-space ||g|| (not the sketch's)
  meta.json                           projection config + adapter fingerprint
  emb_*.partial.npz                   resume state, removed on shard completion

`gnorm_*` is computed in full space during the same walk, so downstream code can
normalize exactly rather than trusting the sketch's norm.

WEIGHTINGS mirror the rest of the pipeline (see compute_mean_chosen_gradient.py):
unweighted, weighted_norm1 (1/(2(1-p))), weighted_norm2 (1/||g_token||_2). One
forward, one backward per requested weighting. Default is weighted_norm1 only,
since that is what GRADIENT_DIRECTION uses in train_weighted_dpo.py.

WHERE THIS SITS IN THE PIPELINE
-------------------------------
  1. THIS SCRIPT              210k chosen gradients -> [210k, 1000] embeddings
  2. cluster_chosen_gradients.py   embeddings -> K assignments (+ proj centroids)
  3. compute_cluster_chosen_gradients.py   a SECOND gradient pass that averages
     full-space chosen gradients per cluster -> K x D centroids. REQUIRED: the
     projected centroids from step 2 live in sketch space and cannot be dotted
     against the rejected-token gradients.
  4. compute_dpo_gradient_products_pissa_lora.py  + the K centroids as extra
     directions -> products_clusters_<w>.pt [K, T] per example
  5. train_weighted_dpo.py    new GRADIENT_DIRECTION "max_cluster_<w>"

Run with --start_idx/--end_idx to shard across GPUs, exactly like
compute_dpo_gradient_products_pissa_lora.py.
"""

import argparse
import json
import os
import re
import time

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
load_dotenv(".env")
from datasets import load_dataset
from tqdm import tqdm

import pissa_lora_common as C

# ── Config ──────────────────────────────────────────────────────────────────
EMB_DIR = "/data/weighted-dpo/pissa-lora-chosen-embeddings"

EMB_DIM = 1000            # projected dimension (see note on K below)
PROJ_METHOD = "hashed"    # "hashed" (viable at 210k) | "dense" (validation only)
PROJ_SEED = 0             # changing this changes the embedding space
PROJ_CHUNK = 1 << 22      # coordinates per seeded chunk; part of the projection's identity
PROJ_TYPE = "rademacher"  # dense backend only: "rademacher" (+-1) | "gaussian"

# Which chosen-gradient weightings to embed. weighted_norm1 is what
# train_weighted_dpo.py's GRADIENT_DIRECTION currently consumes; add the others
# only if you intend to cluster them too (each costs one extra backward).
WEIGHTINGS = ("weighted_norm1",)
ALL_WEIGHTINGS = ("unweighted", "weighted_norm1", "weighted_norm2")

FLUSH_EVERY = 200         # examples between resume checkpoints

# MAINTENANCE MODE. Set to a row count (e.g. 5000, ~20 MB/file) to split every
# oversized checkpoint chunk and finalized shard in EMB_DIR down to that size,
# then exit WITHOUT loading the model or touching the dataset. Leave None for
# normal runs.
#
# Why it exists: a legacy partial adopted by resume_state arrives as ONE chunk of
# ~200k rows (~0.83 GB), while every normally flushed chunk is FLUSH_EVERY rows
# (~0.8 MB). That outlier has to be read whole by anything touching the parts
# directory. Splitting it bounds every read and write in the pipeline.
RECHUNK_MAX_ROWS = None
NUM_EXAMPLES = None       # None -> whole train split; int to limit (smoke test)

# EMB_DIM vs K: random projection preserves the k-means objective within (1+eps)
# at EMB_DIM ~ K/eps^2 (Boutsidis et al.), so 1000 dims comfortably supports
# K up to ~64-100 (eps ~ 0.25-0.3). If you want K >= 256, raise EMB_DIM to 4096 --
# the hashed backend's cost is independent of EMB_DIM, only the stored matrix and
# the [N, EMB_DIM] output grow.

assert set(WEIGHTINGS) <= set(ALL_WEIGHTINGS), f"unknown weighting in {WEIGHTINGS}"
assert PROJ_METHOD in ("hashed", "dense")


# ── Projection backends ─────────────────────────────────────────────────────

def _chunk_generator(chunk_idx):
    """CPU generator for one chunk. CPU (not CUDA) so the projection is identical
    across machines and GPU counts; per-chunk seeding makes chunk order and shard
    layout irrelevant."""
    g = torch.Generator()
    g.manual_seed(PROJ_SEED * 1_000_003 + chunk_idx)
    return g


def _chunk_bounds(total_dim):
    """[(offset, length), ...] partition of the gradient into seeded chunks."""
    return [(off, min(PROJ_CHUNK, total_dim - off))
            for off in range(0, total_dim, PROJ_CHUNK)]


class HashedProjection:
    """CountSketch: every input coordinate -> one output bucket, with a +-1 sign.

    Unbiased on inner products (E[<Sx, Sy>] = <x, y>) with variance ~
    (||x||^2 ||y||^2 + <x,y>^2) / EMB_DIM, so cosines carry ~1/sqrt(EMB_DIM)
    relative error -- the same rate as a dense Gaussian sketch, at 1/EMB_DIM the
    cost. Unlike a very-sparse JL matrix it touches EVERY coordinate exactly once
    (nothing is dropped), which matters for gradients whose mass concentrates in
    a few LoRA tensors.

    The bucket/sign tables are built once from the per-chunk seeds and kept
    resident (D * 5 bytes ~ 0.8 GB at D = 1.6e8), so no random numbers are drawn
    per example.
    """

    name = "hashed"

    def __init__(self, dim, numels, device):
        self.dim, self.numels, self.device = int(dim), list(numels), device
        total = int(sum(numels))
        buckets, signs = [], []
        for chunk_idx, (_, length) in enumerate(_chunk_bounds(total)):
            g = _chunk_generator(chunk_idx)
            buckets.append(torch.randint(0, self.dim, (length,), generator=g,
                                         dtype=torch.int32))
            signs.append(torch.randint(0, 2, (length,), generator=g,
                                       dtype=torch.int8).mul_(2).sub_(1))
        self.bucket = torch.cat(buckets).to(device)
        self.sign = torch.cat(signs).to(device)
        del buckets, signs
        gb = (self.bucket.numel() * 4 + self.sign.numel()) / 1e9
        print(f"  [proj] hashed sketch ready: D={total:,} -> {self.dim}, "
              f"{len(_chunk_bounds(total))} seeded chunks, {gb:.2f} GB resident")

    @torch.no_grad()
    def project(self, grads):
        """(emb [EMB_DIM] fp32 on device, exact full-space ||g|| as a float).

        `grads` is the per-parameter grad list in C.get_lora_param_list order;
        None entries (unused params) are skipped but still advance the offset, so
        the coordinate<->bucket mapping stays aligned.
        """
        out = torch.zeros(self.dim, device=self.device, dtype=torch.float32)
        sq = torch.zeros((), device=self.device, dtype=torch.float64)
        off = 0
        for g, n in zip(grads, self.numels):
            if g is not None:
                v = torch.nan_to_num(g.detach().reshape(-1).float(),
                                     nan=0.0, posinf=0.0, neginf=0.0)
                sq += v.double().pow(2).sum()
                out.scatter_add_(0, self.bucket[off:off + n].long(),
                                 v * self.sign[off:off + n].to(torch.float32))
            off += n
        return out, float(sq.sqrt())


class DenseChunkedProjection:
    """Dense Rademacher/Gaussian JL, regenerated chunk-by-chunk per example.

    The literal "materialize chunks of the random matrix with the same seed"
    approach. Statistically equivalent to the hashed sketch (same O(1/dim)
    variance) but costs D * EMB_DIM RNG draws + MACs PER EXAMPLE, which is
    minutes/example at D ~ 1.6e8. Validation reference only -- do not run this
    over the full 210k.
    """

    name = "dense"

    def __init__(self, dim, numels, device):
        self.dim, self.numels, self.device = int(dim), list(numels), device
        self.total = int(sum(numels))
        self.scale = 1.0 / (self.dim ** 0.5)
        print(f"  [proj] dense chunked sketch: D={self.total:,} -> {self.dim}, "
              f"{len(_chunk_bounds(self.total))} blocks regenerated per example "
              f"({self.total * self.dim / 1e9:.0f}e9 draws/example -- slow by design)")

    def _block(self, chunk_idx, rows):
        g = _chunk_generator(chunk_idx)
        if PROJ_TYPE == "gaussian":
            R = torch.randn(rows, self.dim, generator=g, dtype=torch.float32)
        else:
            R = torch.randint(0, 2, (rows, self.dim), generator=g,
                              dtype=torch.float32).mul_(2).sub_(1)
        return R.mul_(self.scale).to(self.device)

    @torch.no_grad()
    def project(self, grads):
        """Same contract as HashedProjection.project. The gradient is walked as
        one virtual vector and re-cut into PROJ_CHUNK blocks, so block seeding is
        independent of the parameter layout."""
        out = torch.zeros(self.dim, device=self.device, dtype=torch.float32)
        sq = torch.zeros((), device=self.device, dtype=torch.float64)
        buf, buf_len, chunk_idx = [], 0, 0

        def flush(block):
            nonlocal chunk_idx
            out.add_(block @ self._block(chunk_idx, block.numel()))
            chunk_idx += 1

        for g, n in zip(grads, self.numels):
            v = (torch.nan_to_num(g.detach().reshape(-1).float(),
                                  nan=0.0, posinf=0.0, neginf=0.0)
                 if g is not None else torch.zeros(n, device=self.device))
            sq += v.double().pow(2).sum()
            buf.append(v)
            buf_len += n
            while buf_len >= PROJ_CHUNK:
                cat = buf[0] if len(buf) == 1 else torch.cat(buf)
                pos = 0
                while cat.numel() - pos >= PROJ_CHUNK:
                    flush(cat[pos:pos + PROJ_CHUNK])
                    pos += PROJ_CHUNK
                buf = [cat[pos:].clone()] if cat.numel() - pos else []
                buf_len = cat.numel() - pos
        if buf_len:
            flush(buf[0] if len(buf) == 1 else torch.cat(buf))
        return out, float(sq.sqrt())


def make_projection(method, dim, numels, device):
    cls = HashedProjection if method == "hashed" else DenseChunkedProjection
    return cls(dim, numels, device)


# ── Per-example chosen gradients ────────────────────────────────────────────

def chosen_embeddings(model, tokenizer, messages, lora_params, projection,
                      weightings, timing=None):
    """Project this example's chosen gradient under each requested weighting.

    One forward, then one backward per weighting (retain_graph), each projected
    immediately so at most one full-space gradient is alive at a time.
    Returns {weighting: (emb_np [EMB_DIM], gnorm float)} or None if unusable.

    `timing`, if given, is a dict accumulating {"n", "total_s", "proj_s"} so the
    caller can report examples/s and how much of it the projection costs. The
    GPU is synchronized before each projection timer starts, otherwise the
    still-running backward would be billed to the projection.

    The per-token weights match compute_mean_chosen_gradient.py exactly:
      unweighted      sum_i -log p(t_i)
      weighted_norm1  sum_i -log p(t_i) / (2(1 - p(t_i)))      [inverse ||g_i||_1]
      weighted_norm2  sum_i -log p(t_i) / ||g_i||_2
    with ||g_i||_2^2 = 1 - 2p + sum_j p_j^2 and all weights detached.
    """
    input_ids, comp_start, comp_ids = C.build_chat_ids(tokenizer, messages)
    if input_ids is None:
        return None
    device = next(model.parameters()).device
    t_start, proj_s = time.perf_counter(), 0.0

    model.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids.to(device))

    T = len(comp_ids)
    pos = comp_start + torch.arange(T, device=device) - 1
    log_probs = F.log_softmax(outputs.logits[0, pos].float(), dim=-1)
    ids = torch.as_tensor(comp_ids, device=device)
    nll = -log_probs[torch.arange(T, device=device), ids]

    with torch.no_grad():
        p = torch.exp(-nll).clamp(max=1.0 - 1e-6)
        coll = torch.exp(2.0 * log_probs).sum(dim=-1)           # sum_j p_j^2
        w1 = 1.0 / (2.0 * (1.0 - p))
        w2 = 1.0 / torch.sqrt((1.0 - 2.0 * p + coll).clamp(min=1e-12))

    losses = {"unweighted": nll.sum(),
              "weighted_norm1": (nll * w1).sum(),
              "weighted_norm2": (nll * w2).sum()}

    result = {}
    for i, name in enumerate(weightings):
        grads = torch.autograd.grad(
            losses[name], lora_params,
            retain_graph=(i < len(weightings) - 1), allow_unused=True)
        if device.type == "cuda":
            torch.cuda.synchronize()        # bill the backward to the backward
        t_proj = time.perf_counter()
        emb, gnorm = projection.project(grads)
        result[name] = (emb.cpu().numpy(), gnorm)   # .cpu() forces the sync
        proj_s += time.perf_counter() - t_proj
        del grads

    if timing is not None:
        timing["n"] += 1
        timing["total_s"] += time.perf_counter() - t_start
        timing["proj_s"] += proj_s

    del outputs, log_probs, nll, p, coll, w1, w2, losses
    return result


# ── Shard I/O ───────────────────────────────────────────────────────────────

def shard_paths(start_idx, end_idx):
    """(final npz, parts dir, progress json, legacy single-file partial).

    Checkpoints are INCREMENTAL: each flush writes only the rows produced since
    the previous flush, into parts_<range>/chunk_NNNN.npz, and records progress
    in a tiny json. The alternative — rewriting one growing .npz every flush —
    costs O(rows so far) per checkpoint (~800 MB per flush near the end of a
    208k-example shard) and forces resume to materialize every row at once.
    Parts live in a subdirectory so the clustering script's emb_*.npz glob does
    not pick them up; they are concatenated into the final npz and deleted on
    completion.
    """
    stem = os.path.join(EMB_DIR, f"emb_{start_idx:06d}_{end_idx:06d}")
    parts = os.path.join(EMB_DIR, f"parts_{start_idx:06d}_{end_idx:06d}")
    return stem + ".npz", parts, os.path.join(parts, "progress.json"), stem + ".partial.npz"


def pack(rows, weightings):
    """rows: list of (idx, {w: (emb, gnorm)}) -> npz-ready dict of arrays."""
    out = {"idx": np.asarray([r[0] for r in rows], dtype=np.int64)}
    for w in weightings:
        out[f"emb_{w}"] = (np.stack([r[1][w][0] for r in rows]).astype(np.float32)
                           if rows else np.zeros((0, EMB_DIM), dtype=np.float32))
        out[f"gnorm_{w}"] = np.asarray([r[1][w][1] for r in rows], dtype=np.float32)
    return out


CHUNK_RE = re.compile(r"^chunk_\d{4}\.npz$")


def _chunk_files(parts_dir):
    """Completed checkpoint chunks, in write order. Only exact chunk_NNNN.npz
    names count -- anything else in parts_dir is scratch from a crashed write."""
    return sorted(f for f in os.listdir(parts_dir) if CHUNK_RE.match(f))


def write_chunk(parts_dir, seq, rows, weightings):
    """Atomically write `rows` as chunk `seq`. Returns the final path.

    The temp name must ALSO end in .npz: np.savez silently appends the extension
    when it is absent, so a "chunk_0001.npz.tmp" target is actually written to
    "chunk_0001.npz.tmp.npz" and the rename then fails on a missing file. The
    temp is named _tmp_* rather than chunk_* so a crashed write can never be
    mistaken for a completed chunk by _chunk_files.
    """
    tmp = os.path.join(parts_dir, f"_tmp_{seq:04d}.npz")
    final = os.path.join(parts_dir, f"chunk_{seq:04d}.npz")
    np.savez(tmp, **pack(rows, weightings))
    os.replace(tmp, final)
    return final


def resume_state(parts_dir, progress_path, legacy_partial, start_idx):
    """(next_idx, skipped, next_chunk_seq), preparing parts_dir for appending.

    A legacy single-file `<stem>.partial.npz` from an earlier version is adopted
    as chunk 0000 by RENAMING it into parts_dir -- no read of its payload, so a
    part-finished 208k shard costs nothing to recover. Its two extra scalar keys
    (next_idx, skipped) are ignored at concatenation time.
    """
    os.makedirs(parts_dir, exist_ok=True)
    # Anything in parts_dir that is not a completed chunk or the progress file is
    # scratch from a crashed write. Discard it: progress.json is the authority on
    # what was committed, so those rows get recomputed rather than risking a
    # partial or duplicated chunk. (Also self-heals the "chunk_NNNN.npz.tmp.npz"
    # files left by the np.savez extension bug fixed in write_chunk.)
    for stale in sorted(f for f in os.listdir(parts_dir)
                        if not CHUNK_RE.match(f) and f != "progress.json"):
        os.remove(os.path.join(parts_dir, stale))
        print(f"  discarded incomplete checkpoint {stale}")
    if os.path.isfile(progress_path):
        with open(progress_path) as f:
            p = json.load(f)
        seq = len(_chunk_files(parts_dir))
        print(f"  resumed: {p['kept']} rows in {seq} part(s), continuing at "
              f"{p['next_idx']}")
        return int(p["next_idx"]), int(p["skipped"]), seq

    if os.path.isfile(legacy_partial):
        z = np.load(legacy_partial)                  # scalars only; payload not read
        next_idx, skipped = int(z["next_idx"]), int(z["skipped"])
        n_rows = int(z["idx"].shape[0])
        del z
        os.replace(legacy_partial, os.path.join(parts_dir, "chunk_0000.npz"))
        write_progress(progress_path, next_idx, skipped, n_rows)
        print(f"  adopted legacy partial as chunk_0000 ({n_rows} rows, no recompute); "
              f"continuing at {next_idx}")
        return next_idx, skipped, 1

    return start_idx, 0, 0


def write_progress(progress_path, next_idx, skipped, kept):
    tmp = progress_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"next_idx": int(next_idx), "skipped": int(skipped),
                   "kept": int(kept)}, f)
    os.replace(tmp, progress_path)


def finalize(parts_dir, final_path, weightings):
    """Concatenate chunk_*.npz in order -> the shard's final npz. Returns the
    merged dict so the caller can report stats without re-reading."""
    chunks = _chunk_files(parts_dir)
    if not chunks:
        raise RuntimeError(f"No chunk_*.npz in {parts_dir} — nothing to finalize.")
    keys = ["idx"] + [f"{p}_{w}" for w in weightings for p in ("emb", "gnorm")]
    acc = {k: [] for k in keys}
    for name in chunks:
        z = np.load(os.path.join(parts_dir, name))
        missing = [k for k in keys if k not in z]
        if missing:
            raise RuntimeError(
                f"{name} lacks {missing} — it was written with a different "
                f"WEIGHTINGS. Remove {parts_dir} and recompute this shard.")
        for k in keys:
            acc[k].append(z[k])
        del z
    # A single chunk (e.g. an adopted legacy partial that needed no extra work)
    # needs no concatenation -- skipping it halves peak memory on ~1 GB arrays.
    merged = {k: (v[0] if len(v) == 1 else np.concatenate(v)) for k, v in acc.items()}
    del acc
    tmp = final_path + ".tmp.npz"
    np.savez(tmp, **merged)
    os.replace(tmp, final_path)
    for name in chunks:
        os.remove(os.path.join(parts_dir, name))
    for leftover in os.listdir(parts_dir):
        os.remove(os.path.join(parts_dir, leftover))
    os.rmdir(parts_dir)
    return merged


def _split_rows(arrays, keys, max_rows):
    """Yield dicts of row-slices, each at most `max_rows` long."""
    n = len(arrays[keys[0]])
    for s in range(0, n, max_rows):
        yield {k: arrays[k][s:s + max_rows] for k in keys}


def rechunk_parts(parts_dir, weightings, max_rows):
    """Rewrite parts_dir so no chunk holds more than `max_rows` rows.

    Needed because an adopted legacy partial arrives as ONE chunk of ~200k rows
    (~0.83 GB) while every normally-flushed chunk is ~200 rows (~0.8 MB); that
    outlier has to be read whole by anything that touches the directory.

    Safe against interruption: the rewrite goes to a staging directory, is
    verified row-for-row against the source (the concatenated idx must be
    IDENTICAL, not merely the same length), and only then replaces the original
    via two atomic directory renames. A crash at any point leaves either the
    original or the verified replacement intact, never a half-rewritten mix.
    """
    import shutil

    keys = ["idx"] + [f"{p}_{w}" for w in weightings for p in ("emb", "gnorm")]
    chunks = _chunk_files(parts_dir)
    if not chunks:
        print(f"  {parts_dir}: no chunks to re-chunk")
        return
    sizes = [(c, np.load(os.path.join(parts_dir, c))["idx"].shape[0]) for c in chunks]
    total = sum(n for _, n in sizes)
    biggest = max(n for _, n in sizes)
    print(f"  {len(chunks)} chunk(s), {total:,} rows, largest {biggest:,} rows")
    if biggest <= max_rows:
        print(f"  already within {max_rows:,} rows per chunk — nothing to do")
        return

    staging = parts_dir + ".rechunk"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    seq, carry = 0, None
    for name, _ in sizes:
        z = np.load(os.path.join(parts_dir, name))
        missing = [k for k in keys if k not in z]
        if missing:
            shutil.rmtree(staging, ignore_errors=True)
            raise RuntimeError(f"{name} lacks {missing} (different WEIGHTINGS?)")
        arrays = {k: z[k] for k in keys}
        del z
        if carry is not None:                       # prepend last short remainder
            arrays = {k: np.concatenate([carry[k], arrays[k]]) for k in keys}
            carry = None
        for piece in _split_rows(arrays, keys, max_rows):
            if len(piece["idx"]) < max_rows:
                carry = piece                       # hold back; may extend later
                break
            np.savez(os.path.join(staging, f"chunk_{seq:04d}.npz"), **piece)
            seq += 1
        del arrays
    if carry is not None:
        np.savez(os.path.join(staging, f"chunk_{seq:04d}.npz"), **carry)
        seq += 1

    # Verify before destroying anything: same rows, same order, same indices.
    new_idx = np.concatenate([np.load(os.path.join(staging, c))["idx"]
                              for c in _chunk_files(staging)])
    old_idx = np.concatenate([np.load(os.path.join(parts_dir, c))["idx"]
                              for c, _ in sizes])
    if new_idx.shape != old_idx.shape or not np.array_equal(new_idx, old_idx):
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(
            f"re-chunk verification FAILED ({new_idx.shape} vs {old_idx.shape}) — "
            f"original left untouched at {parts_dir}")
    progress = os.path.join(parts_dir, "progress.json")
    if os.path.isfile(progress):
        shutil.copyfile(progress, os.path.join(staging, "progress.json"))

    backup = parts_dir + ".old"
    shutil.rmtree(backup, ignore_errors=True)
    os.rename(parts_dir, backup)
    os.rename(staging, parts_dir)
    shutil.rmtree(backup, ignore_errors=True)
    print(f"  re-chunked into {seq} chunks of <= {max_rows:,} rows "
          f"({total:,} rows verified identical)")


def rechunk_final(final_path, weightings, max_rows):
    """Split an already-finalized shard npz into several range-named shard files.

    The clustering script concatenates every emb_*.npz it finds, so splitting by
    row range is transparent to it. Pieces are written and verified before the
    original is removed, so an interruption leaves the original in place (at
    worst alongside pieces, which would duplicate rows -- the loader warns).
    """
    z = np.load(final_path)
    keys = ["idx"] + [f"{p}_{w}" for w in weightings for p in ("emb", "gnorm")]
    missing = [k for k in keys if k not in z]
    if missing:
        raise RuntimeError(f"{final_path} lacks {missing} (different WEIGHTINGS?)")
    arrays = {k: z[k] for k in keys}
    del z
    n = len(arrays["idx"])
    if n <= max_rows:
        print(f"  {os.path.basename(final_path)}: {n:,} rows — nothing to do")
        return

    written = []
    for piece in _split_rows(arrays, keys, max_rows):
        lo, hi = int(piece["idx"][0]), int(piece["idx"][-1]) + 1
        path = os.path.join(EMB_DIR, f"emb_{lo:06d}_{hi:06d}.npz")
        if os.path.abspath(path) == os.path.abspath(final_path):
            raise RuntimeError(f"piece name collides with the source {path}")
        np.savez(path + ".tmp.npz", **piece)
        os.replace(path + ".tmp.npz", path)
        written.append(path)
    check = np.concatenate([np.load(p)["idx"] for p in written])
    if not np.array_equal(check, arrays["idx"]):
        for p in written:
            os.remove(p)
        raise RuntimeError(f"split verification FAILED — {final_path} left intact")
    os.remove(final_path)
    print(f"  split {os.path.basename(final_path)} ({n:,} rows) into "
          f"{len(written)} files of <= {max_rows:,} rows, verified identical")


def rechunk_all(max_rows):
    """Apply RECHUNK_MAX_ROWS to everything in EMB_DIR.

    Discovers the work by scanning the directory (every parts_* checkpoint dir
    and every finalized emb_*.npz), so it needs no index arguments and covers all
    shards at once.
    """
    if not os.path.isdir(EMB_DIR):
        print(f"Nothing to re-chunk: {EMB_DIR} does not exist")
        return
    entries = sorted(os.listdir(EMB_DIR))
    parts = [e for e in entries
             if e.startswith("parts_") and os.path.isdir(os.path.join(EMB_DIR, e))]
    finals = [e for e in entries if e.startswith("emb_") and e.endswith(".npz")
              and not e.endswith(".partial.npz")]
    if not parts and not finals:
        print(f"Nothing to re-chunk in {EMB_DIR}")
        return
    print(f"Re-chunking to <= {max_rows:,} rows per file in {EMB_DIR}")
    for name in parts:
        print(f"  [parts] {name}")
        rechunk_parts(os.path.join(EMB_DIR, name), WEIGHTINGS, max_rows)
    for name in finals:
        print(f"  [final] {name}")
        rechunk_final(os.path.join(EMB_DIR, name), WEIGHTINGS, max_rows)
    print("Done. Set RECHUNK_MAX_ROWS back to None before the next real run.")


def write_meta(total_dim, n_shards_note=""):
    meta = {
        "emb_dim": EMB_DIM, "proj_method": PROJ_METHOD, "proj_seed": PROJ_SEED,
        "proj_chunk": PROJ_CHUNK, "proj_type": PROJ_TYPE,
        "grad_dim": int(total_dim), "weightings": list(WEIGHTINGS),
        "dataset": C.DATASET_NAME, "model": C.FULL_MODEL_CHECKPOINT,
        "lora_init": C.LORA_INIT, "lora_r": C.LORA_R, "lora_alpha": C.LORA_ALPHA,
        "lora_target_modules": C.LORA_TARGET_MODULES, "pissa_seed": C.PISSA_SEED,
        "mean_chosen_dir": C.MEAN_CHOSEN_DIR, "note": n_shards_note,
    }
    path = os.path.join(EMB_DIR, "meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return path


# ── Validation: hashed vs dense ─────────────────────────────────────────────

def validate_dense(model, tokenizer, dataset, lora_params, numels, device, n, weighting):
    """Embed the first `n` examples with BOTH backends and compare the pairwise
    cosine structure. This is the honest check that the cheap sketch preserves
    what we cluster on. Expect mean |cos_hashed - cos_dense| ~ 1/sqrt(EMB_DIM).
    Costs minutes per example for the dense side -- keep n small (~50-200)."""
    hashed = make_projection("hashed", EMB_DIM, numels, device)
    dense = make_projection("dense", EMB_DIM, numels, device)
    H, Dn = [], []
    for idx in tqdm(range(min(n, len(dataset))), desc="validate hashed vs dense"):
        msgs = [C.normalize_message(m) for m in dataset[idx]["chosen"]]
        for proj, sink in ((hashed, H), (dense, Dn)):
            r = chosen_embeddings(model, tokenizer, msgs, lora_params, proj, (weighting,))
            if r is None:
                break
            sink.append(r[weighting][0])
        torch.cuda.empty_cache()
    if len(H) < 3:
        print("  not enough usable examples to validate")
        return
    def pcos(X):
        X = torch.tensor(np.stack(X))
        X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-12)
        M = X @ X.T
        iu = torch.triu_indices(M.shape[0], M.shape[0], offset=1)
        return M[iu[0], iu[1]]
    ch, cd = pcos(H), pcos(Dn)
    diff = (ch - cd).abs()
    corr = torch.corrcoef(torch.stack([ch, cd]))[0, 1]
    print(f"\n  pairs={ch.numel()}  1/sqrt(EMB_DIM)={1/EMB_DIM**0.5:.4f}")
    print(f"  |cos_hashed - cos_dense|: mean={diff.mean():.4f} max={diff.max():.4f}")
    print(f"  pearson(cos_hashed, cos_dense) = {corr:.4f}")
    print(f"  cos_dense  range [{cd.min():.4f}, {cd.max():.4f}] mean {cd.mean():.4f}")
    print(f"  cos_hashed range [{ch.min():.4f}, {ch.max():.4f}] mean {ch.mean():.4f}")


def self_test(device):
    """Verify the sketch preserves inner products, without touching the model.
    Runs in seconds -- use it to sanity-check a new EMB_DIM/PROJ_CHUNK/seed."""
    numels = [1_000_000] * 8
    total = sum(numels)
    proj = make_projection("hashed", EMB_DIM, numels, device)
    g = torch.Generator().manual_seed(1234)
    err_ip, err_norm = [], []
    for _ in range(8):
        x = [torch.randn(n, generator=g) for n in numels]
        y = [torch.randn(n, generator=g) for n in numels]
        ex, nx = proj.project([t.to(device) for t in x])
        ey, ny = proj.project([t.to(device) for t in y])
        true_ip = sum(float((a * b).sum()) for a, b in zip(x, y))
        true_nx = float(torch.cat(x).norm())
        err_ip.append(abs(float(ex @ ey) - true_ip) / (true_nx * float(torch.cat(y).norm())))
        err_norm.append(abs(nx - true_nx) / true_nx)
    print(f"\nself_test  D={total:,} -> {EMB_DIM}")
    print(f"  inner-product error (normalized): mean={np.mean(err_ip):.4f} "
          f"max={np.max(err_ip):.4f}   expected ~{1/EMB_DIM**0.5:.4f}")
    print(f"  exact-norm error: max={np.max(err_norm):.2e} (must be ~0 -- norms "
          f"are computed in full space, not from the sketch)")
    # Determinism: rebuilding from the same seed must give the identical tables.
    proj2 = make_projection("hashed", EMB_DIM, numels, device)
    same = bool(torch.equal(proj.bucket, proj2.bucket) and torch.equal(proj.sign, proj2.sign))
    print(f"  rebuild from seed {PROJ_SEED} identical: {same}")
    assert same, "projection is not reproducible from its seed"


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_idx", type=int, default=0, help="First example index (inclusive)")
    ap.add_argument("--end_idx", type=int, default=-1, help="Last example index (exclusive); -1 -> end")
    ap.add_argument("--validate_dense", type=int, default=0,
                    help="Embed the first N examples with BOTH backends and report "
                         "cosine agreement, then exit (slow; use N ~ 50-200)")
    ap.add_argument("--self_test", action="store_true",
                    help="Check sketch accuracy/determinism on synthetic vectors, then exit")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.self_test:
        self_test(device)
        return

    if RECHUNK_MAX_ROWS is not None:
        rechunk_all(RECHUNK_MAX_ROWS)
        return

    os.makedirs(EMB_DIR, exist_ok=True)
    print(f"Loading model {C.FULL_MODEL_CHECKPOINT} with PiSSA LoRA (sdpa) ...")
    # sdpa: one plain backward per example, no batched-grad vmap needed.
    model, tokenizer = C.build_model_with_pissa_lora(attn_implementation="sdpa")

    # Adapter identity. What MUST match is the other passes of the CLUSTER
    # pipeline (the full-space centroids and the cluster products), because those
    # dot vectors against each other. This shard therefore verifies against — or
    # establishes — a fingerprint of its own, kept next to the embeddings.
    #
    # The older mean-chosen fingerprint is checked only as a WARNING. A mismatch
    # there does not invalidate anything here: every row of this matrix comes from
    # one adapter instance, and pairwise cosines between chosen gradients are
    # invariant under the PiSSA SVD's sign/rotation ambiguity anyway (a sign flip
    # on coordinate i flips g_i for every example, leaving g_i*h_i unchanged). See
    # the discriminator in pissa_lora_common._mismatch_report for whether it is a
    # basis difference or genuinely different weights.
    own_fp = os.path.join(EMB_DIR, "lora_fingerprint.json")
    if os.path.isfile(own_fp):
        C.verify_fingerprint(model, own_fp)          # fatal: shards must agree
    else:
        C.save_fingerprint(model, own_fp)
        print(f"  established the cluster pipeline's adapter fingerprint -> {own_fp}")
    mean_fp = os.path.join(C.MEAN_CHOSEN_DIR, "lora_fingerprint.json")
    if os.path.isfile(mean_fp):
        try:
            C.verify_fingerprint(model, mean_fp)
            print("  (also matches the mean-chosen/products basis — old aggregated_* "
                  "artifacts are directly reusable)")
        except RuntimeError as e:
            print(f"\n  WARNING: this adapter differs from the mean-chosen/products "
                  f"run. Not fatal for clustering — read the report, then keep the "
                  f"centroid and cluster-products passes on THIS adapter:\n{e}\n")
    else:
        print(f"  note: no fingerprint at {mean_fp}")

    lora_params, names, numels = C.get_lora_param_list(model)
    total_dim = int(sum(numels))
    device = next(model.parameters()).device

    print(f"Loading dataset {C.DATASET_NAME} ...")
    dataset = load_dataset(C.DATASET_NAME, split="train")
    if NUM_EXAMPLES is not None:
        dataset = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))

    if args.validate_dense:
        validate_dense(model, tokenizer, dataset, lora_params, numels, device,
                       args.validate_dense, WEIGHTINGS[0])
        return

    start_idx = max(0, args.start_idx)
    end_idx = len(dataset) if args.end_idx < 0 else min(args.end_idx, len(dataset))
    final_path, parts_dir, progress_path, legacy_partial = shard_paths(start_idx, end_idx)
    if os.path.exists(final_path):
        print(f"Shard already complete: {final_path}")
        return

    print(f"Embedding chosen gradients for [{start_idx}, {end_idx}) of {len(dataset)}")
    print(f"  weightings={list(WEIGHTINGS)}  method={PROJ_METHOD}  dim={EMB_DIM}")
    projection = make_projection(PROJ_METHOD, EMB_DIM, numels, device)
    print(f"  meta -> {write_meta(total_dim)}")

    # `rows` only ever holds the rows since the last flush (<= FLUSH_EVERY), so
    # memory is flat regardless of how far the shard has progressed. Completed
    # rows live in parts_dir/chunk_*.npz and are never read back until finalize.
    next_idx, skipped, chunk_seq = resume_state(
        parts_dir, progress_path, legacy_partial, start_idx)
    rows, kept_before = [], 0
    if os.path.isfile(progress_path):
        with open(progress_path) as f:
            kept_before = int(json.load(f)["kept"])

    def flush(next_i):
        nonlocal chunk_seq, kept_before
        if rows:
            write_chunk(parts_dir, chunk_seq, rows, WEIGHTINGS)
            chunk_seq += 1
            kept_before += len(rows)
            rows.clear()
        # Written only AFTER the chunk lands, so progress never claims rows that
        # are not on disk.
        write_progress(progress_path, next_i, skipped, kept_before)

    # tqdm gives per-shard rate + ETA; `timing` additionally splits out how much
    # of each example is the projection, so the "projection is negligible" claim
    # is measured rather than assumed.
    timing = {"n": 0, "total_s": 0.0, "proj_s": 0.0}
    t_wall = time.perf_counter()
    pbar = tqdm(range(next_idx, end_idx), desc="Chosen-gradient embeddings")
    for n_done, idx in enumerate(pbar, 1):
        msgs = [C.normalize_message(m) for m in dataset[idx]["chosen"]]
        res = chosen_embeddings(model, tokenizer, msgs, lora_params, projection,
                                WEIGHTINGS, timing=timing)
        if res is None:
            skipped += 1
        else:
            rows.append((idx, res))
        if n_done % FLUSH_EVERY == 0:
            flush(idx + 1)
            torch.cuda.empty_cache()
        post = {"kept": kept_before + len(rows), "skipped": skipped}
        if timing["n"]:
            post["s/ex"] = f"{timing['total_s'] / timing['n']:.3f}"
            post["proj%"] = f"{100 * timing['proj_s'] / max(timing['total_s'], 1e-9):.1f}"
        pbar.set_postfix(**post)

    flush(end_idx)                       # persist the tail before concatenating
    if kept_before == 0:
        raise RuntimeError(f"No usable chosen examples in [{start_idx}, {end_idx}).")
    merged = finalize(parts_dir, final_path, WEIGHTINGS)

    n = len(merged["idx"])
    print(f"\nDone. {n} embeddings ({skipped} skipped) -> {final_path}")
    for w in WEIGHTINGS:
        g = merged[f"gnorm_{w}"]
        print(f"  {w}: ||g|| min={g.min():.4f} median={np.median(g):.4f} max={g.max():.4f}")

    # Timing, and what it extrapolates to for the whole split -- run this on a
    # small --end_idx first and read these two lines before launching all shards.
    wall = time.perf_counter() - t_wall
    if timing["n"]:
        per_ex = timing["total_s"] / timing["n"]
        print(f"  timing: {per_ex:.3f} s/example  ({timing['n'] / max(wall, 1e-9):.2f} ex/s), "
              f"projection {100 * timing['proj_s'] / max(timing['total_s'], 1e-9):.1f}% "
              f"({1000 * timing['proj_s'] / timing['n']:.1f} ms/example)")
        print(f"  extrapolated: {per_ex * len(dataset) / 3600:.1f} GPU-hours for all "
              f"{len(dataset):,} examples, i.e. {per_ex * len(dataset) / 3600 / 8:.1f} h "
              f"across 8 shards")
    print(f"  next: cluster_chosen_gradients.py (reads every emb_*.npz in {EMB_DIR})")


if __name__ == "__main__":
    main()
