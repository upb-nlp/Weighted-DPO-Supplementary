"""
Step 3 of the cluster pipeline: rebuild the K cluster centroids in FULL gradient
space, so rejected-token gradients can be dotted against them.

cluster_chosen_gradients.py (step 2) partitions the examples using EMB_DIM-
dimensional random-projection sketches. Its centroids_proj.npy live in sketch
space and cannot be dotted against a real gradient. This script takes only the
LABELS from that run and re-walks the chosen gradients, summing them per cluster
in full space (see cluster_common for what each kind means):

    mean_k      = (1/n_k) sum_{e in k}  g_e
    meannorm_k  = (1/n_k) sum_{e in k}  g_e / ||g_e||

BOTH KINDS COST ONE PASS, NOT TWO
---------------------------------
The gradient is linear in the loss, so

    grad[ sum_e loss_e / ||g_e|| ]  =  sum_e g_e / ||g_e||

and the exact full-space ||g_e|| is ALREADY saved (gnorm_<weighting>) by
compute_chosen_gradient_embeddings.py. The normalized accumulator is therefore a
second backward over the SAME forward with per-example loss scaling -- it does
not force batch size 1, so this pass stays as batched as
compute_mean_chosen_gradient.py.

CLUSTER-MAJOR ORDER
-------------------
Examples are visited cluster by cluster and batched WITHIN a cluster. That buys
three things: a batch is homogeneous, so one backward serves one accumulator;
only ONE cluster's accumulators are resident (2 * D fp32 ~ 1.3 GB, not
2 * K * D ~ 10 GB); and a cluster is finalized and written the moment its members
are done, so --clusters shards across GPUs with no cross-shard merge.

ADAPTER IDENTITY
----------------
FATAL against the cluster branch's own fingerprint (cluster_common.FINGERPRINT_FILE,
established on first run): these centroids get dotted against rejected-token
gradients from compute_cluster_products.py, and both sides of a dot product must
come from ONE adapter instance.

ADVISORY against the older mean-chosen / subsample fingerprints. Those describe
adapters derived months ago whose SVD may no longer be reproducible in this
environment, and a mismatch there does not invalidate anything computed here --
an inner product is invariant under the SVD's sign/rotation ambiguity as long as
BOTH sides transform. What it puts in question is whether cluster products may be
maxed against products_single_*.pt from that older run;
compute_cluster_products.py --validate measures that directly rather than
assuming it. Read pissa_lora_common._mismatch_report's discriminator: if sumsq
matches closely and only sum moves, it is a reparameterisation, not new weights.

To stop the drift permanently: python save_pissa_adapter.py --path <dir>, then
set pissa_lora_common.PISSA_ADAPTER_DIR to it.

OUTPUT (in <cluster run>/centroids_full/)
    centroid_mean_k00.pt       fp32 [D]
    centroid_meannorm_k00.pt   fp32 [D]
    centroid_k00_meta.json     counts, ||c_k||, config
    _resume_k00.pt             running accumulators (removed on completion)

Run:  python compute_cluster_chosen_gradients.py [--clusters 0,1] [--overwrite]
Next: compute_cluster_products.py
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from dotenv import load_dotenv
load_dotenv(".env")
from datasets import load_dataset
from tqdm import tqdm

import pissa_lora_common as C
import cluster_common as CL
from compute_mean_chosen_gradient import build_example, collate

# ── Config ──────────────────────────────────────────────────────────────────
EMB_DIR = "/data/weighted-dpo/pissa-lora-chosen-embeddings"

# Batches are sized by PADDED TOKEN BUDGET, not by a fixed example count.
#
# collate() left-pads to the longest member, so a batch costs B * max_len, and
# activation memory for the backward scales with that product -- not with the
# number of examples. A fixed BATCH_SIZE is therefore unbounded in memory: four
# 2048-token completions cost 16x four 512-token ones. That bites HERE in
# particular because this pass is cluster-major, and the clusters are strongly
# length-stratified (k5's median completion is 2195 chars, k3's is 15). Every
# batch in a long cluster is a worst-case batch, where the mean-chosen pass saw
# lengths interleaved at random and hit the tail only rarely.
#
# 4096 padded tokens ~ two full-length sequences. Raise it if the GPU has room
# (--batch_tokens); short clusters automatically get large batches, and the
# throughput on k3/k6 is much better than a fixed size ever gave.
MAX_BATCH_TOKENS = 4096
MAX_BATCH_SIZE = 16            # cap regardless of length, so short clusters
                               # do not build absurdly wide batches
CHECKPOINT_EVERY = 250         # batches between resume checkpoints (~1.3 GB each)
RESUME = True

# Every fingerprint these centroids must be compatible with. Missing files are
# skipped.
#
# STRICT_FINGERPRINTS makes a mismatch against these FATAL rather than advisory.
#
# Default False, and the reason is worth understanding rather than toggling.
# A dot product is basis-INDEPENDENT when both of its sides come from one basis:
# <D g, D v> = <g, v> for the orthogonal D relating two PiSSA parameterisations.
# The cluster branch computes its own centroids AND its own token gradients, so
# every products_cluster_*.pt value is directly comparable to the saved
# products_single_*.pt even when the two runs used different bases. (Measured on
# this repo: slope +1.008 with ~6% scatter, i.e. unbiased.) What is NOT allowed
# is pairing a STORED vector from one run with gradients from another -- the
# mistake the aggregated_* directions would make, and the one the cluster branch
# structurally cannot make.
#
# So the guard that matters is cluster_common.FINGERPRINT_FILE, which is always
# fatal and pins the cluster branch to ONE adapter. Set this to True as well only
# while the reference adapter is still rebuildable, as a "am I in the right
# container?" check; once it is not, True would block the pass forever.
STRICT_FINGERPRINTS = False
FINGERPRINT_PATHS = (
    os.path.join(C.MEAN_CHOSEN_DIR, "lora_fingerprint.json"),
    "/data/weighted-dpo/groundtruth-eval/"
    "pissa-lora-mean-chosen-SUBSAMPLE/SUBSAMPLE_lora_fingerprint.json",
)

assert set(CL.CENTROID_KINDS) <= {"mean", "meannorm"}


# ── Cluster labels + exact gradient norms ───────────────────────────────────

def load_assignments():
    """(labels [N] int64, idx [N] int64, run_dir) from the step-2 run."""
    rd = CL.run_dir()
    for f in ("assignments.npy", "idx.npy"):
        if not os.path.isfile(os.path.join(rd, f)):
            raise FileNotFoundError(
                f"{os.path.join(rd, f)} not found. Run cluster_chosen_gradients.py "
                f"with K={CL.K} in K_SWEEP and WEIGHTING={CL.WEIGHTING!r} first.")
    labels = np.load(os.path.join(rd, "assignments.npy")).astype(np.int64)
    idx = np.load(os.path.join(rd, "idx.npy")).astype(np.int64)
    if labels.shape != idx.shape:
        raise RuntimeError(f"{rd}: assignments {labels.shape} vs idx {idx.shape}")
    n_k = int(labels.max()) + 1
    if n_k != CL.K:
        raise RuntimeError(f"{rd}/assignments.npy has {n_k} clusters, expected {CL.K}")
    return labels, idx, rd


def load_gnorms(emb_dir, weighting, want_idx):
    """Exact full-space ||g_e|| for each index in `want_idx`, read from the
    embedding shards' gnorm_<weighting> (computed in full space during that pass,
    not from the sketch).

    Paired by index lookup rather than by assuming both files were written in the
    same row order, so a re-sharded EMB_DIR cannot silently mis-pair norms with
    examples. Norms are invariant to the PiSSA SVD's basis ambiguity, so reusing
    them across adapter instances is safe even when a fingerprint differs only by
    an orthogonal reparameterisation.
    """
    shards = sorted(f for f in os.listdir(emb_dir)
                    if f.startswith("emb_") and f.endswith(".npz")
                    and not f.endswith(".partial.npz"))
    if not shards:
        raise FileNotFoundError(f"No emb_*.npz in {emb_dir}")
    key = f"gnorm_{weighting}"
    idxs, gs = [], []
    for name in shards:
        z = np.load(os.path.join(emb_dir, name))
        if key not in z:
            raise KeyError(f"{name} has no {key!r} (has {list(z.keys())})")
        idxs.append(z["idx"])
        gs.append(z[key])
        del z
    emb_idx = np.concatenate(idxs) if len(idxs) > 1 else idxs[0]
    gnorm = np.concatenate(gs) if len(gs) > 1 else gs[0]

    order = np.argsort(emb_idx)
    loc = np.clip(np.searchsorted(emb_idx[order], want_idx), 0, len(order) - 1)
    pos = order[loc]
    if not np.array_equal(emb_idx[pos], want_idx):
        missing = int((emb_idx[pos] != want_idx).sum())
        raise RuntimeError(
            f"{missing} clustered indices have no gnorm in {emb_dir} — the "
            f"embeddings were rebuilt after clustering. Re-run "
            f"cluster_chosen_gradients.py, or drop 'meannorm' from CENTROID_KINDS.")
    g = gnorm[pos].astype(np.float64)
    bad = int((~np.isfinite(g)).sum() + (g <= 0).sum())
    if bad:
        raise RuntimeError(f"{bad} non-positive/non-finite ||g|| in {emb_dir}")
    return g


# ── Per-batch loss ──────────────────────────────────────────────────────────

def per_example_weighted_nll(model, kwargs, labels):
    """Per-example weighted chosen loss, [B]:

        loss_e = sum_i nll_i / (2 (1 - p_i))    over example e's completion tokens

    the `weighted_norm1` weighting (inverse ||g_token||_1), detached — byte-for-
    byte the same quantity as compute_mean_chosen_gradient.batch_losses and the
    embedding pass, just kept per-example so the caller can scale each example
    independently.
    """
    out = model(**kwargs)
    shift_logits = out.logits[:, :-1, :]           # predicts next token
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe = shift_labels.clamp(min=0)

    # Gather on the bf16 logits and cast the [B, S] RESULT, rather than casting
    # the whole [B, S, V] tensor a second time. At B=4, S=2048, V=100k an fp32
    # copy is 3.3 GB and autograd keeps it alive until the backward; gather is a
    # pure selection, so doing it before the cast is exact and costs nothing.
    # The remaining .float() is genuinely needed: logsumexp over a 100k vocab in
    # bf16 loses far too much precision.
    tgt = shift_logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1).float()
    lse = torch.logsumexp(shift_logits.float(), dim=-1)
    nll = (lse - tgt) * mask                                        # [B, S-1]

    with torch.no_grad():
        p = torch.exp(-nll).clamp(max=1.0 - 1e-6)
        w = (1.0 / (2.0 * (1.0 - p))) * mask
    return (nll * w).sum(dim=1)                                     # [B]


# ── Accumulators ────────────────────────────────────────────────────────────

class Accumulator:
    """One flat fp32 [D] buffer per centroid kind, plus the example count."""

    def __init__(self, kinds, total_dim, device, numels):
        self.kinds = list(kinds)
        self.numels = list(numels)
        self.buf = {k: torch.zeros(total_dim, device=device) for k in self.kinds}
        self.n = 0

    def add(self, kind, grads):
        """buf[kind] += flatten(grads), in C.get_lora_param_list order — the one
        order every dot product in this project is aligned to."""
        off, dst = 0, self.buf[kind]
        for g, n in zip(grads, self.numels):
            if g is not None:
                # .to(dst.device) is a no-op on one GPU and the difference between
                # working and crashing when device_map="auto" splits the adapter.
                dst[off:off + n] += torch.nan_to_num(
                    g.detach().reshape(-1).float(), nan=0.0, posinf=0.0,
                    neginf=0.0).to(dst.device)
            off += n

    def state(self, next_pos, skipped):
        return {"buf": {k: v.cpu() for k, v in self.buf.items()}, "n": self.n,
                "next_pos": next_pos, "skipped": skipped, "kinds": self.kinds}

    def load(self, state, device):
        """Restore, or return None if the checkpoint has different kinds."""
        if sorted(state["kinds"]) != sorted(self.kinds):
            return None
        for k in self.kinds:
            self.buf[k] = state["buf"][k].to(device)
        self.n = int(state["n"])
        return int(state["next_pos"])


# ── One cluster ─────────────────────────────────────────────────────────────

def cluster_paths(k_id, kinds):
    p = {kind: CL.centroid_path(kind, k_id) for kind in kinds}
    p["meta"] = CL.centroid_meta_path(k_id)
    p["resume"] = os.path.join(CL.centroids_dir(), f"_resume_k{k_id:02d}.pt")
    return p


def cluster_done(k_id, kinds):
    p = cluster_paths(k_id, kinds)
    return os.path.isfile(p["meta"]) and all(os.path.isfile(p[kind]) for kind in kinds)


def process_cluster(model, tokenizer, dataset, lora_params, numels, device,
                    k_id, members, gnorms, kinds, extra_meta=None):
    """Accumulate and write cluster k_id's centroids. `members` are dataset
    indices; `gnorms` the matching exact ||g_e|| (or None if only "mean")."""
    paths = cluster_paths(k_id, kinds)
    total_dim = int(sum(numels))
    acc = Accumulator(kinds, total_dim, device, numels)
    start_pos, skipped = 0, 0

    if RESUME and os.path.isfile(paths["resume"]):
        state = torch.load(paths["resume"], map_location="cpu", weights_only=False)
        resumed = acc.load(state, device)
        if resumed is None:
            print(f"  k={k_id}: resume checkpoint has different CENTROID_KINDS — restarting")
        else:
            start_pos, skipped = resumed, int(state.get("skipped", 0))
            print(f"  k={k_id}: resumed at member {start_pos}/{len(members)} "
                  f"({acc.n:,} accumulated)")
        del state

    pad_id = tokenizer.pad_token_id
    pending, pending_pos = [], []
    batches = 0
    t0 = time.perf_counter()

    def flush_batch():
        """One forward, one backward per centroid kind."""
        nonlocal batches
        kwargs, labels = collate(pending, pad_id, device)
        per_ex = per_example_weighted_nll(model, kwargs, labels)        # [B]
        losses = {}
        if "mean" in kinds:
            losses["mean"] = per_ex.sum()
        if "meannorm" in kinds:
            inv = torch.tensor([1.0 / gnorms[i] for i in pending_pos],
                               device=device, dtype=per_ex.dtype)
            losses["meannorm"] = (per_ex * inv).sum()
        items = list(losses.items())
        for i, (kind, loss) in enumerate(items):
            grads = torch.autograd.grad(
                loss, lora_params, retain_graph=(i < len(items) - 1),
                allow_unused=True)
            acc.add(kind, grads)
            del grads
        acc.n += len(pending)
        batches += 1
        del kwargs, labels, per_ex, losses, items

    def save_resume(next_pos):
        tmp = paths["resume"] + ".tmp"
        torch.save(acc.state(next_pos, skipped), tmp)
        os.replace(tmp, paths["resume"])

    def flush(next_pos):
        """Run the pending batch, then checkpoint on schedule.

        `next_pos` is the first member NOT yet accumulated — so a crash right
        after this resumes without dropping or double-counting anything.
        """
        nonlocal pending, pending_pos
        bs, longest = len(pending), max(len(ids) for ids, _ in pending)
        flush_batch()
        pending, pending_pos = [], []
        torch.cuda.empty_cache()
        if batches % CHECKPOINT_EVERY == 0:
            save_resume(next_pos)
        pbar.set_postfix(n=acc.n, skipped=skipped, bs=bs, maxlen=longest)

    def would_exceed(new_len):
        """Would adding a sequence of `new_len` tokens blow the padded budget?
        Cost is (batch size) x (LONGEST member), because collate left-pads."""
        longest = max(max(len(ids) for ids, _ in pending), new_len)
        return ((len(pending) + 1) * longest > MAX_BATCH_TOKENS
                or len(pending) + 1 > MAX_BATCH_SIZE)

    pbar = tqdm(range(start_pos, len(members)), desc=f"cluster {k_id}", leave=False)
    for pos in pbar:
        built = build_example(tokenizer, dataset[int(members[pos])]["chosen"])
        if built is None:
            skipped += 1
        else:
            # Flush BEFORE adding when this example would overflow the budget.
            # Everything below `pos` is accumulated at that moment and `pos`
            # itself is not, so the resume position is `pos`, not `pos + 1`.
            if pending and would_exceed(len(built[0])):
                flush(pos)
            pending.append(built)
            pending_pos.append(pos)
        if pending and pos == len(members) - 1:
            flush(pos + 1)

    if acc.n == 0:
        raise RuntimeError(f"cluster {k_id}: no usable chosen examples among "
                           f"{len(members)} members")

    meta = {"k": int(k_id), "K": CL.K, "weighting": CL.WEIGHTING,
            "n_members": int(len(members)), "n_averaged": int(acc.n),
            "n_skipped": int(skipped), "grad_dim": total_dim,
            "kinds": list(kinds), "n_batches": int(batches),
            "batch_tokens": int(MAX_BATCH_TOKENS), "batch_size_cap": int(MAX_BATCH_SIZE),
            "emb_dir": EMB_DIR, "seconds": round(time.perf_counter() - t0, 1)}
    meta.update(extra_meta or {})
    for kind in kinds:
        vec = torch.nan_to_num((acc.buf[kind] / acc.n).cpu(),
                               nan=0.0, posinf=0.0, neginf=0.0)
        meta[f"norm_{kind}"] = float(vec.norm())
        tmp = paths[kind] + ".tmp"
        torch.save(vec, tmp)
        os.replace(tmp, paths[kind])
        del vec
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)
    if os.path.isfile(paths["resume"]):
        os.remove(paths["resume"])
    del acc
    torch.cuda.empty_cache()
    return meta


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global MAX_BATCH_TOKENS          # --batch_tokens overrides the module default
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", type=str, default="",
                    help="Comma-separated cluster ids (default: all). Clusters are "
                         "independent, so this is how you shard across GPUs.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rebuild clusters whose centroids already exist")
    ap.add_argument("--batch_tokens", type=int, default=MAX_BATCH_TOKENS,
                    help=f"Padded tokens per batch, B*max_len (default "
                         f"{MAX_BATCH_TOKENS}). Memory scales with THIS, not with "
                         f"the example count — halve it if a long cluster OOMs, "
                         f"raise it if the GPU is idle.")
    ap.add_argument("--max_per_cluster", type=int, default=0,
                    help="Average at most N members per cluster (0 = all). A "
                         "cluster mean converges as 1/sqrt(n), so N ~ 3000 is "
                         "within a couple of percent of the full direction at "
                         "1/70th the GPU time — enough to decide whether the "
                         "cluster idea beats the single mean before committing "
                         "to the full pass. Members are picked with a fixed seed "
                         "and the result goes to centroids_full_n<N>/ so it can "
                         "never be confused with a complete pass.")
    args = ap.parse_args()
    MAX_BATCH_TOKENS = args.batch_tokens

    if args.max_per_cluster:
        CL.CENTROIDS_SUBDIR = f"centroids_full_n{args.max_per_cluster}"
        print(f"SUBSAMPLED centroids: <= {args.max_per_cluster:,} members per "
              f"cluster -> {CL.CENTROIDS_SUBDIR}/  (set "
              f"cluster_common.CENTROIDS_SUBDIR to this to consume them)")

    kinds = CL.CENTROID_KINDS
    labels, idx, rd = load_assignments()
    os.makedirs(CL.centroids_dir(), exist_ok=True)
    wanted = ([int(s) for s in args.clusters.split(",") if s.strip()]
              if args.clusters else list(range(CL.K)))
    print(f"Cluster run {rd}: {len(labels):,} examples, K={CL.K}; "
          f"building {wanted} with kinds={list(kinds)}")

    todo = []
    for k_id in wanted:
        if not args.overwrite and cluster_done(k_id, kinds):
            print(f"  k={k_id}: already complete — skipping")
        else:
            todo.append(k_id)
    if not todo:
        print("Nothing to do.")
        return

    gnorms = None
    if "meannorm" in kinds:
        print(f"Loading exact ||g|| from {EMB_DIR} ...")
        gnorms = load_gnorms(EMB_DIR, CL.WEIGHTING, idx)
        print(f"  ||g||: min={gnorms.min():.2f} median={np.median(gnorms):.2f} "
              f"max={gnorms.max():.2f}")

    print(f"Loading model {C.FULL_MODEL_CHECKPOINT} with PiSSA LoRA (sdpa) ...")
    # sdpa: plain batched backward, no per-token vmap needed. The parameter space
    # is identical under either attention impl, so the products pass may use eager.
    model, tokenizer = C.build_model_with_pissa_lora(attn_implementation="sdpa")

    # Always fatal against the cluster branch's own fingerprint (centroids and the
    # rejected-token gradients they get dotted with MUST share one adapter). The
    # reference fingerprints are fatal too unless STRICT_FINGERPRINTS is off.
    CL.verify_or_establish_fingerprint(
        model,
        fatal=FINGERPRINT_PATHS if STRICT_FINGERPRINTS else (),
        advisory=() if STRICT_FINGERPRINTS else FINGERPRINT_PATHS)

    lora_params, _, numels = C.get_lora_param_list(model)
    device = next(model.parameters()).device

    print(f"Loading dataset {C.DATASET_NAME} ...")
    dataset = load_dataset(C.DATASET_NAME, split="train")
    if int(idx.max()) >= len(dataset):
        raise RuntimeError(
            f"clustered index {int(idx.max())} exceeds the {len(dataset)}-row "
            f"train split — the clustering was built on a different dataset.")

    for k_id in todo:
        sel = np.flatnonzero(labels == k_id)
        n_total = len(sel)
        if args.max_per_cluster and n_total > args.max_per_cluster:
            # Fixed seed per cluster: the same members every time, so a resumed or
            # re-run shard averages exactly the same set.
            rng = np.random.default_rng(1234 + k_id)
            sel = np.sort(rng.choice(sel, args.max_per_cluster, replace=False))
        print(f"\n[k={k_id}] {n_total:,} members "
              f"({100 * n_total / len(labels):.1f}%)"
              + (f", averaging a random {len(sel):,}" if len(sel) != n_total else ""))
        meta = process_cluster(
            model, tokenizer, dataset, lora_params, numels, device, k_id,
            idx[sel], gnorms[sel] if gnorms is not None else None, kinds,
            extra_meta={"n_cluster_total": int(n_total),
                        "max_per_cluster": int(args.max_per_cluster) or None})
        norms = "  ".join(f"||c_{kind}||={meta[f'norm_{kind}']:.4f}" for kind in kinds)
        print(f"[k={k_id}] done: averaged {meta['n_averaged']:,} "
              f"({meta['n_skipped']} skipped)  {norms}")

    print(f"\nCentroids in {CL.centroids_dir()}/")
    print("Next: compute_cluster_products.py — one forward-mode JVP per centroid "
          "gives every rejected token's dot against it.")


if __name__ == "__main__":
    main()
