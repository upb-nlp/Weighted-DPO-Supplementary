"""
Shared vocabulary for the CLUSTER branch of the gradient pipeline.

The cluster branch replaces the single dataset-mean chosen direction with K
per-cluster mean directions:

  1. compute_chosen_gradient_embeddings.py  210k chosen gradients -> sketches
  2. cluster_chosen_gradients.py            sketches -> K assignments
  3. compute_cluster_chosen_gradients.py    assignments -> K centroids in FULL space
  4. compute_cluster_products.py            g_rejected_token . c_k, per token, per k
  5. 05_correlate.py / train_weighted_dpo.py     max over {own chosen} u {c_k}

Steps 3-5 all need the same constants (which K, which weighting, which centroid
kind, where things live) and the same max-over-clusters arithmetic. Putting them
here keeps the training weights identical to what the ground-truth eval scored.

CENTROID KINDS
--------------
  "mean"      c_k = (1/n_k) sum_{e in k} g_e            raw per-cluster mean; the
              exact per-cluster analogue of mean_grad_<w>.pt, so an example with
              a long completion (hence a big gradient) counts more.
  "meannorm"  c_k = (1/n_k) sum_{e in k} g_e/||g_e||    every example counts
              equally. Invariant to whether the per-example loss sums or averages
              its completion tokens (that convention is a per-example positive
              scalar, which cancels), and matches the cosine geometry the
              spherical k-means actually optimized.

THE SCALE PROBLEM (why CLUSTER_DOT_SCALE exists)
------------------------------------------------
The training score is an elementwise MAX over dot products, and a dot product
scales with the norm of the direction. The K=8 partition is strongly
norm-correlated (inspect_gradient_clusters.py: eta^2(log ||g||) = 0.58, per-cluster
median ||g|| from 466 to 9876), so raw centroids make the max degenerate into
"whichever centroid is longest" for almost every token. Cosines are immune (they
divide the norm out); dots are not, and WEIGHT_METHOD = "sigmoid_dot" is a dot
method. Because a dot is LINEAR in the direction, the fix needs no recompute:
rescaling c_k -> s * c_k/||c_k|| just multiplies the saved dot by s/||c_k||.

  "aggregated"  (default) put every centroid on the norm of the dataset-mean
                chosen gradient. Then max(dot_single, dot_cluster) is exactly the
                current max_weighted_norm1 arithmetic with the one mean direction
                swapped for the best of K -- the apples-to-apples comparison.
  "unit"        ||c_k|| = 1. Ranks clusters identically to the cosine max, but
                puts cluster dots ~1000x below the single-chosen dot, so a max
                that also includes single_* would never pick a cluster. Use with
                cluster-only directions.
  "raw"         leave ||c_k|| alone. The literal "mean of each cluster"; expect
                one cluster to win the max nearly always.
"""

import json
import os
import re

import torch

# ── Which cluster run everything downstream reads ───────────────────────────
CLUSTER_DIR = "/data/weighted-dpo/pissa-lora-chosen-clusters"
WEIGHTING = "weighted_norm1"      # emb_<w> / cluster run, and the chosen-loss weighting

# THE knob for a K sweep. Edit it here, run the centroid pass and the products
# pass, repeat. Nothing else needs changing: every artifact below is named or
# scoped by this value, so K=2, 4, 8, 16 coexist without overwriting each other,
# and 05_correlate.py scores whichever of them it finds side by side.
#
# One caveat of a constant over a CLI flag: the value is read when a job STARTS,
# not when it is submitted. Editing this while an earlier job is still queued
# means that job runs the new K. Either let each job start before editing, or
# check the "Cluster run .../K<k>" line each log prints at startup — it records
# what actually ran.
K = 2
# Which centroid constructions to build and consume. This tuple is the single
# switch: the centroid pass accumulates one buffer per kind, the products pass
# runs one JVP per (kind, cluster), and train/eval derive their direction names
# from it. Dropping a kind therefore halves both passes and the disk.
#   ("mean",)             8 directions/example  -- the analogue of aggregated_*
#   ("mean", "meannorm") 16 directions/example  -- both, for comparing them
# Re-adding "meannorm" later means re-running BOTH passes for that kind; the
# centroid pass also needs gnorm_<w> from the embedding shards to build it.
CENTROID_KINDS = ("mean",)

# Which centroid set to read. compute_cluster_chosen_gradients.py writes
# "centroids_full" for a complete pass and "centroids_full_n<N>" when it averages
# at most N members per cluster (--max_per_cluster). Point this at a subsampled
# set to get an eval signal without paying for the full 208k-example pass; a
# cluster mean converges as 1/sqrt(n), so N = 3000 is within ~2% of the full
# direction. Every consumer reads this one constant, and the products pass
# records the resolved path in cluster_norms.json so a mixed run is detectable.
CENTROIDS_SUBDIR = "centroids_full" #"centroids_full"

# How cluster dot products are rescaled before entering a max. See above.
# A VALUE IS REQUIRED — this is not an optional setting. Commenting it out does
# not mean "no rescaling"; it means every consumer dies with AttributeError, since
# the name is read at import time to build the run directory name. "No rescaling"
# is spelled "raw".
#   "raw"         leave ||c_k|| alone — the literal per-cluster mean
#   "aggregated"  every centroid at ||g_bar||, matching max_weighted_norm1's second arm
#   "unit"        ||c_k|| = 1 — for clusteronly_* only
CLUSTER_DOT_SCALE = "raw"

# ── K as a first-class experimental axis ────────────────────────────────────
#
# Centroids are already K-scoped by directory (CLUSTER_DIR/<w>/K<K>/). Products
# are NOT: they live inside the shared per-example directories of a products run,
# next to products_single_*.pt, so their FILENAMES have to carry K and the
# weighting or a K=4 sweep would silently overwrite K=8's [K, T] matrices with a
# differently-shaped one. Everything below is therefore parameterised by k, with
# the module-level K as the default for the build scripts.
#
# The same reasoning applies to direction NAMES: "maxclu_mean" is ambiguous once
# more than one K exists, so a direction is named maxclu_<kind>_K<k>. That name
# is the single identifier shared by 05_correlate.py's report and
# train_weighted_dpo.py's GRADIENT_DIRECTION, so whichever K wins the eval is
# trained by copying the name across, and no run can be confused for another.

def products_filename(kind, k=None):
    """Per-example [K, T] products file, one per (kind, K)."""
    return f"products_cluster_{kind}_{WEIGHTING}_K{K if k is None else k}.pt"


def cluster_norms_filename(k=None):
    """Per-(weighting, K) sidecar at the root of a products directory, so readers
    (05, training) never need the cluster mount to know ||c_k||."""
    return f"cluster_norms_{WEIGHTING}_K{K if k is None else k}.json"


_NORMS_RE = re.compile(r"^cluster_norms_(?P<w>.+)_K(?P<k>\d+)\.json$")


def discover_cluster_runs(scores_dir, weighting=None):
    """{k: sidecar dict} for every cluster run present in a products directory.

    Lets a consumer score EVERY K that has been computed, side by side in one
    table, instead of being told which one to look at. Sorted by k.
    """
    weighting = WEIGHTING if weighting is None else weighting
    out = {}
    if not os.path.isdir(scores_dir):
        return out
    for name in sorted(os.listdir(scores_dir)):
        m = _NORMS_RE.match(name)
        if not m or m.group("w") != weighting:
            continue
        with open(os.path.join(scores_dir, name)) as f:
            out[int(m.group("k"))] = json.load(f)
    return dict(sorted(out.items()))


# Direction naming, shared by the eval and the trainer.
#   clusteronly_<kind>_K<k>   max over the k centroids alone
#   maxclu_<kind>_K<k>        max over {own chosen gradient} u {the k centroids}
CLUSTER_PREFIXES = ("clusteronly", "maxclu")
_DIRECTION_RE = re.compile(
    r"^(?P<prefix>clusteronly|maxclu)_(?P<kind>[a-z]+)_K(?P<k>\d+)$")


def direction_name(prefix, kind, k):
    return f"{prefix}_{kind}_K{k}"


def parse_direction(name):
    """('maxclu', 'mean', 8) for a cluster direction, or None if it is not one."""
    m = _DIRECTION_RE.match(name)
    return (m.group("prefix"), m.group("kind"), int(m.group("k"))) if m else None

# The adapter identity of the CLUSTER BRANCH. Established by whichever of the two
# cluster passes runs first and verified (fatally) by every later one, exactly as
# compute_chosen_gradient_embeddings.py keeps its own fingerprint.
FINGERPRINT_FILE = os.path.join(CLUSTER_DIR, "lora_fingerprint.json")


def verify_or_establish_fingerprint(model, fatal=(), advisory=()):
    """Check this adapter against the cluster branch's own fingerprint (always
    fatal, established on first run) plus the caller's `fatal` and `advisory` sets.

    WHICH SET A FINGERPRINT BELONGS IN
    ----------------------------------
    `fatal` is for the fingerprints that define the project's reference adapter --
    the mean-chosen one, since every products_*.pt in the repo was computed
    against it. The PiSSA SVD is only reproducible within one environment (a
    different container, torch/peft version or device placement re-derives a
    different basis for the same model), so this check is in practice a "am I in
    the right container?" test, and it is worth failing loudly on. That is the
    default.

    `advisory` is the escape hatch for a fingerprint describing an adapter that
    can no longer be rebuilt at all. A mismatch there does not by itself
    invalidate anything computed here: PiSSA reconstructs the same weight matrix
    either way (W_res := W - BA), so the model's FUNCTION is unchanged and an
    inner product is invariant as long as both of its sides come from one basis.
    What a mismatch questions is MIXING -- maxing new cluster products against
    products_single_* from the other run -- and
    compute_cluster_products.py --validate measures that directly.

    Returns the list of advisory paths that did not match.
    """
    import pissa_lora_common as C

    os.makedirs(os.path.dirname(FINGERPRINT_FILE) or ".", exist_ok=True)
    if os.path.isfile(FINGERPRINT_FILE):
        try:
            C.verify_fingerprint(model, FINGERPRINT_FILE)
        except RuntimeError as e:
            raise RuntimeError(
                f"{e}\n\n  This is the cluster branch's OWN fingerprint. If it was "
                f"established by a run in the wrong environment, delete\n    "
                f"{FINGERPRINT_FILE}\n  AND every centroid built by that run "
                f"({centroids_dir()}), then re-run in the right one — centroids "
                f"from two bases must never end up in the same directory."
            ) from None
    else:
        C.save_fingerprint(model, FINGERPRINT_FILE)
        print(f"  established the cluster branch's adapter fingerprint -> "
              f"{FINGERPRINT_FILE}")

    for path in fatal:
        if os.path.isfile(path):
            C.verify_fingerprint(model, path)                  # raises on mismatch

    mismatched = []
    for path in advisory:
        if not os.path.isfile(path):
            continue
        try:
            C.verify_fingerprint(model, path)
        except RuntimeError as e:
            mismatched.append(path)
            print(f"\n  ADVISORY: this adapter differs from {path}.\n"
                  f"  Read the discriminator below, then decide whether cluster "
                  f"products may be maxed against products computed on that "
                  f"adapter (compute_cluster_products.py --validate answers it "
                  f"empirically):\n{e}\n")
    return mismatched


# ── Artifact layout ─────────────────────────────────────────────────────────

def run_dir(cluster_dir=None, weighting=None, k=None):
    """Late-bound on purpose: a default argument is evaluated at import time, so
    `k=K` would freeze whatever K was configured then and a --k override (or any
    sweep that sets CL.K at runtime) would silently keep writing to the old
    directory."""
    cluster_dir = CLUSTER_DIR if cluster_dir is None else cluster_dir
    weighting = WEIGHTING if weighting is None else weighting
    k = K if k is None else k
    return os.path.join(cluster_dir, weighting, f"K{k}")


def centroids_dir(**kw):
    return os.path.join(run_dir(**kw), CENTROIDS_SUBDIR)


def centroid_path(kind, k_id, **kw):
    return os.path.join(centroids_dir(**kw), f"centroid_{kind}_k{k_id:02d}.pt")


def centroid_meta_path(k_id, **kw):
    return os.path.join(centroids_dir(**kw), f"centroid_k{k_id:02d}_meta.json")


def load_centroid_meta(kinds=None, k=None, **kw):
    """Per-cluster meta from step 3 -> {"norms": {kind: [K floats]}, "counts": [K]}.

    Raises if any cluster is missing, so a half-finished sharded run cannot
    silently produce products against 6 of 8 centroids.
    """
    kinds = CENTROID_KINDS if kinds is None else kinds
    k = K if k is None else k
    norms = {kind: [] for kind in kinds}
    counts = []
    for k_id in range(k):
        path = centroid_meta_path(k_id, k=k, **kw)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} not found — run compute_cluster_chosen_gradients.py "
                f"(cluster {k_id} is missing).")
        with open(path) as f:
            m = json.load(f)
        for kind in kinds:
            if f"norm_{kind}" not in m:
                raise KeyError(
                    f"{path} has no norm_{kind!r} — that centroid kind was not "
                    f"built. Re-run step 3 with {kind!r} in CENTROID_KINDS.")
            norms[kind].append(float(m[f"norm_{kind}"]))
        counts.append(int(m["n_averaged"]))
    return {"norms": norms, "counts": counts}


def load_centroid(kind, k_id, device=None, **kw):
    """One full-space centroid as an fp32 vector [D]."""
    path = centroid_path(kind, k_id, **kw)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found — run compute_cluster_chosen_gradients.py first.")
    v = torch.load(path, map_location="cpu", weights_only=True).float()
    return v if device is None else v.to(device)


# ── The max-over-clusters arithmetic (single source of truth) ───────────────

def dot_scale_factors(cluster_norms, aggregated_norm, mode=None):
    """Multipliers that turn raw cluster dots into CLUSTER_DOT_SCALE dots.

    `cluster_norms` is [K] (||c_k||), `aggregated_norm` the norm of the
    dataset-mean chosen gradient in the SAME artifact set (per-example meta for
    the subsample harness, mean_chosen_norms.json for training).
    """
    mode = CLUSTER_DOT_SCALE if mode is None else mode
    n = torch.as_tensor(cluster_norms, dtype=torch.float32).clamp(min=1e-12)
    if mode == "raw":
        return torch.ones_like(n)
    if mode == "unit":
        return 1.0 / n
    if mode == "aggregated":
        return float(aggregated_norm) / n
    raise ValueError(f"unknown CLUSTER_DOT_SCALE {mode!r}")


def cluster_dot_cos(products, token_norms, cluster_norms, aggregated_norm,
                    mode=None):
    """Per-token (dot, cos, argmax_k) of the BEST cluster for each token.

    products      fp32 [K, T]  raw g_t . c_k from compute_cluster_products.py
    token_norms   fp32 [T]     ||g_t||, from norms_rejected.pt
    cluster_norms [K]          ||c_k||

    The dot is maxed AFTER rescaling (so the max is not decided by ||c_k||); the
    cosine is scale-free and maxed directly. Both are taken independently, which
    is what the existing max_* directions do, so `argmax_k` is reported for the
    dot -- the quantity WEIGHT_METHOD = "sigmoid_dot" actually consumes.
    """
    products = torch.as_tensor(products, dtype=torch.float32)
    token_norms = torch.as_tensor(token_norms, dtype=torch.float32)
    cn = torch.as_tensor(cluster_norms, dtype=torch.float32).clamp(min=1e-12)

    scaled = products * dot_scale_factors(cn, aggregated_norm, mode)[:, None]
    dot, argmax = scaled.max(dim=0)
    cos = (products / (token_norms.clamp(min=1e-12)[None, :] * cn[:, None])).max(dim=0).values
    return dot, cos, argmax
