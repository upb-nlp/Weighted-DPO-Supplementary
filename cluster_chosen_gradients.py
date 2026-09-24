"""
Cluster the projected chosen gradients produced by
compute_chosen_gradient_embeddings.py, under COSINE distance.

Reads every emb_*.npz shard in EMB_DIR, L2-normalizes the rows (which makes
Euclidean k-means equivalent to maximizing cosine), and runs SPHERICAL K-MEANS:
k-means++ init, then alternate
    assign:   a_i = argmax_k  <x_i, c_k>          (cosine, since ||x_i|| = 1)
    update:   c_k = mean of members, RE-NORMALIZED to the unit sphere
The renormalization is the only difference from plain k-means and is what makes
the objective "maximize summed cosine to your centroid" rather than "minimize
squared distance". Implemented directly in torch (a [N, EMB_DIM] @ [EMB_DIM, K]
matmul per iteration -- 210k x 1000 x 64 is milliseconds on GPU), so there is no
faiss/sklearn dependency and the whole K sweep runs in minutes.

WHY SPHERICAL K-MEANS AND NOT SOMETHING ELSE
--------------------------------------------
The downstream use is fixed: every rejected token gets dotted against ALL K
centroids and we take the max. That shapes the requirements:
  - We need CENTROIDS, not just a partition. Rules out DBSCAN/HDBSCAN/spectral
    as the primary method (they give labels; a "centroid" of a density blob is
    not what they optimize, and HDBSCAN on 210k x 1000 with cosine is both slow
    and dominated by a huge noise class here).
  - We need a K we control, because K centroids = K extra dot products per token
    in the products pass and K extra columns to store. Rules out methods that
    discover K freely.
  - Cosine is the right geometry: the downstream score is a cosine, and gradient
    norms vary by orders of magnitude across examples, so Euclidean k-means on
    unnormalized gradients would cluster by norm.
  - Every point must belong somewhere (no noise class), since a centroid built
    from a "noise" subset is still going to be maxed over.
Spherical k-means is exactly the method that optimizes the quantity we will use.
Worth trying as follow-ups, but not first: (a) balanced k-means, if the size
histogram below is very skewed; (b) agglomerative average-linkage on a 20k
subsample, purely to LOOK at the dendrogram and pick K; (c) a mixture of von
Mises-Fisher distributions, the probabilistic version of this, which would give
soft cluster responsibilities -- interesting if you later want a softmax over
clusters instead of a max.

THE CENTERING DIAGNOSTIC (read the printout before trusting any clustering)
--------------------------------------------------------------------------
Gradients of one model over one dataset usually share a large common component:
every chosen gradient partly points "toward more fluent assistant text". If that
shared direction dominates, all pairwise cosines sit near 1.0, the top principal
component eats most of the variance, and k-means will split noise rather than
structure -- while still reporting healthy-looking cohesion. The script measures
this first:
    - distribution of sampled pairwise cosines
    - cosine of each example to the global mean direction
    - variance share of the top principal components
If the mean pairwise cosine is high (say > 0.5) and PC1 dominates, set
CENTER = True: the global mean is subtracted before normalizing, so clustering
sees the RESIDUAL structure. Note that this changes what a cluster means; the
full-space centroids in step 3 should then be built the same way (accumulate
per-cluster means and subtract the global mean, or not) -- record the choice.

OUTPUT (in CLUSTER_DIR/<weighting>/K{K}/)
    assignments.npy   int32 [N]              cluster id per row
    idx.npy           int64 [N]              original train-split index per row
    centroids_proj.npy fp32 [K, EMB_DIM]     SKETCH-space centroids
    metrics.json                             cohesion, separation, sizes, config
CLUSTER_DIR/<weighting>/sweep.json           one row per K, for picking K

The projected centroids are for diagnostics and for the assignment map ONLY.
They cannot be dotted against rejected-token gradients -- those live in full
space. Step 3 (compute_cluster_chosen_gradients.py) re-runs the gradient pass and
averages the full-space chosen gradients within each cluster using assignments.npy.

PICKING K
---------
The printed sweep gives cohesion (mean cosine to own centroid), separation
(cohesion minus mean cosine to the nearest other centroid -- a centroid-based
"simplified silhouette"), and the cluster-size histogram. Two caveats, both
confirmed on planted-cluster synthetic data:
  - COHESION RISES MONOTONICALLY WITH K. It is not a selection signal on its own;
    K=32 always "fits better" than K=8. Only separation and the size histogram
    carry information about K.
  - Separation peaks at the true K only when the clusters are well separated. If
    a shared direction dominates (see above) it collapses to a narrow band at
    every K -- cohesion then reads ~0.6 everywhere and looks healthy while
    telling you nothing. That flatness is itself the signal to set CENTER = True.
So treat the sweep as a shortlist (K where separation stops improving and no
cluster falls below MIN_CLUSTER_SIZE), not a decision. The decisive test is
downstream: the gradient_groundtruth_eval/ harness already scores how well a
direction identifies LLM-labelled wrong tokens, so run steps 3-4 for two or three
candidate K on the SUBSAMPLE and compare AUC there before committing to 210k.
"""

import json
import os

import numpy as np
import torch

# ── Config ──────────────────────────────────────────────────────────────────
EMB_DIR = "/data/weighted-dpo/pissa-lora-chosen-embeddings"
CLUSTER_DIR = "/data/weighted-dpo/pissa-lora-chosen-clusters"

WEIGHTING = "weighted_norm1"     # which emb_<w> to cluster (see WEIGHTINGS there)
K_SWEEP = (2, 4, 8, 16, 32, 64, 128)
CENTER = False                   # subtract the global mean before normalizing
                                 # -- decide from the diagnostic printout
N_ITER = 50                      # spherical k-means iterations (early-stops)
N_RESTARTS = 3                   # keep the best-objective restart per K
SEED = 0
MIN_CLUSTER_SIZE = 200           # flagged (not dropped) in metrics.json
PAIR_SAMPLE = 200_000            # random pairs for the cosine diagnostic
TOL = 1e-5                       # early stop when mean cosine gain < TOL


# ── Load ────────────────────────────────────────────────────────────────────

def load_embeddings(emb_dir, weighting):
    """Concatenate every emb_*.npz shard -> (X [N, dim] fp32, idx [N] int64)."""
    shards = sorted(f for f in os.listdir(emb_dir)
                    if f.startswith("emb_") and f.endswith(".npz")
                    and not f.endswith(".partial.npz"))
    if not shards:
        raise FileNotFoundError(
            f"No emb_*.npz in {emb_dir}. Run compute_chosen_gradient_embeddings.py "
            f"(shard it with --start_idx/--end_idx) first.")
    key = f"emb_{weighting}"
    Xs, idxs = [], []
    for name in shards:
        z = np.load(os.path.join(emb_dir, name))
        if key not in z:
            raise KeyError(f"{name} has no {key!r} (has {list(z.keys())}). "
                           f"Re-run the embedding script with WEIGHTINGS including "
                           f"{weighting!r}.")
        Xs.append(z[key])
        idxs.append(z["idx"])
        print(f"  {name}: {z[key].shape}")
    # Avoid gratuitous copies of an ~825 MB matrix: concatenating a single shard
    # still copies, and so does reordering rows that are already sorted (the
    # common case, since each shard writes ascending indices).
    X = Xs[0] if len(Xs) == 1 else np.concatenate(Xs)
    idx = idxs[0] if len(idxs) == 1 else np.concatenate(idxs)
    del Xs, idxs
    if not np.all(np.diff(idx) > 0):            # deterministic row order
        order = np.argsort(idx)
        X, idx = X[order], idx[order]
    dup = len(idx) - len(np.unique(idx))
    if dup:
        print(f"  WARNING: {dup} duplicate example indices across shards "
              f"(overlapping --start_idx/--end_idx ranges?)")
    print(f"  total: {X.shape[0]:,} examples x {X.shape[1]} dims")
    return X, idx


# ── Diagnostics ─────────────────────────────────────────────────────────────

def diagnose(Xn, device, n_pc=16):
    """Is there structure beyond one shared direction? Returns a dict of stats.
    Xn must already be L2-normalized (rows on the unit sphere)."""
    N = Xn.shape[0]
    g = torch.Generator(device="cpu").manual_seed(SEED)
    i = torch.randint(0, N, (PAIR_SAMPLE,), generator=g)
    j = torch.randint(0, N, (PAIR_SAMPLE,), generator=g)
    i, j = i[i != j], j[i != j]
    cos = torch.cat([                                   # chunked: avoid 2x[P, dim]
        (Xn[i[s:s + 50_000].to(device)] * Xn[j[s:s + 50_000].to(device)]).sum(1)
        for s in range(0, i.numel(), 50_000)])

    mean_dir = Xn.mean(0)
    mean_dir = mean_dir / mean_dir.norm().clamp(min=1e-12)
    cos_mean = Xn @ mean_dir

    # Variance shares of the top principal directions, via the [dim, dim]
    # covariance (dim ~ 1000, so eigvalsh is far cheaper than an SVD of Xc).
    mu = Xn.mean(0, keepdim=True)
    cov = torch.zeros(Xn.shape[1], Xn.shape[1], device=device, dtype=torch.float64)
    for s in range(0, N, 32_768):
        Xc = (Xn[s:s + 32_768] - mu).double()
        cov += Xc.T @ Xc
    var = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0)   # descending
    share = (var / var.sum().clamp(min=1e-30)).float()

    q = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99], device=device)
    stats = {
        "pairwise_cosine_mean": float(cos.mean()),
        "pairwise_cosine_std": float(cos.std()),
        "pairwise_cosine_quantiles": [float(v) for v in torch.quantile(cos, q)],
        "cosine_to_global_mean_dir_mean": float(cos_mean.mean()),
        "cosine_to_global_mean_dir_min": float(cos_mean.min()),
        "pc_variance_share_top": [float(v) for v in share[:n_pc]],
        "pc_variance_share_cumulative_16": float(share[:16].sum()),
    }
    print("\n--- structure diagnostic ---")
    print(f"  pairwise cosine: mean={stats['pairwise_cosine_mean']:.4f} "
          f"std={stats['pairwise_cosine_std']:.4f}  "
          f"q01/q25/q50/q75/q99={'/'.join(f'{v:.3f}' for v in stats['pairwise_cosine_quantiles'])}")
    print(f"  cosine to global mean direction: mean="
          f"{stats['cosine_to_global_mean_dir_mean']:.4f} "
          f"min={stats['cosine_to_global_mean_dir_min']:.4f}")
    print(f"  top-8 PC variance share: "
          f"{'/'.join(f'{v:.3f}' for v in stats['pc_variance_share_top'][:8])}"
          f"   (top-16 cumulative {stats['pc_variance_share_cumulative_16']:.3f})")
    if stats["pairwise_cosine_mean"] > 0.5 and stats["pc_variance_share_top"][0] > 0.5:
        print("  => ONE direction dominates. Set CENTER = True and re-run, or the "
              "clustering will mostly split noise around the shared component.")
    else:
        print("  => no single direction dominates; clustering the raw normalized "
              "rows is reasonable (CENTER = False).")
    return stats


# ── Spherical k-means ───────────────────────────────────────────────────────

def _assign(Xn, Cn, chunk=32_768):
    """(labels [N], cos_to_own [N]) by argmax cosine, chunked over rows."""
    labels = torch.empty(Xn.shape[0], dtype=torch.long, device=Xn.device)
    best = torch.empty(Xn.shape[0], dtype=Xn.dtype, device=Xn.device)
    for s in range(0, Xn.shape[0], chunk):
        sim = Xn[s:s + chunk] @ Cn.T
        b, l = sim.max(dim=1)
        labels[s:s + chunk], best[s:s + chunk] = l, b
    return labels, best


def _sample_cpu_rng(weights, generator):
    """Sample one index ~ weights using a CPU generator, whatever device the
    weights live on (inverse CDF; avoids needing a device-matched generator for
    torch.multinomial, so results are identical on CPU and GPU)."""
    cdf = torch.cumsum(weights, 0)
    u = float(torch.rand(1, generator=generator)) * float(cdf[-1])
    return int(torch.searchsorted(cdf, torch.tensor(u, device=cdf.device,
                                                    dtype=cdf.dtype)))


def _kmeanspp_init(Xn, K, generator):
    """k-means++ on the sphere: seed 1 uniformly, then sample each next centroid
    with probability proportional to its squared cosine distance to the nearest
    already-chosen centroid."""
    N = Xn.shape[0]
    first = int(torch.randint(0, N, (1,), generator=generator).item())
    C = [Xn[first]]
    d2 = (1.0 - Xn @ C[0]).clamp(min=0).pow(2)
    for _ in range(1, K):
        C.append(Xn[_sample_cpu_rng(d2, generator)])
        d2 = torch.minimum(d2, (1.0 - Xn @ C[-1]).clamp(min=0).pow(2))
    return torch.stack(C)


def spherical_kmeans(Xn, K, seed, n_iter=N_ITER):
    """Returns (labels [N], centroids [K, dim] unit-norm, objective float).
    Objective = mean cosine of each point to its own centroid (higher is better).
    Empty clusters are re-seeded to the currently worst-fit points."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    C = _kmeanspp_init(Xn, K, gen)
    C = C / C.norm(dim=1, keepdim=True).clamp(min=1e-12)

    prev = -2.0
    for it in range(n_iter):
        labels, best = _assign(Xn, C)
        obj = float(best.mean())
        # Update: sum members, renormalize (the "spherical" step).
        newC = torch.zeros_like(C)
        newC.index_add_(0, labels, Xn)
        counts = torch.bincount(labels, minlength=K)
        empty = (counts == 0).nonzero().flatten()
        if empty.numel():
            worst = torch.argsort(best)[:empty.numel()]     # least-well-fit points
            newC[empty] = Xn[worst]
        C = newC / newC.norm(dim=1, keepdim=True).clamp(min=1e-12)
        if obj - prev < TOL:
            break
        prev = obj
    labels, best = _assign(Xn, C)
    return labels, C, float(best.mean())


def evaluate(Xn, labels, C, K):
    """Cohesion / separation / size stats for one clustering."""
    coh = torch.zeros(K, device=Xn.device)
    counts = torch.bincount(labels, minlength=K).float()
    own = (Xn * C[labels]).sum(1)
    coh.index_add_(0, labels, own)
    coh = coh / counts.clamp(min=1)
    coh = coh[counts > 0]                       # ignore empty clusters in the min
    # Nearest OTHER centroid, per point -> simplified silhouette.
    second = torch.empty_like(own)
    for s in range(0, Xn.shape[0], 32_768):
        sim = Xn[s:s + 32_768] @ C.T
        sim.scatter_(1, labels[s:s + 32_768, None], -2.0)
        second[s:s + 32_768] = sim.max(dim=1).values
    sizes = counts.long().tolist()
    return {
        "K": K,
        "cohesion_mean_cos_to_own_centroid": float(own.mean()),
        "separation_own_minus_nearest_other": float((own - second).mean()),
        "mean_cos_to_nearest_other_centroid": float(second.mean()),
        "per_cluster_cohesion_min": float(coh.min()),
        "size_min": int(min(sizes)), "size_max": int(max(sizes)),
        "size_median": int(np.median(sizes)),
        "n_clusters_below_min_size": int(sum(s < MIN_CLUSTER_SIZE for s in sizes)),
        "sizes": sizes,
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = os.path.join(CLUSTER_DIR, WEIGHTING)
    os.makedirs(out_root, exist_ok=True)

    print(f"Loading embeddings from {EMB_DIR} (weighting={WEIGHTING!r}) ...")
    X, idx = load_embeddings(EMB_DIR, WEIGHTING)
    Xt = torch.from_numpy(X).to(device)
    del X

    if CENTER:
        mu = Xt.mean(0, keepdim=True)
        Xt = Xt - mu
        print(f"  CENTER=True: subtracted global mean (||mu||={float(mu.norm()):.4f})")
        np.save(os.path.join(out_root, "global_mean_proj.npy"), mu.cpu().numpy())
    Xn = Xt / Xt.norm(dim=1, keepdim=True).clamp(min=1e-12)
    del Xt

    stats = diagnose(Xn, device)
    with open(os.path.join(out_root, "diagnostic.json"), "w") as f:
        json.dump({**stats, "center": CENTER, "n": int(Xn.shape[0]),
                   "emb_dim": int(Xn.shape[1])}, f, indent=2)

    print(f"\nSpherical k-means sweep over K={list(K_SWEEP)} "
          f"({N_RESTARTS} restarts each, <= {N_ITER} iters) on {device} ...")
    sweep = []
    for K in K_SWEEP:
        if K >= Xn.shape[0]:
            print(f"  K={K}: skipped (>= n examples)")
            continue
        best = None
        for r in range(N_RESTARTS):
            labels, C, obj = spherical_kmeans(Xn, K, seed=SEED * 100 + r)
            if best is None or obj > best[2]:
                best = (labels, C, obj)
        labels, C, obj = best
        m = evaluate(Xn, labels, C, K)
        m.update(restarts=N_RESTARTS, objective=obj, center=CENTER, seed=SEED,
                 weighting=WEIGHTING)
        sweep.append({k: v for k, v in m.items() if k != "sizes"})

        kdir = os.path.join(out_root, f"K{K}")
        os.makedirs(kdir, exist_ok=True)
        np.save(os.path.join(kdir, "assignments.npy"), labels.cpu().numpy().astype(np.int32))
        np.save(os.path.join(kdir, "idx.npy"), idx)
        np.save(os.path.join(kdir, "centroids_proj.npy"), C.cpu().numpy())
        with open(os.path.join(kdir, "metrics.json"), "w") as f:
            json.dump(m, f, indent=2)

        flag = "  <-- has tiny clusters" if m["n_clusters_below_min_size"] else ""
        print(f"  K={K:4d}  cohesion={m['cohesion_mean_cos_to_own_centroid']:.4f}  "
              f"separation={m['separation_own_minus_nearest_other']:.4f}  "
              f"sizes[min/med/max]={m['size_min']}/{m['size_median']}/{m['size_max']}{flag}")

    with open(os.path.join(out_root, "sweep.json"), "w") as f:
        json.dump(sweep, f, indent=2)

    print(f"\nWrote {out_root}/{{diagnostic,sweep}}.json and K*/")
    print("Next: compute_cluster_chosen_gradients.py --k K --assignments "
          f"{out_root}/K<K>/assignments.npy   (rebuilds the K centroids in FULL "
          "gradient space -- the projected centroids above cannot be dotted "
          "against rejected-token gradients)")


if __name__ == "__main__":
    main()
