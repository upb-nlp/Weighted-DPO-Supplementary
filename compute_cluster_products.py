"""
Step 4 of the cluster pipeline: dot every REJECTED token's gradient against each
of the K full-space cluster centroids.

    products_cluster_<kind>_<weighting>.pt      fp32 [K, T]
        row k, column t  =  g_t . c_k

written INTO the example directories the existing products passes already
produced, so nothing is recomputed: norms_rejected.pt, products_single_*.pt and
products_aggregated_*.pt stay exactly as they are and the new file just adds K
more directions to max over.

WHY THIS IS CHEAP (forward-mode AD)
-----------------------------------
compute_dpo_gradient_products_pissa_lora.py needs the per-token gradient VECTOR
(to take its norm), so it runs one backward per rejected token -- T backwards per
example, with T in the hundreds. Here we only need SCALARS: the projection of
each token's gradient onto K fixed directions. That is exactly a
Jacobian-vector product. With

    f(theta) = [nll_1(theta), ..., nll_T(theta)]        (one forward's worth)

torch.func.jvp(f, theta, v) returns J v = [<grad nll_t, v>]_t -- ALL T dot
products against direction v from a SINGLE forward-mode pass. So the cost is one
pass per DIRECTION (2 * K = 16 here) instead of one backward per TOKEN, and it is
independent of the completion length. Same identity as
torch.autograd.functional.jvp, computed in true forward mode rather than by the
double-backward trick, so no graph is retained and peak memory is ~2x a forward.

The `loop` backend (exact per-token backward, reusing the existing helper) is
kept as the reference: --validate N cross-checks the two on N examples, AND
re-derives the already-saved aggregated_weighted_norm1 products through the JVP
path, which tests the adapter, the tokenization and the JVP arithmetic end to end
against a file produced months earlier by completely different code.

PRECISION
---------
torch.func.jvp requires the tangent to match the primal's dtype, so each fp32
centroid is cast to the LoRA params' bf16. The resulting quantization noise e
satisfies |<g_t, e>| ~ ||g_t|| ||c|| * eps / sqrt(D) with D ~ 1.6e8, i.e. some
four orders of magnitude below the signal -- but do not take that on faith, run
--validate and read the reported relative error.

ADAPTER IDENTITY
----------------
Both sides of a dot product must come from ONE adapter instance, so this pass
verifies against the same fingerprint the target directory's other products were
built against (fatal on mismatch). See pissa_lora_common._mismatch_report.

Run (subsample eval first — 100 examples):
    python compute_cluster_products.py --mode subsample --validate 2
Then the full set, sharded like the other passes:
    python compute_cluster_products.py --mode full --start_idx 0 --end_idx 20000
"""

import argparse
import json
import os
import sys
import time

import torch
from dotenv import load_dotenv
load_dotenv(".env")
from datasets import load_dataset
from tqdm import tqdm

import pissa_lora_common as C
import cluster_common as CL
from compute_dpo_gradient_products_pissa_lora import (
    SAVE_DIR as FULL_PRODUCTS_DIR,
    compute_rejected_products_and_norms,
)

# ── Config ──────────────────────────────────────────────────────────────────
# Where the bf16 tangents live between examples. "cuda" keeps all 2*K of them
# resident (2 * 8 * D * 2 bytes ~ 5 GB at D = 1.6e8); "cpu" trades ~30 ms per
# direction per example of PCIe traffic for that memory.
TANGENT_DEVICE = "cuda"


# ── Target directories (the two artifact sets that have example dirs) ───────

def resolve_mode(mode):
    """(scores_dir, dirname_fn, index_list_fn, fingerprint_path, mean_grad_path).

    `index_list_fn(dataset)` returns the source dataset indices to process, in
    order. The subsample harness keeps its own frozen index list and its own
    mean-chosen dir, so both are resolved from its config module.
    """
    if mode == "full":
        return (FULL_PRODUCTS_DIR,
                lambda i: f"example_{i:05d}",
                lambda ds: list(range(len(ds))),
                os.path.join(C.MEAN_CHOSEN_DIR, "lora_fingerprint.json"),
                os.path.join(C.MEAN_CHOSEN_DIR, "mean_grad_weighted_norm1.pt"))

    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "gradient_groundtruth_eval"))
    import config as gt_config                     # noqa: E402  (path-dependent)

    def sub_indices(_ds):
        with open(gt_config.SUBSAMPLE_INDICES_FILE) as f:
            return json.load(f)["indices"]

    return (gt_config.GRADIENT_SCORES_DIR,
            gt_config.example_dirname,
            sub_indices,
            os.path.join(gt_config.SUBSAMPLE_MEAN_DIR, gt_config.MEAN_FINGERPRINT_FILE),
            os.path.join(gt_config.SUBSAMPLE_MEAN_DIR,
                         gt_config.MEAN_GRAD_FILES["weighted_norm1"]))


# ── Directions ──────────────────────────────────────────────────────────────

def unflatten_tangent(vec, params):
    """Flat fp32 [D] -> tuple of tensors matching each param's shape and dtype.

    Split in C.get_lora_param_list order, the single order every flattened
    gradient in this project uses. torch.func.jvp requires the tangent to match
    the primal's dtype, so this is where the fp32 centroid becomes bf16 (see the
    PRECISION note in the module docstring). Tangents are parked on
    TANGENT_DEVICE and moved per direction at use time.
    """
    out, off = [], 0
    for p in params:
        n = p.numel()
        dest = p.device if TANGENT_DEVICE == "cuda" else torch.device(TANGENT_DEVICE)
        out.append(vec[off:off + n].view_as(p).to(device=dest, dtype=p.dtype))
        off += n
    if off != vec.numel():
        raise RuntimeError(f"direction has {vec.numel()} dims, adapter has {off}")
    return tuple(out)


def load_directions(params, kinds, k, validate_vec=None):
    """[(name, tangent_tuple), ...] for every centroid, plus the validation
    direction if given. Tangents are materialized once and kept resident."""
    dirs = []
    for kind in kinds:
        for k_id in range(k):
            vec = CL.load_centroid(kind, k_id)
            dirs.append((f"cluster_{kind}_k{k_id}", unflatten_tangent(vec, params)))
            del vec
    if validate_vec is not None:
        dirs.append(("aggregated_weighted_norm1",
                     unflatten_tangent(validate_vec, params)))
    return dirs


# ── The JVP pass ────────────────────────────────────────────────────────────

def make_nll_fn(model, names, input_ids, comp_start, comp_ids):
    """f: (lora params) -> per-token NLL [T], for torch.func.jvp.

    Same quantity as compute_dpo_gradient_products_pissa_lora's
    build_completion_losses, written as logsumexp(z) - z[target] so no [T, V]
    log-prob tensor (and no dual copy of one) is materialized.
    """
    from torch.func import functional_call

    device = input_ids.device
    T = len(comp_ids)
    pos = comp_start + torch.arange(T, device=device) - 1
    ids = torch.as_tensor(comp_ids, device=device)

    def f(*vals):
        out = functional_call(model, dict(zip(names, vals)), (),
                              {"input_ids": input_ids})
        z = out.logits[0, pos].float()                        # [T, V]
        return torch.logsumexp(z, dim=-1) - z.gather(1, ids[:, None]).squeeze(1)

    return f, T


def token_dots_jvp(model, tokenizer, messages, names, params, directions):
    """{direction name: fp32 [T]} of g_t . direction, one forward-mode pass each.

    Returns (dots, T) or (None, None) if the prompt is unusable.

    Deliberately NOT wrapped in torch.no_grad(): forward-mode AD propagates its
    tangent through the same autograd kernels reverse mode uses, so disabling
    grad mode would silently return zeros. Nothing is retained for backward
    either -- the primals are detached, so no reverse graph is built.
    """
    from torch.func import jvp

    input_ids, comp_start, comp_ids = C.build_chat_ids(tokenizer, messages)
    if input_ids is None:
        return None, None
    device = next(model.parameters()).device
    f, T = make_nll_fn(model, names, input_ids.to(device), comp_start, comp_ids)

    primals = tuple(p.detach() for p in params)
    dots = {}
    for name, tangent in directions:
        # Per-parameter device, not one global device: device_map="auto" spreads
        # the LoRA params across GPUs and a tangent must sit with its primal.
        tangent = tuple(t.to(p.device, non_blocking=True)
                        for t, p in zip(tangent, primals))
        _, out = jvp(f, primals, tangent)
        dots[name] = out.detach().float().cpu()
        del out, tangent
    return dots, T


def token_dots_loop(model, tokenizer, messages, directions_flat):
    """Reference implementation: the exact per-token backward from the existing
    products pass, dotted against the same directions in fp32."""
    products, _, T, _ = compute_rejected_products_and_norms(
        model, tokenizer, messages, directions_flat)
    if products is None:
        return None, None
    return {k: v.float() for k, v in products.items()}, T


# ── Validation ──────────────────────────────────────────────────────────────

def rel_report(name, a, b):
    """Agreement between two [T] score vectors, on the scales that matter:
    absolute/relative error, correlation, and Spearman (weights are a monotone
    function of these scores, so rank agreement is what actually decides
    training)."""
    a, b = a.double(), b.double()
    denom = b.abs().clamp(min=1e-12)
    rel = ((a - b).abs() / denom)
    if a.numel() > 1 and a.std() > 0 and b.std() > 0:
        pear = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
        ra = a.argsort().argsort().double()
        rb = b.argsort().argsort().double()
        spear = float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])
    else:
        pear = spear = float("nan")
    print(f"    {name:34s} max|d|={float((a - b).abs().max()):.3e}  "
          f"rel med={float(rel.median()):.2e} max={float(rel.max()):.2e}  "
          f"pearson={pear:.6f} spearman={spear:.6f}")


def validate_basis(model, tokenizer, dataset, names, params, src_idx, ex_dir):
    """Is a dot computed HERE comparable to one saved by the earlier pass?

    This is the question a fingerprint mismatch raises, and it is answerable
    without any assumption about why the adapters differ. Recompute this
    example's own chosen gradient on the CURRENT adapter, project the rejected
    tokens onto it, and compare against the products_single_weighted_norm1.pt
    that the original pass wrote. Both sides of that dot product are built here,
    so if the two adapters differ only by an orthogonal reparameterisation the
    numbers must agree -- an inner product does not care which basis it is
    expressed in, as long as both vectors are in the same one.

      agree  -> reparameterisation only. The saved products_single_* and the new
                products_cluster_* live on the same numerical scale and maxing
                over them is meaningful.
      differ -> the adapters are genuinely different models. The saved products
                are not comparable to anything computed now, and the honest fix
                is to recompute this directory's scores on the pinned adapter
                (cheap for the 100-example subsample, expensive for the full set).

    Contrast with the aggregated check in validate(), which pairs a SAVED vector
    from the old adapter with gradients from this one -- the pairing the codebase
    forbids. Reported alongside on purpose: single agreeing while aggregated
    disagrees means gradients transport but saved vectors do not.
    """
    from compute_dpo_gradient_products_pissa_lora import compute_chosen_gradients

    saved_path = os.path.join(ex_dir, "products_single_weighted_norm1.pt")
    if not os.path.isfile(saved_path):
        print(f"  {ex_dir}: no products_single_weighted_norm1.pt — skipping the "
              f"basis check")
        return
    chosen = [C.normalize_message(m) for m in dataset[src_idx]["chosen"]]
    rejected = [C.normalize_message(m) for m in dataset[src_idx]["rejected"]]

    _, g_w1, _ = compute_chosen_gradients(model, tokenizer, chosen)
    if g_w1 is None:
        print(f"  {ex_dir}: chosen unusable — skipping the basis check")
        return
    direction = [("single_weighted_norm1", unflatten_tangent(g_w1.cpu(), params))]
    d_jvp, T = token_dots_jvp(model, tokenizer, rejected, names, params, direction)
    d_loop, _ = token_dots_loop(model, tokenizer, rejected,
                                {"single_weighted_norm1": g_w1})
    ref = torch.load(saved_path, map_location="cpu", weights_only=True).float()
    if d_jvp is None or len(ref) < T:
        print(f"  {ex_dir}: length mismatch (T={T}, saved={len(ref)}) — skipping")
        return
    print("  [basis] both sides recomputed on the CURRENT adapter vs the saved file:")
    rel_report("loop(now) vs SAVED single", d_loop["single_weighted_norm1"][:T], ref[:T])
    rel_report("jvp(now)  vs SAVED single", d_jvp["single_weighted_norm1"], ref[:T])
    rel_report("jvp(now)  vs loop(now)", d_jvp["single_weighted_norm1"],
               d_loop["single_weighted_norm1"][:T])
    del g_w1
    torch.cuda.empty_cache()


def validate(model, tokenizer, dataset, names, params, directions, indices,
             dirname, scores_dir, n, kinds, mean_vec):
    """Cross-check the JVP against (a) the exact per-token backward loop and
    (b) the aggregated products file saved by the original pass.

    The loop's directions are the ORIGINAL fp32 centroids, not the bf16 tangents
    the JVP consumes, so the reported error is the honest end-to-end one: it
    includes the tangent's dtype cast, which is exactly what the production path
    pays.

    Memory-hungry by design (fp32 directions for the loop AND bf16 tangents for
    the JVP, ~15 GB at K=8 with both kinds, on top of the model) — this is a
    deliberate one-off check, not the production path.
    """
    device = next(model.parameters()).device
    print(f"\n=== validation on {n} example(s) ===")
    print("  loading fp32 reference copies of every direction for the loop backend")
    flat = {}
    for kind in kinds:
        for k_id in range(CL.K):
            flat[f"cluster_{kind}_k{k_id}"] = CL.load_centroid(kind, k_id, device)
    if mean_vec is not None:
        flat["aggregated_weighted_norm1"] = mean_vec.to(device)
    for src_idx in indices[:n]:
        ex_dir = os.path.join(scores_dir, dirname(src_idx))
        msgs = [C.normalize_message(m) for m in dataset[src_idx]["rejected"]]
        t0 = time.perf_counter()
        d_jvp, T = token_dots_jvp(model, tokenizer, msgs, names, params, directions)
        t_jvp = time.perf_counter() - t0
        if d_jvp is None:
            print(f"  example {src_idx}: unusable — skipped")
            continue
        t0 = time.perf_counter()
        d_loop, _ = token_dots_loop(model, tokenizer, msgs, flat)
        t_loop = time.perf_counter() - t0
        print(f"  example {src_idx}: T={T}  jvp={t_jvp:.1f}s "
              f"({len(directions)} directions)  loop={t_loop:.1f}s  "
              f"speedup={t_loop / max(t_jvp, 1e-9):.1f}x")
        for name in d_jvp:
            rel_report(f"jvp vs loop  {name}", d_jvp[name], d_loop[name].cpu())
        saved = os.path.join(ex_dir, "products_aggregated_weighted_norm1.pt")
        if "aggregated_weighted_norm1" in d_jvp and os.path.isfile(saved):
            ref = torch.load(saved, map_location="cpu", weights_only=True).float()
            n_cmp = min(len(ref), T)
            # NOTE: this pairs the OLD adapter's saved mean vector with THIS
            # adapter's gradients — the mix the codebase warns about. Informative
            # next to the basis check below, not a pass/fail on its own.
            rel_report("jvp vs SAVED aggregated (mixed bases)",
                       d_jvp["aggregated_weighted_norm1"][:n_cmp], ref[:n_cmp])
    del flat
    torch.cuda.empty_cache()

    # The decisive check when a fingerprint mismatched: both sides recomputed here.
    for src_idx in indices[:n]:
        validate_basis(model, tokenizer, dataset, names, params, src_idx,
                       os.path.join(scores_dir, dirname(src_idx)))


def coverage_report(scores_dir, dirname, indices, start, end, kinds):
    """Do the cluster directions cover the SAME examples the baseline trains on?

    train_weighted_dpo.py filters the dataset to examples that have every product
    file its GRADIENT_DIRECTION needs. So if this pass covers fewer examples than
    products_single_weighted_norm1.pt does, then max_weighted_norm1 and
    maxclu_<kind>_K<k> train on different subsets and the comparison between them
    silently confounds the direction with the training set. That is invisible in
    the per-example tallies above -- a skip here and a skip in the original pass
    look identical -- so it is measured explicitly.

    The likeliest cause of a gap is a `T` mismatch: this pass re-tokenizes with
    today's transformers/chat template, and a rendering change shifts the token
    count away from the saved norms_rejected.
    """
    baseline = "products_single_weighted_norm1.pt"
    have_base, have_clu, gaps = 0, 0, []
    for pos in range(start, end):
        ex_dir = os.path.join(scores_dir, dirname(indices[pos]))
        if not os.path.isfile(os.path.join(ex_dir, baseline)):
            continue
        have_base += 1
        if all(os.path.isfile(os.path.join(ex_dir, CL.products_filename(kind, CL.K)))
               for kind in kinds):
            have_clu += 1
        elif len(gaps) < 5:
            gaps.append(indices[pos])
    print(f"\n  coverage over [{start}, {end}): {have_base} examples have "
          f"{baseline}, {have_clu} have cluster products")
    if have_clu < have_base:
        print(f"  >>> {have_base - have_clu} examples would be in a "
              f"max_weighted_norm1 run but NOT in a cluster run, e.g. "
              f"{gaps}. Training both and comparing them would confound the "
              f"direction with the training set — rerun those, or filter the "
              f"baseline to the same set before comparing.")
    else:
        print("  cluster directions cover the baseline's training set exactly.")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("full", "subsample"), default="full",
                    help="Which example directories to add cluster products to")
    ap.add_argument("--start_idx", type=int, default=0,
                    help="First position in the target index list (inclusive)")
    ap.add_argument("--end_idx", type=int, default=-1,
                    help="Last position (exclusive); -1 -> end")
    ap.add_argument("--backend", choices=("jvp", "loop"), default="jvp",
                    help="jvp: one forward-mode pass per direction (default). "
                         "loop: the exact per-token backward from the original "
                         "products pass — ~T/(2*K) times slower and it needs the "
                         "directions in fp32 (~10 GB), but it depends only on "
                         "reverse mode, so it is the fallback if some op in this "
                         "model has no forward-AD rule.")
    ap.add_argument("--validate", type=int, default=0,
                    help="Cross-check the JVP against the exact per-token "
                         "backward loop on N examples, then exit")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--strict_basis", action="store_true",
                    help="Make the target directory's fingerprint check FATAL. Off "
                         "by default: cluster products compute both sides of every "
                         "dot here, and such a dot is basis-independent, so they "
                         "are comparable with that directory's saved products even "
                         "across parameterisations. Turn it on while the reference "
                         "adapter is still rebuildable, as a wrong-container check.")
    args = ap.parse_args()

    scores_dir, dirname, index_list, fp_path, mean_path = resolve_mode(args.mode)
    kinds = CL.CENTROID_KINDS
    cmeta = CL.load_centroid_meta(kinds)            # raises if step 3 is incomplete
    print(f"Cluster centroids: {CL.centroids_dir()}")
    for kind in kinds:
        norms = cmeta["norms"][kind]
        print(f"  {kind:9s} ||c_k|| min={min(norms):.4f} max={max(norms):.4f}  "
              f"({max(norms) / max(min(norms), 1e-12):.1f}x spread)")
    print(f"  cluster sizes: {cmeta['counts']}")

    print(f"\nLoading model {C.FULL_MODEL_CHECKPOINT} with PiSSA LoRA (eager) ...")
    # eager: same attention impl the existing products in these directories were
    # computed with, and every op in its backward/forward has a forward-AD rule.
    model, tokenizer = C.build_model_with_pissa_lora(attn_implementation="eager")
    # ALWAYS fatal against the cluster branch's own fingerprint: the centroids and
    # these token gradients are the two sides of one dot product, so they must
    # share an adapter. The target directory's fingerprint is advisory unless
    # --strict_basis, because a dot with both sides computed here is
    # basis-independent and therefore comparable with that directory's saved
    # values regardless of which parameterisation produced them.
    mismatched = CL.verify_or_establish_fingerprint(
        model,
        fatal=(fp_path,) if args.strict_basis else (),
        advisory=() if args.strict_basis else (fp_path,))
    if mismatched and not args.validate:
        print("  >>> This directory's existing products came from a different "
              "adapter instance. That is expected to be harmless — both sides of "
              "every cluster dot are computed here, and such a dot is "
              "basis-independent. --validate 1 confirms it numerically by "
              "recomputing products_single_weighted_norm1 and checking it "
              "reproduces the saved file at slope ~ +1.\n")

    params, names, numels = C.get_lora_param_list(model)
    grad_dim = int(sum(numels))

    validate_vec = None
    if args.validate:
        v = torch.load(mean_path, map_location="cpu", weights_only=True).float()
        if v.numel() != grad_dim:
            raise RuntimeError(f"{mean_path} dim {v.numel()} != adapter {grad_dim}")
        validate_vec = v

    directions, flat_directions = None, None
    if args.backend == "jvp" or args.validate:
        print(f"Materializing {len(kinds) * CL.K} tangents on {TANGENT_DEVICE} "
              f"({len(kinds) * CL.K * grad_dim * 2 / 1e9:.1f} GB bf16) ...")
        directions = load_directions(params, kinds, CL.K, validate_vec)
    if args.backend == "loop" and not args.validate:
        dev = next(model.parameters()).device
        print(f"Loading {len(kinds) * CL.K} directions in fp32 on {dev} "
              f"({len(kinds) * CL.K * grad_dim * 4 / 1e9:.1f} GB) ...")
        flat_directions = {f"cluster_{kind}_k{k_id}": CL.load_centroid(kind, k_id, dev)
                           for kind in kinds for k_id in range(CL.K)}

    print(f"Loading dataset {C.DATASET_NAME} ...")
    dataset = load_dataset(C.DATASET_NAME, split="train")
    indices = index_list(dataset)

    if args.validate:
        validate(model, tokenizer, dataset, names, params, directions, indices,
                 dirname, scores_dir, args.validate, kinds, validate_vec)
        return

    os.makedirs(scores_dir, exist_ok=True)
    # Products already in this directory were dotted against one specific centroid
    # set. Adding rows from a different set (a subsampled centroids_full_n3000 on
    # top of a full centroids_full, say) would leave a directory whose [K, T]
    # matrices are not comparable example to example, and nothing downstream could
    # tell. Refuse unless the caller says to replace the lot.
    norms_path = os.path.join(scores_dir, CL.cluster_norms_filename())
    if os.path.isfile(norms_path) and not args.overwrite:
        with open(norms_path) as f:
            prev = json.load(f)
        if prev.get("centroids_dir") != CL.centroids_dir():
            raise RuntimeError(
                f"{scores_dir} already holds cluster products built against\n"
                f"  {prev.get('centroids_dir')}\nbut this run would use\n"
                f"  {CL.centroids_dir()}\nRe-run with --overwrite to rebuild them "
                f"all, or point cluster_common.CENTROIDS_SUBDIR back.")

    with open(norms_path, "w") as f:
        json.dump({"K": CL.K, "weighting": CL.WEIGHTING, "kinds": list(kinds),
                   "norms": cmeta["norms"], "counts": cmeta["counts"],
                   "centroids_dir": CL.centroids_dir(), "grad_dim": grad_dim}, f,
                  indent=2)

    start = max(0, args.start_idx)
    end = len(indices) if args.end_idx < 0 else min(args.end_idx, len(indices))
    print(f"\nCluster products for positions [{start}, {end}) of {len(indices)} "
          f"in {scores_dir}")

    done = missing = skipped = 0
    timing = {"n": 0, "s": 0.0}
    pbar = tqdm(range(start, end), desc="Cluster products")
    for pos in pbar:
        src_idx = indices[pos]
        ex_dir = os.path.join(scores_dir, dirname(src_idx))
        # Only ADD to directories the earlier pass completed: this script does not
        # produce norms_rejected.pt (a JVP yields projections, not the gradient
        # vector), and without it there is no cosine.
        if not os.path.isfile(os.path.join(ex_dir, "norms_rejected.pt")):
            missing += 1
            continue
        out_paths = {kind: os.path.join(ex_dir, CL.products_filename(kind, CL.K))
                     for kind in kinds}
        if not args.overwrite and all(os.path.isfile(p) for p in out_paths.values()):
            done += 1
            continue

        msgs = [C.normalize_message(m) for m in dataset[src_idx]["rejected"]]
        t0 = time.perf_counter()
        if args.backend == "jvp":
            dots, T = token_dots_jvp(model, tokenizer, msgs, names, params, directions)
        else:
            dots, T = token_dots_loop(model, tokenizer, msgs, flat_directions)
        if dots is None:
            skipped += 1
            continue
        timing["n"] += 1
        timing["s"] += time.perf_counter() - t0

        norms = torch.load(os.path.join(ex_dir, "norms_rejected.pt"),
                           map_location="cpu", weights_only=True)
        if len(norms) != T:
            # A length mismatch means the two passes tokenized differently; the
            # per-token pairing would be silently wrong, so refuse this example.
            tqdm.write(f"  {ex_dir}: T={T} but norms_rejected has {len(norms)} — "
                       f"skipping (tokenization drift)")
            skipped += 1
            continue

        for kind in kinds:
            mat = torch.stack([dots[f"cluster_{kind}_k{k_id}"] for k_id in range(CL.K)])
            tmp = out_paths[kind] + ".tmp"
            torch.save(mat, tmp)
            os.replace(tmp, out_paths[kind])
        done += 1
        if timing["n"]:
            pbar.set_postfix(done=done, skipped=skipped, no_dir=missing,
                             s_ex=f"{timing['s'] / timing['n']:.2f}")
        torch.cuda.empty_cache()

    n_dirs = len(kinds) * CL.K
    print(f"\nDone. {done} examples have cluster products "
          f"({skipped} skipped, {missing} had no existing products dir).")
    coverage_report(scores_dir, dirname, indices, start, end, kinds)
    if timing["n"]:
        print(f"  {args.backend}: {timing['s'] / timing['n']:.2f} s/example for "
              f"{n_dirs} directions "
              f"({timing['s'] / timing['n'] / n_dirs:.3f} s per direction)")
    print(f"  ||c_k|| recorded in {norms_path}")


if __name__ == "__main__":
    main()
