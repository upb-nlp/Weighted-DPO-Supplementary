"""
Build the PiSSA-LoRA adapter ONCE, PROVE it is the adapter that produced the
existing gradient products, and write it to disk so no later pass re-derives it.

WHY PINNING
-----------
pissa_lora_common derives the adapter by running torch.linalg.svd over every
targeted weight matrix. That is deterministic within one environment and NOT
across environments: an SVD is unique only up to sign flips of paired singular
vectors and rotations inside near-degenerate singular subspaces, so a different
container, torch/peft version or device placement yields a different
PARAMETERISATION of the same model. Saving the tensors once removes the SVD from
the critical path forever.

WHY VERIFY BEFORE SAVING
------------------------
Pinning freezes whatever basis you happen to be in. Freeze the wrong one and
every products_*.pt in the repo silently stops being comparable to anything
computed afterwards. So this script refuses to save until it has shown that the
live adapter reproduces artifacts the reference pass already wrote.

THREE LEVELS, AND WHY EACH IS NEEDED
------------------------------------
1. FINGERPRINT (fast, no gradients). Per-tensor (numel, sum, sumsq) against
   MEAN_CHOSEN_DIR/lora_fingerprint.json -- exactly the check
   compute_dpo_gradient_products_pissa_lora.py runs before it will process
   anything. Cheap, but it is two scalars per tensor.

2. BASIS-INVARIANT reproduction: ||g_chosen||, norms_rejected, and
   products_single_* (this example's own chosen gradient dotted with its own
   rejected-token gradients). Both sides of those are recomputed here, and an
   inner product does not care which basis it is expressed in as long as both
   vectors share it. So level 2 answers "is this the same MODEL?" and would
   still pass on a rotated adapter.

3. BASIS-SENSITIVE reproduction: products_aggregated_*, which dots the STORED
   mean_grad_*.pt vector against freshly computed token gradients. A stored
   vector does not rotate with the adapter, so this is the one check that fails
   if the basis drifted. It is the reason level 2 alone is not enough.

Reading the verdict:
    2 and 3 pass -> same model, same coordinates. Safe to pin; everything in
                    SAVE_DIR stays valid.
    2 passes, 3 fails -> same model, DIFFERENT SVD basis. Pinning is still
                    coherent going forward, but products_aggregated_* can no
                    longer be extended (the stored mean vector belongs to the old
                    basis); it would have to be rebuilt with
                    compute_mean_chosen_gradient.py.
    2 fails -> different weights. Do not pin. You are in the wrong container.

The messages come from each example's own metadata.json, so the check depends on
nothing but the adapter -- not on the HF dataset being reachable.

WHAT IS SAVED (~13 GB for all-linear on a 7B, bf16)
    the LoRA A/B factors (~0.3 GB at r=64), AND the PiSSA-modified base weight of
    every targeted module. The residuals are included because the LoRA gradients
    depend on the forward activations, hence on the residual; reconstructing it
    as W_hf - (alpha/r) B@A would reintroduce a dependence on rounding order.

USAGE
    python save_pissa_adapter.py --check_only            # just answer "am I in the right container?"
    python save_pissa_adapter.py --path /data/weighted-dpo/pissa-lora-adapter-r64
then set in pissa_lora_common.py:
    PISSA_ADAPTER_DIR = "/data/weighted-dpo/pissa-lora-adapter-r64"
With PISSA_ADAPTER_DIR already set, --check_only verifies the PINNED adapter
instead of a freshly derived one -- the same three levels, now proving the file
on disk reproduces the products.
"""

import argparse
import json
import os

import torch
from dotenv import load_dotenv
load_dotenv(".env")

import pissa_lora_common as C
from compute_dpo_gradient_products_pissa_lora import (
    SAVE_DIR,
    compute_chosen_gradients,
    compute_rejected_products_and_norms,
    is_example_complete,
    load_mean_gradients,
)

# A reproduction counts as matching when the median relative difference is below
# this and the vectors are essentially perfectly correlated. Slack is for bf16
# gradients and non-deterministic CUDA reduction order, which move a dot product
# over D ~ 1.6e8 terms at the 1e-4..1e-3 level between identical runs.
REL_TOL = 1e-2
CORR_TOL = 0.999
# Correlation floor on the UNAMPLIFIED channels below which the model itself is
# considered different. Two unrelated 240-d vectors correlate at 0 +- 1/sqrt(240)
# ~ 0.065, so anything above ~0.99 there is the same geometry seen through
# different rounding, not a different model.
CLEAN_CORR_TOL = 0.995


# ── Comparison ──────────────────────────────────────────────────────────────

def compare(label, got, ref, clean=False):
    """Agreement between a freshly computed and a saved quantity. Returns
    (ok, stats) and prints one line.

    `clean` marks the UNAMPLIFIED channels -- norms_rejected and the unweighted
    products. The weighted variants divide by (1 - p), so for a token the model
    is confident about, a bf16-level difference in the logits becomes a large
    difference in the weight and hence in the product. Those channels are
    therefore useless for deciding "is this the same model"; the clean ones are
    the ones classify() trusts.
    """
    got = torch.as_tensor(got, dtype=torch.float64).reshape(-1)
    ref = torch.as_tensor(ref, dtype=torch.float64).reshape(-1)
    n = min(got.numel(), ref.numel())
    got, ref = got[:n], ref[:n]
    rel = ((got - ref).abs() / ref.abs().clamp(min=1e-12))
    med, mx = float(rel.median()), float(rel.max())
    # SLOPE is the statistic that decides: least squares through the origin, so
    # slope == 1 means "reproduces". Correlation is invariant to any positive
    # affine rescaling -- it measures SHAPE and by construction ignores the thing
    # under test. A mixed-basis comparison can sit at |corr| = 0.98 while the
    # values are off by a factor of -0.5, because projecting the same token
    # gradients onto a sign-flipped copy of a vector preserves most of the
    # per-token shape while negating the result. Both are printed; read slope.
    slope = float((got * ref).sum() / (ref * ref).sum().clamp(min=1e-30))
    # A single scalar has no correlation. Recorded as NaN rather than faked, so
    # classify() can exclude it instead of reading a fabricated 0.0 as evidence
    # that the model changed.
    if n > 1 and got.std() > 0 and ref.std() > 0:
        corr = float(torch.corrcoef(torch.stack([got, ref]))[0, 1])
    else:
        corr = float("nan")
    ok = med < REL_TOL and (corr > CORR_TOL or n == 1)
    print(f"    {'OK ' if ok else 'BAD'} {label:38s} n={n:5d} "
          f"rel med={med:.2e} max={mx:.2e} slope={slope:+.4f} corr="
          + ("  n/a   " if n == 1 else f"{corr:.6f}"))
    return ok, {"n": n, "rel_median": med, "rel_max": mx, "corr": corr,
                "slope": slope, "ok": ok, "clean": clean, "label": label}


def load_pt(example_dir, name):
    return torch.load(os.path.join(example_dir, name), map_location="cpu",
                      weights_only=True).float()


def verify_example(model, tokenizer, example_dir, means):
    """Recompute this example's saved artifacts and diff them.

    Returns (level2_stats, level3_stats) as lists of the per-check dicts, or
    (None, None) if the example is unusable. The CALLER classifies, because the
    verdict turns on correlation as much as on relative error: a rounding-level
    perturbation of the adapter shows up as a few percent of median relative
    error while correlation stays at 0.999+, whereas genuinely different weights
    destroy the correlation too.
    """
    with open(os.path.join(example_dir, "metadata.json")) as f:
        meta = json.load(f)
    # The messages the reference pass actually used, so tokenization cannot drift.
    chosen, rejected = meta["chosen"], meta["rejected"]

    print(f"  {os.path.basename(example_dir)}  "
          f"({meta['num_rejected_tokens']} rejected tokens)")
    g_uw, g_w1, g_w2 = compute_chosen_gradients(model, tokenizer, chosen)
    if g_uw is None:
        print("    unusable — skipped")
        return None, None

    inv, basis = [], []
    # Level 2a: the chosen-gradient norms (scalars, basis-invariant).
    for key, vec in (("unweighted", g_uw), ("weighted_norm1", g_w1),
                     ("weighted_norm2", g_w2)):
        _, st = compare(f"||g_chosen_{key}||", float(vec.norm()),
                        float(meta[f"norm_{key}"]), clean=(key == "unweighted"))
        inv.append(st)

    # One backward per rejected token, dotted against everything at once. This is
    # the reference implementation itself, so a mismatch is the adapter, nothing
    # else.
    directions = {
        "single_unweighted": g_uw,
        "single_weighted_norm1": g_w1,
        "single_weighted_norm2": g_w2,
        "aggregated_unweighted": means["unweighted"],
        "aggregated_weighted_norm1": means["weighted_norm1"],
        "aggregated_weighted_norm2": means["weighted_norm2"],
    }
    products, norms, T, _ = compute_rejected_products_and_norms(
        model, tokenizer, rejected, directions)
    if products is None:
        print("    rejected unusable — skipped")
        return None, None

    # Level 2b: per-token gradient norms and the single_* products. Both sides
    # recomputed here, so these survive a change of basis.
    _, st = compare("norms_rejected", norms,
                    load_pt(example_dir, "norms_rejected.pt"), clean=True)
    inv.append(st)
    for name in ("single_unweighted", "single_weighted_norm1", "single_weighted_norm2"):
        _, st = compare(f"products_{name}", products[name],
                        load_pt(example_dir, f"products_{name}.pt"),
                        clean=(name == "single_unweighted"))
        inv.append(st)

    # Level 3: the STORED mean vector against fresh gradients — does not survive
    # a change of basis, which is exactly what makes it informative. A rotated
    # adapter gives correlation ~0 here; per-singular-vector SIGN FLIPS (PiSSA may
    # return (U,S,V) or (-U,S,-V), both reconstructing the same B@A) give
    # correlation ~ -1, since the stored vector is in the old sign convention.
    for name in ("aggregated_unweighted", "aggregated_weighted_norm1",
                 "aggregated_weighted_norm2"):
        _, st = compare(f"products_{name}", products[name],
                        load_pt(example_dir, f"products_{name}.pt"))
        basis.append(st)

    del g_uw, g_w1, g_w2, products, norms
    torch.cuda.empty_cache()
    return inv, basis


def fast_probe(ref_path):
    """Decide whether THIS environment reproduces the reference basis, using one
    LoRA module instead of 224.

    For triaging containers. PiSSA's init is per-module independent -- each
    module's A and B come from the SVD of that module's own weight -- so
    attaching LoRA to a single module produces byte-identical tensors for it to
    what the full run produced, at 1/224th of the SVD cost. The basis-sensitive
    `sum` stat then answers "would the full fingerprint match?" in about a minute.

    A negative result is conclusive: that module is enough to prove the basis
    differs. A positive result is strong but must be confirmed with the real
    check (--check_only --verify 0) before pinning anything.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    with open(ref_path) as f:
        ref = json.load(f)
    ref_stats = dict(zip(ref["names"], ref["stats"]))
    # "base_model.model.<module path>.lora_A.default.weight" -> "<module path>".
    # peft matches target_modules by suffix, so the full path selects exactly one.
    module = ref["names"][0].split(".lora_")[0].replace("base_model.model.", "", 1)
    print(f"  probing a single module: {module}")

    model = AutoModelForCausalLM.from_pretrained(
        C.FULL_MODEL_CHECKPOINT, config=C._load_base_config(),
        dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
        attn_implementation="eager")
    for p in model.parameters():
        p.requires_grad_(False)
    # Mirrors build_model_with_pissa_lora exactly except for target_modules.
    torch.manual_seed(C.PISSA_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(C.PISSA_SEED)
    model = get_peft_model(model, LoraConfig(
        r=C.LORA_R, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT,
        bias="none", target_modules=[module], task_type="CAUSAL_LM",
        init_lora_weights=C.LORA_INIT))
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    cur = C.lora_fingerprint(model)
    hit = False
    for name, stat in zip(cur["names"], cur["stats"]):
        if name not in ref_stats:
            print(f"    {name}: not in the reference fingerprint (name scheme "
                  f"changed?)")
            continue
        hit = True
        r = ref_stats[name]
        d_sum = abs(stat[1] - r[1]) / max(abs(r[1]), 1e-30)
        d_sq = abs(stat[2] - r[2]) / max(abs(r[2]), 1e-30)
        print(f"    {name.split('.')[-3:][0]:12s} sumsq rel={d_sq:.3e}  "
              f"sum rel={d_sum:.3e}  {'MATCH' if d_sum < 1e-4 else 'differs'}")
    if not hit:
        raise SystemExit("Could not match any tensor name — cannot probe.")
    print("\n  sum is basis-SENSITIVE: if it matches here, this environment very "
          "likely\n  reproduces the reference basis — confirm with "
          "--check_only --verify 0 before pinning.\n  If it differs, this "
          "environment is conclusively not the one.")


def classify(inv, basis):
    """(kind, explanation) from the level-2 and level-3 check statistics.

    Correlation, not relative error, is what separates the cases. The PiSSA
    residual is W_res := W - BA, so ANY basis the SVD lands on reconstructs the
    same weight matrix and therefore the same model -- but with different
    rounding, which the near-p=1 token weights (1/(2(1-p))) amplify from bf16
    noise into percent-level differences in the derived quantities. That looks
    "BAD" against a tight relative tolerance while the correlation stays at
    0.999+. Genuinely different weights destroy the correlation as well.
    """
    inv_vec = [s for s in inv if s["n"] > 1]        # scalars carry no correlation
    basis_vec = [s for s in basis if s["n"] > 1]
    clean = [s for s in inv_vec if s["clean"]]      # unamplified channels only
    inv_corr = min(s["corr"] for s in clean) if clean else float("nan")
    inv_rel = max(s["rel_median"] for s in clean) if clean else float("nan")
    basis_corr = min(s["corr"] for s in basis_vec) if basis_vec else 1.0
    # Furthest-from-1 slope: the only statistic that distinguishes "reproduces"
    # from "same shape, wrong value".
    basis_slope = (max((s["slope"] for s in basis_vec), key=lambda v: abs(v - 1.0))
                   if basis_vec else 1.0)

    if all(s["ok"] for s in inv) and all(s["ok"] for s in basis):
        return "EXACT", (f"This IS the adapter behind {SAVE_DIR} — same weights, "
                         f"same basis. Safe to pin; every existing artifact stays "
                         f"valid.")
    if inv_corr > CLEAN_CORR_TOL:
        note = (f"the unamplified basis-invariant channels (norms_rejected, "
                f"products_single_unweighted) track the saved ones at correlation "
                f">= {inv_corr:.5f}, median relative error {inv_rel:.1%}")
        if abs(basis_slope - 1.0) > 0.1:
            return "SAME MODEL, DIFFERENT PARAMETERISATION", (
                f"{note}, while the basis-SENSITIVE products_aggregated_* "
                f"reproduce at slope {basis_slope:+.3f} instead of +1 "
                f"(correlation {basis_corr:+.3f} — high |corr| there is NOT "
                f"reassurance: those compare the same token gradients projected "
                f"onto a fixed vector and a sign-flipped copy of it, which "
                f"preserves shape while changing the value"
                + (", hence the negative slope: PiSSA returned (-U,S,-V) for a "
                   "SUBSET of singular pairs where it once returned (U,S,V) — "
                   "same B@A, same model, flipped gradient coordinates on those "
                   "blocks, so |slope| < 1" if basis_slope < 0 else "") + ").\n"
                f"     The model is the same; its LoRA coordinates are not. The "
                f"original adapter was never saved, so it cannot be recovered from "
                f"disk — only by finding the exact environment that derived it.\n"
                f"     Everything in {SAVE_DIR} was computed within ONE basis and "
                f"remains valid. You can still ADD quantities computed here, as "
                f"long as BOTH sides of each new dot product are computed here: "
                f"such a dot is basis-independent (<Dg,Dv> = <g,v>) and the "
                f"single_* slope above shows the agreement is unbiased. The one "
                f"forbidden operation is pairing a STORED vector from that run "
                f"(mean_grad_*.pt) with gradients computed here — which is exactly "
                f"what the aggregated_* row above did, on purpose, to detect this.")
        return "SAME MODEL, DRIFTED SUBSPACE", (
            f"{note}, and the aggregated products track too. The rank-r subspaces "
            f"differ slightly (near-tied singular values), so derived values agree "
            f"only to ~{inv_rel:.1%}. Pinning is coherent, but do not mix these "
            f"values with the saved ones at finer resolution than that.")
    return "DIFFERENT WEIGHTS", (
        f"correlation on the unamplified basis-invariant channels falls to "
        f"{inv_corr:.4f} — this is not the same model. Wrong container, wrong base "
        f"checkpoint, or a changed LoRA config. Do not pin this one.")


def find_examples(save_dir, n):
    """The first n complete example dirs in the reference products directory."""
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(
            f"{save_dir} does not exist — there is nothing to verify against. "
            f"Run with --skip_verify if you really mean to pin an unverified "
            f"adapter.")
    out = []
    for name in sorted(os.listdir(save_dir)):
        if not name.startswith("example_"):
            continue
        d = os.path.join(save_dir, name)
        if is_example_complete(d):
            out.append(d)
        if len(out) >= n:
            break
    if not out:
        raise FileNotFoundError(f"No complete example dirs in {save_dir}")
    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", help="Directory to write adapter_state.pt into "
                                   "(not needed with --check_only)")
    ap.add_argument("--check_only", action="store_true",
                    help="Run the three verification levels and exit without "
                         "saving. With PISSA_ADAPTER_DIR set this checks the "
                         "PINNED adapter; otherwise a freshly derived one.")
    ap.add_argument("--verify", type=int, default=1,
                    help="How many example dirs to reproduce (each costs one "
                         "backward per rejected token, so ~minutes). 0 = "
                         "fingerprint only, which is the fast probe for sweeping "
                         "containers or GPU counts looking for the environment "
                         "that derived the reference adapter.")
    ap.add_argument("--skip_verify", action="store_true",
                    help="Save without reproducing anything. Only when there are "
                         "no reference products to check against.")
    ap.add_argument("--force", action="store_true",
                    help="Save even if verification failed, or overwrite an "
                         "existing adapter")
    ap.add_argument("--save_fingerprint", nargs="?", const="", metavar="PATH",
                    help="Record THIS adapter's fingerprint and exit, without "
                         "deriving or writing the 13 GB adapter itself. Defaults "
                         "to cluster_common.FINGERPRINT_FILE. Unlike the June one, "
                         "it carries torch/transformers/peft/device_count, so a "
                         "future mismatch is diagnosable instead of a guessing "
                         "game. Never overwrites without --force.")
    ap.add_argument("--fast_probe", action="store_true",
                    help="Attach LoRA to ONE module instead of 224 and compare "
                         "just that module against the reference fingerprint. ~1 "
                         "minute per container instead of ~10 — use it to triage "
                         "candidate singularity images, then confirm a hit with "
                         "--check_only --verify 0.")
    args = ap.parse_args()

    if args.fast_probe:
        fp = os.path.join(C.MEAN_CHOSEN_DIR, "lora_fingerprint.json")
        print(f"environment: {json.dumps(C._env_versions())}")
        print(f"reference:   {fp}")
        fast_probe(fp)
        return

    if args.save_fingerprint is not None:
        import cluster_common as CL
        out = args.save_fingerprint or CL.FINGERPRINT_FILE
        if os.path.isfile(out) and not args.force:
            raise SystemExit(
                f"{out} already exists — it is the anchor a later pass will be "
                f"checked against, so overwriting it silently would re-open "
                f"exactly the ambiguity it exists to close. Pass --force if the "
                f"adapter it describes is genuinely obsolete.")
        print(f"environment: {json.dumps(C._env_versions())}")
        model, _ = C.build_model_with_pissa_lora(attn_implementation="eager")
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        C.save_fingerprint(model, out)
        ref = os.path.join(C.MEAN_CHOSEN_DIR, "lora_fingerprint.json")
        print(f"\nWrote {out}\n"
              f"  This adapter is now the cluster branch's reference; every later "
              f"cluster pass verifies against it and fails loudly on drift.\n"
              f"  The original reference at {ref} is untouched — nothing in this "
              f"pipeline writes there.")
        return

    if not args.check_only and not args.path:
        raise SystemExit("--path is required unless --check_only")

    pinned = C.PISSA_ADAPTER_DIR is not None and os.path.isdir(C.PISSA_ADAPTER_DIR)
    if args.path:
        state = os.path.join(args.path, "adapter_state.pt")
        if os.path.isfile(state) and not args.force:
            raise SystemExit(
                f"{state} already exists. Point PISSA_ADAPTER_DIR at it rather "
                f"than rebuilding — a rebuild would produce a different basis and "
                f"invalidate everything computed against the existing one. Pass "
                f"--force only if you intend exactly that.")
        if pinned:
            raise SystemExit(
                f"pissa_lora_common.PISSA_ADAPTER_DIR is already "
                f"{C.PISSA_ADAPTER_DIR!r}, so the model would LOAD that adapter "
                f"and this would just copy it. Unset it to derive fresh, or use "
                f"--check_only to verify the pinned one.")

    print(f"{'Loading the pinned' if pinned else 'Deriving the'} PiSSA adapter "
          f"(init={C.LORA_INIT!r}, r={C.LORA_R}, target={C.LORA_TARGET_MODULES!r})"
          + ("" if pinned else " — the exact SVD over ~224 matrices takes minutes")
          + " ...")
    # eager: the attention impl the reference products were computed with, so the
    # numbers being diffed below are apples to apples.
    model, tokenizer = C.build_model_with_pissa_lora(attn_implementation="eager")

    # Printed so two containers can be diffed by eye. The reference fingerprint
    # predates env recording, so this is the only way to hunt for the environment
    # that derived it. device_count matters as much as the versions:
    # device_map="auto" runs each weight's SVD on whichever device holds it.
    print(f"\n  environment: {json.dumps(C._env_versions())}")

    # ── Level 1 ──
    print("\n=== level 1: fingerprint ===")
    fp = os.path.join(C.MEAN_CHOSEN_DIR, "lora_fingerprint.json")
    fp_ok = True
    if os.path.isfile(fp):
        try:
            C.verify_fingerprint(model, fp)
        except RuntimeError as e:
            fp_ok = False
            print(f"{e}\n")
    else:
        print(f"  {fp} not found — skipping")

    verdict = None
    if args.verify > 0 and not args.skip_verify:
        _, _, numels = C.get_lora_param_list(model)
        means = load_mean_gradients(next(model.parameters()).device, sum(numels))
        print(f"\n=== levels 2 and 3: reproduce saved products from {SAVE_DIR} ===")
        results = [verify_example(model, tokenizer, d, means)
                   for d in find_examples(SAVE_DIR, args.verify)]
        results = [r for r in results if r[0] is not None]
        if not results:
            raise SystemExit("No example could be reproduced — nothing verified.")
        inv = [s for r in results for s in r[0]]
        basis = [s for r in results for s in r[1]]
        verdict = classify(inv, basis)

    # ── Verdict ──
    print("\n=== verdict ===")
    print(f"  fingerprint              : {'match' if fp_ok else 'MISMATCH'}")
    safe = fp_ok
    if verdict is not None:
        kind, detail = verdict
        print(f"  reproduction             : {kind}")
        print(f"  {detail}")
        safe = fp_ok and kind == "EXACT"

    if args.check_only:
        return
    if not safe and not args.force:
        raise SystemExit(
            "\nRefusing to save: this is not the adapter that produced "
            f"{SAVE_DIR}.\n"
            "  If the verdict is DIFFERENT WEIGHTS, find the right container.\n"
            "  If it is SAME MODEL, DIFFERENT PARAMETERISATION, the reference "
            "adapter no longer\n"
            "  exists anywhere (it was never saved). Pinning THIS one is then a "
            "deliberate\n"
            "  choice to start a new basis: legitimate, but it means every "
            "direction you want\n"
            "  must be rebuilt here rather than read from the old dumps. Pass "
            "--force to do that.")

    os.makedirs(args.path, exist_ok=True)
    C.save_adapter(model, args.path)
    print("  round-tripping the saved tensors back into the model ...")
    C.load_adapter(model, args.path)          # re-verifies against its own fingerprint
    print(f"\nNow set in pissa_lora_common.py:\n"
          f"    PISSA_ADAPTER_DIR = {args.path!r}\n"
          f"Every later pass loads these exact tensors instead of re-running the "
          f"SVD, so no fingerprint can drift again. Re-run this script with "
          f"--check_only at any time to re-prove the pinned adapter still "
          f"reproduces {SAVE_DIR}.")


if __name__ == "__main__":
    main()
