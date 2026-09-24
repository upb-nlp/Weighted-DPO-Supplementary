"""
Shared building blocks for the PiSSA-LoRA gradient-product experiments.

Both `compute_mean_chosen_gradient.py` (which aggregates the mean chosen
gradient over the dataset) and `compute_dpo_gradient_products_meanchosen.py`
(which dots each rejected token's gradient against that mean) MUST operate on
the *byte-identical* LoRA adapter, or the dot products are meaningless.

The adapter here is NOT a trained checkpoint. It is derived directly from the
pretrained SFT weights with PiSSA, exactly like
`../Compare-Different-Gradients-Projections/modeling.py::attach_lora`:
PEFT's default init sets B=0, so the gradient w.r.t. A is identically zero and
the LoRA gradient is half-degenerate; PiSSA initialises A and B from the SVD of
each weight matrix so both receive gradients.

Reproducibility: with the default exact `pissa` init, `torch.linalg.svd` is
deterministic, so two independent runs build the identical residual base + A + B
(this is what lets the two scripts share an adapter without any save/load of the
PiSSA residual). `lora_fingerprint` / `verify_fingerprint` guard against any
drift: the products script refuses to run if its freshly-built adapter does not
match the one the mean-gradient script used.
"""

import json
import os
import warnings

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ── Shared config ─────────────────────────────────────────────────────────
FULL_MODEL_CHECKPOINT = "allenai/Olmo-3-7B-Instruct-SFT"
DATASET_NAME = "anonymous/Dolci-Instruct-DPO-en"
MAX_SEQ_LENGTH = 2048

# Where the mean-chosen-gradient artifacts (mean vectors + fingerprint + meta)
# live. The products script reads the fingerprint + mean vectors from here.
MEAN_CHOSEN_DIR = "/data/weighted-dpo/pissa-lora-mean-chosen"

# LoRA derived from the weights — mirrors Compare-Different-Gradients-Projections.
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = "all-linear"
# "pissa" = exact SVD (deterministic -> guaranteed identical across runs).
# "pissa_niter_16" = fast randomized SVD (matches the Compare project's default,
# but reproducibility then relies on RNG determinism; the fingerprint check will
# catch a mismatch). Switch to the niter variant if exact init is too slow.
LORA_INIT = "pissa"
PISSA_SEED = 0

# ── Persisted adapter ───────────────────────────────────────────────────────
# Deriving the adapter from the SVD makes it reproducible ONLY as long as
# torch.linalg.svd returns the identical factorisation, which in practice drifts
# across library versions and across device placements (device_map="auto" runs
# each weight's SVD on whichever device holds it). An SVD is unique only up to
# sign flips of paired singular vectors and rotations inside near-degenerate
# singular subspaces, so a drifted run yields a DIFFERENT PARAMETERISATION of the
# same update B@A -- which is enough to make verify_fingerprint fail and enough to
# invalidate any dot product that mixes vectors from the two runs.
#
# Setting PISSA_ADAPTER_DIR removes the SVD from the critical path entirely:
# save_adapter() writes the exact tensors once, and every later pass loads them
# byte-for-byte instead of re-deriving them. It is also much faster to start
# (the exact PiSSA SVD over ~224 matrices of a 7B model is minutes).
#
# What gets saved: the LoRA A/B tensors (~0.3 GB at r=64) AND the PiSSA-modified
# base weights of every targeted module (~13 GB for "all-linear" on a 7B model,
# bf16). The base residuals are included because the gradients w.r.t. A and B
# depend on the forward activations, hence on the residual -- reconstructing it as
# W_hf - (alpha/r)*B@A would re-introduce a dependence on rounding order.
PISSA_ADAPTER_DIR = '/data/weighted-dpo/pissa-lora-adapter-r64'    # e.g. "/data/weighted-dpo/pissa-lora-adapter-r64"


# ── Tokenizer / model / adapter ─────────────────────────────────────────────

def load_tokenizer():
    """Tokenizer with bos as the pad token (never eos) and left padding."""
    tok = AutoTokenizer.from_pretrained(
        FULL_MODEL_CHECKPOINT, trust_remote_code=True, padding_side="left"
    )
    # Pad with bos (project convention) — do NOT add a new token.
    tok.pad_token = tok.bos_token
    return tok


def _load_base_config():
    # Olmo-3's config.json stores rope_parameters beta_fast/beta_slow as ints;
    # transformers now requires floats. Cast them in the loaded config.
    config = AutoConfig.from_pretrained(FULL_MODEL_CHECKPOINT, trust_remote_code=True)
    rope_params = getattr(config, "rope_parameters", None)
    if isinstance(rope_params, dict):
        for k in ("beta_fast", "beta_slow"):
            if k in rope_params:
                rope_params[k] = float(rope_params[k])
    return config


def build_model_with_pissa_lora(attn_implementation, dtype=torch.bfloat16,
                                adapter_dir=None):
    """Load the frozen SFT base and attach a PiSSA-derived LoRA adapter.

    `attn_implementation` is "sdpa" (fast — for the mean-gradient pass) or
    "eager" (required by the products script's is_grads_batched backward). The
    adapter construction is identical regardless of attention impl, so the
    fingerprint matches across both scripts.

    `adapter_dir` (defaulting to PISSA_ADAPTER_DIR) loads a previously saved
    adapter instead of deriving it from the SVD, which is the only way to
    GUARANTEE two passes share a parameterisation. When loading, the LoRA is
    attached with the cheap default init (B=0) purely to create the right module
    structure; every A, B and base-layer weight is then overwritten from disk.

    Returns (peft_model, tokenizer). Only LoRA params require grad.
    """
    from peft import LoraConfig, get_peft_model

    adapter_dir = PISSA_ADAPTER_DIR if adapter_dir is None else adapter_dir
    load_saved = adapter_dir is not None and os.path.isdir(adapter_dir)

    tokenizer = load_tokenizer()
    config = _load_base_config()

    base_model = AutoModelForCausalLM.from_pretrained(
        FULL_MODEL_CHECKPOINT,
        config=config,
        dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    for p in base_model.parameters():
        p.requires_grad_(False)

    lconf = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias="none",
        target_modules=LORA_TARGET_MODULES, task_type="CAUSAL_LM",
        # Skip the (slow, environment-sensitive) SVD when the exact tensors are
        # about to be loaded over the top of it anyway.
        init_lora_weights=True if load_saved else LORA_INIT,
    )

    # Seed right before init so the (randomized) PiSSA SVD is reproducible.
    torch.manual_seed(PISSA_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(PISSA_SEED)

    if load_saved:
        print(f"Attaching LoRA structure (r={LORA_R}, target={LORA_TARGET_MODULES!r}) "
              f"then loading the saved adapter from {adapter_dir} ...")
    else:
        print(f"Attaching PiSSA LoRA (init={LORA_INIT!r}, r={LORA_R}, "
              f"target={LORA_TARGET_MODULES!r}, seed={PISSA_SEED}) ...")
    model = get_peft_model(base_model, lconf)
    if load_saved:
        load_adapter(model, adapter_dir)
    model.config.use_cache = False
    model.eval()

    # Belt-and-suspenders: only LoRA params trainable.
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name

    params, _, numels = get_lora_param_list(model)
    print(f"Trainable LoRA parameters: {sum(numels):,} across {len(params)} tensors")
    return model, tokenizer


def get_lora_param_list(model):
    """Return (params, names, numels) for the trainable LoRA params, in a fixed
    iteration order. ALL gradient flattening (mean vector, per-example chosen,
    per-token rejected) uses this exact order so every dot product is aligned
    dimension-for-dimension."""
    params, names, numels = [], [], []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            params.append(param)
            names.append(name)
            numels.append(param.numel())
    return params, names, numels


# ── Persisting the exact adapter ────────────────────────────────────────────

def _adapter_state_keys(model):
    """Keys of every tensor that defines this adapter's gradient geometry: the
    LoRA factors, plus the PiSSA-modified base weight of each targeted module
    (the forward activations — hence the LoRA gradients — depend on it)."""
    return [k for k in model.state_dict()
            if "lora_" in k or k.endswith(".base_layer.weight")]


def save_adapter(model, path):
    """Write the exact adapter tensors + fingerprint to `path`.

    Run this ONCE from the pass that establishes the basis, then point
    PISSA_ADAPTER_DIR at it so every later pass loads rather than re-derives.
    Size is dominated by the base residuals (~13 GB for all-linear on 7B, bf16).
    """
    os.makedirs(path, exist_ok=True)
    sd = model.state_dict()
    keys = _adapter_state_keys(model)
    tensors = {k: sd[k].detach().cpu() for k in keys}
    nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
    tmp = os.path.join(path, "adapter_state.pt.tmp")
    torch.save(tensors, tmp)
    os.replace(tmp, os.path.join(path, "adapter_state.pt"))
    save_fingerprint(model, os.path.join(path, "lora_fingerprint.json"))
    with open(os.path.join(path, "adapter_meta.json"), "w") as f:
        json.dump({"model": FULL_MODEL_CHECKPOINT, "lora_init": LORA_INIT,
                   "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
                   "lora_dropout": LORA_DROPOUT,
                   "lora_target_modules": LORA_TARGET_MODULES,
                   "pissa_seed": PISSA_SEED, "n_tensors": len(keys),
                   "bytes": int(nbytes), "env": _env_versions()}, f, indent=2)
    print(f"  saved adapter ({len(keys)} tensors, {nbytes / 1e9:.2f} GB) -> {path}")


def load_adapter(model, path):
    """Overwrite this model's LoRA factors and base residuals from `path`.

    Verifies against the fingerprint saved alongside, so a partial or mismatched
    load fails loudly rather than silently producing a third basis.
    """
    state_path = os.path.join(path, "adapter_state.pt")
    if not os.path.isfile(state_path):
        raise FileNotFoundError(
            f"{state_path} not found. Build the adapter once and call "
            f"save_adapter(model, {path!r}) before pointing PISSA_ADAPTER_DIR here.")
    tensors = torch.load(state_path, map_location="cpu", weights_only=True)
    expected = set(_adapter_state_keys(model))
    missing = expected - set(tensors)
    extra = set(tensors) - expected
    if missing or extra:
        raise RuntimeError(
            f"Saved adapter does not match this model's structure "
            f"({len(missing)} missing, {len(extra)} unexpected keys). It was "
            f"saved from a different LoRA config or model. Missing e.g. "
            f"{sorted(missing)[:3]}")
    sd = model.state_dict()
    with torch.no_grad():
        for k, v in tensors.items():
            sd[k].copy_(v.to(sd[k].device, sd[k].dtype))
    del tensors
    fp = os.path.join(path, "lora_fingerprint.json")
    if os.path.isfile(fp):
        verify_fingerprint(model, fp)
    else:
        print(f"  WARNING: no fingerprint in {path}; load not verified")


def _env_versions():
    """Library/device context, recorded so a future mismatch is diagnosable.
    None of this is compared automatically — it exists to be read by a human."""
    import transformers
    try:
        import peft
        peft_v = peft.__version__
    except ImportError:
        peft_v = None
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft_v, "cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None)}


# ── Fingerprint (adapter identity check) ────────────────────────────────────

def lora_fingerprint(model):
    """A compact, init-sensitive signature of the LoRA adapter: per-tensor
    (numel, sum, sum-of-squares) in fp64, plus the ordered names. Two adapters
    with the same fingerprint are identical to within fp rounding."""
    params, names, numels = get_lora_param_list(model)
    stats = []
    for p in params:
        pf = p.detach().double()
        stats.append([float(p.numel()), float(pf.sum()), float((pf * pf).sum())])
    return {"names": names, "numels": numels, "stats": stats,
            "total_dim": int(sum(numels)),
            "lora_init": LORA_INIT, "lora_r": LORA_R, "seed": PISSA_SEED,
            # Not compared by verify_fingerprint — recorded so that a future
            # mismatch can be traced to a version/device change by eye.
            "env": _env_versions()}


def save_fingerprint(model, path):
    with open(path, "w") as f:
        json.dump(lora_fingerprint(model), f, indent=2)


def verify_fingerprint(model, path, rtol=1e-4, atol=1e-3):
    """Assert the model's current adapter matches the saved fingerprint. Raises
    RuntimeError on any structural mismatch or out-of-tolerance stat."""
    with open(path) as f:
        ref = json.load(f)
    cur = lora_fingerprint(model)
    if cur["names"] != ref["names"] or cur["numels"] != ref["numels"]:
        raise RuntimeError(
            "LoRA adapter structure differs from the one used to build the mean "
            f"gradient (fingerprint: {path}). Names/numels mismatch — the two "
            "scripts must use identical LoRA config/init/seed."
        )
    ref_stats = torch.tensor(ref["stats"], dtype=torch.float64)
    cur_stats = torch.tensor(cur["stats"], dtype=torch.float64)
    if not torch.allclose(cur_stats, ref_stats, rtol=rtol, atol=atol):
        raise RuntimeError(_mismatch_report(ref, cur, ref_stats, cur_stats, path))
    print(f"  fingerprint OK: adapter matches {path} (total_dim={cur['total_dim']:,})")


def _mismatch_report(ref, cur, ref_stats, cur_stats, path, rel_match=1e-5,
                     rel_drift=1e-2):
    """Explain a VALUES mismatch, and say whether the old artifacts survive it.

    The discriminator is which stat moved. Per tensor the fingerprint stores
    (numel, sum, sumsq), and PiSSA's SVD is unique only up to sign flips of
    paired singular vectors and rotations inside near-degenerate singular
    subspaces. Any such ambiguity is an orthogonal reparameterisation
    (A' = QA, B' = BQ^T, leaving B'A' = BA), under which
        sumsq = ||A||_F^2   is INVARIANT
        sum   = 1^T A 1     is NOT
    Gradients transform the same way (dL/dA' = Q dL/dA) and orthogonal maps
    preserve inner products and norms, so every products_*.pt / norms_*.pt /
    cosine stays numerically valid in either basis — as long as both sides of a
    dot product come from ONE adapter instance. Mixing bases (a saved
    mean_grad_*.pt against freshly computed token gradients) is what breaks.
    """
    rel_all = ((cur_stats - ref_stats).abs()
               / ref_stats.abs().clamp(min=1e-30))          # elementwise
    rel_sum, rel_sq = rel_all[:, 1], rel_all[:, 2]
    n = len(ref["names"])
    n_sq_bad = int((rel_sq >= rel_match).sum())
    lines = [
        "LoRA adapter VALUES differ from the fingerprint at " + path,
        f"  max abs stat diff {(cur_stats - ref_stats).abs().max().item():.3e}; "
        f"worst RELATIVE diff {float(rel_all.max()):.2e} (the rtol this check "
        f"would have needed)",
        f"  sumsq (basis-INVARIANT): {n - n_sq_bad}/{n} tensors match within "
        f"{rel_match:g}, max rel diff {rel_sq.max():.3e}",
        f"  sum   (basis-SENSITIVE): "
        f"{int((rel_sum < rel_match).sum())}/{n} match, max rel diff {rel_sum.max():.3e}",
    ]
    if n_sq_bad == 0 and int((rel_sum >= rel_match).sum()) == 0:
        lines += [
            f"  => EVERY stat matches within {rel_match:g}; only this check's own",
            "     rtol/atol failed. The adapters are the same to fp resolution —",
            "     nothing is wrong with the artifacts.",
        ]
    elif n_sq_bad == 0:
        lines += [
            "  => SAME ADAPTER, DIFFERENT SVD BASIS (sign flips / rotations in",
            "     degenerate subspaces). Frobenius norms all match. Cosines and",
            "     dot products computed WITHIN one basis remain valid, so saved",
            "     products_*.pt are still usable — but never pair a saved vector",
            "     from one run with gradients from another.",
        ]
    elif float(rel_sq.max()) < rel_drift:
        # sumsq is basis-INVARIANT, so a deviation this small cannot be "different
        # weights": it is the SVD picking a slightly different rank-r subspace
        # where singular values are nearly tied, plus bf16 rounding. Distinguished
        # from the branch below because the practical consequences are opposite.
        lines += [
            f"  => DIFFERENT SVD BASIS + NUMERICAL DRIFT. Frobenius norms agree to",
            f"     {rel_sq.max():.1e} relative (they are basis-invariant, so a",
            "      deviation this small is not a weight change); only the",
            "      basis-sensitive sums moved. Typical cause: a torch/peft version",
            "      change or different device placement re-running the SVD.",
            "  =>  Artifacts computed ENTIRELY within one run stay valid. Whether",
            "      values from the two runs are comparable is an empirical",
            "      question at the ~this-relative-order level — recompute one saved",
            "      quantity here and diff it (compute_cluster_products.py",
            "      --validate does exactly this) rather than assuming either way.",
        ]
    else:
        lines += [
            f"  => {n_sq_bad} tensors differ in Frobenius norm by up to "
            f"{rel_sq.max():.1e}:",
            "     these are genuinely different weights, not a reparameterisation.",
            "     Nothing computed before is comparable to anything computed now.",
        ]
    ref_env, cur_env = ref.get("env"), cur.get("env")
    if ref_env:
        diffs = [f"{k}: recorded={ref_env.get(k)!r} live={cur_env.get(k)!r}"
                 for k in cur_env if ref_env.get(k) != cur_env.get(k)]
        lines.append("  environment drift: " + ("; ".join(diffs) if diffs else "none"))
    else:
        lines += ["  (this fingerprint predates env recording — compare torch/",
                  "   transformers/peft versions and GPU count by hand)"]
    lines += [
        "  FIX: build the adapter once, call save_adapter(model, dir), and set",
        f"  PISSA_ADAPTER_DIR in {__name__} so no pass re-derives the SVD.",
    ]
    return "\n".join(lines)


# ── Chat tokenization (shared with the legacy script's split logic) ─────────

def normalize_message(msg):
    # Strip extra fields and keep only what the Olmo-3 chat template consumes.
    return {"role": msg["role"], "content": msg["content"] or ""}


def build_chat_ids(tokenizer, messages):
    """Tokenize a full chat and return (input_ids[1,L], comp_start, completion_ids).

    Identical split logic to the legacy compute_dpo_gradient_products.py: the
    prompt ends right after the final `<|im_start|>assistant` header; the
    completion runs from there to the end of the assistant turn. Returns
    (None, None, None) if the prompt alone exceeds MAX_SEQ_LENGTH.
    """
    norm_messages = [normalize_message(m) for m in messages]
    full_text = tokenizer.apply_chat_template(
        norm_messages, tokenize=False, add_generation_prompt=False,
    )
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    header_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    L = len(header_ids)
    last_idx = -1
    for i in range(len(full_ids) - L, -1, -1):
        if full_ids[i:i + L] == header_ids:
            last_idx = i
            break
    if last_idx == -1:
        warnings.warn("Could not locate '<|im_start|>assistant' header in tokenized chat")
        return None, None, None

    comp_start = last_idx + L
    completion_ids = full_ids[comp_start:]
    total_len = comp_start + len(completion_ids)
    if total_len > MAX_SEQ_LENGTH:
        max_comp = MAX_SEQ_LENGTH - comp_start
        if max_comp <= 0:
            warnings.warn(f"Prompt too long: {full_text[:100]}...")
            return None, None, None
        completion_ids = completion_ids[:max_comp]

    input_ids = full_ids[:comp_start] + completion_ids
    return torch.tensor([input_ids]), comp_start, completion_ids
