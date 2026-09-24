"""
Like compute_dpo_gradient_products.py, but (a) the LoRA is PiSSA-derived from
the pretrained weights (NOT a trained checkpoint — see pissa_lora_common.py),
and (b) in addition to the per-example chosen gradient, each rejected token is
also compared against the AGGREGATED (dataset-mean) chosen gradient produced by
compute_mean_chosen_gradient.py.

Three weightings are used everywhere — unweighted, norm-1 (weight 1/(2(1-p)),
i.e. inverse ||g_token||_1) and norm-2 (weight 1/sqrt(1 - 2p + sum_j p_j^2),
i.e. inverse ||g_token||_2). See compute_mean_chosen_gradient.py for the math.

For each example, per rejected token t with gradient g_t:
  SINGLE per-example chosen gradient g_unweighted / g_weighted_norm1 / g_weighted_norm2:
    - products_single_unweighted[t]         = g_t · g_unweighted
    - products_single_weighted_norm1[t]     = g_t · g_weighted_norm1
    - products_single_weighted_norm2[t]     = g_t · g_weighted_norm2
    - norms_rejected[t]                     = ||g_t||
    - norm_unweighted / norm_weighted_norm1 / norm_weighted_norm2   (scalars)
  AGGREGATED mean chosen gradient ḡ_unweighted / ḡ_weighted_norm1 / ḡ_weighted_norm2:
    - products_aggregated_unweighted[t]     = g_t · ḡ_unweighted
    - products_aggregated_weighted_norm1[t] = g_t · ḡ_weighted_norm1
    - products_aggregated_weighted_norm2[t] = g_t · ḡ_weighted_norm2
    - norm_mean_unweighted / norm_mean_weighted_norm1 / norm_mean_weighted_norm2
      (scalars, global; from meta)

Cosine vs the aggregated chosen direction is then e.g.
    products_aggregated_weighted_norm1 / (norms_rejected * norm_mean_weighted_norm1).

Run compute_mean_chosen_gradient.py FIRST. This script verifies, via a saved
fingerprint, that its freshly-built PiSSA adapter is identical to the one the
mean gradient was computed on, and refuses to run otherwise.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
load_dotenv(".env")
from datasets import load_dataset
from tqdm import tqdm

import pissa_lora_common as C

# ── Config ──────────────────────────────────────────────────────────────
SAVE_DIR = "/data/weighted-dpo/dpo-gradient-products-pissa-lora"
NUM_EXAMPLES = None


# ── Helpers ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer():
    """PiSSA-LoRA model (eager attention — required for the batched backward)."""
    return C.build_model_with_pissa_lora(attn_implementation="eager")


def collect_lora_grads(model, to_cpu=False) -> torch.Tensor:
    """Flatten all LoRA parameter gradients into one fp32 vector, in
    C.get_lora_param_list order (so it aligns with the batched rejected grads
    and the loaded mean-gradient vectors)."""
    grads = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad and param.grad is not None:
            grads.append(param.grad.detach().float().flatten())
    grad_vec = torch.cat(grads)
    nan_count = torch.isnan(grad_vec).sum().item()
    inf_count = torch.isinf(grad_vec).sum().item()
    if nan_count > 0 or inf_count > 0:
        print(f"    WARNING: {nan_count} NaN, {inf_count} Inf in gradient — replacing with 0")
        grad_vec = torch.nan_to_num(grad_vec, nan=0.0, posinf=0.0, neginf=0.0)
    if to_cpu:
        grad_vec = grad_vec.cpu()
    return grad_vec


def _flatten_batched_grads(grads, numels, batch, device):
    """Flatten per-param batched grads (each [batch, *shape] or None) into one
    [batch, D] fp32 matrix, in C.get_lora_param_list order. NaN/Inf -> 0."""
    cols = []
    for g, n in zip(grads, numels):
        if g is None:
            cols.append(torch.zeros(batch, n, device=device))
        else:
            cols.append(g.reshape(batch, -1).float())
    flat = torch.cat(cols, dim=1)
    return torch.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)


def build_completion_losses(outputs, comp_start, comp_token_ids, device):
    """Vectorized per-token NLL vector L of shape [T], L[i] = -log p(token_i),
    plus the per-token collision probability coll[i] = sum_j p_j^2 (used for the
    norm-2 weight). Returns (nll, comp_ids, coll)."""
    comp_ids = torch.as_tensor(comp_token_ids, device=device)
    T = comp_ids.shape[0]
    pos = comp_start + torch.arange(T, device=device) - 1
    sel_logits = outputs.logits[0, pos].float()
    log_probs = F.log_softmax(sel_logits, dim=-1)
    nll = -log_probs[torch.arange(T, device=device), comp_ids]
    coll = torch.exp(2.0 * log_probs).sum(dim=-1)            # sum_j p_j^2, [T]
    return nll, comp_ids, coll


REQUIRED_TENSOR_FILES = (
    "products_single_unweighted.pt",
    "products_single_weighted_norm1.pt",
    "products_single_weighted_norm2.pt",
    "products_aggregated_unweighted.pt",
    "products_aggregated_weighted_norm1.pt",
    "products_aggregated_weighted_norm2.pt",
    "norms_rejected.pt",
    "norm_unweighted.pt",
    "norm_weighted_norm1.pt",
    "norm_weighted_norm2.pt",
)


def is_example_complete(example_dir: str) -> bool:
    if not os.path.isdir(example_dir):
        return False
    metadata_path = os.path.join(example_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        return False
    try:
        with open(metadata_path, "r") as f:
            json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    for fname in REQUIRED_TENSOR_FILES:
        fpath = os.path.join(example_dir, fname)
        if not os.path.isfile(fpath):
            return False
        try:
            obj = torch.load(fpath, map_location="cpu", weights_only=True)
        except Exception:
            return False
        del obj
    return True


# ── Chosen gradient computation (per-example) ───────────────────────────

def token_weights(nll, coll):
    """Detached per-token norm-1 and norm-2 weights from the per-token NLL and
    collision probability coll = sum_j p_j^2:
      - norm-1: 1/(2(1 - p))                    [inverse ||g_token||_1]
      - norm-2: 1/sqrt(1 - 2p + sum_j p_j^2)    [inverse ||g_token||_2]
    """
    with torch.no_grad():
        p = torch.exp(-nll).clamp(max=1.0 - 1e-6)
        w_norm1 = 1.0 / (2.0 * (1.0 - p))
        g2 = (1.0 - 2.0 * p + coll).clamp(min=1e-12)        # ||g_token||_2^2
        w_norm2 = 1.0 / torch.sqrt(g2)
    return w_norm1, w_norm2


def compute_chosen_gradients(model, tokenizer, messages):
    """Per-example chosen gradient under three weightings (unweighted, norm-1,
    norm-2), each a GPU fp32 vector. Returns (g_unweighted, g_w1, g_w2) or
    (None, None, None) if the prompt is unusable."""
    input_ids, comp_start, comp_token_ids = C.build_chat_ids(tokenizer, messages)
    if input_ids is None:
        return None, None, None
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    model.zero_grad()
    outputs = model(input_ids=input_ids)
    nll, _, coll = build_completion_losses(outputs, comp_start, comp_token_ids, device)
    w_norm1, w_norm2 = token_weights(nll, coll)

    unweighted_loss = nll.sum()
    w1_loss = (nll * w_norm1).sum()
    w2_loss = (nll * w_norm2).sum()

    w1_loss.backward(retain_graph=True)
    g_w1 = collect_lora_grads(model)
    model.zero_grad()
    w2_loss.backward(retain_graph=True)
    g_w2 = collect_lora_grads(model)
    model.zero_grad()
    unweighted_loss.backward()
    g_unweighted = collect_lora_grads(model)

    del outputs, unweighted_loss, w1_loss, w2_loss, nll, coll, w_norm1, w_norm2
    torch.cuda.empty_cache()
    return g_unweighted, g_w1, g_w2


# ── Rejected per-token products and norms ───────────────────────────────

def compute_rejected_products_and_norms(model, tokenizer, messages, directions):
    """Per-token dot products against every direction in `directions` (a dict
    name -> GPU fp32 vector of dim D) plus the per-token gradient norm.

    One forward pass, then one plain torch.autograd.grad backward per rejected
    token (retain_graph reuses the shared forward graph). NO is_grads_batched:
    vmapping the backward (b>1) silently corrupts gradients on this Olmo-3 model
    — its eager-attention backward has a batching rule that mixes gradients
    across the batched tokens (proven by diagnose_grad_batch.py: b=1 is bit-exact
    vs this loop, b=8 diverges). Each token is therefore computed independently.

    Returns (products: dict name -> Tensor[T], norms_rejected: Tensor[T], T, D)
    or (None, None, None, None) if skipped.
    """
    input_ids, comp_start, comp_token_ids = C.build_chat_ids(tokenizer, messages)
    if input_ids is None:
        return None, None, None, None
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    T = len(comp_token_ids)
    products = {name: torch.zeros(T) for name in directions}
    norms = torch.zeros(T)

    lora_params, _, numels = C.get_lora_param_list(model)
    D = sum(numels)

    model.zero_grad()
    outputs = model(input_ids=input_ids)
    nll, _, _ = build_completion_losses(outputs, comp_start, comp_token_ids, device)

    for t in range(T):
        grads = torch.autograd.grad(
            nll[t], lora_params, retain_graph=True, allow_unused=True)
        flat = _flatten_batched_grads(
            [g.unsqueeze(0) if g is not None else None for g in grads],
            numels, 1, device)                      # [1, D]
        for name, vec in directions.items():
            products[name][t] = (flat @ vec).item()
        norms[t] = flat.norm().item()
        del grads, flat

    del outputs, nll
    torch.cuda.empty_cache()
    return products, norms, T, D


def load_mean_gradients(device, expected_dim):
    """Load ḡ_unweighted / ḡ_weighted_norm1 / ḡ_weighted_norm2 (fp32, GPU).
    Asserts each dimension matches the freshly-built adapter. Returns a dict
    name -> GPU vector."""
    files = {
        "unweighted": "mean_grad_unweighted.pt",
        "weighted_norm1": "mean_grad_weighted_norm1.pt",
        "weighted_norm2": "mean_grad_weighted_norm2.pt",
    }
    means = {}
    for name, fname in files.items():
        path = os.path.join(C.MEAN_CHOSEN_DIR, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Missing {path}. Run compute_mean_chosen_gradient.py first.")
        v = torch.load(path, map_location="cpu", weights_only=True).float()
        if v.numel() != expected_dim:
            raise RuntimeError(
                f"mean_grad_{name} dim {v.numel()} != adapter grad dim "
                f"{expected_dim} — the mean gradient was built on a different "
                "adapter. Rebuild it with the current pissa_lora_common config.")
        means[name] = v.to(device)
    return means


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, required=True, help="First example index (inclusive)")
    parser.add_argument("--end_idx", type=int, required=True, help="Last example index (exclusive)")
    args = parser.parse_args()

    print(f"Loading model {C.FULL_MODEL_CHECKPOINT} with PiSSA LoRA (eager) ...")
    model, tokenizer = load_model_and_tokenizer()

    # Guarantee this adapter is the same one the mean gradient was built on.
    fp_path = os.path.join(C.MEAN_CHOSEN_DIR, "lora_fingerprint.json")
    if not os.path.isfile(fp_path):
        raise FileNotFoundError(
            f"Missing {fp_path}. Run compute_mean_chosen_gradient.py first.")
    C.verify_fingerprint(model, fp_path)

    device = next(model.parameters()).device
    _, _, numels = C.get_lora_param_list(model)
    grad_dim = sum(numels)
    means = load_mean_gradients(device, grad_dim)
    norm_mean = {name: float(v.norm()) for name, v in means.items()}
    print(f"  ||mean_grad_unweighted||     = {norm_mean['unweighted']:.6f}")
    print(f"  ||mean_grad_weighted_norm1|| = {norm_mean['weighted_norm1']:.6f}")
    print(f"  ||mean_grad_weighted_norm2|| = {norm_mean['weighted_norm2']:.6f}")

    print(f"Loading dataset {C.DATASET_NAME} ...")
    dataset = load_dataset(C.DATASET_NAME, split="train")
    if NUM_EXAMPLES is not None:
        dataset = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))

    start_idx = max(0, args.start_idx)
    end_idx = min(args.end_idx, len(dataset))
    print(f"  Processing examples [{start_idx}, {end_idx}) out of {len(dataset)}\n")

    os.makedirs(SAVE_DIR, exist_ok=True)
    # Record the global mean-chosen norms once for downstream cosine math.
    with open(os.path.join(SAVE_DIR, "mean_chosen_norms.json"), "w") as f:
        json.dump({"norm_mean_unweighted": norm_mean["unweighted"],
                   "norm_mean_weighted_norm1": norm_mean["weighted_norm1"],
                   "norm_mean_weighted_norm2": norm_mean["weighted_norm2"],
                   "grad_dim": grad_dim,
                   "mean_chosen_dir": C.MEAN_CHOSEN_DIR}, f, indent=2)

    for idx in tqdm(range(start_idx, end_idx), desc="Processing examples"):
        example_dir = os.path.join(SAVE_DIR, f"example_{idx:05d}")
        if is_example_complete(example_dir):
            print(f"=== Example {idx}/{len(dataset)} === already complete — skipping")
            continue

        example = dataset[idx]
        chosen_messages = [C.normalize_message(m) for m in example["chosen"]]
        rejected_messages = [C.normalize_message(m) for m in example["rejected"]]

        print(f"=== Example {idx}/{len(dataset)} ===")
        print(f"  Chosen   ({len(chosen_messages)} msgs): {chosen_messages[-1]['content'][:200]}")
        print(f"  Rejected ({len(rejected_messages)} msgs): {rejected_messages[-1]['content'][:200]}")

        print("  Computing chosen gradients...")
        g_unweighted, g_w1, g_w2 = compute_chosen_gradients(model, tokenizer, chosen_messages)
        if g_unweighted is None:
            print("  SKIPPED (prompt too long)")
            continue

        directions = {
            "single_unweighted": g_unweighted,
            "single_weighted_norm1": g_w1,
            "single_weighted_norm2": g_w2,
            "aggregated_unweighted": means["unweighted"],
            "aggregated_weighted_norm1": means["weighted_norm1"],
            "aggregated_weighted_norm2": means["weighted_norm2"],
        }

        print("  Computing rejected per-token gradient products...")
        products, norms_rejected, T, D = compute_rejected_products_and_norms(
            model, tokenizer, rejected_messages, directions)
        if products is None:
            print("  SKIPPED (prompt too long)")
            continue

        norm_chosen = {
            "unweighted": g_unweighted.norm(),
            "weighted_norm1": g_w1.norm(),
            "weighted_norm2": g_w2.norm(),
        }

        print(f"  Rejected tokens: {T}, grad dim: {D}")
        print(f"  ||g_unweighted||={norm_chosen['unweighted'].item():.4f}  "
              f"||g_weighted_norm1||={norm_chosen['weighted_norm1'].item():.4f}  "
              f"||g_weighted_norm2||={norm_chosen['weighted_norm2'].item():.4f}")
        print(f"  ||X|| row norms: min={norms_rejected.min().item():.4f}, max={norms_rejected.max().item():.4f}")

        os.makedirs(example_dir, exist_ok=True)
        for name, vec in products.items():
            torch.save(vec, os.path.join(example_dir, f"products_{name}.pt"))
        torch.save(norms_rejected, os.path.join(example_dir, "norms_rejected.pt"))
        for name, val in norm_chosen.items():
            torch.save(val.cpu(), os.path.join(example_dir, f"norm_{name}.pt"))

        metadata = {
            "chosen": chosen_messages,
            "rejected": rejected_messages,
            "num_rejected_tokens": T,
            "grad_dim": D,
            "norm_unweighted": norm_chosen["unweighted"].item(),
            "norm_weighted_norm1": norm_chosen["weighted_norm1"].item(),
            "norm_weighted_norm2": norm_chosen["weighted_norm2"].item(),
            "norm_mean_unweighted": norm_mean["unweighted"],
            "norm_mean_weighted_norm1": norm_mean["weighted_norm1"],
            "norm_mean_weighted_norm2": norm_mean["weighted_norm2"],
        }
        with open(os.path.join(example_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"  Saved to {example_dir}\n")
        del g_unweighted, g_w1, g_w2, products, norms_rejected
        torch.cuda.empty_cache()

    print(f"\nDone. All results saved to {SAVE_DIR}/")


if __name__ == "__main__":
    main()
