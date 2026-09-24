"""
Compute the AGGREGATED (mean over the dataset) chosen gradient for a
PiSSA-derived LoRA on allenai/Olmo-3-7B-Instruct-SFT, over the chosen
completions of anonymous/Dolci-Instruct-DPO-en.

This is the reference direction the rejected-token gradients are later dotted
against by compute_dpo_gradient_products_meanchosen.py. The adapter is derived
from the pretrained weights with PiSSA (see pissa_lora_common.py) — NOT a
trained checkpoint — so both A and B receive gradients.

Three means are produced (all averaged over examples; per-example loss is the
SUM of completion-token terms, matching the per-example chosen gradient in the
products script):
  - mean_grad_unweighted     = mean_e  grad[ sum_i -log p(t_i) ]
  - mean_grad_weighted_norm1 = mean_e  grad[ sum_i -log p(t_i) / (1 - p(t_i)) ]
  - mean_grad_weighted_norm2 = mean_e  grad[ sum_i -log p(t_i) / ||g_i||_2 ]

The two weighted variants normalize each completion token's contribution by the
norm of its per-token logit gradient g_i = softmax(z_i) - onehot(t_i):
  - norm-1: ||g_i||_1 = 2(1 - p(t_i))               -> weight 1/(2(1 - p))
  - norm-2: ||g_i||_2 = sqrt(1 - 2 p + sum_j p_j^2)  -> weight 1/||g_i||_2
where sum_j p_j^2 (the collision probability) is computed stably as
exp(logsumexp(2 z) - 2 logsumexp(z)). All weights are detached.

For speed this runs on real BATCHES (batch size > 1, left-padded), one forward
+ three backward (norm-1 weighted, norm-2 weighted, unweighted) per batch. A
resume checkpoint is written every --checkpoint_every batches so a long run
survives interruption.

Outputs (in pissa_lora_common.MEAN_CHOSEN_DIR):
  - mean_grad_weighted_norm1.pt  fp32 [D]
  - mean_grad_weighted_norm2.pt  fp32 [D]
  - mean_grad_unweighted.pt      fp32 [D]
  - lora_fingerprint.json      adapter identity (verified by the products script)
  - meta.json                  num_examples, D, norms, param layout, config
  - _resume.pt                 running accumulators (removed on success)
"""

import json
import os

import torch
from dotenv import load_dotenv
load_dotenv(".env")
from datasets import load_dataset
from tqdm import tqdm

import pissa_lora_common as C

NUM_EXAMPLES = None       # None -> whole train split; int to limit
BATCH_SIZE = 4
CHECKPOINT_EVERY = 50     # batches between resume checkpoints
RESUME = True             # False -> ignore any existing resume checkpoint, start over


# ── Batch construction ──────────────────────────────────────────────────────

def build_example(tokenizer, messages):
    """(input_ids[L], labels[L]) for one chosen chat, or None if unusable.
    labels = -100 on prompt tokens, real id on completion tokens (so the
    supervised span is exactly build_chat_ids' completion — identical to the
    per-example chosen gradient in the products script)."""
    input_ids, comp_start, comp_ids = C.build_chat_ids(tokenizer, messages)
    if input_ids is None:
        return None
    ids = input_ids[0].tolist()
    labels = [-100] * comp_start + ids[comp_start:]
    return ids, labels


def collate(batch, pad_id, device):
    """Left-pad a list of (ids, labels) into model kwargs + labels tensor."""
    maxL = max(len(ids) for ids, _ in batch)
    input_ids, attn, labels = [], [], []
    for ids, labs in batch:
        pad = maxL - len(ids)
        input_ids.append([pad_id] * pad + ids)
        attn.append([0] * pad + [1] * len(ids))
        labels.append([-100] * pad + labs)
    to = lambda x: torch.tensor(x, dtype=torch.long, device=device)
    return {"input_ids": to(input_ids), "attention_mask": to(attn)}, to(labels)


def batch_losses(model, kwargs, labels):
    """Return (w_norm1_loss, w_norm2_loss, unweighted_loss, n_examples).

    All three losses SUM completion-token terms over the whole batch, so a
    single backward gives the sum of per-example gradients (divide by example
    count at the end to get the mean). Both per-token weights are detached:
      - norm-1: 1/(2(1 - p))                      [inverse ||g||_1]
      - norm-2: 1/sqrt(1 - 2p + sum_j p_j^2)      [inverse ||g||_2]
    """
    out = model(**kwargs)
    shift_logits = out.logits[:, :-1, :]          # predicts next token
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe = shift_labels.clamp(min=0)

    # NLL without materialising a full log-softmax tensor.
    lse = torch.logsumexp(shift_logits.float(), dim=-1)              # [B, S-1]
    tgt = shift_logits.float().gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    nll = (lse - tgt) * mask                                         # [B, S-1]

    unweighted_loss = nll.sum()
    with torch.no_grad():
        p = torch.exp(-nll).clamp(max=1.0 - 1e-6)
        # collision prob sum_j p_j^2, stable via logsumexp of doubled logits.
        lse2 = torch.logsumexp(2.0 * shift_logits.float(), dim=-1)   # [B, S-1]
        coll = torch.exp(lse2 - 2.0 * lse)                           # [B, S-1]
        w_norm1 = (1.0 / (2.0 * (1.0 - p))) * mask
        g2 = (1.0 - 2.0 * p + coll).clamp(min=1e-12)                 # ||g||_2^2
        w_norm2 = (1.0 / torch.sqrt(g2)) * mask
    w_norm1_loss = (nll * w_norm1).sum()
    w_norm2_loss = (nll * w_norm2).sum()
    return w_norm1_loss, w_norm2_loss, unweighted_loss, kwargs["input_ids"].shape[0]


# ── Accumulation ─────────────────────────────────────────────────────────────

def add_grads_(accum, grads):
    """accum[i] += grads[i] (flattened to match the 1-D accumulator; None -> 0).

    autograd.grad returns each grad with the param's shape, but accum[i] is a
    flat [numel] buffer, so flatten before adding."""
    for i, g in enumerate(grads):
        if g is not None:
            accum[i] += g.detach().float().reshape(-1)


def flatten(accum):
    vec = torch.cat([a.reshape(-1) for a in accum]).float()
    return torch.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def _next_backup(path):
    """First free '<path>.bakN' — numbered rather than timestamped so repeated
    runs stay ordered and nothing is ever overwritten."""
    stem, ext = os.path.splitext(path)
    n = 1
    while os.path.exists(f"{stem}.bak{n}{ext}"):
        n += 1
    return f"{stem}.bak{n}{ext}"


def main():
    os.makedirs(C.MEAN_CHOSEN_DIR, exist_ok=True)
    fp_path = os.path.join(C.MEAN_CHOSEN_DIR, "lora_fingerprint.json")
    resume_path = os.path.join(C.MEAN_CHOSEN_DIR, "_resume.pt")

    print(f"Loading model {C.FULL_MODEL_CHECKPOINT} with PiSSA LoRA (sdpa) ...")
    # sdpa is fine here (fast) — no batched backward; gradients are in the same
    # parameter space as the products script regardless of attention impl.
    model, tokenizer = C.build_model_with_pissa_lora(attn_implementation="sdpa")
    # This file is the identity of the adapter every products_*.pt in the repo was
    # computed against, and the PiSSA SVD is only reproducible within one
    # environment. Overwriting it from a container that derives a different basis
    # would destroy the only remaining record of the old one -- and that record is
    # what makes a later mismatch diagnosable at all. So preserve before writing.
    if os.path.isfile(fp_path):
        backup = _next_backup(fp_path)
        os.replace(fp_path, backup)
        print(f"  NOTE: preserved the previous adapter fingerprint -> {backup}\n"
              f"        (it identifies the adapter the existing products were "
              f"built on; this run is about to replace it)")
    C.save_fingerprint(model, fp_path)
    print(f"  saved adapter fingerprint -> {fp_path}")

    lora_params, names, numels = C.get_lora_param_list(model)
    device = next(model.parameters()).device

    print(f"Loading dataset {C.DATASET_NAME} ...")
    dataset = load_dataset(C.DATASET_NAME, split="train")
    if NUM_EXAMPLES is not None:
        dataset = dataset.select(range(min(NUM_EXAMPLES, len(dataset))))
    N = len(dataset)

    # Accumulators (fp32, on GPU) + bookkeeping.
    acc_w1 = [torch.zeros(n, device=device) for n in numels]
    acc_w2 = [torch.zeros(n, device=device) for n in numels]
    acc_uw = [torch.zeros(n, device=device) for n in numels]
    n_examples = 0
    start_idx = 0

    if RESUME and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        if ckpt.get("names") == names and ckpt.get("numels") == numels:
            acc_w1 = [t.to(device) for t in ckpt["acc_w1"]]
            acc_w2 = [t.to(device) for t in ckpt["acc_w2"]]
            acc_uw = [t.to(device) for t in ckpt["acc_uw"]]
            n_examples = ckpt["n_examples"]
            start_idx = ckpt["next_idx"]
            print(f"  resumed from {resume_path}: {n_examples} examples done, "
                  f"continuing at index {start_idx}")
        else:
            print("  resume checkpoint has a different adapter layout — starting over")

    def save_resume(next_idx):
        tmp = resume_path + ".tmp"
        torch.save({"acc_w1": [a.cpu() for a in acc_w1],
                    "acc_w2": [a.cpu() for a in acc_w2],
                    "acc_uw": [a.cpu() for a in acc_uw],
                    "n_examples": n_examples, "next_idx": next_idx,
                    "names": names, "numels": numels}, tmp)
        os.replace(tmp, resume_path)

    bs = BATCH_SIZE
    pad_id = tokenizer.pad_token_id
    skipped = 0
    pending, pending_end = [], start_idx

    pbar = tqdm(range(start_idx, N), desc="Mean chosen gradient")
    batches_done = 0
    for idx in pbar:
        ex = dataset[idx]
        built = build_example(tokenizer, ex["chosen"])
        if built is not None:
            pending.append(built)
        else:
            skipped += 1
        pending_end = idx + 1

        if len(pending) >= bs or (idx == N - 1 and pending):
            kwargs, labels = collate(pending, pad_id, device)
            w1_loss, w2_loss, unweighted_loss, nb = batch_losses(model, kwargs, labels)

            grads_w1 = torch.autograd.grad(
                w1_loss, lora_params, retain_graph=True, allow_unused=True)
            add_grads_(acc_w1, grads_w1)
            grads_w2 = torch.autograd.grad(
                w2_loss, lora_params, retain_graph=True, allow_unused=True)
            add_grads_(acc_w2, grads_w2)
            grads_uw = torch.autograd.grad(
                unweighted_loss, lora_params, allow_unused=True)
            add_grads_(acc_uw, grads_uw)

            n_examples += nb
            pending = []
            batches_done += 1
            del kwargs, labels, w1_loss, w2_loss, unweighted_loss
            del grads_w1, grads_w2, grads_uw
            torch.cuda.empty_cache()

            if batches_done % CHECKPOINT_EVERY == 0:
                save_resume(pending_end)
            pbar.set_postfix(examples=n_examples, skipped=skipped)

    if n_examples == 0:
        raise RuntimeError("No usable chosen examples — nothing to average.")

    mean_w1 = flatten(acc_w1) / n_examples
    mean_w2 = flatten(acc_w2) / n_examples
    mean_uw = flatten(acc_uw) / n_examples

    torch.save(mean_w1.cpu(), os.path.join(C.MEAN_CHOSEN_DIR, "mean_grad_weighted_norm1.pt"))
    torch.save(mean_w2.cpu(), os.path.join(C.MEAN_CHOSEN_DIR, "mean_grad_weighted_norm2.pt"))
    torch.save(mean_uw.cpu(), os.path.join(C.MEAN_CHOSEN_DIR, "mean_grad_unweighted.pt"))

    meta = {
        "dataset": C.DATASET_NAME,
        "model": C.FULL_MODEL_CHECKPOINT,
        "n_examples_averaged": n_examples,
        "n_skipped": skipped,
        "grad_dim": int(mean_w1.numel()),
        "norm_mean_weighted_norm1": float(mean_w1.norm()),
        "norm_mean_weighted_norm2": float(mean_w2.norm()),
        "norm_mean_unweighted": float(mean_uw.norm()),
        "lora_init": C.LORA_INIT, "lora_r": C.LORA_R, "lora_alpha": C.LORA_ALPHA,
        "lora_target_modules": C.LORA_TARGET_MODULES, "pissa_seed": C.PISSA_SEED,
        "param_names": names, "param_numels": numels,
        "batch_size": bs,
    }
    with open(os.path.join(C.MEAN_CHOSEN_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    if os.path.exists(resume_path):
        os.remove(resume_path)

    print(f"\nDone. Averaged {n_examples} examples ({skipped} skipped).")
    print(f"  ||mean_grad_weighted_norm1|| = {meta['norm_mean_weighted_norm1']:.6f}")
    print(f"  ||mean_grad_weighted_norm2|| = {meta['norm_mean_weighted_norm2']:.6f}")
    print(f"  ||mean_grad_unweighted||     = {meta['norm_mean_unweighted']:.6f}")
    print(f"  saved to {C.MEAN_CHOSEN_DIR}/")


if __name__ == "__main__":
    main()
