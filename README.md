# Weighted DPO — Gradient-Aligned Per-Token Reweighting of the DPO Loss

Reference implementation accompanying the paper submission, together with the preference-optimization
baselines it is compared against.

Standard DPO treats a preference pair `(y_w, y_l) | x` as two atomic sequences: every token in the
rejected response is pushed down with equal force, including the many tokens that are semantically
indistinguishable from their counterparts in the chosen response (formatting and control tokens,
closed-form factual tokens, function words, long shared prefixes). Weighted DPO assigns each rejected
token a weight derived from how its parameter-space gradient aligns with the update direction implied
by the chosen completion: tokens whose gradient *opposes* the chosen direction are penalized, tokens
that *agree* with it are suppressed.

This supplement contains the method, the gradient-alignment pipeline that produces its weights, and
the baseline trainers. The ground-truth token-labelling harness, the benchmark evaluation harness and
the plotting scripts are not included.

---

## 1. Method

Let `π_θ` be the policy and `g_i = ∇_θ nll_i(y)` the per-token gradient, collected over an untrained
low-rank probe (§3) rather than the full parameter set.

**Aggregate chosen direction.** For the chosen completion a single aggregate gradient summarizes what
the model would learn from the example:

```
g_w = Σ_i 1/(1 - p_i) · ∇_θ ( -log π_θ(y_w^(i) | x, y_w^(<i)) ),    p_i = π_θ(y_w^(i) | x, y_w^(<i))
```

The `1/(1-p_i)` factor up-weights tokens the model is already confident in but is still asked to push
higher. Unweighted and `norm2` variants are computed alongside it for ablations.

**Per-token rejected alignment.** For each rejected token `j` the pipeline stores its dot products
against the chosen directions and its norm — never the full `(T_l, D)` matrix:

```
p_j = ⟨g_j^(l), g_w⟩,    n_j = ‖g_j^(l)‖₂,    c_j = p_j / (n_j ‖g_w‖)
```

A *positive* `p_j` means penalizing token `j` would move parameters the same way learning the chosen
completion does — the token should not be penalized. A *negative* `p_j` marks the tokens the loss
should focus on.

**Weights.** `train_weighted_dpo.py` exposes three orthogonal knobs:

1. `WEIGHT_FORMULA` — `f(c, p, pn) → raw weight`, where `pn = p / max|p|` over the example. Cosine
   based: `token_scale` `(1−c)/2`, `exp_half`, `exp_m1`, `dist_sq`, `sigmoid_c` `σ(−c·k)`,
   `one_minus_c`, `raw_cos`. Dot based: `raw_dot`, `norm_dot`, `neg_dot`, `dot_scale` `(1−pn)/2`,
   `sigmoid_dot` `σ(−pn·k)`.
2. `WEIGHT_NORM` — per-sequence normalization: `none` / `softmax(·/T)` / `l1`.
3. `WEIGHT_SUM_MODE` — a trainer-side rescale applied inside the loss, where reference log-probs and
   post-truncation token counts are visible: `normalize`, `reward_match`, `token_count`,
   `token_scale`, `ignore`.

`WEIGHT_METHOD` names a `(formula, norm)` pair. `FORCE_CHOSEN_WEIGHT_ONE = True`: only rejected tokens
carry weighting.

**Loss.**

```
L = -log σ( β [ A_w Σ_i w_i^(w) log(π_θ/π_ref)(y_w^(i)) − A_l Σ_j w_j^(l) log(π_θ/π_ref)(y_l^(j)) ] )
```

`A_w`, `A_l` are per-sequence adjustments that rescale the weighted reward to match the magnitude of
the unweighted one. They are computed under `torch.no_grad()` and treated as detached constants: they
preserve the *scale* of the implicit reward without changing which tokens carry signal. When weights
are uniform (`WEIGHT_MODE = "uniform"`) the adjustment is exactly `1.0` and the loss reduces exactly to
standard DPO — a property worth keeping as a regression test when modifying the trainer.

---

## 2. Contents

```
── method ──
train_weighted_dpo.py                 Direction registry + weight registry + full fine-tune
weighted_dpo_trainer.py               WeightedDPOTrainer + weighted preference collator

── baselines ──
train_standard_dpo.py                 DPO           (TRL DPOTrainer)
train_standard_ipo.py                 IPO           (TRL DPOTrainer, loss_type="ipo")
train_standard_kto.py                 KTO           (TRL KTOTrainer)
train_standard_simpo.py               SimPO         + simpo_trainer.py
train_standard_tdpo.py                TDPO          + tidpo_trainer.py (TDPOTrainer)
train_tidpo.py                        TI-DPO        + tidpo_trainer.py (TIDPOTrainer)

── gradient pipeline (§3) ──
pissa_lora_common.py                  Probe definition, tokenization, fingerprinting
save_pissa_adapter.py                 Pin the probe once; verify it on every later pass
compute_mean_chosen_gradient.py       Dataset-mean chosen gradient ḡ, three weightings
compute_dpo_gradient_products_pissa_lora.py   Per-token ⟨g_t, ·⟩ and ‖g_t‖ dumps

── cluster branch (§4) ──
cluster_common.py                     Shared constants + max-over-clusters arithmetic
compute_chosen_gradient_embeddings.py Chosen gradients → CountSketch embeddings
cluster_chosen_gradients.py           Spherical k-means sweep over K → assignments
compute_cluster_chosen_gradients.py   Assignments → K centroids in full space
compute_cluster_products.py           JVP: ⟨g_t, c_k⟩ for all t, per k
inspect_gradient_clusters.py          Cluster diagnostics (η², distinctive terms, medoids)
```

Some comments reference scripts from the full research repository that are outside the scope of this
supplement (the ground-truth correlation harness, the visualizer, the plotting scripts). They are
retained as provenance for the constants they explain; nothing in the shipped code path imports them,
with one exception: `compute_cluster_products.py --mode subsample` resolves its paths from the
ground-truth harness config and therefore only `--mode full` runs here.

---

## 3. Gradient pipeline

Per-token gradients against a full 7B parameter set are intractable, so alignment is measured on an
**untrained PiSSA-derived LoRA probe** (`r=64`, `α=128`, `dropout=0`, `target_modules="all-linear"`,
`init="pissa"` → `D = 159,907,840` across 448 tensors). The probe is a measurement instrument, never a
training adapter: DPO training itself is full fine-tuning.

Engineering points that matter for reproducing the numbers:

- **Two-stage backward.** The chosen completion is forwarded once and back-propagated twice
  (`retain_graph=True`), once per aggregate, avoiding a duplicate forward.
- **Chunked per-token rejected backward.** Each rejected token needs its own backward call. Tokens are
  processed in chunks that share a forward pass and retain the graph only within the chunk, bounding
  peak memory to one chunk instead of the full sequence.
- **Dot products on GPU, scalars to CPU.** `g_j^(l)` lives on the GPU only long enough to produce its
  scalars.
- **NaN/Inf guard.** Gradients are sanitized (`nan_to_num`) before each dot product so a single
  pathological example cannot poison later statistics.

**The probe must be byte-identical across passes.** An SVD is unique only up to sign flips of paired
singular vectors and rotations inside near-degenerate singular subspaces, so re-deriving it drifts
across library versions and device placements. A drifted run is a *different parameterization of the
same update* — enough to silently invalidate any dot product mixing vectors from two runs. Hence
`save_pissa_adapter.py` writes the exact tensors (LoRA A/B **and** the PiSSA-modified base weights)
once, and every later pass loads them and verifies a fingerprint. Do not skip this step.

Outputs land in `example_NNNNN/` directories: `products_{weighted,unweighted}.pt` `(T_l,)`,
`norms_rejected.pt` `(T_l,)`, the scalar norms, and `metadata.json`.

## 4. Cluster branch

Replaces the single dataset-mean chosen direction with K per-cluster means, so each token follows
whichever reference direction it aligns with most strongly. Centroids are built as `mean` (raw
per-cluster mean) or `meannorm` (mean of unit-norm gradients).

Because the training score is an elementwise max over *dot products*, and a dot scales with the norm of
the direction, raw centroids make the max degenerate into "whichever centroid is longest". The
partition is strongly norm-correlated, so `CLUSTER_DOT_SCALE` rescales every centroid onto a common
norm first. A dot is linear in the direction, so this needs no recompute. Directions are named
`maxclu_<kind>_K<k>` / `clusteronly_<kind>_K<k>` and parsed rather than looked up, so several K coexist
in one products directory.

---

## 5. Setup

### Configuration

Paths are module-level constants at the top of each script. Set them before running:

| Constant | File | What it holds |
|---|---|---|
| `PISSA_ADAPTER_DIR` | `pissa_lora_common.py` | The pinned probe (~13 GB) |
| `MEAN_CHOSEN_DIR` | `pissa_lora_common.py` | Mean chosen gradients + fingerprint |
| `SAVE_DIR` | `compute_dpo_gradient_products_pissa_lora.py` | Per-token product dumps |
| `GRADIENT_PRODUCTS_DIR` | `train_weighted_dpo.py` | Must match the `SAVE_DIR` above |
| `CLUSTER_DIR`, `K` | `cluster_common.py` | Cluster run to read; K is the sweep knob |
| `EMB_DIR` | `compute_chosen_gradient_embeddings.py` | Gradient sketches |
| `OUTPUT_DIR` | each training script | Checkpoints |

Defaults are placeholders under `/data/weighted-dpo` and `/checkpoints/weighted-dpo`.

`DATASET_NAME` is set to the placeholder `anonymous/Dolci-Instruct-DPO-en`: a 300K-example
English-only single-turn filter of the public `allenai/Dolci-Instruct-DPO`, keeping `(user, assistant)`
pairs of at most 8000 characters whose prompt is detected as English by FastText `lid.176`. Point this
constant at your own copy. The prompt boundary must sit immediately after the final
`<|im_start|>assistant` header so the completion begins with the canonical leading `\n` — this
alignment is what matches the per-token weight vector to the rejected token IDs at training time.

Credentials are read from a `.env` in the repository root (see `.env.example`).

### Environment

- `transformers >= 4.45` (the Olmo-3 `rope_parameters` schema requires float values; the scripts coerce
  ints to floats at load time)
- `trl` exposing `DPOTrainer`, `DPOConfig`, `DataCollatorForPreference`, `selective_log_softmax`
- `peft`, `datasets`, `accelerate`, `torch`, `wandb`, `bitsandbytes` (for `adamw_8bit`)
- `numpy`, `scipy`, `scikit-learn`, `tqdm`, `python-dotenv`

Multi-GPU (H100/A100 class) is assumed; the per-token gradient pass is the expensive stage and is
sharded by index range.

### Hyperparameters

Following the OLMo 3 paper (arXiv:2512.13961) Table 48, "7B Instruct DPO" column, with β scaled for
TRL's non-length-normalized sigmoid loss.

| Parameter | Value |
|---|---|
| Base model | `allenai/Olmo-3-7B-Instruct-SFT` (full fine-tune, bf16) |
| `max_seq_length` | 2048 |
| `learning_rate` | `5e-6` |
| `num_train_epochs` | 1 |
| `per_device_train_batch_size` / effective | 1 / 128 |
| `β` | 0.02 |
| `warmup_ratio` / scheduler / optim | 0.1 / linear / `adamw_8bit` |
| `GRADIENT_DIRECTION` | `max_weighted_norm1` |
| `WEIGHT_METHOD` / `WEIGHT_SUM_MODE` | `sigmoid_dot` / `token_scale` |
| `SIGMOID_SLOPE` / `WEIGHT_TEMPERATURE` | 1.0 / 0.1 |

Gradient accumulation is derived from `EFFECTIVE_BATCH_SIZE / (BATCH_SIZE × num_gpus)` and must divide
evenly. `OUTPUT_DIR` encodes direction, method, sum-mode, temperature, β and lr so runs cannot collide.
The tokenizer uses `pad_token = bos_token` (never `eos`) and `padding_side="left"`; no new tokens are
added. Keep shared hyperparameters in sync across the training scripts — they are the A/B pair.

---

## 6. Running

```bash
# 1. Pin the alignment probe — ONCE, and never from a different environment.
python save_pissa_adapter.py --check_only
python save_pissa_adapter.py --path /data/weighted-dpo/pissa-lora-adapter-r64
#    then set PISSA_ADAPTER_DIR in pissa_lora_common.py

# 2. Reference directions.
python compute_mean_chosen_gradient.py
python compute_dpo_gradient_products_pissa_lora.py --start_idx 0     --end_idx 26099
python compute_dpo_gradient_products_pissa_lora.py --start_idx 26099 --end_idx 52198
#    ... shard across GPUs; one backward per rejected token, this is the expensive pass

# 3. Cluster branch (optional, adds K modal directions).
python compute_chosen_gradient_embeddings.py
python cluster_chosen_gradients.py
#    set K / CENTROID_KINDS in cluster_common.py, then:
python compute_cluster_chosen_gradients.py
python compute_cluster_products.py --mode full --start_idx 0 --end_idx 20000
python inspect_gradient_clusters.py

# 4. Train. Set GRADIENT_DIRECTION, WEIGHT_METHOD and WEIGHT_SUM_MODE
#    in train_weighted_dpo.py first.
accelerate launch train_weighted_dpo.py

# 5. Baselines.
accelerate launch train_standard_dpo.py
accelerate launch train_standard_ipo.py
accelerate launch train_standard_kto.py
accelerate launch train_standard_simpo.py
accelerate launch train_standard_tdpo.py
accelerate launch train_tidpo.py
```

### Sanity checks

- `WEIGHT_MODE = "uniform"` in `train_weighted_dpo.py` must reproduce `train_standard_dpo.py` exactly.
- `compute_cluster_products.py --validate 2` compares the JVP path against explicit dot products and
  reports relative error.
- `save_pissa_adapter.py --check_only` confirms the probe matches the fingerprint the dumps were built
  against. A mismatch invalidates every product computed against the other adapter.

---

## 7. References

- Rafailov et al., *Direct Preference Optimization*, arXiv:2305.18290.
- OLMo 3, arXiv:2512.13961 (base checkpoint and DPO hyperparameters).
- PiSSA, arXiv:2404.02948 (probe initialization).
- Hugging Face TRL, `DPOTrainer` / `KTOTrainer` / `CPOTrainer`.
