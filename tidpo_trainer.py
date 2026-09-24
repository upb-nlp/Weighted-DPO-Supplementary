"""
TIDPOTrainer / TDPOTrainer: TI-DPO and TDPO as TRL DPOTrainer subclasses.

Both live here because TDPO *is* TI-DPO's base objective -- TI-DPO adds token
importance weights on top of the TDPO2 loss, and the TIDPO release is literally a
fork of the TDPO codebase. Sharing one `_compute_loss` means the two baselines
cannot drift apart in the plumbing (shift convention, completion masking, chunked
KL), so any measured difference is the method. See TDPOTrainer at the bottom.

TDPO: Zeng et al., "Token-level Direct Preference Optimization", ICML 2024
(arXiv:2404.11999), https://github.com/Vance0124/Token-level-Direct-Preference-Optimization


Reference: Yang et al., "Token-Importance Guided Direct Preference Optimization",
ICLR 2026 (arXiv:2505.19653), and the authors' release at
https://github.com/gracefulning/TIDPO.

The authors' code is a fork of the original (hydra + FSDP) DPO codebase pinned to
torch 2.0.1 / transformers 4.29.2, so it cannot load OLMo-3. This is a port of the
method onto the same TRL stack every other baseline here uses, so the A/B against
train_standard_dpo.py and train_weighted_dpo.py is exact.

Two components, matching the authors' config/loss/tidpo.yaml:

1. Token-importance weights (paper Eq. 5-8). The attribution target is the max
   logit at the last completion position; its gradient w.r.t. the input
   embeddings gives a raw per-token score I_i = ||d target / d e_i||_1. Scores are
   normalized over the completion span and mixed with a position-based Gaussian
   prior:

       W = lam * I_norm + (1 - lam) * P_prior
       P_prior(t) ~ exp(-0.5 ((t - mu) / sigma)^2),  mu = (n-1)/2,  sigma = n / sigma_div

   then rescaled to mean 1 over the completion, so the weighted log-ratio sum stays
   on the same scale as the unweighted one (i.e. uniform weights reduce exactly to
   the unweighted margin). The weights are constants -- no gradient flows through
   the attribution pass.

2. A TDPO2 base objective. The authors' `tidpo` loss inherits TDPO2 (alpha=0.5,
   if_tdpo2=true), not vanilla sigmoid DPO, despite what the paper text describes:

       margin_y = sum_t w_t * (log pi_theta(y_t) - log pi_ref(y_t))
       kl_y     = sum_t KL( pi_ref(.|y_<t) || pi_theta(.|y_<t) )      [unweighted, full-vocab]
       logits   = (margin_c - margin_r) - alpha * (kl_r - kl_c.detach())
       loss     = -logsigmoid(beta * logits)

   Set alpha=0.0 to drop to plain weighted sigmoid DPO.

The triplet term (paper Eq. 13) is deliberately NOT implemented. It requires
sampling an anchor completion from the policy every step; at 7B with
per_device_train_batch_size=1 the autoregressive decode costs more than the rest
of training combined. This trainer therefore corresponds to the authors' own
"No Triplet Loss" ablation row (paper Table 2), which should be stated as such
when the numbers are reported.

The paper and the authors' shipped config/loss/tidpo.yaml disagree on three of
four method hyperparameters. Defaults here follow the PAPER: Table B13 is
explicitly "the final hyperparameters used in our experiments", whereas
config/loss/tidpo.yaml is a gpt2_small demo config (config.yaml alongside it
sets max_length=256, effective batch 2, datasets=[hh]).

                            paper                 config/loss/tidpo.yaml
    beta   temperature      0.1   (Table B13)     0.2
    alpha  TDPO KL weight   0.5   (Table B13)     0.5     agree
    lambda weight mix       0.7   (Table B13)     0.2
    gamma  triplet weight   0.1   (Table B13)     0.001   (not implemented here)
    sigma  prior width      n/4   (Eq. 7)         n/8     (prior_sigma_div)

Paper Table B11 sweeps lambda over {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0} and reports
stability across [0.3, 0.7]; lambda=0 is the prior-only control. Table B12 sweeps
alpha over {0.1, 0.2, 0.3, 0.5}.

Appendix D.1 states the expected cost: "the computational cost per training
iteration is approximately double that of standard DPO", from the one extra
backward pass per sequence.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from trl import DPOTrainer
from trl.trainer.utils import selective_log_softmax


# ---------------------------------------------------------------------------
# Token importance (paper Eq. 5-8)
# ---------------------------------------------------------------------------

def _last_true_index(mask: torch.Tensor) -> torch.LongTensor:
    """Index of the last nonzero entry in each row of a [B, T] mask.

    TRL's collator left-pads the prompt and right-pads the completion, so a row
    has padding on *both* ends and `mask.sum(dim=1) - 1` (what the authors' code
    uses) is not the last valid position.
    """
    m = mask.to(torch.long)
    flipped = m.flip(dims=[1])
    # argmax returns the first maximal entry, i.e. the first 1 scanning backwards.
    return mask.shape[1] - 1 - flipped.argmax(dim=1)


def gradient_attribution_scores(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor,
    target_positions: torch.LongTensor,
) -> torch.FloatTensor:
    """Raw per-token importance I_i = ||d max(logits[target_pos]) / d e_i||_1.

    Returns a detached [B, T] tensor. Padding positions are zeroed.

    Runs on the *unwrapped* module so DDP's reducer hooks are not involved, and
    takes grads only w.r.t. the embedding tensor, so no parameter `.grad` is
    touched and nothing is accumulated into the optimizer state.

    Two memory choices, both load-bearing at 7B / T=2048 (this pass OOM'd an
    H200 before them). check_tidpo_port.py asserts the result is bit-identical
    to the naive `model(...).logits` formulation:

    1. The decoder is called directly and lm_head is applied ONLY at the target
       positions, so the full [B, T, V] logits (~0.8 GB bf16 at OLMo-3's ~100k
       vocab) and their gradient are never materialised -- the target is one
       scalar per sequence, so all but one row of that tensor is dead weight.
       Calling get_decoder() also bypasses accelerate's ConvertOutputsToFp32
       wrapper, which had been doubling it again in fp32.
    2. The model is left in whatever mode it is already in rather than forced to
       eval(). HF's GradientCheckpointingLayer only checkpoints when
       self.training is True, so eval() would silently disable checkpointing for
       exactly the pass that needs it most. Safe because OLMo-3 sets
       attention_dropout=0.0 -- verify before reusing this with another model.

    Called before the policy forward so this pass's activations are freed by
    autograd.grad() before the training graph is built; the two never overlap.
    """
    decoder = model.get_decoder()
    lm_head = model.get_output_embeddings()
    with torch.enable_grad():
        embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
        hidden = decoder(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state                                           # [B, T, H]
        batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
        target_logits = lm_head(hidden[batch_idx, target_positions])  # [B, V], not [B, T, V]
        target = target_logits.float().max(dim=-1).values             # [B]
        grads = torch.autograd.grad(target.sum(), embeddings)[0]

    scores = grads.detach().abs().sum(dim=-1).to(torch.float32)
    return scores * attention_mask.to(scores.dtype)


def mix_importance_with_prior(
    scores: torch.FloatTensor,
    span_mask: torch.Tensor,
    lambda_importance: float,
    prior_sigma_div: float,
) -> torch.FloatTensor:
    """Combine normalized attribution scores with a Gaussian position prior (Eq. 7-8).

    `span_mask` selects the tokens the weights apply to (the completion). Weights
    are rescaled to mean 1 over that span and zeroed outside it, so a uniform
    importance distribution gives exactly the unweighted log-ratio sum.
    """
    weights = torch.zeros_like(scores)

    for i in range(scores.shape[0]):
        idx = torch.nonzero(span_mask[i], as_tuple=False).squeeze(-1)
        n = int(idx.numel())
        if n == 0:
            continue
        if n == 1:
            weights[i, idx] = 1.0
            continue

        # Gaussian prior over the completion span only (positions 0..n-1 within it).
        pos = torch.arange(n, device=scores.device, dtype=torch.float32)
        center = (n - 1) / 2.0
        sigma = max(1.0, n / prior_sigma_div)
        prior = torch.exp(-0.5 * ((pos - center) / sigma) ** 2)
        prior = prior / prior.sum()

        raw = scores[i, idx].clamp_min(0)
        total = raw.sum()
        if lambda_importance > 0.0 and total > 0:
            mixed = lambda_importance * (raw / total) + (1.0 - lambda_importance) * prior
        else:
            # No usable attribution signal (or lambda=0): prior only.
            mixed = prior

        # Mean 1 over the span, not sum 1, so the margin keeps the scale of the
        # unweighted sum and beta stays comparable with the DPO baseline.
        weights[i, idx] = mixed / mixed.sum() * float(n)

    return weights


# ---------------------------------------------------------------------------
# TDPO2 position-wise KL
# ---------------------------------------------------------------------------

def position_kl(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Per-position KL( pi_ref || pi_theta ), differentiable w.r.t. policy_logits.

    Written as

        KL = sum_v p_ref log p_ref  -  [ sum_v p_ref * z  -  logsumexp(z) ]

    so the policy side never needs an explicit [B, T, V] log_softmax tensor. The
    time axis is chunked because V is ~100k for OLMo-3 and the fp32 intermediates
    would otherwise spike several GB at once.

    Returns [B, T].
    """
    outs = []
    for start in range(0, policy_logits.shape[1], chunk_size):
        z = policy_logits[:, start : start + chunk_size, :].float()
        with torch.no_grad():
            r = ref_logits[:, start : start + chunk_size, :].float()
            ref_logps = r.log_softmax(-1)
            ref_ps = ref_logps.exp()
            neg_entropy = (ref_ps * ref_logps).sum(-1)
            del r, ref_logps
        cross = (ref_ps * z).sum(-1) - torch.logsumexp(z, dim=-1)
        outs.append(neg_entropy - cross)
    return torch.cat(outs, dim=1)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TIDPOTrainer(DPOTrainer):
    """DPOTrainer implementing TI-DPO's weighted TDPO2 objective.

    Args:
        lambda_importance: mixing weight on the gradient-attribution component
            (paper Table B13: 0.7). 1.0 = attribution only, 0.0 = Gaussian prior
            only (the useful control: it tests whether the weights carry any
            content signal at all).
        prior_sigma_div: sigma = span_length / prior_sigma_div (paper Eq. 7: 4).
        tdpo_alpha: TDPO2 coefficient on the KL-difference term (paper Table B13:
            0.5). 0.0 reduces the objective to weighted sigmoid DPO.
        enable_gradient_attribution: when False, skips the extra forward/backward
            entirely and uses the prior alone (implies lambda_importance = 0).
        kl_chunk_size: time-axis chunk for the full-vocab KL. Does not change the
            memory *retained* for the backward (that is set by the total sequence
            length either way), but it does bound the size of each individual
            allocation. At ~100k vocab a 512-chunk asks for a ~0.4 GB contiguous
            fp32 block per chunk; 128 asks for ~0.1 GB. On a fragmented allocator
            the smaller request is the one that succeeds.

    Requires a live `ref_model`; `precompute_ref_log_probs` cannot be used because
    the TDPO2 KL term needs full reference logits, not just per-token log-probs.
    """

    def __init__(
        self,
        *args,
        lambda_importance: float = 0.7,
        prior_sigma_div: float = 4.0,
        tdpo_alpha: float = 0.5,
        enable_gradient_attribution: bool = True,
        token_weighting: bool = True,
        if_tdpo2: bool = True,
        kl_chunk_size: int = 128,
        **kwargs,
    ):
        if not 0.0 <= lambda_importance <= 1.0:
            raise ValueError(f"lambda_importance must be in [0, 1], got {lambda_importance}")
        if prior_sigma_div < 1.0:
            raise ValueError(f"prior_sigma_div must be >= 1, got {prior_sigma_div}")

        self.lambda_importance = 0.0 if not enable_gradient_attribution else lambda_importance
        self.prior_sigma_div = prior_sigma_div
        self.tdpo_alpha = tdpo_alpha
        self.enable_gradient_attribution = enable_gradient_attribution
        self.token_weighting = token_weighting
        self.if_tdpo2 = if_tdpo2
        self.kl_chunk_size = kl_chunk_size
        super().__init__(*args, **kwargs)

        if getattr(self.args, "precompute_ref_log_probs", False):
            raise ValueError(
                "TIDPOTrainer needs full reference logits for the TDPO2 position-KL "
                "term; precompute_ref_log_probs must be False."
            )
        if self.ref_model is None:
            raise ValueError("TIDPOTrainer requires an explicit ref_model.")

        # Fail loudly if this TRL does not route the loss through `_compute_loss`.
        # The hook is undocumented and has moved between TRL versions (some releases
        # dispatch via `get_batch_loss_metrics` instead). Without this check a
        # version bump would silently run plain DPO under the TI-DPO run name --
        # a mislabelled baseline is far worse than a crash on startup.
        if not any("_compute_loss" in klass.__dict__ for klass in type(self).__mro__[1:]):
            raise RuntimeError(
                "This TRL's DPOTrainer does not define `_compute_loss`, so the TI-DPO "
                "objective would never be called and the run would silently be plain DPO. "
                "Re-point the override at whatever hook this TRL uses (likely "
                "`get_batch_loss_metrics`). NOTE: weighted_dpo_trainer.py overrides the "
                "same hook, so it is affected identically -- check it too."
            )

    def _compute_loss(self, model, inputs, return_outputs=False):
        mode = "train" if self.model.training else "eval"

        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps", "token_weights"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]

        # --- token importance weights (constants; computed before the graph-building
        #     forward so the attribution pass's activations are freed first) ---
        # token_weighting=False is plain TDPO: every completion token gets weight 1,
        # and neither the attribution pass nor the Gaussian prior runs at all.
        if not self.token_weighting:
            weights = None
        elif self.enable_gradient_attribution:
            unwrapped = self.accelerator.unwrap_model(model)
            target_positions = _last_true_index(completion_mask.bool())
            scores = gradient_attribution_scores(
                unwrapped, input_ids, attention_mask, target_positions
            )
        else:
            scores = torch.zeros_like(input_ids, dtype=torch.float32)

        if self.token_weighting:
            weights = mix_importance_with_prior(
                scores,
                completion_mask.bool(),
                self.lambda_importance,
                self.prior_sigma_div,
            )

        # --- policy forward ---
        # The logit slices are kept as VIEWS, not .contiguous() copies as in
        # WeightedDPOTrainer. TDPO2 needs full-vocab policy AND reference logits
        # live at the same time, and at OLMo-3's ~100k vocab each [2, 2047, V] bf16
        # copy is ~0.8 GB; contiguous() would add ~1.6 GB of peak VRAM for nothing,
        # since both selective_log_softmax and position_kl handle strided input.
        outputs = model(**model_kwargs)
        shift_logits = outputs.logits[..., :-1, :]
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()
        # Weights are indexed by input_ids position; shift to align with the
        # log-probs, matching WeightedDPOTrainer.
        if weights is None:
            shift_weights = torch.ones_like(shift_completion_mask, dtype=shift_logits.dtype)
        else:
            shift_weights = weights[..., 1:].contiguous().to(shift_logits.dtype)

        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps = per_token_logps * shift_completion_mask

        # --- reference forward ---
        with torch.no_grad():
            ref_outputs = self.ref_model(**model_kwargs)
        ref_shift_logits = ref_outputs.logits[..., :-1, :]
        ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
        ref_per_token_logps = ref_per_token_logps * shift_completion_mask

        # --- weighted log-ratio margin (Eq. 11) ---
        per_token_logratio = per_token_logps - ref_per_token_logps
        margins = (per_token_logratio * shift_weights * shift_completion_mask).sum(dim=1)
        chosen_margin, rejected_margin = margins.chunk(2, dim=0)

        # --- sequential KL, unweighted over the completion (TDPO Eq. 10) ---
        # TDPO1 has no alpha at all, so the KL is needed whenever if_tdpo2 is False
        # even though tdpo_alpha may be 0.
        use_kl = (not self.if_tdpo2) or (self.tdpo_alpha != 0.0)
        if use_kl:
            per_pos_kl = position_kl(shift_logits, ref_shift_logits, self.kl_chunk_size)
            kl_sums = (per_pos_kl * shift_completion_mask).sum(dim=1)
            chosen_kl, rejected_kl = kl_sums.chunk(2, dim=0)
            if self.if_tdpo2:
                # Eq. 17/18: delta_2 uses the stop-gradient operator on the chosen KL,
                # and alpha scales the whole term.
                logits = (chosen_margin - rejected_margin) - self.tdpo_alpha * (
                    rejected_kl - chosen_kl.detach()
                )
            else:
                # Eq. 14/15: TDPO1 propagates gradient through BOTH KL terms and has
                # no alpha. beta multiplies the whole bracket, as in Eq. 13-15.
                logits = (chosen_margin - rejected_margin) - (rejected_kl - chosen_kl)
        else:
            chosen_kl = rejected_kl = torch.zeros_like(chosen_margin)
            logits = chosen_margin - rejected_margin

        loss = -F.logsigmoid(self.beta * logits).mean()

        # --- metrics (names kept aligned with WeightedDPOTrainer so the sweep
        #     aggregation in compare_olmes_to_paper.py / plot_beta_sweep.py sees
        #     the same series) ---
        unweighted = (per_token_logratio * shift_completion_mask).sum(dim=1)
        unweighted_chosen, unweighted_rejected = unweighted.detach().chunk(2, dim=0)
        logps = (per_token_logps * shift_completion_mask).sum(dim=1)
        chosen_logps, rejected_logps = logps.detach().chunk(2, dim=0)

        chosen_rewards = self.beta * chosen_margin.detach()
        rejected_rewards = self.beta * rejected_margin.detach()

        def _log(key, value):
            self._metrics[mode][key].append(self.accelerator.gather_for_metrics(value).mean().item())

        _log("rewards/chosen", chosen_rewards)
        _log("rewards/rejected", rejected_rewards)
        _log("rewards/accuracies", (chosen_rewards > rejected_rewards).float())
        _log("rewards/margins", chosen_rewards - rejected_rewards)
        _log("logps/chosen", chosen_logps)
        _log("logps/rejected", rejected_logps)

        _log("rewards/chosen_unweighted", self.beta * unweighted_chosen)
        _log("rewards/rejected_unweighted", self.beta * unweighted_rejected)
        _log("rewards/margins_unweighted", self.beta * (unweighted_chosen - unweighted_rejected))
        _log("rewards/accuracies_unweighted", (unweighted_chosen > unweighted_rejected).float())

        _log("tdpo/kl_chosen", chosen_kl.detach().float())
        _log("tdpo/kl_rejected", rejected_kl.detach().float())
        _log("tdpo/kl_margin", (rejected_kl - chosen_kl).detach().float())

        _, rejected_w_mask = shift_completion_mask.chunk(2, dim=0)
        _, rejected_w = shift_weights.chunk(2, dim=0)
        rej_w = rejected_w[rejected_w_mask.bool()].float()
        if rej_w.numel() > 1:
            self._metrics[mode]["weights/rejected_mean"].append(rej_w.mean().item())
            self._metrics[mode]["weights/rejected_std"].append(rej_w.std().item())
            self._metrics[mode]["weights/rejected_max"].append(rej_w.max().item())
            self._metrics[mode]["weights/rejected_min"].append(rej_w.min().item())

        if mode == "train":
            num_tokens_in_batch = (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        return (loss, outputs) if return_outputs else loss


class TDPOTrainer(TIDPOTrainer):
    """Plain TDPO (Zeng et al., ICML 2024, arXiv:2404.11999).

    TDPO is DPO plus a sequential-KL term. With uniform token weights this class
    reproduces the authors' `tdpo_loss` exactly. Writing beta out of the bracket
    as the reference implementation does:

        u      = beta * [ (sum_t log ratio)_c - (sum_t log ratio)_r ]       Eq. 13
        kl_y   = sum_t KL( pi_ref(.|y_<t) || pi_theta(.|y_<t) )             Eq. 10

        TDPO1: loss = -logsigmoid( beta * [ du - (kl_r - kl_c)          ] ) Eq. 14/15
        TDPO2: loss = -logsigmoid( beta * [ du - alpha*(kl_r - sg(kl_c))] ) Eq. 17/18

    TDPO1 propagates gradient through both KL terms and has no alpha. TDPO2 stops
    the gradient on the chosen-side KL (treating it as a baseline) and scales the
    term by alpha; the paper's §4.4 argues this is what prevents the chosen-side
    sequential KL from growing unchecked.

    It inherits from TIDPOTrainer only to share the loss plumbing -- the actual
    lineage runs the other way. `token_weighting=False` disables the importance
    weights and the attribution pass entirely, so this costs the same as standard
    DPO plus the KL term (no second backward pass).

    Hyperparameters (authors' README command and config/loss/tdpo.yaml):
    beta=0.1, alpha=0.5. Their README adds a selection rule worth heeding:
    "When the learning rate is low, we recommend TDPO1; conversely, for higher
    learning rates, TDPO2 is preferable."
    """

    def __init__(self, *args, if_tdpo2: bool = True, tdpo_alpha: float = 0.5, **kwargs):
        kwargs.pop("token_weighting", None)
        kwargs.pop("enable_gradient_attribution", None)
        super().__init__(
            *args,
            token_weighting=False,
            enable_gradient_attribution=False,
            if_tdpo2=if_tdpo2,
            tdpo_alpha=tdpo_alpha,
            **kwargs,
        )
