"""
WeightedDPOTrainer: a DPOTrainer subclass that accepts per-token weights
for chosen and rejected completions.

The dataset must contain two extra columns:
  - chosen_weights:   list[float] of length len(chosen_ids)
  - rejected_weights: list[float] of length len(rejected_ids)

Each weight scales the corresponding token's log-probability before the
per-sequence sum, so the DPO loss becomes:

  loss = -log σ( β * (Σ_i w_c_i log π(c_i) - Σ_i w_c_i log π_ref(c_i)
                     - Σ_j w_r_j log π(r_j) + Σ_j w_r_j log π_ref(r_j)) )

When all weights are 1.0, this reduces to the standard DPO loss.
"""

from dataclasses import dataclass
from typing import Any

import torch
from trl import DPOTrainer, DPOConfig
from trl.trainer.dpo_trainer import DataCollatorForPreference
from trl.trainer.utils import pad, selective_log_softmax
from datasets import Dataset, IterableDataset
try:
    from transformers.utils import is_peft_model
except ImportError:
    from peft import PeftModel
    def is_peft_model(model):
        return isinstance(model, PeftModel)

try:
    from trl.trainer.utils import disable_gradient_checkpointing
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def disable_gradient_checkpointing(model, kwargs):
        yield


try:
    from peft import use_adapter
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def use_adapter(model, adapter_name=None):
        yield


# ---------------------------------------------------------------------------
# Data collator that also pads token_weights
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorForWeightedPreference(DataCollatorForPreference):
    """Extends DataCollatorForPreference to carry per-token weights."""

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        # Let the parent build input_ids, attention_mask, completion_mask, etc.
        output = super().torch_call(examples)

        # Build token_weights in the same chosen-first-then-rejected order.
        # Weights apply only to completion tokens; prompt positions get weight 0
        # (they are masked out anyway).
        has_weights = "chosen_weights" in examples[0] and "rejected_weights" in examples[0]
        if has_weights:
            chosen_weights = [
                [0.0] * len(example["prompt_ids"]) + list(example["chosen_weights"])
                for example in examples
            ]
            rejected_weights = [
                [0.0] * len(example["prompt_ids"]) + list(example["rejected_weights"])
                for example in examples
            ]

            # Apply same truncation as parent
            max_length = getattr(self, "max_length", None)
            if max_length is not None:
                truncation_mode = getattr(self, "truncation_mode", "keep_start")
                if truncation_mode == "keep_start":
                    sl = slice(None, max_length)
                else:
                    sl = slice(-max_length, None)
                chosen_weights = [w[sl] for w in chosen_weights]
                rejected_weights = [w[sl] for w in rejected_weights]

            all_weights = chosen_weights + rejected_weights
            all_weights = [torch.tensor(w, dtype=torch.float32) for w in all_weights]

            token_weights = pad(
                all_weights,
                padding_value=0.0,
                padding_side="right",
                pad_to_multiple_of=getattr(self, "pad_to_multiple_of", None),
            )

            # Ensure token_weights matches the sequence length of input_ids
            # (TRL may append EOS or other tokens during collation)
            if "input_ids" in output:
                seq_len = output["input_ids"].shape[1]
                w_len = token_weights.shape[1]
                if w_len != seq_len:
                    print(f"WARNING: token_weights length {w_len} != input_ids length {seq_len}, adjusting by {'padding' if w_len < seq_len else 'truncating'}")
                if w_len < seq_len:
                    token_weights = torch.nn.functional.pad(token_weights, (0, seq_len - w_len), value=0.0)
                elif w_len > seq_len:
                    token_weights = token_weights[:, :seq_len]

            output["token_weights"] = token_weights

        return output


# ---------------------------------------------------------------------------
# WeightedDPOTrainer
# ---------------------------------------------------------------------------

class WeightedDPOTrainer(DPOTrainer):
    """DPOTrainer that weights each token's log-probability contribution.

    The per-token weights are always first normalized to sum to 1 per sequence.
    ``weight_sum_mode`` then controls how that normalized vector is rescaled
    before being applied to the log-probs:

      - "normalize":    leave weights summing to 1.
      - "reward_match": rescale (per sequence, detached) so the weighted reward
                        sum equals the unweighted reward sum, i.e. the standard
                        DPO reward is preserved (the original default behavior).
                        Requires reference per-token log-probs (not available
                        in precompute_ref_logps mode).
      - "token_count":  rescale so weights sum to the number of completion
                        tokens, i.e. the mean per-token weight is 1.
      - "token_scale":  apply the provided per-token weights as-is, WITHOUT the
                        per-sequence sum-to-1 normalization. Intended for weights
                        already scaled into [0, 1] per token (e.g. the cosine
                        distance (1 - cos) / 2 from train_weighted_dpo.py), so
                        each token's weight is independent of the token count.
      - "ignore":       ignore the provided weights entirely; every token gets
                        weight 1, recovering standard (unweighted) DPO.
    """

    VALID_WEIGHT_SUM_MODES = ("normalize", "reward_match", "token_count", "token_scale", "ignore")

    def __init__(self, *args, weight_sum_mode: str = "reward_match",
                 force_chosen_weight_one: bool = False, **kwargs):
        if weight_sum_mode not in self.VALID_WEIGHT_SUM_MODES:
            raise ValueError(
                f"weight_sum_mode must be one of {self.VALID_WEIGHT_SUM_MODES}, "
                f"got {weight_sum_mode!r}"
            )
        self.weight_sum_mode = weight_sum_mode
        # When True, every chosen-half token gets weight 1 regardless of
        # weight_sum_mode, so the chosen log-probs are the plain summed log-probs
        # (standard DPO on the chosen side); only rejected tokens are weighted.
        self.force_chosen_weight_one = force_chosen_weight_one
        super().__init__(*args, **kwargs)

        # Replace the data collator with our weighted variant.
        # The parent __init__ already created a DataCollatorForPreference;
        # we swap it out, preserving its settings.
        if isinstance(self.data_collator, DataCollatorForPreference):
            parent_collator = self.data_collator
            collator_kwargs = {
                "pad_token_id": getattr(parent_collator, "pad_token_id", 0),
            }
            for attr in ("max_length", "truncation_mode", "pad_to_multiple_of"):
                if hasattr(parent_collator, attr):
                    collator_kwargs[attr] = getattr(parent_collator, attr)
            self.data_collator = DataCollatorForWeightedPreference(**collator_kwargs)

    # ---- ensure weight columns are not dropped by the trainer ----

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        # Add weight columns so they survive remove_unused_columns filtering
        for col in ("chosen_weights", "rejected_weights"):
            if col not in self._signature_columns:
                self._signature_columns.append(col)

    # ---- override dataset preparation to keep weight columns ----

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        """Call parent _prepare_dataset, then ensure weight columns survive."""
        # Stash weight columns before the parent's .map() calls might drop them
        has_chosen_weights = "chosen_weights" in dataset.column_names if hasattr(dataset, "column_names") else False
        has_rejected_weights = "rejected_weights" in dataset.column_names if hasattr(dataset, "column_names") else False

        dataset = super()._prepare_dataset(dataset, processing_class, args, dataset_name)

        # If the original dataset had weight columns and they survived, great.
        # If they were dropped by tokenize_fn's map (which only returns
        # prompt_ids/chosen_ids/rejected_ids), we need to verify.
        # In practice, HF map preserves columns not returned by the function,
        # so they should still be there. But let's check.
        if hasattr(dataset, "column_names"):
            cols = dataset.column_names
            if has_chosen_weights and "chosen_weights" not in cols:
                raise ValueError(
                    "chosen_weights column was lost during dataset preparation. "
                    "Ensure your dataset map functions don't drop extra columns."
                )
            if has_rejected_weights and "rejected_weights" not in cols:
                raise ValueError(
                    "rejected_weights column was lost during dataset preparation. "
                    "Ensure your dataset map functions don't drop extra columns."
                )
        return dataset

    # ---- override loss computation to apply token weights ----

    def _compute_loss(self, model, inputs, return_outputs=False):
        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device

        # --- forward pass on policy model ---
        _non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps", "token_weights"}
        model_kwargs = {k: v for k, v in inputs.items() if k not in _non_model_keys}
        model_kwargs["use_cache"] = False
        outputs = model(**model_kwargs)

        input_ids = inputs["input_ids"]
        completion_mask = inputs["completion_mask"]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()

        per_token_logps = selective_log_softmax(shift_logits, shift_labels)
        per_token_logps[shift_completion_mask == 0] = 0.0

        # --- build per-token weights (normalized so they sum to 1 per sequence) ---
        # In "ignore" mode we skip weighting entirely (shift_weights stays None),
        # so the loss uses raw log-probs == standard unweighted DPO.
        shift_weights = None
        if "token_weights" in inputs and self.weight_sum_mode != "ignore":
            shift_weights = inputs["token_weights"][..., 1:].contiguous().to(per_token_logps.dtype).to(device)
            if self.weight_sum_mode != "token_scale":
                # Normalize: w_i = w_i_raw / sum(w_i_raw), so sum(w_i) = 1
                weight_sums = (shift_weights * shift_completion_mask).sum(dim=1, keepdim=True).clamp(min=1e-8)
                shift_weights = shift_weights / weight_sums
            # token_scale: use the per-token weights as-is (each already in [0, 1],
            # the cosine-distance scaling from prepare_dataset); no per-sequence
            # renormalization, so the weight does not depend on the token count.

        # Unweighted logps for diagnostic metrics
        unweighted_logps = per_token_logps.sum(dim=1)
        unweighted_chosen_logps, unweighted_rejected_logps = unweighted_logps.chunk(2, dim=0)

        # --- reference model log-probs (per-token, needed for WEIGHT_ADJUSTMENT) ---
        ref_per_token_logps = None
        if self.precompute_ref_logps:
            ref_chosen_logps = inputs["ref_chosen_logps"]
            ref_rejected_logps = inputs["ref_rejected_logps"]
            # Precomputed ref logps are unweighted
            unweighted_ref_chosen_logps = ref_chosen_logps
            unweighted_ref_rejected_logps = ref_rejected_logps
        else:
            with torch.no_grad(), disable_gradient_checkpointing(
                self.model, self.args.gradient_checkpointing_kwargs
            ):
                if is_peft_model(model) and self.ref_model is None:
                    unwrapped = self.accelerator.unwrap_model(model)
                    with use_adapter(
                        unwrapped,
                        adapter_name="ref" if "ref" in unwrapped.peft_config else None,
                    ):
                        ref_outputs = self.model(**model_kwargs)
                else:
                    ref_outputs = self.ref_model(**model_kwargs)

            ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
            ref_per_token_logps = selective_log_softmax(ref_shift_logits, shift_labels)
            ref_per_token_logps[shift_completion_mask == 0] = 0.0

            # Capture unweighted ref logps before applying weights
            unweighted_ref_logps = ref_per_token_logps.sum(dim=1)
            unweighted_ref_chosen_logps, unweighted_ref_rejected_logps = unweighted_ref_logps.chunk(2, dim=0)

        # --- token_count: rescale the (sum-to-1) weights so they sum to the number
        #     of completion tokens, i.e. the mean per-token weight is 1. ---
        if shift_weights is not None and self.weight_sum_mode == "token_count":
            comp_lengths = shift_completion_mask.sum(dim=1, keepdim=True).to(shift_weights.dtype)
            shift_weights = shift_weights * comp_lengths

        # --- reward_match: per-sequence scalar that rescales token weights so the
        #     weighted reward sum matches the unweighted (standard DPO) reward sum.
        #     Computed for both chosen and rejected halves as a detached constant;
        #     gradient structure (which tokens contribute how much) is preserved.
        #     Invariant: when input weights are uniform, sum_weighted == sum_unweighted,
        #     so adjustment == 1.0 and the weights are unchanged (standard DPO preserved).
        #     Skipped when ref per-token logps aren't available (precompute mode). ---
        if (
            shift_weights is not None
            and self.weight_sum_mode == "reward_match"
            and ref_per_token_logps is not None
        ):
            with torch.no_grad():
                per_token_reward = per_token_logps - ref_per_token_logps  # [B, T]
                sum_rewards_unweighted = (per_token_reward * shift_completion_mask).sum(dim=1)
                sum_rewards_weighted = (per_token_reward * shift_weights * shift_completion_mask).sum(dim=1)
                # Fall back to adjustment = 1.0 when |sum_weighted| < 1e-6 (e.g. policy
                # == ref at init when starting from the chosen checkpoint). Without this,
                # 0/0 would collapse weights to zero, killing gradients and stalling training.
                eps = 1e-6
                denom_sign = torch.where(
                    sum_rewards_weighted >= 0,
                    torch.ones_like(sum_rewards_weighted),
                    -torch.ones_like(sum_rewards_weighted),
                )
                safe_denom = denom_sign * sum_rewards_weighted.abs().clamp(min=eps)
                raw_adjustment = sum_rewards_unweighted / safe_denom
                weight_adjustment = torch.where(
                    sum_rewards_weighted.abs() < eps,
                    torch.ones_like(raw_adjustment),
                    raw_adjustment,
                )  # [B]
            shift_weights = shift_weights * weight_adjustment.unsqueeze(1)

            chosen_adj, rejected_adj = weight_adjustment.chunk(2, dim=0)
            self._metrics[mode]["weights/adjustment_chosen"].append(
                self.accelerator.gather_for_metrics(chosen_adj).mean().item()
            )
            self._metrics[mode]["weights/adjustment_rejected"].append(
                self.accelerator.gather_for_metrics(rejected_adj).mean().item()
            )

        # --- force chosen-side weights to 1 (standard DPO on the chosen half) ---
        # Applied last so it overrides every weight_sum_mode rescaling above:
        # each chosen completion token contributes its raw log-prob (weight 1),
        # while the rejected half keeps the weighting scheme.
        if shift_weights is not None and self.force_chosen_weight_one:
            n_chosen = shift_weights.shape[0] // 2
            shift_weights[:n_chosen] = shift_completion_mask[:n_chosen].to(shift_weights.dtype)

        # --- apply weights to policy (and reference) log-probs, then aggregate ---
        if shift_weights is not None:
            weighted_per_token_logps = per_token_logps * shift_weights
        else:
            weighted_per_token_logps = per_token_logps
        logps = weighted_per_token_logps.sum(dim=1)
        chosen_logps, rejected_logps = logps.chunk(2, dim=0)

        if ref_per_token_logps is not None:
            if shift_weights is not None:
                ref_per_token_logps = ref_per_token_logps * shift_weights
            ref_logps = ref_per_token_logps.sum(dim=1)
            ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

        # --- log-ratios and scores ---
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps

        if self.f_divergence_type == "reverse_kl":
            chosen_scores = chosen_logratios
            rejected_scores = rejected_logratios
        elif self.f_divergence_type == "forward_kl":
            chosen_scores = -torch.exp(-chosen_logratios)
            rejected_scores = -torch.exp(-rejected_logratios)
        elif self.f_divergence_type == "js_divergence":
            chosen_scores = torch.nn.functional.logsigmoid(chosen_logratios)
            rejected_scores = torch.nn.functional.logsigmoid(rejected_logratios)
        elif self.f_divergence_type == "alpha_divergence":
            if abs(self.f_alpha_divergence_coef - 1.0) < 1e-6:
                chosen_scores = chosen_logratios
                rejected_scores = rejected_logratios
            else:
                coef = 1.0 / (self.f_alpha_divergence_coef - 1.0)
                t_chosen = (self.f_alpha_divergence_coef - 1.0) * chosen_logratios
                t_rejected = (self.f_alpha_divergence_coef - 1.0) * rejected_logratios
                dtype = t_chosen.dtype
                clamp_max = {torch.float16: 11.0, torch.bfloat16: 80.0, torch.float32: 80.0}[dtype]
                chosen_scores = torch.exp(torch.clamp(t_chosen.float(), max=clamp_max)).to(dtype) * coef
                rejected_scores = torch.exp(torch.clamp(t_rejected.float(), max=clamp_max)).to(dtype) * coef
        else:
            raise ValueError(f"Unknown f_divergence_type: {self.f_divergence_type}")

        delta_score = chosen_scores - rejected_scores

        # --- loss computation ---
        import torch.nn.functional as F
        from trl.trainer.utils import entropy_from_logits

        loss = 0.0
        for loss_type, loss_weight in zip(self.loss_types, self.loss_weights, strict=True):
            if loss_type == "sigmoid":
                per_sequence_loss = -F.logsigmoid(self.beta * delta_score)
            elif loss_type == "hinge":
                per_sequence_loss = torch.relu(1 - self.beta * delta_score)
            elif loss_type == "ipo":
                chosen_mask_ipo, rejected_mask_ipo = completion_mask.chunk(2, dim=0)
                chosen_avg = chosen_scores / chosen_mask_ipo.sum(dim=1).clamp(min=1.0)
                rejected_avg = rejected_scores / rejected_mask_ipo.sum(dim=1).clamp(min=1.0)
                per_sequence_loss = (chosen_avg - rejected_avg - 1 / (2 * self.beta)) ** 2
            elif loss_type == "robust":
                clean = -(1 - self.label_smoothing) * F.logsigmoid(self.beta * delta_score)
                flipped = -self.label_smoothing * F.logsigmoid(-self.beta * delta_score)
                per_sequence_loss = (clean - flipped) / (1 - 2 * self.label_smoothing)
            elif loss_type == "exo_pair":
                epsilon = torch.tensor(self.label_smoothing, device=device)
                qw = torch.sigmoid(self.beta * delta_score)
                log_qw = F.logsigmoid(self.beta * delta_score)
                log_pw = torch.log1p(-epsilon)
                ql = torch.sigmoid(-self.beta * delta_score)
                log_ql = F.logsigmoid(-self.beta * delta_score)
                log_pl = torch.log(epsilon)
                per_sequence_loss = qw * (log_qw - log_pw) + ql * (log_ql - log_pl)
            elif loss_type == "nca_pair":
                cr = self.beta * chosen_scores
                rr = self.beta * rejected_scores
                per_sequence_loss = -F.logsigmoid(cr) - 0.5 * F.logsigmoid(-cr) - 0.5 * F.logsigmoid(-rr)
            elif loss_type == "bco_pair":
                cr = self.beta * chosen_scores
                rr = self.beta * rejected_scores
                per_sequence_loss = -F.logsigmoid(cr) - F.logsigmoid(-rr)
            elif loss_type == "sppo_hard":
                per_sequence_loss = (chosen_scores - 0.5 / self.beta) ** 2 + (rejected_scores + 0.5 / self.beta) ** 2
            elif loss_type == "aot":
                logratios = chosen_logps - rejected_logps
                ref_logratios_aot = ref_chosen_logps - ref_rejected_logps
                logratios_sorted, _ = torch.sort(logratios, dim=0)
                ref_sorted, _ = torch.sort(ref_logratios_aot, dim=0)
                delta = logratios_sorted - ref_sorted
                per_sequence_loss = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
                )
            elif loss_type == "aot_unpaired":
                cs, _ = torch.sort(chosen_logratios, dim=0)
                rs, _ = torch.sort(rejected_logratios, dim=0)
                delta = cs - rs
                per_sequence_loss = (
                    -F.logsigmoid(self.beta * delta) * (1 - self.label_smoothing)
                    - F.logsigmoid(-self.beta * delta) * self.label_smoothing
                )
            elif loss_type == "apo_zero":
                per_sequence_loss = (
                    1 - torch.sigmoid(self.beta * chosen_logratios)
                    + torch.sigmoid(self.beta * rejected_logratios)
                )
            elif loss_type == "apo_down":
                per_sequence_loss = (
                    torch.sigmoid(self.beta * chosen_logratios)
                    + 1 - torch.sigmoid(self.beta * delta_score)
                )
            elif loss_type == "discopop":
                logits_dp = delta_score * self.beta
                mod = torch.sigmoid(logits_dp / self.args.discopop_tau)
                per_sequence_loss = -F.logsigmoid(logits_dp) * (1 - mod) + torch.exp(-logits_dp) * mod
            elif loss_type == "sft":
                chosen_logits_sft, _ = shift_logits.chunk(2, dim=0)
                chosen_labels_sft, _ = shift_labels.chunk(2, dim=0)
                chosen_mask_sft, _ = shift_completion_mask.chunk(2, dim=0)
                batch_loss = F.cross_entropy(
                    chosen_logits_sft[chosen_mask_sft.bool()],
                    chosen_labels_sft[chosen_mask_sft.bool()],
                )
                per_sequence_loss = batch_loss.expand(chosen_logits_sft.size(0))
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")

            if self.use_weighting:
                comp_lengths = shift_completion_mask.sum(dim=1).clamp_min(1)
                with torch.no_grad():
                    lse1 = torch.logsumexp(shift_logits, dim=-1)
                    lse2 = torch.logsumexp(2.0 * shift_logits, dim=-1)
                    log_denom = lse2 - 2.0 * lse1
                    aligned = (per_token_logps - log_denom) * shift_completion_mask
                mean_logps = aligned.sum(dim=1) / comp_lengths
                weights = torch.exp(mean_logps)
                cw, rw = weights.chunk(2, dim=0)
                per_sequence_loss *= cw * rw

            loss += per_sequence_loss.mean() * loss_weight

        # --- metrics (mirrors parent) ---
        per_token_entropy = entropy_from_logits(shift_logits.detach())
        entropy = per_token_entropy[shift_completion_mask.bool()].mean()
        entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
        self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            num_tokens_in_batch = (
                self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            )
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        chosen_logits_m, rejected_logits_m = shift_logits.detach().chunk(2, dim=0)
        chosen_mask_m, rejected_mask_m = shift_completion_mask.chunk(2, dim=0)
        total_chosen_logits = chosen_logits_m[chosen_mask_m.bool()].mean(-1).sum()
        total_chosen_tokens = chosen_mask_m.sum()
        total_rejected_logits = rejected_logits_m[rejected_mask_m.bool()].mean(-1).sum()
        total_rejected_tokens = rejected_mask_m.sum()
        total_chosen_logits = self.accelerator.gather_for_metrics(total_chosen_logits).sum().item()
        total_chosen_tokens = self.accelerator.gather_for_metrics(total_chosen_tokens).sum().item()
        total_rejected_logits = self.accelerator.gather_for_metrics(total_rejected_logits).sum().item()
        total_rejected_tokens = self.accelerator.gather_for_metrics(total_rejected_tokens).sum().item()
        self._metrics[mode]["logits/chosen"].append(
            total_chosen_logits / total_chosen_tokens if total_chosen_tokens > 0 else 0.0
        )
        self._metrics[mode]["logits/rejected"].append(
            total_rejected_logits / total_rejected_tokens if total_rejected_tokens > 0 else 0.0
        )

        predictions = chosen_logits_m.argmax(dim=-1)
        c_mask = shift_completion_mask[: len(shift_completion_mask) // 2].bool()
        c_labels = shift_labels[: len(shift_labels) // 2]
        correct = (predictions == c_labels) & c_mask
        total_t = self.accelerator.gather_for_metrics(c_mask.sum())
        correct_t = self.accelerator.gather_for_metrics(correct.sum())
        total_sum = total_t.sum()
        self._metrics[mode]["mean_token_accuracy"].append(
            (correct_t.sum() / total_sum).item() if total_sum > 0 else 0.0
        )

        chosen_rewards = self.beta * chosen_logratios.detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        self._metrics[mode]["rewards/chosen"].append(
            self.accelerator.gather(chosen_rewards).mean().item()
        )
        self._metrics[mode]["rewards/rejected"].append(
            self.accelerator.gather(rejected_rewards).mean().item()
        )
        reward_acc = (chosen_rewards > rejected_rewards).float()
        self._metrics[mode]["rewards/accuracies"].append(
            self.accelerator.gather(reward_acc).mean().item()
        )
        margins = chosen_rewards - rejected_rewards
        self._metrics[mode]["rewards/margins"].append(
            self.accelerator.gather(margins).mean().item()
        )
        self._metrics[mode]["logps/chosen"].append(
            self.accelerator.gather(chosen_logps).mean().item()
        )
        self._metrics[mode]["logps/rejected"].append(
            self.accelerator.gather(rejected_logps).mean().item()
        )

        # --- unweighted metrics (comparable across weighted / uniform runs) ---
        self._metrics[mode]["logps/chosen_unweighted"].append(
            self.accelerator.gather(unweighted_chosen_logps.detach()).mean().item()
        )
        self._metrics[mode]["logps/rejected_unweighted"].append(
            self.accelerator.gather(unweighted_rejected_logps.detach()).mean().item()
        )

        unweighted_chosen_rewards = self.beta * (unweighted_chosen_logps - unweighted_ref_chosen_logps).detach()
        unweighted_rejected_rewards = self.beta * (unweighted_rejected_logps - unweighted_ref_rejected_logps).detach()
        self._metrics[mode]["rewards/chosen_unweighted"].append(
            self.accelerator.gather(unweighted_chosen_rewards).mean().item()
        )
        self._metrics[mode]["rewards/rejected_unweighted"].append(
            self.accelerator.gather(unweighted_rejected_rewards).mean().item()
        )
        unweighted_margins = unweighted_chosen_rewards - unweighted_rejected_rewards
        self._metrics[mode]["rewards/margins_unweighted"].append(
            self.accelerator.gather(unweighted_margins).mean().item()
        )
        unweighted_acc = (unweighted_chosen_rewards > unweighted_rejected_rewards).float()
        self._metrics[mode]["rewards/accuracies_unweighted"].append(
            self.accelerator.gather(unweighted_acc).mean().item()
        )

        # --- token weight statistics (normalized weights as seen by the loss) ---
        if shift_weights is not None:
            _, rejected_mask_w = shift_completion_mask.chunk(2, dim=0)
            _, rejected_shift_weights = shift_weights.chunk(2, dim=0)
            rej_w = rejected_shift_weights[rejected_mask_w.bool()]
            if rej_w.numel() > 0:
                self._metrics[mode]["weights/rejected_mean"].append(rej_w.mean().item())
                self._metrics[mode]["weights/rejected_std"].append(
                    rej_w.std().item() if rej_w.numel() > 1 else 0.0
                )
                self._metrics[mode]["weights/rejected_max"].append(rej_w.max().item())
                self._metrics[mode]["weights/rejected_min"].append(rej_w.min().item())

        return (loss, outputs) if return_outputs else loss
