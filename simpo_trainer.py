"""
SimPOTrainer: TRL's experimental CPOTrainer with one fix — the SimPO
length-normalization denominator is clamped to >= 1.

TRL's get_batch_logps computes the (length-normalized) average log-prob as:

    (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)

If an example's completion has 0 unmasked tokens, this is 0/0 -> NaN. The NaN
is produced in the *forward* (so logps go NaN) and, crucially, also in the
*gradient*, which then poisons every weight on the next optimizer step and kills
the run. Sanitizing the output afterwards is too late (the backward graph
already contains the NaN), so we clamp the denominator before the division.

Everything else is the upstream CPO/SimPO loss, unchanged. The override is the
exact upstream get_batch_logps body (TRL v0.29) with `.clamp(min=1)` added and a
diagnostic print so we can confirm whether 0-token completions are the cause.
"""

import torch
from trl.experimental.cpo import CPOTrainer
from trl.trainer.utils import selective_log_softmax


class SimPOTrainer(CPOTrainer):
    _warned_empty = False

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != -100
        labels[labels == -100] = 0
        per_token_logps = selective_log_softmax(logits, labels)

        # Diagnostic: how many sequences in this batch have a 0-token completion?
        n_empty = int((loss_mask.sum(-1) == 0).sum())
        if n_empty > 0:
            print(f"[SimPO guard] {n_empty} sequence(s) with 0 completion tokens "
                  f"this batch -> clamping denominator (would have been 0/0 NaN)", flush=True)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1)
        else:
            return (per_token_logps * loss_mask).sum(-1)
