"""
TDPO baseline on anonymous/Dolci-Instruct-DPO-en, starting from
allenai/Olmo-3-7B-Instruct-SFT.

Method: Zeng et al., "Token-level Direct Preference Optimization", ICML 2024
(arXiv:2404.11999) / https://github.com/Vance0124/Token-level-Direct-Preference-Optimization

TDPO is DPO plus a sequential forward-KL term, D_SeqKL(x, y; pi_ref || pi_theta),
computed per token over the completion and summed. See tidpo_trainer.py for the
loss; TDPO is TI-DPO's base objective, so the two baselines share one
_compute_loss and cannot drift apart in the plumbing.

    TDPO1  loss = -logsigmoid( beta * [ du - (kl_r - kl_c)           ] )   Eq. 14/15
    TDPO2  loss = -logsigmoid( beta * [ du - alpha*(kl_r - sg(kl_c)) ] )   Eq. 17/18

TDPO_VARIANT=tdpo2 (default) or tdpo1. The authors' README: "When the learning
rate is low, we recommend TDPO1; conversely, for higher learning rates, TDPO2 is
preferable." At lr=5e-6 -- which is exactly their own config.yaml value -- that
rule is ambiguous, and their shipped config/loss/tdpo.yaml actually defaults to
if_tdpo2: false while their README's recommended command is TDPO2. Both variants
are one env var apart; run TDPO2 first (the paper's headline method, and the base
objective of the TI-DPO run) and TDPO1 as the cheap follow-up.

HYPERPARAMETER PROVENANCE. Same policy as train_tidpo.py: the method knobs are
the authors' own. Unlike TI-DPO, the paper and the release AGREE here -- Appendix
B ("TDPO Implementation Details and Hyperparameters") states outright:

    "Unless specified otherwise, we use a alpha = 0.5, beta = 0.1, batch size of
     64, and the RMSprop optimizer with a learning rate of 5e-6. We linearly warm
     up the learning rate from 0 to 5e-6 over 150 steps."

and Sec. 5.1 adds "For the remainder of this paper, we set alpha = 0.5" and beta
= 0.1 following DPO's official implementation. Appendix B also prints the
reference PyTorch loss, which tidpo_trainer.position_kl / the TDPO1-TDPO2 branch
reproduce line for line (check_tidpo_port.py asserts this against a verbatim
transcription).

    knob                    value    source
    ---------------------------------------------------------------------------
    beta   (temperature)    0.1      paper Appendix B + Sec. 5.1; README; tdpo.yaml
    alpha  (SeqKL weight)   0.5      paper Appendix B + Sec. 5.1; README; tdpo.yaml
                                     (TDPO1 has no alpha; ignored when variant=tdpo1)
    learning_rate           5e-6     paper Appendix B; config.yaml (this repo's DPO: 1e-5)
    num_train_epochs        1        repo config.yaml (paper does not state)
    max_grad_norm           10.0     repo config.yaml (paper does not state;
                                     TRL default is 1.0)

    KEPT FROM THIS REPO, by explicit choice -- optimization infrastructure:
    optimizer               adamw_8bit   (paper Appendix B says RMSprop)
    warmup                  ratio 0.1    (paper Appendix B says linear 0 -> 5e-6
                                          over 150 absolute steps; at 1632 total
                                          steps, ratio 0.1 = 163, so this is a
                                          near-match by coincidence rather than
                                          by construction)
    effective_batch_size    128          (paper Appendix B says 64)
    per_device_batch_size   1
    grad_accum_steps        derived      same rule as train_standard_dpo.py

    Note the batch size is the one place where paper and repo were both explicit
    and we still chose ours: 64 vs 128. Halving it doubles the step count to
    ~3264 and roughly doubles wall-clock. TDPO_EFF_BATCH=64 runs it their way if
    the reviewer asks.

    DELIBERATE DEVIATION:
    max_length              2048     repo config.yaml says 512, sized for Anthropic-HH.
                                     The paper does not state a sequence length.
                                     Dolci-Instruct-DPO-en is filtered at 8000 chars
                                     (~2000 tokens), so 512 would truncate most
                                     completions.

    NOT FOLLOWED, and worth stating if reported: the authors' README says the
    preference stage runs "multiple episodes (e.g., three episodes)", while their
    config.yaml ships n_epochs: 1. We use 1 epoch, matching their config and every
    other baseline in this repo. Three epochs would triple the cost and break
    comparability with train_standard_dpo.py.

COST. TDPO needs no second backward pass, so unlike TI-DPO it costs about the same
as standard DPO. The only extra is the full-vocab sequential KL, which adds a few
GB of activation but no meaningful compute.

Dataset columns after preprocessing:
    prompt, chosen, rejected
"""
from dotenv import load_dotenv
load_dotenv(".env")
import os

# Must be set before torch is imported -- the caching allocator reads this at CUDA
# init. expandable_segments lets a segment grow instead of demanding one contiguous
# free block, which is what actually killed the first TDPO run: it died asking for
# 10.83 GiB with 9.70 GiB free AND 15.04 GiB "reserved but unallocated", i.e. there
# was enough memory, just too fragmented to hand out in one piece. Completion
# lengths here run from a few hundred tokens to 2048, so every step requests a
# differently-sized block -- the textbook fragmentation case, and the reason the
# job survived 650 steps before dying rather than failing immediately. Allocation
# strategy only; numerics are untouched. Export PYTORCH_ALLOC_CONF to override.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from trl import DPOConfig

from tidpo_trainer import TDPOTrainer

SEED = 42
set_seed(SEED)

# ---- Config ----
MODEL_NAME = "allenai/Olmo-3-7B-Instruct-SFT"
DATASET_NAME = "anonymous/Dolci-Instruct-DPO-en"

# ---- TDPO method knobs (paper Appendix B; README; config/loss/tdpo.yaml all agree) ----
TDPO_VARIANT = os.environ.get("TDPO_VARIANT", "tdpo2").lower()
if TDPO_VARIANT not in ("tdpo1", "tdpo2"):
    raise ValueError(f"TDPO_VARIANT must be tdpo1 or tdpo2, got {TDPO_VARIANT!r}")
IF_TDPO2 = TDPO_VARIANT == "tdpo2"
# alpha scales the SeqKL term in TDPO2 only; TDPO1 has no alpha (paper Eq. 14).
TDPO_ALPHA = float(os.environ.get("TDPO_ALPHA", "0.5"))

# ---- Optimization ----
BETA = float(os.environ.get("TDPO_BETA", "0.1"))         # paper Appendix B
LEARNING_RATE = 5.0e-6                                   # paper Appendix B
MAX_GRAD_NORM = 10.0                                     # TDPO config.yaml (paper silent)
NUM_EPOCHS = 1                                           # TDPO config.yaml (paper silent)
# Kept from this repo (optimization infrastructure, not method).
OPTIM = "adamw_8bit"
WARMUP_RATIO = 0.1
BATCH_SIZE = 1
EFFECTIVE_BATCH_SIZE = int(os.environ.get("TDPO_EFF_BATCH", "128"))
# Deliberate deviation from TDPO config.yaml (512); see provenance above.
MAX_SEQ_LENGTH = int(os.environ.get("TDPO_MAX_LEN", "2048"))

# ---- Memory / restart knobs (infrastructure; no effect on the objective) ----
# Gradient checkpointing trades ~25-35% throughput for a large drop in activation
# memory (all decoder layers -> boundary hidden states + one recomputed layer).
# use_reentrant=False is REQUIRED: reentrant checkpointing does not support
# torch.autograd.grad(..., inputs=...), which the TI-DPO attribution pass uses.
# OLMo-3 has no dropout, so recomputation is deterministic and numerics are
# bit-identical -- only throughput changes, so A/B comparability is unaffected.
# TDPO OOM'd at step 651/1632 in the main backward without it, so this defaults ON.
TDPO_GRAD_CHECKPOINT = bool(int(os.environ.get("TDPO_GRAD_CHECKPOINT", "1")))
# "1"/"true" resumes from the newest checkpoint in OUTPUT_DIR; a path resumes from
# that checkpoint; unset starts fresh.
TDPO_RESUME = os.environ.get("TDPO_RESUME", "")

_alpha_tag = f"_alpha_{TDPO_ALPHA}" if IF_TDPO2 else ""
OUTPUT_DIR = (
    f"/checkpoints/weighted-dpo/"
    f"olmo3-7b-{TDPO_VARIANT}-full{_alpha_tag}_beta_{BETA}_lr_{LEARNING_RATE}"
)

local_rank = int(os.environ.get("LOCAL_RANK", 0))
num_gpus = int(os.environ.get("WORLD_SIZE", 1))
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // (BATCH_SIZE * num_gpus)
assert EFFECTIVE_BATCH_SIZE % (BATCH_SIZE * num_gpus) == 0

run_name = OUTPUT_DIR.split("/")[-1]
if local_rank == 0:
    wandb_config = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "method": TDPO_VARIANT,
        "paper": "arXiv:2404.11999",
        "max_seq_length": MAX_SEQ_LENGTH,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "beta": BETA,
        "warmup_ratio": WARMUP_RATIO,
        "max_grad_norm": MAX_GRAD_NORM,
        "optim": OPTIM,
        "gradient_checkpointing": TDPO_GRAD_CHECKPOINT,
        "num_gpus": num_gpus,
        "if_tdpo2": IF_TDPO2,
        "tdpo_alpha": TDPO_ALPHA if IF_TDPO2 else None,
        "hparam_source": "paper arXiv:2404.11999 Appendix B (alpha=0.5, beta=0.1, lr=5e-6)",
    }
    wandb.init(project="weighted-dpo", name=run_name, config=wandb_config)


# ---- Dataset preparation ----

def prepare_dataset(tokenizer):
    """Load the DPO dataset and format prompt / chosen / rejected completions.

    Identical to train_standard_dpo.py, including the prompt boundary, so the
    tokenization is bit-for-bit the same across the two runs.
    """
    dataset = load_dataset(DATASET_NAME, split="train")

    def normalize_message(msg):
        # Strip extra fields (reasoning_content, function_call, audio, ...) and
        # keep only what the Olmo-3 chat template consumes.
        return {"role": msg["role"], "content": msg["content"] or ""}

    # Prompt ends right after the last "<|im_start|>assistant" header (no trailing
    # newline), so the completion starts with the "\n" token.
    HEADER_STR = "<|im_start|>assistant"

    def format_example(example):
        chosen_messages = [normalize_message(m) for m in example["chosen"]]
        rejected_messages = [normalize_message(m) for m in example["rejected"]]

        chosen_full = tokenizer.apply_chat_template(
            chosen_messages, tokenize=False, add_generation_prompt=False,
        )
        rejected_full = tokenizer.apply_chat_template(
            rejected_messages, tokenize=False, add_generation_prompt=False,
        )

        chosen_pos = chosen_full.rfind(HEADER_STR)
        rejected_pos = rejected_full.rfind(HEADER_STR)
        if chosen_pos == -1 or rejected_pos == -1:
            raise RuntimeError(
                f"Could not locate '{HEADER_STR}' in chat-templated example; "
                "chat template format may have changed."
            )

        return {
            "prompt": chosen_full[: chosen_pos + len(HEADER_STR)],
            "chosen": chosen_full[chosen_pos + len(HEADER_STR):],
            "rejected": rejected_full[rejected_pos + len(HEADER_STR):],
        }

    dataset = dataset.map(
        format_example,
        num_proc=16,
        desc="Formatting dataset",
        remove_columns=dataset.column_names,
    )
    return dataset


# ---- Model loading ----

def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token

    # Olmo-3's config.json stores rope_parameters beta_fast/beta_slow as ints;
    # transformers now requires floats. Cast them in the loaded config.
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    rope_params = getattr(config, "rope_parameters", None)
    if isinstance(rope_params, dict):
        for k in ("beta_fast", "beta_slow"):
            if k in rope_params:
                rope_params[k] = float(rope_params[k])

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, config=config, dtype=torch.bfloat16, trust_remote_code=True,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, config=config, dtype=torch.bfloat16, trust_remote_code=True,
    )

    # Olmo-3's generation_config ships with temperature/top_p but do_sample=False,
    # which fails strict validation on checkpoint save in newer transformers.
    if model.generation_config is not None:
        model.generation_config.do_sample = True

    return model, ref_model, tokenizer


# ---- Training ----

def main():
    model, ref_model, tokenizer = load_models()
    dataset = prepare_dataset(tokenizer)

    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        max_length=MAX_SEQ_LENGTH,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=MAX_GRAD_NORM,
        optim=OPTIM,
        bf16=True,
        gradient_checkpointing=TDPO_GRAD_CHECKPOINT,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, int(0.2 * len(dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS * num_gpus))),
        save_total_limit=1,
        report_to="wandb",
        run_name=run_name,
        dataloader_num_workers=8,
        dataset_num_proc=16,
        ddp_find_unused_parameters=False,
        loss_type="sigmoid",  # unused: TDPOTrainer overrides _compute_loss
        seed=SEED,
        data_seed=SEED,
    )

    trainer = TDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        if_tdpo2=IF_TDPO2,
        tdpo_alpha=TDPO_ALPHA,
    )

    resume = TDPO_RESUME
    if resume.lower() in ("1", "true", "yes"):
        resume = True
    elif not resume:
        resume = None
    trainer.train(resume_from_checkpoint=resume)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
