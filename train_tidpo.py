"""
TI-DPO baseline on anonymous/Dolci-Instruct-DPO-en, starting from
allenai/Olmo-3-7B-Instruct-SFT.

Method: Yang et al., "Token-Importance Guided Direct Preference Optimization",
ICLR 2026 (arXiv:2505.19653) / https://github.com/gracefulning/TIDPO.
See tidpo_trainer.py for what is and is not ported -- in particular the triplet
term (Eq. 13) is omitted, so this reproduces the authors' own "No Triplet Loss"
ablation and must be reported under that name.

HYPERPARAMETER PROVENANCE. The method knobs are the authors' own, not this repo's,
so TI-DPO is evaluated at the settings its authors chose. Where the paper and the
release disagree the paper wins: Appendix B.5 / Table B13 is explicitly "the final
hyperparameters used in our experiments", whereas config/loss/tidpo.yaml sits next
to a config.yaml for gpt2_small on HH (max_length 256, effective batch 2).

    knob                    value    source
    ---------------------------------------------------------------------------
    beta   (temperature)    0.1      paper Table B13   (repo config: 0.2)
    alpha  (TDPO KL)        0.5      paper Table B13   (repo config: 0.5, agrees)
    lambda (weight mix)     0.7      paper Table B13   (repo config: 0.2)
    sigma  (prior width)    n/4      paper Eq. 7       (repo config: n/8)
    gamma  (triplet)        0.1      paper Table B13   -- term not implemented
    learning_rate           5e-6     repo config.yaml  (this repo's DPO: 1e-5)
    max_grad_norm           10.0     repo config.yaml  (TRL default: 1.0)
    num_train_epochs        1        repo config.yaml  (agrees with this repo)

    KEPT FROM THIS REPO, by explicit choice -- optimization infrastructure that is
    orthogonal to the method, and that every other baseline here already shares:
    optimizer               adamw_8bit   (repo config.yaml says RMSprop)
    warmup                  ratio 0.1    (repo config.yaml says 150 absolute steps)
    effective_batch_size    128          (repo config.yaml says 2)
    per_device_batch_size   1            (agrees with repo config.yaml)
    grad_accum_steps        derived      EFFECTIVE_BATCH_SIZE / (BATCH_SIZE x num_gpus),
                                         the same rule as train_standard_dpo.py

    The batch size is deliberately ours and not theirs. The paper states no batch
    size for its Llama-3.1-8B / Mistral-7B runs, so nothing is being overridden;
    config.yaml's effective batch of 2 belongs to a gpt2_small demo, and at that
    size one epoch over ~300k pairs is ~150k optimizer steps on a 7B full
    fine-tune. Holding it at 128 also keeps the optimizer trajectory comparable
    with every other run in sweep_comparison.csv.

    DELIBERATE DEVIATION -- one repo value that is an artefact of a gpt2-scale demo
    and would make the run meaningless:

    max_length              2048     repo config.yaml says 256. Dolci-Instruct-DPO-en
                                     is filtered at 8000 chars (~2000 tokens), so 256
                                     would truncate away most of every completion --
                                     for TI-DPO especially, since the attribution
                                     target is taken at the END of the completion.

Env-overridable if you want to check them: TIDPO_MAX_LEN, TIDPO_EFF_BATCH.

WARNING on beta. The authors' beta=0.1 was chosen at max_length=256. TRL's DPO
loss is a non-length-normalized SUM of per-token log-ratios, so at 2048 tokens
beta*delta is roughly an order of magnitude larger than in their setting and
logsigmoid may saturate into an effective hinge. This repo's own DPO baseline uses
beta=0.002 for exactly that reason (see train_standard_dpo.py). Watch
rewards/margins over the first ~100 steps; if it saturates, sweep TIDPO_BETA with
the existing plot_beta_sweep.py infrastructure rather than silently keeping a
value that cannot train.

Dataset columns after preprocessing:
    prompt, chosen, rejected
"""
from dotenv import load_dotenv
load_dotenv(".env")
import os

# Must be set before torch is imported -- the caching allocator reads this at CUDA
# init. See train_standard_tdpo.py for the full reasoning: the TDPO run OOM'd at
# step 651/1632 asking for 10.83 GiB while 15.04 GiB sat reserved-but-unallocated,
# i.e. fragmentation, not capacity. TI-DPO is the more exposed of the two -- the
# attribution pass allocates and frees a full activation set every step on top of
# the training graph, which is exactly what fragments the allocator. Allocation
# strategy only; numerics are untouched. Export PYTORCH_ALLOC_CONF to override.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
from trl import DPOConfig

from tidpo_trainer import TIDPOTrainer

SEED = 42
set_seed(SEED)

# ---- Config ----
MODEL_NAME = "allenai/Olmo-3-7B-Instruct-SFT"
DATASET_NAME = "anonymous/Dolci-Instruct-DPO-en"

# ---- TI-DPO method knobs (paper Table B13 / Eq. 7; see provenance above) ----
# Table B11 sweeps lambda over {0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0}; lambda=0 is the
# Gaussian-prior-only control, which isolates whether the weights carry any content
# signal at all.
LAMBDA_IMPORTANCE = float(os.environ.get("TIDPO_LAMBDA", "0.7"))
PRIOR_SIGMA_DIV = float(os.environ.get("TIDPO_SIGMA_DIV", "4.0"))
# TDPO2 KL coefficient. Table B12 sweeps it over {0.1, 0.2, 0.3, 0.5}.
# 0.0 drops the objective to weighted sigmoid DPO.
TDPO_ALPHA = float(os.environ.get("TIDPO_ALPHA", "0.5"))
# Triplet weight from Table B13. Recorded for provenance only -- the triplet term
# is not implemented (it needs a policy sample per step); see tidpo_trainer.py.
TRIPLET_GAMMA_UNIMPLEMENTED = 0.1
# Skips the attribution forward/backward entirely (~2x -> ~1x step cost, per the
# authors' Appendix D.1), leaving the prior alone.
ENABLE_ATTRIBUTION = bool(int(os.environ.get("TIDPO_ATTRIBUTION", "1")))

# ---- Optimization ----
BETA = float(os.environ.get("TIDPO_BETA", "0.1"))       # paper Table B13
LEARNING_RATE = 5.0e-6                                   # TIDPO config.yaml
MAX_GRAD_NORM = 10.0                                     # TIDPO config.yaml
NUM_EPOCHS = 1                                           # TIDPO config.yaml
# Kept from this repo (optimization infrastructure, not method); GRAD_ACCUM_STEPS
# is derived below by the same rule as train_standard_dpo.py.
OPTIM = "adamw_8bit"
WARMUP_RATIO = 0.1
BATCH_SIZE = 1
EFFECTIVE_BATCH_SIZE = int(os.environ.get("TIDPO_EFF_BATCH", "128"))
# Deliberate deviation from TIDPO config.yaml (256); see provenance above.
MAX_SEQ_LENGTH = int(os.environ.get("TIDPO_MAX_LEN", "2048"))

# ---- Memory / restart knobs (infrastructure; no effect on the objective) ----
# Gradient checkpointing trades ~25-35% throughput for a large drop in activation
# memory (all decoder layers -> boundary hidden states + one recomputed layer).
# use_reentrant=False is REQUIRED: reentrant checkpointing does not support
# torch.autograd.grad(..., inputs=...), which the TI-DPO attribution pass uses.
# OLMo-3 has no dropout, so recomputation is deterministic and numerics are
# bit-identical -- only throughput changes, so A/B comparability is unaffected.
# The TI-DPO run completed steps without it, so this defaults OFF to keep a resume
# identical to the live job. Set 1 if a resume OOMs -- numerics are unchanged.
TIDPO_GRAD_CHECKPOINT = bool(int(os.environ.get("TIDPO_GRAD_CHECKPOINT", "0")))
# "1"/"true" resumes from the newest checkpoint in OUTPUT_DIR; a path resumes from
# that checkpoint; unset starts fresh.
TIDPO_RESUME = os.environ.get("TIDPO_RESUME", "")

_attr_tag = "attr" if ENABLE_ATTRIBUTION else "prioronly"
OUTPUT_DIR = (
    f"/checkpoints/weighted-dpo/"
    f"olmo3-7b-tidpo-notriplet-{_attr_tag}_lam_{LAMBDA_IMPORTANCE}_sig_{PRIOR_SIGMA_DIV}"
    f"_alpha_{TDPO_ALPHA}_beta_{BETA}_lr_{LEARNING_RATE}"
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
        "method": "ti-dpo (no triplet)",
        "paper": "arXiv:2505.19653",
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
        "gradient_checkpointing": TIDPO_GRAD_CHECKPOINT,
        "num_gpus": num_gpus,
        "lambda_importance": LAMBDA_IMPORTANCE,
        "prior_sigma_div": PRIOR_SIGMA_DIV,
        "tdpo_alpha": TDPO_ALPHA,
        "enable_gradient_attribution": ENABLE_ATTRIBUTION,
        "triplet_gamma_unimplemented": TRIPLET_GAMMA_UNIMPLEMENTED,
        "hparam_source": "paper Table B13 / Eq. 7; lr+max_grad_norm from TIDPO config.yaml",
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
        gradient_checkpointing=TIDPO_GRAD_CHECKPOINT,
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
        loss_type="sigmoid",  # unused: TIDPOTrainer overrides _compute_loss
        seed=SEED,
        data_seed=SEED,
    )

    trainer = TIDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        lambda_importance=LAMBDA_IMPORTANCE,
        prior_sigma_div=PRIOR_SIGMA_DIV,
        tdpo_alpha=TDPO_ALPHA,
        enable_gradient_attribution=ENABLE_ATTRIBUTION,
    )

    resume = TIDPO_RESUME
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
