"""
Standard SimPO training on anonymous/Dolci-Instruct-DPO-en, starting from
allenai/Olmo-3-7B-Instruct-SFT, using the HuggingFace TRL CPOTrainer with
loss_type="simpo".

SimPO (Simple Preference Optimization, arXiv:2405.14734, NeurIPS 2024) is a
*reference-free* preference method: the reward is the length-normalized average
log-probability of a completion (no reference model), and a target margin gamma
is subtracted from the chosen-minus-rejected reward difference. In TRL it is
implemented as CPO with cpo_alpha=0.0 (drops the SFT/NLL term) and loss_type
"simpo".

Hyperparameters follow the official SimPO repo (princeton-nlp/SimPO) v2 config
(beta=10), effective batch size 128, with LR raised to 2e-6:
    beta = 10, gamma_beta_ratio = 0.3, learning_rate = 2.0e-6
TRL's CPOConfig takes the absolute margin `simpo_gamma`, so we pass
    simpo_gamma = beta * gamma_beta_ratio = 10 * 0.3 = 3.0
(The v1 beta=2.5 / lr=1e-6 recipe produced grad_norm ~0.4 and zero learning over
a full epoch on Olmo-3 — the length-normalized SimPO gradient was too weak.)

Dataset columns after preprocessing:
    prompt, chosen, rejected
"""
from dotenv import load_dotenv
load_dotenv(".env")
import os
import torch
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
# TRL 0.29 moved CPO into the experimental namespace (same SimPO implementation
# as upstream v1.5.1; only the import path changed from `from trl import ...`).
from trl.experimental.cpo import CPOConfig
# SimPOTrainer = CPOTrainer with the length-norm denominator clamped to >=1,
# guarding against the 0-token-completion 0/0 -> NaN that poisons training.
from simpo_trainer import SimPOTrainer

SEED = 42
set_seed(SEED)

# ---- Config ----
MODEL_NAME = "allenai/Olmo-3-7B-Instruct-SFT"
DATASET_NAME = "anonymous/Dolci-Instruct-DPO-en"
OUTPUT_DIR = "/checkpoints/weighted-dpo/olmo3-7b-standard-simpo-full_beta_10_gbr_0.3_lr_2e-6"

# SimPO repo (princeton-nlp/SimPO) v2 config (beta=10). The v1 beta=2.5 / lr=1e-6
# gave essentially no gradient signal on Olmo-3 (length-normalized SimPO grads
# are ~100x smaller than summed DPO/IPO; beta=2.5 didn't compensate -> grad_norm
# ~0.4 and zero learning over a full epoch). v2's larger beta + a 2e-6 LR restore
# an update magnitude comparable to the IPO run that did learn.
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 2.0e-6
NUM_EPOCHS = 1
BATCH_SIZE = 1
EFFECTIVE_BATCH_SIZE = 128
# SimPO needs a much larger beta than DPO; gamma is the target reward margin.
BETA = 10.0
GAMMA_BETA_RATIO = 0.3
SIMPO_GAMMA = BETA * GAMMA_BETA_RATIO  # TRL takes the absolute margin (3.0)
WARMUP_RATIO = 0.1

local_rank = int(os.environ.get("LOCAL_RANK", 0))
num_gpus = int(os.environ.get("WORLD_SIZE", 1))
GRAD_ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // (BATCH_SIZE * num_gpus)
assert EFFECTIVE_BATCH_SIZE % (BATCH_SIZE * num_gpus) == 0

run_name = OUTPUT_DIR.split("/")[-1]
if local_rank == 0:
    wandb_config = {
        "model": MODEL_NAME,
        "dataset": DATASET_NAME,
        "max_seq_length": MAX_SEQ_LENGTH,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRAD_ACCUM_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "beta": BETA,
        "gamma_beta_ratio": GAMMA_BETA_RATIO,
        "simpo_gamma": SIMPO_GAMMA,
        "warmup_ratio": WARMUP_RATIO,
        "num_gpus": num_gpus,
        "loss_type": "simpo",
    }
    wandb.init(project="weighted-dpo-olmo_clean", name=run_name, config=wandb_config)


# ---- Dataset preparation ----

def prepare_dataset(tokenizer):
    """Load the DPO dataset and format prompt / chosen / rejected completions."""
    dataset = load_dataset(DATASET_NAME, split="train")
    #dataset = dataset.shuffle(seed=SEED).select(range(200))

    def normalize_message(msg):
        # Strip extra fields (reasoning_content, function_call, audio, ...) and
        # keep only what the Olmo-3 chat template consumes.
        return {"role": msg["role"], "content": msg["content"] or ""}

    # Match the boundary used by train_weighted_dpo.py and
    # compute_dpo_gradient_products.build_chat_ids: prompt ends right after the
    # last "<|im_start|>assistant" header (no trailing newline), so the
    # completion starts with the "\n" token. The cl100k-style pre-tokenizer
    # splits at "\n", making this a stable BPE boundary that satisfies TRL's
    # tokenized-prefix invariant.
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
        num_proc=4,
        desc="Formatting dataset",
        remove_columns=dataset.column_names,
    )

    # SimPO length-normalizes the reward by the completion token count, so a
    # 0-token completion (empty response, or one truncated away when the prompt
    # fills max_length) produces 0/0 -> NaN, which then poisons the weights.
    # Drop those rows: require >=1 real completion token on each side and room
    # for the completion under max_length.
    def _keep(example):
        chosen_ids = tokenizer(example["chosen"], add_special_tokens=False)["input_ids"]
        rejected_ids = tokenizer(example["rejected"], add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        return (
            len(chosen_ids) > 0
            and len(rejected_ids) > 0
            and len(prompt_ids) < MAX_SEQ_LENGTH
        )

    n_before = len(dataset)
    dataset = dataset.filter(_keep, num_proc=4, desc="Dropping empty/over-long rows")
    n_dropped = n_before - len(dataset)
    if local_rank == 0 and n_dropped:
        print(f"Filtered {n_dropped}/{n_before} degenerate rows (empty or over-long completions)")

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

    # SimPO is reference-free: no ref_model needed (also halves model memory).
    # Force SDPA attention: eager attention materializes the full per-head score
    # matrix and overflows in bf16 on some batches (-> NaN log-probs in the
    # forward pass). SDPA uses a numerically stable online softmax, and also
    # avoids the per-head mask recompute that caused the original CheckpointError.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, config=config, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Olmo-3's generation_config ships with temperature/top_p but do_sample=False,
    # which fails strict validation on checkpoint save in newer transformers.
    if model.generation_config is not None:
        model.generation_config.do_sample = True

    return model, tokenizer


# ---- Training ----

def main():

    model, tokenizer = load_models()
    dataset = prepare_dataset(tokenizer)

    # SimPO via CPO: loss_type="simpo" + cpo_alpha=0.0 drops the BC/NLL term,
    # leaving the pure SimPO loss (reference-free, length-normalized, margin
    # gamma). simpo_gamma is the absolute target reward margin.
    training_args = CPOConfig(
        output_dir=OUTPUT_DIR,
        loss_type="simpo",
        cpo_alpha=0.0,
        beta=BETA,
        simpo_gamma=SIMPO_GAMMA,
        max_length=MAX_SEQ_LENGTH,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_ratio=WARMUP_RATIO,
        optim="adamw_8bit",
        bf16=True,
        # Olmo-3's per-head attention mask is rebuilt differently on the
        # checkpoint recompute, which trips non-reentrant checkpointing's strict
        # metadata check. Reentrant checkpointing recomputes via normal autograd
        # and avoids the comparison.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        max_grad_norm=1.0,  # clip gradients; insurance against one-off spikes
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, int(0.2 * len(dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS * num_gpus))),
        save_total_limit=1,
        report_to="wandb",
        run_name=run_name,
        dataloader_num_workers=8,
        ddp_find_unused_parameters=False,
        seed=SEED,
        data_seed=SEED,
    )

    # SimPOTrainer (CPOTrainer subclass) is reference-free (no ref_model argument).
    trainer = SimPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    print(f"Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
