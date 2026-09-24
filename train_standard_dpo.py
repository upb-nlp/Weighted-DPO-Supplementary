"""
Standard DPO training on anonymous/Dolci-Instruct-DPO-en, starting from
allenai/Olmo-3-7B-Instruct-SFT, using the HuggingFace TRL DPOTrainer.

Hyperparameters follow the Olmo 3 paper (arXiv:2512.13961), Table 48,
"7B Instruct DPO" column, with β scaled down to 0.05 for TRL's
non-length-normalized sigmoid DPO loss.

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
from trl import DPOConfig, DPOTrainer

SEED = 42
set_seed(SEED)

# ---- Config ----
MODEL_NAME = "allenai/Olmo-3-7B-Instruct-SFT"
DATASET_NAME = "anonymous/Dolci-Instruct-DPO-en"
# REVERSE_DPO=1 swaps chosen<->rejected to train the "negative" contrastive model
# for TIS-DPO(D) weight estimation. Default (0) = standard forward/positive DPO.
REVERSE = bool(int(os.environ.get("REVERSE_DPO", "0")))
_variant = "reverse" if REVERSE else "full"


# Olmo 3 paper, Table 48, 7B Instruct DPO column.
MAX_SEQ_LENGTH = 2048
LEARNING_RATE = 1.0e-5
NUM_EPOCHS = 1
BATCH_SIZE = 1
EFFECTIVE_BATCH_SIZE = 128
BETA = 0.002
WARMUP_RATIO = 0.1

OUTPUT_DIR = f"/checkpoints/weighted-dpo/olmo3-7b-standard-dpo-{_variant}_beta_{BETA}_lr_{LEARNING_RATE}"

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
        "warmup_ratio": WARMUP_RATIO,
        "num_gpus": num_gpus,
        "reverse": REVERSE,
    }
    wandb.init(project="weighted-dpo", name=run_name, config=wandb_config)


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

        # Reverse DPO: swap so the model is trained to PREFER the originally
        # rejected response -> a deliberately "negative" model. (Both share the
        # same prompt, so swapping the completions is all that's needed.)
        if REVERSE:
            chosen_messages, rejected_messages = rejected_messages, chosen_messages

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
        optim="adamw_8bit",
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=max(1, int(0.2 * len(dataset) // (BATCH_SIZE * GRAD_ACCUM_STEPS * num_gpus))),
        save_total_limit=1,
        report_to="wandb",
        run_name=run_name,
        dataloader_num_workers=8,
        dataset_num_proc=16,  # parallelize TRL's internal "Tokenizing train dataset" step
        ddp_find_unused_parameters=False,
        loss_type="sigmoid",
        seed=SEED,
        data_seed=SEED,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
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
