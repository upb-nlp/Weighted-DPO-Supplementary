"""
Standard KTO training on anonymous/Dolci-Instruct-DPO-en, starting from
allenai/Olmo-3-7B-Instruct-SFT, using the HuggingFace TRL KTOTrainer.

KTO (Kahneman-Tversky Optimization, arXiv:2402.01306) is an *unpaired* method:
each example is (prompt, completion, label) where label is a bool (desirable /
undesirable). We derive it from the paired Dolci preference data by unpairing
each row into two KTO rows:
    (prompt, chosen,   label=True)
    (prompt, rejected, label=False)
This gives a perfectly balanced set (equal desirable/undesirable), so
desirable_weight = undesirable_weight = 1.0.

KTO uses a reference model (like DPO/IPO), so this is NOT reference-free.

Important KTO-specific settings (per the TRL docs):
  - per_device_train_batch_size must be >= 4: KTO estimates a KL term per
    device batch, and a too-small per-step batch gives a poor KL estimate.
  - For beta = 0.1, the learning rate should not exceed 1e-6.

Dataset columns after preprocessing:
    prompt, completion, label
"""
from dotenv import load_dotenv
load_dotenv(".env")
import os
import torch
import wandb

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, set_seed
# KTO is top-level in TRL 0.29 (it moves to trl.experimental.kto only at v1.0).
from trl import KTOConfig, KTOTrainer

SEED = 42
set_seed(SEED)

# ---- Config ----
MODEL_NAME = "allenai/Olmo-3-7B-Instruct-SFT"
DATASET_NAME = "anonymous/Dolci-Instruct-DPO-en"
OUTPUT_DIR = "/checkpoints/weighted-dpo/olmo3-7b-standard-kto-full_beta_0.1_lr_1e-6"

MAX_SEQ_LENGTH = 2048     # KTOConfig only exposes max_length (prompt+completion)
LEARNING_RATE = 1.0e-6    # KTO: <= 1e-6 for beta=0.1 (LR-sensitive)
NUM_EPOCHS = 1
# KTO needs per-device batch >= 4 for a usable KL estimate (not 1 like the
# other methods). effective = BATCH_SIZE * num_gpus * GRAD_ACCUM_STEPS = 128.
BATCH_SIZE = 4
EFFECTIVE_BATCH_SIZE = 128
BETA = 0.1
# Balanced after unpairing (1 desirable + 1 undesirable per source pair), so the
# (desirable_weight * n_desirable) : (undesirable_weight * n_undesirable) ratio
# is 1:1 (recommended range is 1:1 .. 4:3).
DESIRABLE_WEIGHT = 1.0
UNDESIRABLE_WEIGHT = 1.0
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
        "desirable_weight": DESIRABLE_WEIGHT,
        "undesirable_weight": UNDESIRABLE_WEIGHT,
        "warmup_ratio": WARMUP_RATIO,
        "num_gpus": num_gpus,
        "loss_type": "kto",
    }
    wandb.init(project="weighted-dpo-olmo_clean", name=run_name, config=wandb_config)


# ---- Dataset preparation ----

def prepare_dataset(tokenizer):
    """Load the paired DPO dataset, format it, then unpair into KTO format."""
    dataset = load_dataset(DATASET_NAME, split="train")
    #dataset = dataset.shuffle(seed=SEED).select(range(200))

    def normalize_message(msg):
        # Strip extra fields (reasoning_content, function_call, audio, ...) and
        # keep only what the Olmo-3 chat template consumes.
        return {"role": msg["role"], "content": msg["content"] or ""}

    # Same boundary as the DPO/IPO/SimPO scripts: prompt ends right after the
    # last "<|im_start|>assistant" header, completion starts with the "\n" token.
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

    # Drop degenerate rows (empty completion either side, or prompt too long).
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
    if local_rank == 0 and n_before - len(dataset):
        print(f"Filtered {n_before - len(dataset)}/{n_before} degenerate rows")

    # Unpair: each (prompt, chosen, rejected) -> two KTO rows.
    def unpair(batch):
        prompts, completions, labels = [], [], []
        for p, c, r in zip(batch["prompt"], batch["chosen"], batch["rejected"]):
            prompts.append(p); completions.append(c); labels.append(True)
            prompts.append(p); completions.append(r); labels.append(False)
        return {"prompt": prompts, "completion": completions, "label": labels}

    dataset = dataset.map(
        unpair,
        batched=True,
        num_proc=4,
        desc="Unpairing for KTO",
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

    # SDPA attention (numerically stable bf16 softmax; avoids the eager per-head
    # mask issues we hit elsewhere).
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, config=config, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, config=config, dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="sdpa",
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

    training_args = KTOConfig(
        output_dir=OUTPUT_DIR,
        beta=BETA,
        desirable_weight=DESIRABLE_WEIGHT,
        undesirable_weight=UNDESIRABLE_WEIGHT,
        max_length=MAX_SEQ_LENGTH,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="linear",
        warmup_ratio=WARMUP_RATIO,
        optim="adamw_8bit",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        max_grad_norm=1.0,
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

    trainer = KTOTrainer(
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
