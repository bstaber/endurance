"""Script to fine-tune a LLaMA/Mistral model using QLoRA and PEFT."""

import torch
from datasets import load_dataset  # noqa F401
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,  # noqa F401
    Trainer,  # noqa F401
    TrainingArguments,
)

base_model_id = "mistralai/Mistral-7B-v0.1"  # or "meta-llama/Llama-3-8B"

# --- 4-bit quantization config for QLoRA ---
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- Model & tokenizer ---
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_cfg,
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"

# Ensure model knows pad token and disable cache with checkpointing
model.config.pad_token_id = tok.pad_token_id
model.config.use_cache = False  # required when gradient_checkpointing=True

# Prepare for k-bit training (QLoRA best practice)
model = prepare_model_for_kbit_training(model)

# --- LoRA config ---
peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # drop to 8 if you need more headroom
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # LLaMA/Mistral naming
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()

# --- Data ---
# ds = load_dataset("tatsu-lab/alpaca", split="train[:1%]")  # example only

# def fmt(example):
#     text = (
#         "### Instruction:\n" + example["instruction"] +
#         "\n\n### Input:\n" + (example["input"] or "") +
#         "\n\n### Response:\n" + example["output"]
#     )
#     enc = tok(text, truncation=True, max_length=1024, padding="max_length")
#     enc["labels"] = enc["input_ids"].copy()  # causal LM: predict next token
#     return enc

# ds = ds.map(fmt, remove_columns=ds.column_names)

# collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

# --- Training ---
args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=1,  # micro-batch
    gradient_accumulation_steps=8,  # effective batch = 8
    learning_rate=2e-4,
    num_train_epochs=2,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_8bit",  # bitsandbytes optimizer (great with QLoRA)
    max_grad_norm=0.3,
    report_to="none",  # or "wandb"/"tensorboard"
)

# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=ds,
#     data_collator=collator,
# )
# trainer.train()
