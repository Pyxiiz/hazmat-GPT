# train_lora.py
import os
import json
from dataclasses import dataclass, field
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

@dataclass
class Config:
    base_model: str = "bigcode/starcoder"  # change to your chosen model
    dataset_path: str = "data/merged.jsonl"
    output_dir: str = "out-lora"
    per_device_train_batch_size: int = 2
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_seq_length: int = 1024
    use_4bit: bool = True

def build_tokenized_dataset(tokenizer, path, max_length):
    raw = load_dataset("json", data_files=path, split="train")
    def preprocess(ex):
        # combine prompt and completion and target for causal LM
        prompt = ex["prompt"].rstrip() + "\n\n### Solution\n"
        target = ex["completion"].lstrip()
        text = prompt + target + tokenizer.eos_token
        tokenized = tokenizer(text, truncation=True, max_length=max_length)
        return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}
    tokenized = raw.map(preprocess, remove_columns=raw.column_names)
    return tokenized

def main():
    cfg = Config()
    # quantization config if using bitsandbytes
    bnb_config = None
    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    print("Loading tokenizer and model:", cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16 if cfg.use_4bit else None,
        low_cpu_mem_usage=True,
    )

    # Prepare model for k-bit training if quantized
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # PEFT LoRA config
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    dataset = build_tokenized_dataset(tokenizer, cfg.dataset_path, cfg.max_seq_length)
    print("Dataset size:", len(dataset))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        dataloader_pin_memory=True,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
    )

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

if __name__ == "__main__":
    main()
