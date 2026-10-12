import argparse
import json
import os
from typing import List, Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen on WebGuard dataset with QLoRA/LoRA.")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="Base Qwen model ID.")
    parser.add_argument("--data_path", type=str, default="data/webguard/monitor_factory_train_a11y.json", help="Path to WebGuard training data.")
    parser.add_argument("--output_dir", type=str, default="models/webguard_qwen_lora", help="Directory to save trained LoRA adapter.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per device batch size.")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional sample cap for fast testing.")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum token length.")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    return parser.parse_args()

def load_webguard_dataset(data_path: str, max_samples: int = None):
    if not os.path.exists(data_path):
        from .download_webguard_data import download_all_webguard_datasets
        download_all_webguard_datasets()
        
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if max_samples and max_samples > 0:
        data = data[:max_samples]
        
    print(f"Loaded {len(data)} training samples from {data_path}.")
    return data

def train(args):
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    print(f"=== Starting WebGuard Fine-Tuning ===")
    print(f"Base Model: {args.model_id}")
    print(f"Output Dir: {args.output_dir}")
    print(f"CUDA Available: {torch.cuda.is_available()}")

    raw_data = load_webguard_dataset(args.data_path, args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format training texts using chat template
    formatted_texts = []
    for item in raw_data:
        messages = item.get("messages", [])
        if len(messages) >= 2:
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
            formatted_texts.append({"text": chat_text})

    dataset = Dataset.from_list(formatted_texts)

    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"],
            max_length=args.max_length,
            truncation=True,
            padding=False
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Load model with optional 4-bit quantization
    kwargs = {"trust_remote_code": True}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        try:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
            print("Using 4-bit BitsAndBytes quantization (QLoRA).")
        except Exception as e:
            print(f"BitsAndBytes not available, using float16: {e}")
            kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **kwargs)

    if torch.cuda.is_available() and hasattr(model, "is_loaded_in_4bit") and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        optim="paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch",
        report_to="none",
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    print("Starting training loop...")
    trainer.train()

    print(f"Saving trained LoRA adapter to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training complete!")

if __name__ == "__main__":
    args = parse_args()
    train(args)
