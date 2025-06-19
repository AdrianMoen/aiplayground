"""
Fine-tune the Phi-3 Mini model using QLoRA.
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from trl import SFTTrainer
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Phi-3 Mini using QLoRA")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct", 
                        help="Base model to fine-tune")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", 
                        help="Dataset to use for fine-tuning")
    parser.add_argument("--output_dir", type=str, default="./phi3-mini-finetuned", 
                        help="Directory to save the fine-tuned model")
    
    # Training arguments
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, 
                        help="Gradient accumulation steps")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Additional arguments
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="phi3-mini-finetuning", 
                        help="wandb project name")
    
    return parser.parse_args()

def prepare_dataset(dataset_name, tokenizer, max_length):
    """Prepare the dataset for training."""
    dataset = load_dataset(dataset_name)
    
    # You may need to adjust this formatting based on your specific dataset structure
    def format_instruction(example):
        return f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    
    def tokenize_function(examples):
        formatted_texts = [format_instruction({"instruction": inst, "input": inp, "output": out}) 
                           for inst, inp, out in zip(examples["instruction"], 
                                                    examples["input"] if "input" in examples else [""] * len(examples["instruction"]), 
                                                    examples["output"])]
        return tokenizer(formatted_texts, padding="max_length", truncation=True, max_length=max_length)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    
    return tokenized_dataset

def init_model(base_model_name):
    """Initialize the model with QLoRA configuration."""
    
    # Configure quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization config
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def create_peft_config(args):
    """Create Parameter-Efficient Fine-Tuning config."""
    
    target_modules = [
        "q_proj",
        "k_proj", 
        "v_proj", 
        "o_proj", 
        "gate_proj", 
        "up_proj", 
        "down_proj"
    ]
    
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    return config

def train(args):
    """Train the model using QLoRA."""
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(project=args.wandb_project)
    
    # Initialize model and tokenizer
    print(f"Loading base model: {args.base_model}")
    model, tokenizer = init_model(args.base_model)
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Create LoRA config
    peft_config = create_peft_config(args)
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Prepare dataset
    print(f"Loading dataset: {args.dataset}")
    train_dataset = prepare_dataset(args.dataset, tokenizer, args.max_seq_length)["train"]
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"phi3-mini-qlora-{args.lora_r}" if args.use_wandb else None,
    )
    
    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        tokenizer=tokenizer,
        peft_config=peft_config,
        dataset_text_field="text",  # Adjust based on your dataset
        max_seq_length=args.max_seq_length,
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    # End wandb run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    train(args)
    print("Training completed!")
