"""
Enhanced fine-tuning script with validation support and better training configuration.
"""

import os
import argparse
import torch
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Fine-tune Phi-3 Mini using QLoRA")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--train_dataset", type=str, default="./data/splits/train.json")
    parser.add_argument("--val_dataset", type=str, default="./data/splits/validation.json")
    parser.add_argument("--output_dir", type=str, default="./phi3-mini-cypher-v2")
    parser.add_argument("--merged_model_dir", type=str, default="./phi3-mini-cypher-v2-merged")
    
    # Training arguments - Extended for longer training
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=12)  # Increased for longer training
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    # LoRA arguments - Enhanced
    parser.add_argument("--lora_r", type=int, default=16)  # Increased
    parser.add_argument("--lora_alpha", type=int, default=32)  # Increased
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Validation and monitoring
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="phi3-cypher-enhanced")
    parser.add_argument("--save_merged_model", action="store_true")
    
    return parser.parse_args()

def load_dataset_from_json(file_path):
    """Load dataset from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return Dataset.from_list(data)

# def format_instruction(example):
#     """Format training example."""
#     return f"""<|user|>
# {example['input']}
# <|assistant|>
# {example['output']}<|end|>"""


def prepare_datasets(train_file, val_file, tokenizer, max_length):
    """Prepare training and validation datasets."""
    
    # Load datasets
    train_dataset = load_dataset_from_json(train_file)
    val_dataset = load_dataset_from_json(val_file) if val_file else None
    
    def format_instruction(example):
        """Format training example - FIXED to match original format."""
        return f"""### Instruction:
{example['instruction']}

### Input:
{example.get('input', '')}

### Response:
{example['output']}"""
    
    def tokenize_function(examples):
        """FIXED: Handle batch processing correctly."""
        formatted_texts = []
        
        # Handle batch processing correctly
        for i in range(len(examples["instruction"])):
            instruction = examples["instruction"][i]
            output = examples["output"][i]
            
            # Handle input field (may not exist in all examples)
            input_text = examples["input"][i] if "input" in examples and examples["input"][i] else ""
            
            formatted_text = format_instruction({
                "instruction": instruction,
                "input": input_text,
                "output": output
            })
            formatted_texts.append(formatted_text)
        
        # Tokenize
        return tokenizer(
            formatted_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
    
    # Tokenize datasets
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    val_tokenized = None
    if val_dataset:
        val_tokenized = val_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=val_dataset.column_names
        )
    
    return train_tokenized, val_tokenized

# def prepare_datasets(train_file, val_file, tokenizer, max_length):
#     """Prepare training and validation datasets."""
    
#     # Load datasets
#     train_dataset = load_dataset_from_json(train_file)
#     val_dataset = load_dataset_from_json(val_file) if val_file else None
    
#     def tokenize_function(examples):
#         # Format the text
#         texts = [format_instruction(ex) for ex in examples]
        
#         # Tokenize
#         tokenized = tokenizer(
#             texts,
#             padding="max_length",
#             truncation=True,
#             max_length=max_length,
#             return_tensors="pt"
#         )
        
#         # For causal LM, labels are the same as input_ids
#         tokenized["labels"] = tokenized["input_ids"].clone()
        
#         return tokenized
    
#     # Tokenize datasets
#     train_tokenized = train_dataset.map(
#         tokenize_function,
#         batched=True,
#         remove_columns=train_dataset.column_names
#     )
    
#     val_tokenized = None
#     if val_dataset:
#         val_tokenized = val_dataset.map(
#             tokenize_function,
#             batched=True,
#             remove_columns=val_dataset.column_names
#         )
    
#     return train_tokenized, val_tokenized

def init_model(base_model_name):
    """Initialize model with optimized configuration."""
    
    # Enhanced quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # Better than float16
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Optimized attention
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def create_peft_config(args):
    """Create enhanced PEFT configuration."""
    
    # Target more modules for better adaptation
    target_modules = [
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # "lm_head"  # Added language modeling head
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
    """Enhanced training with validation support."""
    
    # Initialize wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )
    
    # Initialize model and tokenizer
    print(f"Loading base model: {args.base_model}")
    model, tokenizer = init_model(args.base_model)
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Create and apply PEFT config
    peft_config = create_peft_config(args)
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(
        args.train_dataset, 
        args.val_dataset, 
        tokenizer, 
        args.max_seq_length
    )
    
    print(f"Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    
    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        
        # Training configuration
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        
        # Learning rate and optimization
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        
        # Precision and optimization
        fp16=True,
        bf16=False,  # Better for RTX 4090
        
        # Evaluation and monitoring
        eval_strategy="steps" if val_dataset else "no",# COMMENT THIS OUT AGAIN IF IT FAILS!
        # eval_strategy="no",
        eval_steps=args.eval_steps if val_dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=25,
        
        # Model saving
        # save_total_limit=3,
        #load_best_model_at_end=True if val_dataset else False,
        #metric_for_best_model="eval_loss" if val_dataset else None,
        
        # Reporting
        report_to="wandb" if args.use_wandb else "none",
        run_name=f"phi3-cypher-r{args.lora_r}-{args.epochs}ep",
        
        # Memory optimization
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Callbacks
    callbacks = []
    # if val_dataset and args.early_stopping_patience > 0:
    #     callbacks.append(
    #         EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
    #     )
    ##NO EARLY STOPPING; SO COMMENTED THIS OUT
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )
    
    # Train
    print("🚀 Starting enhanced training...")
    trainer.train()
    
    # Save model
    print(f"💾 Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    
    # Save merged model if requested
    if args.save_merged_model:
        print(f"🔗 Saving merged model to {args.merged_model_dir}")
        model = model.merge_and_unload()
        model.save_pretrained(args.merged_model_dir)
        tokenizer.save_pretrained(args.merged_model_dir)
    
    # End wandb
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    train(args)
    print("✅ Enhanced training completed!")