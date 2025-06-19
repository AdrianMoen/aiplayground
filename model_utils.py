"""
Utility script to save and load fine-tuned models.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Save or load fine-tuned models")
    
    parser.add_argument("--action", type=str, choices=["save", "load"], required=True,
                        help="Whether to save or load a model")
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct",
                        help="Base model name")
    parser.add_argument("--adapter_path", type=str, default="./phi3-mini-finetuned",
                        help="Path to the trained adapter model")
    parser.add_argument("--output_path", type=str, default="./phi3-mini-finetuned-merged",
                        help="Path to save the merged model (for save action)")
    parser.add_argument("--load_8bit", action="store_true",
                        help="Load model in 8-bit precision")
    parser.add_argument("--load_4bit", action="store_true",
                        help="Load model in 4-bit precision")
    
    return parser.parse_args()

def save_model(args):
    """Save a fine-tuned model by merging adapter weights with base model."""
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load PEFT adapter
    if os.path.exists(args.adapter_path):
        print(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        raise ValueError(f"Adapter path {args.adapter_path} not found")
    
    # Merge adapter weights with base model
    print("Merging adapter weights with base model")
    model = model.merge_and_unload()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Save the merged model
    print(f"Saving merged model to {args.output_path}")
    model.save_pretrained(args.output_path, safe_serialization=True)
    tokenizer.save_pretrained(args.output_path)
    
    print("Model saved successfully!")

def load_model(args):
    """Load a fine-tuned model for verification."""
    if args.load_4bit:
        print("Loading model in 4-bit precision")
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        if os.path.exists(args.output_path):
            # If merged model exists, load it
            print(f"Loading merged model from {args.output_path}")
            model = AutoModelForCausalLM.from_pretrained(
                args.output_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.output_path, trust_remote_code=True)
        else:
            # Otherwise load base model + adapter
            print(f"Loading base model: {args.base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
            
            # Load PEFT adapter
            if os.path.exists(args.adapter_path):
                print(f"Loading adapter from {args.adapter_path}")
                model = PeftModel.from_pretrained(model, args.adapter_path)
            else:
                raise ValueError(f"Neither merged model nor adapter path found")
    
    elif args.load_8bit:
        print("Loading model in 8-bit precision")
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        
        if os.path.exists(args.output_path):
            print(f"Loading merged model from {args.output_path}")
            model = AutoModelForCausalLM.from_pretrained(
                args.output_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.output_path, trust_remote_code=True)
        else:
            print(f"Loading base model: {args.base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
            
            if os.path.exists(args.adapter_path):
                print(f"Loading adapter from {args.adapter_path}")
                model = PeftModel.from_pretrained(model, args.adapter_path)
            else:
                raise ValueError(f"Neither merged model nor adapter path found")
    
    else:
        # Load in FP16
        if os.path.exists(args.output_path):
            print(f"Loading merged model from {args.output_path}")
            model = AutoModelForCausalLM.from_pretrained(
                args.output_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(args.output_path, trust_remote_code=True)
        else:
            print(f"Loading base model: {args.base_model}")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            
            tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
            
            if os.path.exists(args.adapter_path):
                print(f"Loading adapter from {args.adapter_path}")
                model = PeftModel.from_pretrained(model, args.adapter_path)
            else:
                raise ValueError(f"Neither merged model nor adapter path found")
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model loaded successfully!")
    
    # Test the model with a simple prompt
    test_prompt = "Explain the concept of neural networks in simple terms:"
    
    # Format according to Phi-3's expected format
    formatted_prompt = f"<|user|>\n{test_prompt}\n<|assistant|>"
    
    print("\nRunning test generation...")
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nInput: {test_prompt}")
    print(f"Output: {response}")

def main():
    args = parse_args()
    
    if args.action == "save":
        save_model(args)
    elif args.action == "load":
        load_model(args)

if __name__ == "__main__":
    main()
