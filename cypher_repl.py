"""
Simplified command-line REPL for the fine-tuned Phi-3 Mini model.
Specifically designed for natural language to Cypher query conversion.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
from utils.model_management import load_model_from_saved

def parse_args():
    parser = argparse.ArgumentParser(description="Cypher REPL for fine-tuned Phi-3 Mini")
    
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct", 
                        help="Base model name")
    parser.add_argument("--adapter_path", type=str, default="./phi3-mini-finetuned", 
                        help="Path to the trained adapter model")
    parser.add_argument("--saved_model_dir", type=str, default=None,
                        help="Path to a saved model directory with metadata (overrides base_model and adapter_path)")
    parser.add_argument("--use_4bit", action="store_true", 
                        help="Whether to load the model in 4-bit precision")
    parser.add_argument("--max_length", type=int, default=1024, 
                        help="Maximum length for generation")
    parser.add_argument("--temperature", type=float, default=0.3, 
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p sampling parameter")
    
    return parser.parse_args()

def load_model(args):
    """Load the fine-tuned model."""
    if args.saved_model_dir and os.path.exists(args.saved_model_dir):
        print(f"Loading model from saved directory: {args.saved_model_dir}")
        model, tokenizer, metadata = load_model_from_saved(args.saved_model_dir, args.use_4bit)
        return model, tokenizer
    
    print(f"Loading base model: {args.base_model}")
    
    # Set up quantization if requested
    if args.use_4bit:
        print("Using 4-bit quantization")
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,  # Disable KV cache to avoid DynamicCache errors
        )
    else:
        # Load in fp16 if not using quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, 
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False,  # Disable KV cache to avoid DynamicCache errors
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load PEFT adapter
    if os.path.exists(args.adapter_path):
        print(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        print(f"Warning: Adapter path {args.adapter_path} not found. Using base model only.")
    
    # Merge adapter weights with base model for faster inference (if not using 4-bit)
    if not args.use_4bit and os.path.exists(args.adapter_path):
        print("Merging adapter weights with base model")
        model = model.merge_and_unload()
    
    return model, tokenizer

def format_prompt(user_input):
    """Format the prompt for Cypher generation."""
    system_prompt = """You are a Neo4j Cypher query generator. 
Your task is to convert natural language questions about movies into Cypher queries.
Only return the Cypher query without any explanation or additional text.
The database has a movie graph model with the following node types:
- Movie (properties: title, released, tagline)
- Person (properties: name, born)
- Genre (properties: name)
And the following relationship types:
- ACTED_IN (Person to Movie, properties: roles)
- DIRECTED (Person to Movie)
- IN_GENRE (Movie to Genre)"""
    
    prompt = f"""<|system|>
{system_prompt}
<|user|>
{user_input}
<|assistant|>
"""
    return prompt

def generate_response(model, tokenizer, prompt, args):
    """Generate a response from the model using a simpler approach."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(inputs.input_ids)
    
    # Use a simpler generation approach
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": 512,  # Control only new tokens, not total length
        "do_sample": args.temperature > 0,
        "use_cache": False,     # Disable KV cache to avoid DynamicCache errors
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    # Only add temperature and top_p if we're sampling
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    
    # Generate response with direct generation
    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
    
    # Decode the response
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response after the prompt
    # Find the last occurrence of the prompt parts
    if "<|assistant|>" in prompt:
        parts = generated_text.split("<|assistant|>")
        if len(parts) > 1:
            return parts[-1].strip()
    
    # Fallback: return everything after the user's input
    user_input_part = prompt.split("<|user|>")[-1].split("<|assistant|>")[0].strip()
    if user_input_part in generated_text:
        return generated_text.split(user_input_part)[-1].strip()
    
    return generated_text

def cli_interface(model, tokenizer, args):
    """Command-line interface for interacting with the model."""
    print("\nPhi-3 Mini Cypher Generator")
    print("---------------------------")
    print("Type 'exit' or 'quit' to end the session")
    print("Enter your natural language question about movies to get a Cypher query\n")
    
    while True:
        user_input = input("\nQuestion: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        prompt = format_prompt(user_input)
        
        print("\nGenerating Cypher query...")
        response = generate_response(model, tokenizer, prompt, args)
        
        print("-------------")
        print(response)
        print("-------------")

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model(args)
    
    # Prepare model for inference
    model.eval()
    
    # Launch CLI interface
    cli_interface(model, tokenizer, args)

if __name__ == "__main__":
    main()
