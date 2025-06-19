"""
Simple command-line REPL for the fine-tuned Phi-3 Mini model for Cypher generation.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser(description="Command-line REPL for Phi-3 Mini Cypher Generator")
    
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct", 
                        help="Base model name")
    parser.add_argument("--adapter_path", type=str, default="./model_output/phi3-mini-cypher", 
                        help="Path to the trained adapter model")
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
        )
    else:
        # Load in fp16 if not using quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, 
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
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

def generate_text(model, tokenizer, prompt, max_length=1024, temperature=0.3, top_p=0.9):
    """Simple text generation function."""
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Simple greedy generation to avoid compatibility issues
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature if temperature > 0 else 1.0,
            top_p=top_p if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the assistant's response
    if "<|assistant|>" in prompt:
        parts = full_text.split("<|assistant|>")
        if len(parts) > 1:
            return parts[-1].strip()
    
    return full_text

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model(args)
    model.eval()  # Set model to evaluation mode
    
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
        try:
            response = generate_text(
                model, 
                tokenizer, 
                prompt, 
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            print("-------------")
            print(response)
            print("-------------")
        except Exception as e:
            print(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()
