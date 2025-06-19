"""
Extremely simplified command-line REPL for the fine-tuned Phi-3 Mini model for Cypher generation.
Uses the legacy generation API to avoid compatibility issues.
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
    parser.add_argument("--max_tokens", type=int, default=512, 
                        help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3, 
                        help="Sampling temperature")
    
    return parser.parse_args()

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

def main():
    args = parse_args()
    
    # Load base model
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Load PEFT adapter
    if os.path.exists(args.adapter_path):
        print(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        # Merge adapter weights for faster inference
        print("Merging adapter weights with base model")
        model = model.merge_and_unload()
    else:
        print(f"Warning: Adapter path {args.adapter_path} not found. Using base model only.")
    
    # Set model to evaluation mode
    model.eval()
    
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
            # Tokenize the prompt
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Simple greedy generation
            with torch.no_grad():
                outputs = model.forward(
                    **inputs, 
                    max_new_tokens=args.max_tokens, 
                    return_dict_in_generate=True,
                    output_scores=False,
                )
                
                token_ids = outputs.sequences[0]
                
            # Decode generated tokens
            generated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            
            # Extract assistant's response
            if "<|assistant|>" in generated_text:
                response = generated_text.split("<|assistant|>")[-1].strip()
            else:
                response = generated_text.strip()
            
            print("-------------")
            print(response)
            print("-------------")
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            print("Trying an alternative approach...")
            
            try:
                # Try a simpler approach with just input IDs
                input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=args.max_tokens,
                        do_sample=(args.temperature > 0),
                        temperature=args.temperature,
                        num_return_sequences=1,
                    )
                
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Extract assistant's response
                if "<|assistant|>" in generated_text:
                    response = generated_text.split("<|assistant|>")[-1].strip()
                else:
                    response = generated_text.strip()
                
                print("-------------")
                print(response)
                print("-------------")
            except Exception as e2:
                print(f"Alternative approach also failed: {str(e2)}")

if __name__ == "__main__":
    main()
