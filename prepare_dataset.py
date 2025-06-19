"""
Prepare custom training data for fine-tuning Phi-3 Mini.
"""

import argparse
import json
import os
import pandas as pd
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare custom data for fine-tuning")
    
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to input data file (JSON or CSV)")
    parser.add_argument("--output_file", type=str, default="./custom_dataset.json", 
                        help="Path to save the processed dataset")
    parser.add_argument("--format", type=str, choices=["alpaca", "chatml"], default="alpaca",
                        help="Format of the processed data")
    
    return parser.parse_args()

def convert_to_alpaca_format(data):
    """Convert data to Alpaca format (instruction, input, output)."""
    alpaca_data = []
    
    for item in data:
        # Handle different input formats flexibly
        instruction = item.get("instruction", item.get("prompt", ""))
        input_text = item.get("input", "")
        output = item.get("output", item.get("response", item.get("completion", "")))
        
        alpaca_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output
        })
    
    return alpaca_data

def convert_to_chatml_format(data):
    """Convert data to ChatML format with user/assistant messages."""
    chatml_data = []
    
    for item in data:
        # Handle different input formats flexibly
        instruction = item.get("instruction", item.get("prompt", ""))
        input_text = item.get("input", "")
        output = item.get("output", item.get("response", item.get("completion", "")))
        
        # Combine instruction and input for user message
        user_message = instruction
        if input_text:
            user_message += f"\n\n{input_text}"
        
        chatml_data.append({
            "messages": [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": output}
            ]
        })
    
    return chatml_data

def load_data(file_path):
    """Load data from JSON or CSV file."""
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return df.to_dict("records")
    else:
        raise ValueError("Unsupported file format. Please provide JSON or CSV file.")

def save_dataset(data, output_path, format_type):
    """Save the processed dataset."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(data)} examples to {output_path} in {format_type} format")

def main():
    args = parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}")
    data = load_data(args.input_file)
    
    # Convert to specified format
    if args.format == "alpaca":
        processed_data = convert_to_alpaca_format(data)
    elif args.format == "chatml":
        processed_data = convert_to_chatml_format(data)
    
    # Save processed data
    save_dataset(processed_data, args.output_file, args.format)
    
    # Create a Hugging Face dataset object (useful for verification)
    dataset = Dataset.from_pandas(pd.DataFrame(processed_data))
    print(f"Dataset created with {len(dataset)} examples")
    print(f"Dataset features: {dataset.features}")

if __name__ == "__main__":
    main()
