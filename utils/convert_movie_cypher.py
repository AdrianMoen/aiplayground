"""
Convert movie_nl_cypher_1000.csv to a format suitable for fine-tuning Phi-3 Mini.
"""

import argparse
import pandas as pd
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Convert movie_nl_cypher_1000.csv to fine-tuning format")
    
    parser.add_argument("--input_file", type=str, default="data/movie_nl_cypher_1000.csv", 
                        help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, default="data/movie_nl_cypher_dataset.json", 
                        help="Path to save the processed dataset")
    parser.add_argument("--format", type=str, choices=["alpaca", "chatml"], default="alpaca",
                        help="Format of the processed data")
    parser.add_argument("--split", type=float, default=0.1,
                        help="Percentage of data to use for validation (0.0-1.0)")
    
    return parser.parse_args()

def convert_to_alpaca_format(df):
    """Convert dataframe to Alpaca format (instruction, input, output)."""
    alpaca_data = []
    
    for _, row in df.iterrows():
        instruction = "Convert the following natural language query about movies to a Cypher query for Neo4j graph database."
        input_text = row['utterance']
        output = row['cypher']
        
        alpaca_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output
        })
    
    return alpaca_data

def convert_to_chatml_format(df):
    """Convert dataframe to ChatML format."""
    chatml_data = []
    
    for _, row in df.iterrows():
        messages = [
            {"role": "system", "content": "You are a helpful assistant that converts natural language queries about movies into Cypher queries for Neo4j graph database."},
            {"role": "user", "content": row['utterance']},
            {"role": "assistant", "content": row['cypher']}
        ]
        
        chatml_data.append({"messages": messages})
    
    return chatml_data

def main():
    args = parse_args()
    
    print(f"Reading data from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    print(f"Found {len(df)} examples")
    
    # Create train and validation splits
    val_size = int(len(df) * args.split)
    train_df = df.iloc[:-val_size] if val_size > 0 else df
    val_df = df.iloc[-val_size:] if val_size > 0 else None
    
    print(f"Converting to {args.format} format...")
    if args.format == "alpaca":
        train_data = convert_to_alpaca_format(train_df)
        val_data = convert_to_alpaca_format(val_df) if val_df is not None else None
    else:  # chatml
        train_data = convert_to_chatml_format(train_df)
        val_data = convert_to_chatml_format(val_df) if val_df is not None else None
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Write training data
    train_output = args.output_file
    with open(train_output, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2)
    
    print(f"Saved {len(train_data)} training examples to {train_output}")
    
    # Write validation data if it exists
    if val_data is not None:
        val_output = args.output_file.replace('.json', '_val.json')
        with open(val_output, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2)
        
        print(f"Saved {len(val_data)} validation examples to {val_output}")
    
    print("\nTo use this dataset for fine-tuning, run:")
    print(f"python finetune_phi3_qlora.py --dataset {train_output} --output_dir model_output/phi3-mini-cypher")
    
    if val_data is not None:
        print(f"\nValidation dataset saved to {val_output}")

if __name__ == "__main__":
    main()
