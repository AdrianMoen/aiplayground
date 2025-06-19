"""
Combined script to run the full Phi-3 Mini fine-tuning pipeline.
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run the full Phi-3 Mini fine-tuning pipeline")
    
    # Setup arguments
    parser.add_argument("--setup_env", action="store_true", 
                        help="Setup the virtual environment and install dependencies")
    
    # Dataset arguments
    parser.add_argument("--prepare_data", action="store_true", 
                        help="Prepare custom training data")
    parser.add_argument("--input_file", type=str, 
                        help="Path to input data file for custom dataset")
    parser.add_argument("--output_file", type=str, default="./custom_dataset.json", 
                        help="Path to save the processed dataset")
    parser.add_argument("--data_format", type=str, choices=["alpaca", "chatml"], default="alpaca",
                        help="Format of the processed data")
    
    # Fine-tuning arguments
    parser.add_argument("--run_finetuning", action="store_true", 
                        help="Run the fine-tuning process")
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct", 
                        help="Base model to fine-tune")
    parser.add_argument("--dataset", type=str, 
                        help="Dataset to use for fine-tuning (HF dataset or path to custom dataset)")
    parser.add_argument("--output_dir", type=str, default="./phi3-mini-finetuned", 
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size")
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Whether to use wandb for logging")
    
    # REPL arguments
    parser.add_argument("--run_repl", action="store_true", 
                        help="Run the REPL interface after fine-tuning")
    parser.add_argument("--use_4bit", action="store_true", 
                        help="Whether to load the model in 4-bit precision for REPL")
    parser.add_argument("--web_ui", action="store_true", 
                        help="Launch Gradio web interface instead of CLI")
    
    # Model saving/loading arguments
    parser.add_argument("--save_merged_model", action="store_true",
                        help="Save the fine-tuned model by merging adapter with base model")
    parser.add_argument("--merged_model_dir", type=str, default="./phi3-mini-finetuned-merged",
                        help="Directory to save the merged model")
    parser.add_argument("--use_merged_model", action="store_true",
                        help="Use the merged model instead of adapter for REPL")
    
    return parser.parse_args()

def setup_environment():
    """Setup the virtual environment and install dependencies."""
    venv_path = Path("./venv")
    
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Determine the pip path based on the OS
    if os.name == "nt":  # Windows
        pip_path = venv_path / "Scripts" / "pip"
    else:  # Unix/Linux/MacOS
        pip_path = venv_path / "bin" / "pip"
    
    print("Installing dependencies...")
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"])
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"])
    
    print("Environment setup complete!")

def prepare_data(args):
    """Prepare custom training data."""
    if not args.input_file:
        print("Error: --input_file is required when using --prepare_data")
        return False
    
    print("Preparing custom training data...")
    cmd = [
        "python", "prepare_dataset.py",
        "--input_file", args.input_file,
        "--output_file", args.output_file,
        "--format", args.data_format
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("Data preparation complete!")
        return True
    else:
        print("Error during data preparation.")
        return False

def run_finetuning(args):
    """Run the fine-tuning process."""
    # Use either the provided dataset or the custom dataset
    dataset = args.dataset if args.dataset else args.output_file
    
    print(f"Running fine-tuning using dataset: {dataset}")
    cmd = [
        "python", "finetune_phi3_qlora.py",
        "--base_model", args.base_model,
        "--dataset", dataset,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
    ]
    
    if args.use_wandb:
        cmd.append("--use_wandb")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("Fine-tuning complete!")
        return True
    else:
        print("Error during fine-tuning.")
        return False

def save_merged_model(args):
    """Save the fine-tuned model by merging adapter with base model."""
    print("Saving merged model...")
    cmd = [
        "python", "model_utils.py",
        "--action", "save",
        "--base_model", args.base_model,
        "--adapter_path", args.output_dir,
        "--output_path", args.merged_model_dir
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("Model saving complete!")
        return True
    else:
        print("Error during model saving.")
        return False

def run_repl(args):
    """Run the REPL interface."""
    print("Starting REPL interface...")
    cmd = [
        "python", "phi3_repl.py",
        "--base_model", args.base_model,
    ]
    
    # Use either the adapter path or the merged model
    if args.use_merged_model:
        cmd.extend(["--adapter_path", args.merged_model_dir])
    else:
        cmd.extend(["--adapter_path", args.output_dir])
    
    if args.use_4bit:
        cmd.append("--use_4bit")
    
    if args.web_ui:
        cmd.append("--web_ui")
    
    subprocess.run(cmd)

def main():
    args = parse_args()
    
    if args.setup_env:
        setup_environment()
    
    if args.prepare_data:
        success = prepare_data(args)
        if not success:
            return
    
    if args.run_finetuning:
        success = run_finetuning(args)
        if not success:
            return
    
    if args.save_merged_model:
        success = save_merged_model(args)
        if not success:
            return
    
    if args.run_repl:
        run_repl(args)

if __name__ == "__main__":
    main()
