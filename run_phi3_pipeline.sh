#!/bin/bash

echo "Phi-3 Mini Fine-tuning Pipeline"
echo "=============================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed or not in your PATH."
    echo "Please install Python 3.8 or higher and try again."
    exit 1
fi

menu() {
    echo "Choose an option:"
    echo "1. Setup environment"
    echo "2. Prepare custom dataset"
    echo "3. Fine-tune model"
    echo "4. Save merged model"
    echo "5. Run REPL interface"
    echo "6. Run full pipeline"
    echo "7. Exit"
    echo
    
    read -p "Enter your choice (1-7): " choice
    
    case $choice in
        1) setup ;;
        2) prepare ;;
        3) finetune ;;
        4) save_model ;;
        5) repl ;;
        6) full ;;
        7) exit 0 ;;
        *) echo "Invalid choice. Please try again."; menu ;;
    esac
}

setup() {
    echo "Setting up environment..."
    python3 run_pipeline.py --setup_env
    echo
    menu
}

prepare() {
    echo "Preparing custom dataset..."
    read -p "Enter path to input data file: " input_file
    read -p "Enter path to save processed dataset (or press Enter for default): " output_file
    read -p "Enter format (alpaca or chatml) (or press Enter for default): " format
    
    cmd="python3 run_pipeline.py --prepare_data --input_file \"$input_file\""
    if [ ! -z "$output_file" ]; then
        cmd="$cmd --output_file \"$output_file\""
    fi
    if [ ! -z "$format" ]; then
        cmd="$cmd --data_format $format"
    fi
    
    eval $cmd
    echo
    menu
}

finetune() {
    echo "Fine-tuning the model..."
    read -p "Enter dataset path or name (or press Enter to use the prepared dataset): " dataset
    read -p "Enter output directory for fine-tuned model (or press Enter for default): " output_dir
    read -p "Enter number of epochs (or press Enter for default): " epochs
    read -p "Enter batch size (or press Enter for default): " batch_size
    read -p "Use Weights & Biases for logging? (y/n) (default: n): " use_wandb
    
    cmd="python3 run_pipeline.py --run_finetuning"
    if [ ! -z "$dataset" ]; then
        cmd="$cmd --dataset \"$dataset\""
    fi
    if [ ! -z "$output_dir" ]; then
        cmd="$cmd --output_dir \"$output_dir\""
    fi
    if [ ! -z "$epochs" ]; then
        cmd="$cmd --epochs $epochs"
    fi
    if [ ! -z "$batch_size" ]; then
        cmd="$cmd --batch_size $batch_size"
    fi
    if [ "$use_wandb" = "y" ] || [ "$use_wandb" = "Y" ]; then
        cmd="$cmd --use_wandb"
    fi
    
    eval $cmd
    echo
    menu
}

save_model() {
    echo "Saving merged model..."
    read -p "Enter base model name (or press Enter for default): " base_model
    read -p "Enter adapter path (or press Enter for default): " adapter_path
    read -p "Enter merged model directory (or press Enter for default): " merged_model_dir
    
    cmd="python3 run_pipeline.py --save_merged_model"
    if [ ! -z "$base_model" ]; then
        cmd="$cmd --base_model \"$base_model\""
    fi
    if [ ! -z "$adapter_path" ]; then
        cmd="$cmd --output_dir \"$adapter_path\""
    fi
    if [ ! -z "$merged_model_dir" ]; then
        cmd="$cmd --merged_model_dir \"$merged_model_dir\""
    fi
    
    eval $cmd
    echo
    menu
}

repl() {
    echo "Running REPL interface..."
    read -p "Use 4-bit quantization? (y/n) (default: n): " use_4bit
    read -p "Use web UI? (y/n) (default: n): " web_ui
    read -p "Use merged model instead of adapter? (y/n) (default: n): " use_merged
    
    cmd="python3 run_pipeline.py --run_repl"
    if [ "$use_4bit" = "y" ] || [ "$use_4bit" = "Y" ]; then
        cmd="$cmd --use_4bit"
    fi
    if [ "$web_ui" = "y" ] || [ "$web_ui" = "Y" ]; then
        cmd="$cmd --web_ui"
    fi
    if [ "$use_merged" = "y" ] || [ "$use_merged" = "Y" ]; then
        cmd="$cmd --use_merged_model"
    fi
    
    eval $cmd
    echo
    menu
}

full() {
    echo "Running full pipeline..."
    read -p "Enter path to input data file: " input_file
    read -p "Enter number of epochs (or press Enter for default): " epochs
    read -p "Use web UI for REPL? (y/n) (default: n): " web_ui
    read -p "Save merged model? (y/n) (default: n): " save_merged
    
    cmd="python3 run_pipeline.py --setup_env --prepare_data --input_file \"$input_file\" --run_finetuning --run_repl"
    if [ ! -z "$epochs" ]; then
        cmd="$cmd --epochs $epochs"
    fi
    if [ "$web_ui" = "y" ] || [ "$web_ui" = "Y" ]; then
        cmd="$cmd --web_ui"
    fi
    if [ "$save_merged" = "y" ] || [ "$save_merged" = "Y" ]; then
        cmd="$cmd --save_merged_model"
    fi
    
    eval $cmd
    echo
    menu
}

# Start the menu
menu
