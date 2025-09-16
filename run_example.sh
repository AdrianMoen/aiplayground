#!/bin/bash

echo "Running full Phi-3 Mini fine-tuning pipeline with example dataset..."
echo "This will:"
echo "  1. Set up the environment"
echo "  2. Fine-tune Phi-3 Mini on the example dataset"
echo "  3. Save the fine-tuned model"
echo "  4. Launch the REPL interface"
echo

input_file="example_dataset.json"
epochs=3
batch_size=4
output_dir="./phi3-mini-finetuned"
merged_dir="./phi3-mini-finetuned-merged"

echo "Step 1: Setting up environment..."
python3 run_pipeline.py --setup_env

echo "Step 2: Fine-tuning on example dataset..."
python3 run_pipeline.py --run_finetuning --dataset "$input_file" --epochs $epochs --batch_size $batch_size --output_dir "$output_dir"

echo "Step 3: Saving merged model..."
python3 run_pipeline.py --save_merged_model --output_dir "$output_dir" --merged_model_dir "$merged_dir"

echo "Step 4: Launching REPL interface..."
python3 run_pipeline.py --run_repl --use_merged_model --merged_model_dir "$merged_dir" --web_ui

echo "Done!"
