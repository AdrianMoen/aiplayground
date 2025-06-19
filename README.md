# Phi-3 Mini QLoRA Fine-tuning Environment

This project enables fine-tuning of the Microsoft Phi-3 Mini model using Quantized Low-Rank Adaptation (QLoRA) and provides a REPL (Read-Eval-Print Loop) interface for interacting with the fine-tuned model. The project is designed to work optimally with an NVIDIA RTX 4090 GPU.

## System Requirements

- Windows 10/11 (64-bit)
- NVIDIA RTX 4090 GPU (24GB VRAM)
- 32GB+ RAM recommended
- 100GB+ free disk space
- Python 3.10 or newer

## Project Structure

```
aiplayground/
├── setup.bat                  # Main setup menu
├── run_example_pipeline.bat   # Run example fine-tuning pipeline
├── phi3_repl.py               # Interactive REPL for using the model
├── finetune_phi3_qlora.py     # Fine-tuning script
├── requirements.txt           # Python dependencies
├── setup/
│   ├── cuda/                  # CUDA and GPU setup scripts
│   │   ├── fix_cuda_bnb.bat
│   │   ├── install_cuda.bat
│   │   ├── install_pytorch.bat
│   │   └── update_drivers.bat
│   ├── utils/                 # Utility scripts
│   │   └── check_system.bat
│   └── tests/                 # Test scripts
│       ├── full_environment_test.bat
│       ├── test_phi3_load.bat
│       ├── verify_bitsandbytes.bat
│       ├── verify_cuda.bat
│       └── verify_pytorch.bat
└── docs/
    ├── CUDA_SETUP_GUIDE.md
    └── rtx4090_optimization_tips.md
```

## Environment Setup

### Using the Setup Menu

The easiest way to set up your environment is to use the provided setup menu:

```bash
setup.bat
```

This interactive menu allows you to:
1. Check system requirements
2. Update NVIDIA drivers
3. Install CUDA Toolkit
4. Install PyTorch with CUDA
5. Fix bitsandbytes CUDA support
6. Verify your installation
7. Test the Phi-3 model loading
8. Run the full environment test
9. Run the example fine-tuning pipeline

### Manual Setup Steps

If you prefer to run the steps manually:

1. Check system requirements:
   ```bash
   setup\utils\check_system.bat
   ```

2. Update NVIDIA drivers:
   ```bash
   setup\cuda\update_drivers.bat
   ```

3. Install CUDA Toolkit:
   ```bash
   setup\cuda\install_cuda.bat
   ```

4. Install PyTorch with CUDA support:
   ```bash
   setup\cuda\install_pytorch.bat
   ```

5. Fix bitsandbytes CUDA support:
   ```bash
   setup\cuda\fix_cuda_bnb.bat
   ```

6. Verify your installation:
   ```bash
   setup\tests\full_environment_test.bat
   ```

## Fine-tuning the Model

### Using the Example Pipeline

The easiest way to run the fine-tuning pipeline:

```bash
run_example_pipeline.bat
```

This script will:
1. Check if Python and CUDA are installed
2. Create necessary directories
3. Fine-tune Phi-3 Mini on the example dataset
4. Save both the adapter and the merged model
5. Launch the REPL interface with the fine-tuned model

### Manual Fine-tuning

To fine-tune the model directly:

```bash
python finetune_phi3_qlora.py --dataset example_dataset.json --output_dir ./model_output/phi3-mini-finetuned
```

### Fine-tuning Options

- `--base_model`: The model to fine-tune (default: "microsoft/Phi-3-mini-4k-instruct")
- `--dataset`: Path to the dataset file or Hugging Face dataset name
- `--output_dir`: Directory to save the fine-tuned model
- `--epochs`: Number of training epochs (default: 3)
- `--batch_size`: Batch size for training (default: 8)
- `--learning_rate`: Learning rate (default: 2e-4)
- `--save_merged_model`: Save the merged model after fine-tuning
- `--merged_model_dir`: Directory to save the merged model

## Using the REPL Interface

### Command-line Interface

```bash
python phi3_repl.py
```

### Web UI Interface

```bash
python phi3_repl.py --web_ui
```

### Using a Fine-tuned Model

```bash
python phi3_repl.py --model_path ./model_output/phi3-mini-finetuned --web_ui
```

### REPL Options

- `--model_path`: Path to the fine-tuned model
- `--base_model`: Base model to use (default: "microsoft/Phi-3-mini-4k-instruct")
- `--use_4bit`: Load the model in 4-bit precision (saves VRAM)
- `--web_ui`: Use web interface instead of CLI
- `--temperature`: Sampling temperature (default: 0.7)
- `--top_p`: Top-p sampling parameter (default: 0.9)
- `--max_length`: Maximum new tokens to generate (default: 500)

## Troubleshooting

If you encounter any issues:

1. Run the verification scripts to check your environment:
   ```bash
   setup\tests\verify_cuda.bat
   setup\tests\verify_pytorch.bat
   setup\tests\verify_bitsandbytes.bat
   ```

2. For bitsandbytes CUDA issues, run the fix script:
   ```bash
   setup\cuda\fix_cuda_bnb.bat
   ```

3. Run the full environment test to validate your setup:
   ```bash
   setup\tests\full_environment_test.bat
   ```

## RTX 4090 Optimization

This project includes optimization tips for RTX 4090 GPUs in `rtx4090_optimization_tips.md`, including:

- Memory management strategies
- QLoRA parameter recommendations
- Inference optimizations
- Generation parameters
- Performance monitoring

## References

- [Microsoft Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
