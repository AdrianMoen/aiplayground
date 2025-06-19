# RTX 4090 Optimization Tips for Phi-3 Mini Fine-tuning

This document provides optimization tips specifically for users with an NVIDIA RTX 4090 GPU when fine-tuning and running the Phi-3 Mini model.

## Hardware Specifications of RTX 4090

- CUDA Cores: 16,384
- Memory: 24GB GDDR6X
- Memory Bandwidth: 1,008 GB/s
- TDP: 450W

## Fine-tuning Optimizations

### Memory Management

1. **Batch Size Optimization**
   - The RTX 4090's 24GB VRAM allows for larger batch sizes compared to smaller GPUs
   - Recommended batch size: 8-16 (depending on sequence length)
   - For longer sequences (>2048 tokens), reduce batch size accordingly

2. **Gradient Accumulation**
   - Use gradient accumulation steps of 4-8 to simulate larger batch sizes
   - This can improve training stability while keeping memory usage manageable

3. **Mixed Precision Training**
   - Always use mixed precision (fp16) for maximum performance
   - This is enabled by default in the finetune_phi3_qlora.py script

### QLoRA Parameters

1. **Rank (r) and Alpha**
   - With the RTX 4090, you can afford to use higher rank values
   - Recommended: r=64, alpha=128 for better adaptation while still being efficient
   - Default in the script: r=8, alpha=16 (more conservative)

2. **Target Modules**
   - The script targets key modules for adaptation
   - On the 4090, you can expand this to include more layers if needed

### Parallel Processing

1. **DataLoader Workers**
   - Set num_workers in DataLoader to 4-8 to leverage CPU parallelism
   - This helps keep the GPU fed with data

2. **Prefetch Factor**
   - Consider using a prefetch factor of 2-4 to queue batches ahead of time

## Inference Optimizations

### For Maximum Speed

1. **Merge Weights**
   - After fine-tuning, merge the LoRA weights with the base model using `model.merge_and_unload()`
   - This is done automatically in the REPL script when not using 4-bit mode
   - Provides faster inference compared to using the adapter separately

2. **Flash Attention**
   - The RTX 4090 supports Flash Attention 2
   - Add `attn_implementation="flash_attention_2"` when loading the model for faster attention computation

### For Maximum Efficiency (Longer Context)

1. **4-bit Quantization**
   - Use 4-bit quantization with `--use_4bit` flag for REPL
   - Allows handling longer contexts and saves VRAM

2. **Efficient Attention Patterns**
   - When generating long outputs, consider using efficient attention mechanisms like sliding window attention

## Generation Parameters

1. **Optimized Generation Settings for RTX 4090**
   - Temperature: 0.7 (good balance between creativity and coherence)
   - Top-p: 0.9 (maintains good diversity while filtering unlikely tokens)
   - Typical-p: 0.2 (optional, can improve coherence)

2. **Speculative Decoding**
   - Consider implementing speculative decoding for faster generation
   - This requires more complex code but can provide 2-3x speedup on RTX 4090

## Monitoring Performance

1. **GPU Utilization**
   - Use `nvidia-smi` to monitor GPU utilization
   - Target >80% utilization during training for optimal performance

2. **Memory Usage**
   - Monitor VRAM usage to optimize batch size
   - Command: `nvidia-smi --query-gpu=memory.used --format=csv -l 1`

## CUDA and Library Versions

For optimal performance with RTX 4090, use:

- CUDA 12.1+
- PyTorch 2.1.0+
- Transformers 4.36.0+
- bitsandbytes 0.40.0+

## Custom Configurations for RTX 4090

Add these flags to your fine-tuning command for optimal RTX 4090 performance:

```bash
python finetune_phi3_qlora.py \
  --base_model microsoft/Phi-3-mini-4k-instruct \
  --dataset your_dataset.json \
  --batch_size 12 \
  --gradient_accumulation_steps 4 \
  --lora_r 64 \
  --lora_alpha 128 \
  --lr 2e-4 \
  --max_seq_length 2048
```
