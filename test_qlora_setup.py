import os
import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "Not available")
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("BitsAndBytes version:", bnb.__version__)

# Create a small test tensor on GPU
if torch.cuda.is_available():
    print("\nCreating test tensor on GPU...")
    test_tensor = torch.ones(10, 10).cuda()
    print("Test tensor device:", test_tensor.device)
    print("Test tensor:", test_tensor)

print("\nTesting QLora components...")
# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("BitsAndBytesConfig created successfully")

# Try a small model to test
try:
    print("\nTrying to load a small test model with quantization...")
    model_id = "facebook/opt-125m"  # Small model for testing
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print("Model loaded successfully with quantization!")
    
    # Test inference
    print("\nTesting inference...")
    inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    print("Generated text:", tokenizer.decode(outputs[0], skip_special_tokens=True))
    
    print("\nAll tests passed! Your environment is ready for QLoRA fine-tuning.")
except Exception as e:
    print(f"Error: {e}")
