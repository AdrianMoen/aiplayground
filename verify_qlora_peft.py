import torch
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda if torch.cuda.is_available() else "Not available")
print("GPU device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("BitsAndBytes version:", bnb.__version__)

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("\nLoading a small model...")
model_id = "microsoft/phi-1_5"  # Much smaller model than Phi-3
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("\nLoading with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

print("\nPreparing model for kbit training...")
model = prepare_model_for_kbit_training(model)

print("\nCreating LoRA configuration...")
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "dense"]
)

print("\nCreating PEFT model...")
model = get_peft_model(model, peft_config)

print("\nModel parameters:")
print(f"Trainable params: {model.print_trainable_parameters()}")

print("\nTesting inference...")
inputs = tokenizer("Hello, my name is", return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=20)
print("Generated text:", tokenizer.decode(outputs[0], skip_special_tokens=True))

print("\nEnvironment is ready for QLoRA fine-tuning!")
