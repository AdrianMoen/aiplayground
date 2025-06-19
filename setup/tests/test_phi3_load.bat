@echo off
echo Testing Phi-3 Model Loading with QLoRA
echo ==================================
echo.

cd %~dp0\..\..

echo Checking required libraries...
python -c "import torch; import transformers; import peft; import bitsandbytes as bnb; import accelerate; print('All required libraries are installed!')"

if %ERRORLEVEL% neq 0 (
    echo Some required libraries are missing. Please install them with:
    echo pip install -r requirements.txt
    exit /b 1
)

echo.
echo Attempting to load Phi-3 Mini in 4-bit with QLoRA adapter...

python -c "
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print('PyTorch CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))

print('\nConfiguring quantization...')
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print('\nLoading Phi-3 Mini tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-3-mini-4k-instruct', trust_remote_code=True)

print('\nLoading Phi-3 Mini model in 4-bit...')
model = AutoModelForCausalLM.from_pretrained(
    'microsoft/Phi-3-mini-4k-instruct',
    device_map='auto',
    trust_remote_code=True,
    quantization_config=bnb_config
)

print('\nModel successfully loaded in 4-bit!')
print('Model type:', type(model).__name__)
print('Model parameters:', sum(p.numel() for p in model.parameters()))

print('\nTesting inference...')
inputs = tokenizer('Hello, I am Phi-3. How can I help you today?', return_tensors='pt').to(model.device)
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print('\nSample generation:')
print(response)
print('\nModel loaded and tested successfully!')
"

if %ERRORLEVEL% neq 0 (
    echo Failed to load Phi-3 Mini model.
    echo This may indicate an issue with your environment configuration.
    exit /b 1
)

echo.
echo Phi-3 Model Load Test Complete!
echo.
