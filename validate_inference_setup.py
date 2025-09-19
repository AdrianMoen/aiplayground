from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="eager",
).eval()

messages = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a C++ class for a 3D vector with magnitude()."}
]
prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.2,
        do_sample=False,
        use_cache=True,  # KV cache enabled
    )
print(tok.decode(out[0], skip_special_tokens=True))

