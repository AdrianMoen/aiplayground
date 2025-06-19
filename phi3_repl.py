"""
REPL (Read-Eval-Print Loop) interface for the fine-tuned Phi-3 Mini model.
"""

import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
import gradio as gr
from utils.model_management import load_model_from_saved

def parse_args():
    parser = argparse.ArgumentParser(description="REPL for fine-tuned Phi-3 Mini")
    
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct", 
                        help="Base model name")
    parser.add_argument("--adapter_path", type=str, default="./phi3-mini-finetuned", 
                        help="Path to the trained adapter model")
    parser.add_argument("--saved_model_dir", type=str, default=None,
                        help="Path to a saved model directory with metadata (overrides base_model and adapter_path)")
    parser.add_argument("--use_4bit", action="store_true", 
                        help="Whether to load the model in 4-bit precision")
    parser.add_argument("--max_length", type=int, default=4096, 
                        help="Maximum length for generation")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p sampling parameter")
    parser.add_argument("--web_ui", action="store_true", 
                        help="Launch Gradio web interface instead of CLI")
    
    return parser.parse_args()

def load_model(args):
    """Load the fine-tuned model."""
    if args.saved_model_dir and os.path.exists(args.saved_model_dir):
        print(f"Loading model from saved directory: {args.saved_model_dir}")
        model, tokenizer, metadata = load_model_from_saved(args.saved_model_dir, args.use_4bit)
        return model, tokenizer
    
    print(f"Loading base model: {args.base_model}")
    
    # Set up quantization if requested
    if args.use_4bit:
        print("Using 4-bit quantization")
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # Load in fp16 if not using quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, 
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load PEFT adapter
    if os.path.exists(args.adapter_path):
        print(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
    else:
        print(f"Warning: Adapter path {args.adapter_path} not found. Using base model only.")
    
    # Merge adapter weights with base model for faster inference (if not using 4-bit)
    if not args.use_4bit and os.path.exists(args.adapter_path):
        print("Merging adapter weights with base model")
        model = model.merge_and_unload()
    
    return model, tokenizer

def format_prompt(user_input, system_prompt=None):
    """Format the prompt according to Phi-3's expected format."""
    if system_prompt:
        prompt = f"""<|system|>
{system_prompt}
<|user|>
{user_input}
<|assistant|>
"""
    else:
        prompt = f"""<|user|>
{user_input}
<|assistant|>
"""
    return prompt

def generate_response(model, tokenizer, prompt, args):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Set up streamer for token-by-token generation
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    
    generation_kwargs = dict(
        inputs=inputs.input_ids,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
        streamer=streamer,
    )
    
    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield tokens as they're generated
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield generated_text
    
    thread.join()

def cli_interface(model, tokenizer, args):
    """Command-line interface for interacting with the model."""
    print("\nPhi-3 Mini REPL Interface")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'system: <prompt>' to set a system prompt\n")
    
    system_prompt = None
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if user_input.startswith("system: "):
            system_prompt = user_input[8:]
            print(f"System prompt set to: {system_prompt}")
            continue
        
        prompt = format_prompt(user_input, system_prompt)
        
        print("\nAssistant: ", end="", flush=True)
        for text in generate_response(model, tokenizer, prompt, args):
            # Print only the new part (replace previous output with new)
            print("\rAssistant: " + text, end="", flush=True)
        print()  # New line after generation is complete

def gradio_interface(model, tokenizer, args):
    """Gradio web interface for interacting with the model."""
    
    system_prompt = gr.State("")
    
    def respond(message, chat_history, system):
        prompt = format_prompt(message, system)
        bot_message = ""
        for text in generate_response(model, tokenizer, prompt, args):
            bot_message = text
            yield chat_history + [[message, bot_message]]
    
    def set_system(new_system):
        return new_system
    
    with gr.Blocks(title="Phi-3 Mini Chat Interface") as demo:
        gr.Markdown("# Phi-3 Mini Fine-tuned Chat Interface")
        
        with gr.Row():
            with gr.Column():
                system_input = gr.Textbox(
                    placeholder="Optional: Set a system prompt here",
                    label="System Prompt",
                    lines=2,
                )
                system_button = gr.Button("Set System Prompt")
        
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(placeholder="Type your message here...", label="Your Message", lines=3)
        
        msg.submit(respond, [msg, chatbot, system_prompt], chatbot)
        system_button.click(set_system, system_input, system_prompt)
        
    demo.queue().launch(share=True)

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model(args)
    
    # Prepare model for inference
    model.eval()
    
    if args.web_ui:
        # Launch web interface
        gradio_interface(model, tokenizer, args)
    else:
        # Launch CLI interface
        cli_interface(model, tokenizer, args)

if __name__ == "__main__":
    main()
