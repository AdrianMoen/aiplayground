"""
REPL (Read-Eval-Print Loop) interface for the fine-tuned Phi-3 Mini model.
"""

import os
import argparse
import torch
import time
import traceback
import statistics
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
    parser.add_argument("--use_streamer", action="store_true", 
                        help="use streamer for printing, allows to meassure ttft(time to first token)")
    parser.add_argument("--base_only", action="store_true",
                        help="Use only the base model, skip loading any adapters")
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
        return model, tokenizer, metadata
    
    print(f"Loading base model: {args.base_model}")
    
    # Set up quantization if requested
    if args.use_4bit:
        print("Using 4-bit quantization")
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, #Changed from float16, might be better
            bnb_4bit_use_double_quant=True,
            attn_implementation="sdpa" # added this
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True, # changed this back to true
            attn_implementation="sdpa" # added this
        )
    else:
        # Load in fp16 if not using quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, 
            torch_dtype=torch.bfloat16, #Changed from float16, might be better
            device_map="auto",
            trust_remote_code=True,
        )

        # Print debug info to see what's actually loaded
    print(f"✓ Model device: {next(model.parameters()).device}")
    print(f"✓ Model dtype: {next(model.parameters()).dtype}")
    print(f"✓ Model config: {type(model.config).__name__}")
    
    if hasattr(model.config, 'attn_implementation'):
        print(f"✓ Attention implementation: {model.config.attn_implementation}")
    else:
        print("⚠️  Attention implementation: unknown/default")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear cache between tests
        print(f"✓ GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load PEFT adapter only if not using base model (base_only flag)
    if not args.base_only and os.path.exists(args.adapter_path):
        print(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

        # Merge adapter weights with base model for faster inference (if not using 4-bit)
        if not args.use_4bit and os.path.exists(args.adapter_path):
            print("Merging adapter weights with base model")
            model = model.merge_and_unload()
        
    else:
        if args.base_only:
            print(f"using base model: {args.base_model}")
        else:
            print(f"Warning: Adapter path {args.adapter_path} not found. Using base model only.")
    
    return model, tokenizer, None

# copilot suggested this.
def test_attention_speed(model, tokenizer):
    """Quick test to verify attention implementation."""
    print("🧪 Testing attention speed...")
    
    # Simple test prompt
    test_prompt = "<|user|>\nHello world\n<|assistant|>\n"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    # Warmup
    with torch.no_grad():
        _ = model.generate(inputs.input_ids, max_new_tokens=10, use_cache=True)
    
    # Measure
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(inputs.input_ids, max_new_tokens=50, use_cache=True)
    end = time.time()
    
    tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
    speed = tokens_generated / (end - start)
    print(f"🚀 Attention test: {speed:.1f} tok/s ({tokens_generated} tokens in {end-start:.2f}s)")
    return speed

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

def generate_response(model, tokenizer, prompt, args, measure_latency:bool=False):
    """Generate a response from the model."""

    # start time
    start_time = time.time()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Create proper attention mask to avoid warnings
    attention_mask = torch.ones_like(inputs.input_ids)
    
    # Set up streamer for token-by-token generation
    # streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    
    tokenization_time = time.time()

    generation_kwargs = dict(
        input_ids=inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=min(512, args.max_length - len(inputs.input_ids[0])),  # Use max_new_tokens instead of max_length
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.temperature > 0,
        # trust_remote_code=True,
        # streamer=streamer, # removed cause it clutters the screen.
        use_cache=True,  # Disable KV cache to avoid DynamicCache errors
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # generate, no stream!
    generation_start = time.time()
    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)
    generation_end = time.time()
    
    # Extract only the new tokens (response)
    input_length = len(inputs.input_ids[0])
    response_tokens = outputs[0][input_length:]
    decoding_start = time.time()
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)
    decoding_end = time.time()

    response = response.strip()
    if '\n' in response:
        response = response.split('\n')[0]

    end_time = time.time()
    # Calculate metrics
    if measure_latency:
        metrics = {
            'total_time': end_time - start_time,
            'tokenization_time': tokenization_time - start_time,
            'generation_time': generation_end - generation_start,
            'decoding_time': decoding_end - decoding_start,
            'input_tokens': len(inputs.input_ids[0]),
            'output_tokens': len(response_tokens),
            'tokens_per_second': len(response_tokens) / (generation_end - generation_start) if generation_end > generation_start else 0,
            'response': response
        }
    return response, metrics

    # Start generation in a separate thread, fuck this thread shit
    # thread = Thread(target=model.generate, kwargs=generation_kwargs)
    # thread.start()
    # Yield tokens as they're generated
    # generated_text = ""
    # for new_text in streamer:
    #     generated_text += new_text
    #     yield generated_text
    # thread.join()



class LatencyBenchmark:
    ''' Handle latency benchmarking. '''

    def __init__(self):
        self.measurements: list[dict] = []

    def add_measurement(self, metrics: dict):
        """Add a measurement to the benchmark."""
        self.measurements.append(metrics)
    
    def get_statistics(self) -> dict:
        """Calculate statistics across all measurements."""
        if not self.measurements:
            return {}
        
        total_times = [m['total_time'] for m in self.measurements]
        generation_times = [m['generation_time'] for m in self.measurements]
        tokens_per_sec = [m['tokens_per_second'] for m in self.measurements if m['tokens_per_second'] > 0]
        output_tokens = [m['output_tokens'] for m in self.measurements]
        
        return {
            'total_requests': len(self.measurements),
            'avg_total_time': statistics.mean(total_times),
            'median_total_time': statistics.median(total_times),
            'p95_total_time': self._percentile(total_times, 0.95),
            'p99_total_time': self._percentile(total_times, 0.99),
            'avg_generation_time': statistics.mean(generation_times),
            'avg_tokens_per_second': statistics.mean(tokens_per_sec) if tokens_per_sec else 0,
            'avg_output_tokens': statistics.mean(output_tokens),
            'min_total_time': min(total_times),
            'max_total_time': max(total_times)
        }
    
    def _percentile(self, data: list[float], percentile: float) -> float:
        """Calculate percentile of data."""
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def print_stats(self):
        """Print benchmark statistics."""
        stats = self.get_statistics()
        if not stats:
            print("No measurements recorded.")
            return
        
        print("\n" + "="*50)
        print("LATENCY BENCHMARK RESULTS")
        print("="*50)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Average Total Time: {stats['avg_total_time']:.3f}s")
        print(f"Median Total Time: {stats['median_total_time']:.3f}s")
        print(f"95th Percentile: {stats['p95_total_time']:.3f}s")
        print(f"99th Percentile: {stats['p99_total_time']:.3f}s")
        print(f"Min Time: {stats['min_total_time']:.3f}s")
        print(f"Max Time: {stats['max_total_time']:.3f}s")
        print(f"Average Generation Time: {stats['avg_generation_time']:.3f}s")
        print(f"Average Tokens/Second: {stats['avg_tokens_per_second']:.1f}")
        print(f"Average Output Length: {stats['avg_output_tokens']:.1f} tokens")
        print("="*50)
    

def cli_interface(model, tokenizer, args):
    """Command-line interface for interacting with the model."""
    print("\nPhi-3 Mini REPL Interface")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'system: <prompt>' to set a system prompt\n")
    
    benchmark = LatencyBenchmark()

    system_prompt = """You are a Neo4j Cypher query generator.
    Your task is to convert natural language questions about movies into Cypher queries.
    Only return the Cypher query without any explanation or additional text.
    The database has a movie graph model with the following node types:
    - Movie (properties: title, released, tagline)
    - Person (properties: name, born)
    - Genre (properties: name)
    And the following relationship types:
    - ACTED_IN (Person to Movie, properties: roles)
    - DIRECTED (Person to Movie)
    - IN_GENRE (Movie to Genre)
    **IMPORTANT:** ONLY REPLY WITH THE CYPHER QUERY, DO NOT PROVIDE ADDITIONAL INFORMATION UNLESS EXPLICITLY ASKED"""
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "benchmark":
            benchmark.print_stats()
            continue
        
        if user_input.lower() == "clear_benchmark":
            benchmark = LatencyBenchmark()
            print("Benchmark statistics cleared.")
            continue

        if user_input.startswith("system: "):
            system_prompt = user_input[8:]
            print(f"System prompt set to: {system_prompt}")
            continue
        
        prompt = format_prompt(user_input, system_prompt)
        
        print("\nAssistant: ", end="", flush=True)
        try: # new method, without streaming
            response, metrics = generate_response(model, tokenizer, prompt, args, measure_latency=True)
            print(response)
            benchmark.add_measurement(metrics) # add to benchmark
            print(f"[{metrics['generation_time']:.3f}s, {metrics['tokens_per_second']:.1f} tok/s]")
        except Exception as e:
            traceback.print_exc()
            print(f"Error generating response: {e}")
        # old method, with streaming
        # for text in generate_response(model, tokenizer, prompt, args):
        #     # Print only the new part (replace previous output with new)
        #     print("\rAssistant: " + text, end="", flush=True)
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
        
        chatbot = gr.Chatbot(height=500, type='messages')
        msg = gr.Textbox(placeholder="Type your message here...", label="Your Message", lines=3)
        
        print("before submit")
        msg.submit(respond, [msg, chatbot, system_prompt], chatbot)
        print("after submit!")
        system_button.click(set_system, system_input, system_prompt)
        
    demo.queue().launch(share=True)

def main():
    args = parse_args()
    
    # Load model and tokenizer
    model, tokenizer, metadata = load_model(args)
    
    print(f"metadata {metadata}")

    # Prepare model for inference
    model.eval()

    # Quick attention speed test
    test_speed = test_attention_speed(model, tokenizer)
    print(f"Expected improvement with SDPA: {test_speed:.1f} tok/s (should be 80+ tok/s)")
    
    if args.web_ui:
        # Launch web interface
        gradio_interface(model, tokenizer, args)
    else:
        # Launch CLI interface
        cli_interface(model, tokenizer, args)

if __name__ == "__main__":
    main()
