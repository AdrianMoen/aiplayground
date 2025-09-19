# validate_phi3_basemodel.py
import os, time, math
import torch
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

metrics= {
    "cache":[],
    "no_cache":[]
}


BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct" # base model path
MERGED_MODEL = "./phi3-mini-cypher-v4-merged" # merged model path
ADAPTER_PATH = "./phi3-mini-cypher-v4"  # adapter path
TEST_DATA_FILE = "/home/adrianmoen/workspace/master/aiplayground-anders/data/test_data_benchmark/test_data.json"
NUM_TEST_ELEM = 30

# System prompt constant
SYSTEM_PROMPT = """You are a Neo4j Cypher query generator.
Your task is to convert natural language questions about movies into Cypher queries.
Only return the Cypher query without any explanation or additional text.
The database has a movie graph model with the following node types:
- Movie (properties: title, released, tagline)
- Person (properties: name, born)
- Genre (properties: name)
And the following relationship types:
- ACTED_IN (Person to Movie, properties: roles)
- DIRECTED (Person to Movie)
- IN_GENRE (Movie to Genre)"""

# adapter_path:str
# base_model:str
# saved_model_dir:str
# max_length:int = 4096
# temperature:float = 0.9
# top_p:float=0.9

def banner(msg):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

def tokens_per_sec(num_new_tokens, elapsed):
    return float(num_new_tokens) / max(elapsed, 1e-9)

def clear_cache():
    """Clear GPU memory and reset cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_model_from_path(model_path: str):
    """Load any model from a path (works for both base model ID and local merged model)."""
    print(f"Loading model: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def load_base_model(model_id: str):
    """Load base Phi-3 model."""
    print(f"Loading base model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

# NOT USED AT THE MOMENT
def load_finetuned_model(adapter_path: str, base_model_id: str):
    """Load fine-tuned model with adapter."""
    print(f"Loading fine-tuned model: {adapter_path}")
    from peft import PeftModel
    
    # Load base model first
    model, tokenizer = load_base_model(base_model_id)
    
    # Load and merge adapter
    if os.path.exists(adapter_path):
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Merging adapter weights...")
        model = model.merge_and_unload()
    else:
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
    
    return model, tokenizer


def benchmark_model(model, tokenizer, model_name: str, test_prompts: list):
    """Benchmark a model with multiple test prompts."""
    benchmark = LatencyBenchmark()
    
    print(f"\n🚀 Benchmarking {model_name}")
    print(f"Running {len(test_prompts)} test prompts...")
    
    # Model info for saving
    model_info = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'dtype': str(next(model.parameters()).dtype),
        'device': str(next(model.parameters()).device),
        'attention_impl': getattr(model, "_attn_implementation", "unknown")
    }
    
    for i, prompt_data in enumerate(test_prompts):
        user_query = prompt_data['prompt']
        expected_output = prompt_data.get('expected_output', '')
        
        print(f"Test {i+1}/{len(test_prompts)}: {user_query[:50]}...")
        
        # Format with system prompt (same as phi3_repl.py)
        formatted_prompt = format_prompt_with_system(user_query, SYSTEM_PROMPT)
        
        # Tokenize input
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        input_tokens = inputs.input_ids.shape[1]
        
        # Generation parameters
        gen_kwargs = {
            'max_new_tokens': 100,
            'do_sample': False,
            'use_cache': True,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id
        }
        
        # Benchmark generation
        start_time = time.time()
        
        with torch.inference_mode():
            outputs = model.generate(inputs.input_ids, **gen_kwargs)
        
        end_time = time.time()
        
        # Extract response
        response_tokens = outputs[0][input_tokens:]
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Calculate metrics
        generation_time = end_time - start_time
        output_tokens = len(response_tokens)
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
        
        metrics = {
            'test_id': i + 1,
            'user_query': user_query,
            'expected_output': expected_output,
            'input_text': formatted_prompt,
            'output_text': response_text.strip(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'generation_time': generation_time,
            'tokens_per_second': tokens_per_second,
            'model_name': model_name
        }
        
        benchmark.add_measurement(metrics)
        print(f"  → {tokens_per_second:.1f} tok/s ({output_tokens} tokens in {generation_time:.3f}s)")
    
    return benchmark, model_info

class LatencyBenchmark:
    ''' Handle latency benchmarking. '''

    def __init__(self):
        self.measurements: list[dict] = []

    def add_measurement(self, metrics: dict):
        """Add a measurement to the benchmark."""
        metrics['timestamp'] = time.time()
        self.measurements.append(metrics)
    
    def get_statistics(self) -> dict:
        """Calculate statistics across all measurements."""
        if not self.measurements:
            return {}
        
        generation_times = [m['generation_time'] for m in self.measurements]
        tokens_per_sec = [m['tokens_per_second'] for m in self.measurements if m['tokens_per_second'] > 0]
        output_tokens = [m['output_tokens'] for m in self.measurements]
        
        return {
            'total_requests': len(self.measurements),
            'avg_generation_time': statistics.mean(generation_times),
            'median_generation_time': statistics.median(generation_times),
            'p95_generation_time': self._percentile(generation_times, 0.95),
            'p99_generation_time': self._percentile(generation_times, 0.99),
            'avg_tokens_per_second': statistics.mean(tokens_per_sec) if tokens_per_sec else 0,
            'avg_output_tokens': statistics.mean(output_tokens),
            'min_generation_time': min(generation_times),
            'max_generation_time': max(generation_times)
        }
    
    def _percentile(self, data: list[float], percentile: float) -> float:
        """Calculate percentile of data."""
        sorted_data = sorted(data)
        index = int(percentile * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def save_results(self, filename:str, model_info:dict = None):
        import json
        from datetime import datetime, timezone

        filename = filename+datetime.now().strftime("_%d-%H:%M")+".json"

        results = {
            'benchmark_info' : {
                'timestamp': datetime.now().isoformat(),
                'model_info': model_info or {},
                'total_measurements': len(self.measurements)
            },
            'statistics': self.get_statistics(),
            'raw_measurements': self.measurements
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filename}")

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
        print(f"Average Generation Time: {stats['avg_generation_time']:.3f}s")
        print(f"Median Generation Time: {stats['median_generation_time']:.3f}s")
        print(f"95th Percentile: {stats['p95_generation_time']:.3f}s")
        print(f"99th Percentile: {stats['p99_generation_time']:.3f}s")
        print(f"Min Time: {stats['min_generation_time']:.3f}s")
        print(f"Max Time: {stats['max_generation_time']:.3f}s")
        print(f"Average Tokens/Second: {stats['avg_tokens_per_second']:.1f}")
        print(f"Average Output Length: {stats['avg_output_tokens']:.1f} tokens")
        print("="*50)


def load_test_prompts_from_file(file_path: str, max_prompts: int = None):
    """Load test prompts from your training data file."""
    import json
    
    print(f"Loading test prompts from: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert training data to test prompts
    test_prompts = []
    for i, item in enumerate(data):
        if max_prompts and i >= max_prompts:
            break
            
        # Extract just the natural language query
        prompt_text = item.get('input')
        if not prompt_text:
            continue
            
        test_prompts.append({
            'prompt': prompt_text,
            'expected_output': item.get('output')  # Store expected Cypher for comparison
        })
    
    print(f"Loaded {len(test_prompts)} test prompts")
    return test_prompts

def format_prompt_with_system(user_input: str, system_prompt: str):
    """Format the prompt according to Phi-3's expected format with system prompt."""
    return f"""<|system|>
{system_prompt}
<|user|>
{user_input}
<|assistant|>
"""

def main():
    banner("Model Benchmarking Suite")
    
    # Environment info
    print("torch:", torch.__version__)
    try:
        import transformers
        print("transformers:", transformers.__version__)
    except Exception:
        pass
    
    # Model paths (update these to your actual paths)
    base_model_id = BASE_MODEL
    merged_model = MERGED_MODEL
    adapter_path = ADAPTER_PATH  # Update this path
    
    # load test data
    test_prompts = load_test_prompts_from_file(TEST_DATA_FILE, max_prompts=NUM_TEST_ELEM)

    results = {}
    
    # # Benchmark base model
    # banner("Benchmarking Base Model")
    # base_model, base_tokenizer = load_base_model(base_model_id)
    # base_benchmark, base_info = benchmark_model(base_model, base_tokenizer, "base_model", test_prompts)
    # base_benchmark.print_stats()
    # base_benchmark.save_results("benchmark_base_model", base_info)
    # results['base'] = base_benchmark.get_statistics()
    
    # # Clear memory
    # del base_model, base_tokenizer
    # clear_cache()
    
    # Temporary return, must re fine tune the model. with this version.

    # Benchmark fine-tuned model
    banner("Benchmarking Fine-tuned Model")
    try:
        ft_model, ft_tokenizer = load_model_from_path(MERGED_MODEL)
        ft_benchmark, ft_info = benchmark_model(ft_model, ft_tokenizer, "finetuned_model", test_prompts)
        ft_benchmark.print_stats()
        ft_benchmark.save_results("benchmark_finetuned_model", ft_info)
        results['finetuned'] = ft_benchmark.get_statistics()
        
        # Comparison
        banner("Model Comparison")
        base_avg_speed = results['base']['avg_tokens_per_second']
        ft_avg_speed = results['finetuned']['avg_tokens_per_second']
        
        print(f"Base model average speed:      {base_avg_speed:.1f} tok/s")
        print(f"Fine-tuned model average speed: {ft_avg_speed:.1f} tok/s")
        print(f"Speed difference: {((ft_avg_speed - base_avg_speed) / base_avg_speed * 100):+.1f}%")
        
    except FileNotFoundError as e:
        print(f"Could not load fine-tuned model: {e}")
        print("Skipping fine-tuned model benchmark")



if __name__ == "__main__":
    print(f"basemodel path: {BASE_MODEL}")
    print(f"finetuned path: {MERGED_MODEL}")
    main()