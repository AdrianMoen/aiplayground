# validate_phi3_basemodel.py
import os, time, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

metrics= {
    "cache":[],
    "no_cache":[]
}

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

def main():
    banner("Environment")
    print("torch:", torch.__version__)
    try:
        import transformers
        print("transformers:", transformers.__version__)
        import accelerate
        print("accelerate:", accelerate.__version__)
    except Exception:
        pass
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    

    banner("Load tokenizer & model (no flash-attn, eager attention)")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda",
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # FLASH ATTENTION WORKING!!!!!!!
    ).eval()

    # Helpful printouts
    cfg = model.config
    print("dtype:", next(model.parameters()).dtype)
    print("attn impl:", getattr(model, "_attn_implementation", None))
    print("attn impl (config):", getattr(model.config, "_attn_implementation_internal", None))
    print("max_position_embeddings (context):", getattr(cfg, "max_position_embeddings", "unknown"))
    print("rope_theta:", getattr(cfg, "rope_theta", "n/a"))

    # Chat-format a prompt
    sys_msg = {"role": "system", "content": "You are a careful, concise assistant."}
    user_msg = {"role": "user", "content": "Give me two bullet points about the moon."}
    chat = tok.apply_chat_template([sys_msg, user_msg], tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)

    banner("Smoke test: forward pass + logits sanity (no NaNs/Infs)")
    with torch.inference_mode():
        out = model(**inputs)
        logits = out.logits
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        print("logits shape:", tuple(logits.shape))
        print("NaNs:", has_nan, "Infs:", has_inf)
        assert not has_nan and not has_inf, "Numerical issue: logits contain NaN/Inf"

    banner("Deterministic generation + KV cache speed check")
    set_seed(0)  # make greedy path deterministic
    gen_kwargs = dict(max_new_tokens=200, do_sample=False, use_cache=True, temperature=None, top_p=None)

    # warmup
    with torch.inference_mode():
        _ = model.generate(**inputs, **{**gen_kwargs, "max_new_tokens": 100})

    for _ in range(5):
        clear_cache()
        set_seed(42)  # Ensure deterministic results
        # with cache (normal generate)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.inference_mode():
            out_cache = model.generate(**inputs, **gen_kwargs)
        t_cache = time.time() - t0
        metrics["cache"].append(t_cache)
        mem_cache = torch.cuda.max_memory_allocated()


        clear_cache()
        set_seed(42)  # Ensure deterministic results

        # no cache (force)
        from copy import deepcopy
        gcfg = deepcopy(model.generation_config)
        gcfg.use_cache = False
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.inference_mode():
            out_nocache = model.generate(**inputs, generation_config=gcfg, max_new_tokens=200, do_sample=True, use_cache=False)
        t_nocache = time.time() - t0
        metrics["no_cache"].append(t_nocache)
        mem_nocache = torch.cuda.max_memory_allocated()

    # new_tokens = out_cache.shape[1] - inputs["input_ids"].shape[1]
    # print(f"Produced {new_tokens} tokens.")
    # print(f"with cache : {t_cache:6.3f}s  | ~{tokens_per_sec(new_tokens, t_cache):.1f} tok/s | peak mem {mem_cache/1e9:.2f} GB")
    # print(f"no  cache : {t_nocache:6.3f}s | ~{tokens_per_sec(new_tokens, t_nocache):.1f} tok/s | peak mem {mem_nocache/1e9:.2f} GB")

    # # Equivalence check: split-prefix vs full prompt (uses KV under the hood)
    # banner("Equivalence check: split-prefix vs full-prompt")
    # # Split prompt roughly in half by tokens
    # full_ids = inputs["input_ids"][0]
    # mid = max(1, full_ids.shape[0] // 2)
    # first_half = full_ids[:mid].unsqueeze(0)
    # second_half = full_ids[mid:].unsqueeze(0)

    # # Generate from full prompt (reference)
    # set_seed(0)
    # with torch.inference_mode():
    #     ref = model.generate(input_ids=full_ids.unsqueeze(0).to(model.device),
    #                          max_new_tokens=100, do_sample=False, use_cache=True)

    # # Now feed first_half, then append second_half and continue
    # set_seed(0)
    # with torch.inference_mode():
    #     # step 1: prime cache with first half
    #     _ = model(input_ids=first_half.to(model.device))
    #     # step 2: continue with second half (simulate streaming input)
    #     cont = model.generate(input_ids=full_ids.unsqueeze(0).to(model.device),
    #                           max_new_tokens=100, do_sample=False, use_cache=True)

    # same = torch.equal(ref, cont)
    # print("Split-prefix equals full-prompt:", bool(same))
    # if not same:
    #     print("Note: tiny numerical differences can cause divergences after many steps;")
    #     print("for a strict test we keep do_sample=False and same seed.")

    # # Show decoded sample (trim for brevity)
    # text = tok.decode(out_cache[0], skip_special_tokens=True)
    # print("\n--- Sample generation (first 400 chars) ---")
    # print(text[:400])
    # print("------------------------------------------")

    # # Long-context test (~3.4k tokens) to ensure near-limit works
    # banner("Long-context (~3.4k tokens) test")
    # long_text = " ".join(["When in the course of human events,"] * 200)  # repetitive but long
    # long_chat = tok.apply_chat_template(
    #     [{"role": "user", "content": long_text}],
    #     tokenize=False,
    #     add_generation_prompt=True,
    # )
    # long_inputs = tok(long_chat, return_tensors="pt", truncation=False).to(model.device)
    # n_ctx = long_inputs["input_ids"].shape[1]
    # print("Long prompt token length:", n_ctx)
    # if hasattr(cfg, "max_position_embeddings"):
    #     max_ctx = cfg.max_position_embeddings
    #     print("Model context limit:", max_ctx)
    #     if n_ctx >= max_ctx:
    #         print("Warning: prompt touches/exceeds context window; generation may truncate/slide.")
    # with torch.inference_mode():
    #     t0 = time.time()
    #     long_out = model.generate(**long_inputs, max_new_tokens=32, do_sample=False, use_cache=True)
    #     t1 = time.time()
    # print(f"Long-context generate OK in {t1-t0:.2f}s")

    # banner("All checks completed")
    # print("✔ Base model loads, runs forward, generates deterministically, cache speeds things up, and handles long prompts.")

if __name__ == "__main__":
    main()

    print("avg. cache: ", sum(metrics["cache"]) / len(metrics["cache"]))
    print("avg. no_cache: ", sum(metrics["no_cache"]) / len(metrics["no_cache"]))
