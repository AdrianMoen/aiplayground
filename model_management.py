"""
Utility functions for saving and loading fine-tuned models.
"""

import os
import json
import torch
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def save_model_with_metadata(
    base_model_name,
    adapter_path,
    output_dir,
    metadata=None,
    save_full_model=False
):
    """
    Save a fine-tuned model with metadata for easy reloading.
    
    Args:
        base_model_name (str): The name or path of the base model
        adapter_path (str): Path to the adapter model
        output_dir (str): Directory to save the model and metadata
        metadata (dict, optional): Additional metadata to save with the model
        save_full_model (bool, optional): Whether to save the full merged model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create metadata
    metadata = metadata or {}
    metadata.update({
        "base_model_name": base_model_name,
        "adapter_path": adapter_path,
        "save_date": None,  # Will be filled in by datetime.now().isoformat() in a real environment
        "save_full_model": save_full_model,
    })
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    if save_full_model:
        # Load and merge model
        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        
        print("Merging adapter with base model")
        model = model.merge_and_unload()
        
        # Save the full model
        print(f"Saving full model to {output_dir}")
        model.save_pretrained(os.path.join(output_dir, "full_model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "full_model"))
    else:
        # Just copy the adapter files
        print(f"Copying adapter files from {adapter_path} to {output_dir}/adapter")
        shutil.copytree(adapter_path, os.path.join(output_dir, "adapter"), dirs_exist_ok=True)
    
    print(f"Model saved to {output_dir} with metadata")
    return metadata_path

def load_model_from_saved(
    saved_dir,
    use_4bit=False,
):
    """
    Load a model from a saved directory with metadata.
    
    Args:
        saved_dir (str): Directory containing the saved model and metadata
        use_4bit (bool, optional): Whether to load the model in 4-bit precision
        
    Returns:
        tuple: (model, tokenizer) The loaded model and tokenizer
    """
    # Load metadata
    metadata_path = os.path.join(saved_dir, "model_metadata.json")
    if not os.path.exists(metadata_path):
        raise ValueError(f"No metadata found in {saved_dir}")
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    base_model_name = metadata.get("base_model_name")
    saved_full_model = metadata.get("save_full_model", False)
    
    # Check if we have the full model or just the adapter
    if saved_full_model and os.path.exists(os.path.join(saved_dir, "full_model")):
        print(f"Loading full model from {saved_dir}/full_model")
        if use_4bit:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                os.path.join(saved_dir, "full_model"),
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                os.path.join(saved_dir, "full_model"),
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(saved_dir, "full_model"), trust_remote_code=True)
    else:
        # Load base model and adapter
        print(f"Loading base model: {base_model_name}")
        if use_4bit:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        adapter_path = os.path.join(saved_dir, "adapter")
        if os.path.exists(adapter_path):
            print(f"Loading adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            
            # Merge adapter weights with base model for faster inference (if not using 4-bit)
            if not use_4bit:
                print("Merging adapter weights with base model")
                model = model.merge_and_unload()
        else:
            print(f"Warning: Adapter not found at {adapter_path}")
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, metadata
