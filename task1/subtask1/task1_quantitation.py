import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch

@torch.no_grad()
def measure_inference_time(model, batch, tokenizer, max_new_tokens):
    """Measures inference time and memory usage."""
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to("cuda")
    start_time = time.time()
    
    generated = tokenized_batch["input_ids"]
    for _ in range(max_new_tokens):
        outputs = model(**tokenized_batch)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        tokenized_batch["input_ids"] = torch.cat([tokenized_batch["input_ids"], next_token], dim=-1)
        tokenized_batch["attention_mask"] = torch.cat([tokenized_batch["attention_mask"], torch.ones_like(next_token)], dim=-1)
        generated = torch.cat([generated, next_token], dim=-1)

    end_time = time.time()
    return generated, end_time - start_time

def run_experiment(models, tokenizer, prompts, max_new_tokens=100):
    """Runs inference for each model configuration and records time and memory usage."""
    results = {}
    for model_name, model in models.items():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        print(f"Running inference for {model_name}...")
        _, inference_time = measure_inference_time(model, prompts, tokenizer, max_new_tokens)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
        results[model_name] = {"inference_time": inference_time, "peak_memory": peak_memory}
        print(f"{model_name}: Time = {inference_time:.2f}s, Memory = {peak_memory:.2f} MB")
    
    return results

if __name__ == "__main__":
    # Parameters
    MODEL_NAME = "openai-community/gpt2"
    MAX_NEW_TOKENS = 100
    BATCH_SIZE = 16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    with open("data.txt") as f:
        prompt_dataset = [line.strip() for line in f.readlines()]
    
    # Load models with different quantization levels
    print("Loading models with various quantization levels...")
    fp32_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cuda").eval()
    fp16_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cuda", torch_dtype=torch.float16).eval()
    bnb_int8 = BitsAndBytesConfig(
        load_in_8bit=True, 
    )
    
    bnb_int4 = BitsAndBytesConfig(
        load_in_4bit=True, 
    )
        

    int8_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_int8,  
        device_map="cuda",              
        low_cpu_mem_usage=True,        
    ).eval()
    
    int4_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_int4,  
        device_map="cuda",              
        low_cpu_mem_usage=True,        
    ).eval()

    models = {
        "FP32": fp32_model,
        "FP16": fp16_model,
        "INT8": int8_model,
        "INT4": int4_model,
    }

    # Run experiments
    prompts = prompt_dataset[:BATCH_SIZE]  # Take a batch of prompts
    results = run_experiment(models, tokenizer, prompts, MAX_NEW_TOKENS)

    # Display results
    print("\n--- Results Summary ---")
    for model_name, metrics in results.items():
        print(f"{model_name}: Time = {metrics['inference_time']:.2f}s, Memory = {metrics['peak_memory']:.2f} MB")
