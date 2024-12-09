import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

# 加载模型和分词器
def load_model_and_tokenizer(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# 记录显存使用情况
def get_memory_usage(device):
    if device.type == "cuda":
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # 转为 MB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)  # 转为 MB
        return memory_allocated, memory_reserved
    else:
        return 0, 0

# Baseline 推理
def baseline_inference(model, tokenizer, device, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start_time = time.time()
    
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=max_length, 
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )
    
    end_time = time.time()
    inference_time = end_time - start_time
    total_tokens = outputs.size(-1)
    tps = total_tokens / inference_time

    memory_allocated, memory_reserved = get_memory_usage(device)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text, inference_time, tps, memory_allocated, memory_reserved

# 使用 KV-cache 的推理
def kv_cache_inference(model, tokenizer, device, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    past_key_values = None
    start_time = time.time()

    generated_ids = inputs["input_ids"]
    for _ in range(max_length):
        outputs = model(
            input_ids=generated_ids[:, -1:],  # 只输入最后一个 token
            past_key_values=past_key_values,  # 使用缓存
            use_cache=True
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values  # 更新缓存
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    
    end_time = time.time()
    inference_time = end_time - start_time
    total_tokens = generated_ids.size(-1)
    tps = total_tokens / inference_time

    memory_allocated, memory_reserved = get_memory_usage(device)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text, inference_time, tps, memory_allocated, memory_reserved

# 从文件读取多行 prompt
def read_prompts_from_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        prompts = file.readlines()
    return [prompt.strip() for prompt in prompts if prompt.strip()]  # 去除空行和空格

# 比较性能
def compare_inference_with_file(filename):
    model_name = "gpt2"  # 选择的模型名称
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    
    # 读取文件中的所有 prompts
    prompts = read_prompts_from_file(filename)
    
    # 初始化累积统计
    total_baseline_time, total_baseline_tps = 0, 0
    total_kv_time, total_kv_tps = 0, 0
    total_baseline_mem_alloc, total_baseline_mem_res = 0, 0
    total_kv_mem_alloc, total_kv_mem_res = 0, 0
    valid_baseline_count, valid_kv_count = 0, 0  # 有效记录的计数

    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i + 1}/{len(prompts)}: {prompt[:50]}...")
        
        # Baseline 推理
        baseline_text, baseline_time, baseline_tps, baseline_mem_alloc, baseline_mem_res = baseline_inference(
            model, tokenizer, device, prompt, max_length=100
        )
        if baseline_tps <= 300:
            print(f"Baseline Text: {baseline_text[:50]}...")  # 打印生成内容的前50字符
            total_baseline_time += baseline_time
            total_baseline_tps += baseline_tps
            total_baseline_mem_alloc += baseline_mem_alloc
            total_baseline_mem_res += baseline_mem_res
            valid_baseline_count += 1  # 有效记录计数
        else:
            print(f"Baseline TPS too high ({baseline_tps:.2f}), skipping this prompt.")

        # KV-cache 推理
        kv_cache_text, kv_cache_time, kv_cache_tps, kv_cache_mem_alloc, kv_cache_mem_res = kv_cache_inference(
            model, tokenizer, device, prompt, max_length=100
        )
        if kv_cache_tps <= 300:
            print(f"KV-cache Text: {kv_cache_text[:50]}...")  # 打印生成内容的前50字符
            total_kv_time += kv_cache_time
            total_kv_tps += kv_cache_tps
            total_kv_mem_alloc += kv_cache_mem_alloc
            total_kv_mem_res += kv_cache_mem_res
            valid_kv_count += 1  # 有效记录计数
        else:
            print(f"KV-cache TPS too high ({kv_cache_tps:.2f}), skipping this prompt.")

    # 汇总性能结果
    print("\n--- Performance Summary ---")
    if valid_baseline_count > 0:
        print("Baseline Inference:")
        print(f"  Avg Time: {total_baseline_time / valid_baseline_count:.4f} seconds")
        print(f"  Avg Tokens/sec: {total_baseline_tps / valid_baseline_count:.4f}")
        print(f"  Avg GPU Memory Allocated: {total_baseline_mem_alloc / valid_baseline_count:.2f} MB")
        print(f"  Avg GPU Memory Reserved: {total_baseline_mem_res / valid_baseline_count:.2f} MB")
    else:
        print("Baseline Inference: No valid data after filtering.")

    if valid_kv_count > 0:
        print("\nKV-cache Inference:")
        print(f"  Avg Time: {total_kv_time / valid_kv_count:.4f} seconds")
        print(f"  Avg Tokens/sec: {total_kv_tps / valid_kv_count:.4f}")
        print(f"  Avg GPU Memory Allocated: {total_kv_mem_alloc / valid_kv_count:.2f} MB")
        print(f"  Avg GPU Memory Reserved: {total_kv_mem_res / valid_kv_count:.2f} MB")
    else:
        print("KV-cache Inference: No valid data after filtering.")

if __name__ == "__main__":
    # 提供文件名
    compare_inference_with_file("data.txt")

