import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

# 加载模型和分词器
def load_model_and_tokenizer(model_name="gpt2", quantize=False):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if quantize:
    # 动态量化模型（只对线性层进行量化，使用 int8）
        model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear},  # 只量化线性层
            dtype=torch.qint8  # 使用 int8 进行量化（不支持 float16）
        )
        print("Model quantized with int8.")
    
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

# 推理函数（通用框架）
def inference(model, tokenizer, device, prompt, max_length=50, use_kv_cache=False):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start_time = time.time()

    if use_kv_cache:
        past_key_values = None
        generated_ids = inputs["input_ids"]
        for _ in range(max_length):
            outputs = model(
                input_ids=generated_ids[:, -1:], 
                past_key_values=past_key_values, 
                use_cache=True
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    else:
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
        generated_ids = outputs

    end_time = time.time()
    inference_time = end_time - start_time
    total_tokens = generated_ids.size(-1)
    tps = total_tokens / inference_time

    memory_allocated, memory_reserved = get_memory_usage(device)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text, inference_time, tps, memory_allocated, memory_reserved

# 比较性能
def compare_inference_with_file(filename):
    model_name = "gpt2"
    
    # 加载标准模型
    model, tokenizer, device = load_model_and_tokenizer(model_name)
    # 加载量化模型
    quant_model, _, _ = load_model_and_tokenizer(model_name, quantize=True)
    
    # 读取文件中的所有 prompts
    prompts = read_prompts_from_file(filename)
    
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i + 1}/{len(prompts)}: {prompt[:50]}...")

        # 标准推理
        baseline_text, baseline_time, baseline_tps, baseline_mem_alloc, baseline_mem_res = inference(
            model, tokenizer, device, prompt, max_length=100, use_kv_cache=False
        )
        print(f"Baseline: {baseline_text[:50]}...")

        # 量化推理
        quant_text, quant_time, quant_tps, quant_mem_alloc, quant_mem_res = inference(
            quant_model, tokenizer, device, prompt, max_length=100, use_kv_cache=False
        )
        print(f"Quantization: {quant_text[:50]}...")

        # 量化 + KV-cache 推理
        quant_kv_text, quant_kv_time, quant_kv_tps, quant_kv_mem_alloc, quant_kv_mem_res = inference(
            quant_model, tokenizer, device, prompt, max_length=100, use_kv_cache=True
        )
        print(f"Quantization + KV-cache: {quant_kv_text[:50]}...")

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

    # 假设量化推理的相关统计信息（quantization_time, quantization_tps, quantization_mem_alloc, quantization_mem_res）需要在量化推理函数中处理
    if valid_quantization_count > 0:  # 如果有量化推理有效记录
        print("\nQuantization Inference:")
        print(f"  Avg Time: {total_quantization_time / valid_quantization_count:.4f} seconds")
        print(f"  Avg Tokens/sec: {total_quantization_tps / valid_quantization_count:.4f}")
        print(f"  Avg GPU Memory Allocated: {total_quantization_mem_alloc / valid_quantization_count:.2f} MB")
        print(f"  Avg GPU Memory Reserved: {total_quantization_mem_res / valid_quantization_count:.2f} MB")
    else:
        print("Quantization Inference: No valid data after filtering.")

    # 假设KV-cache与量化联合使用的推理的相关统计信息（kv_quantization_time, kv_quantization_tps, kv_quantization_mem_alloc, kv_quantization_mem_res）需要在联合推理函数中处理
    if valid_kv_quantization_count > 0:  # 如果有KV-cache和量化联合推理有效记录
        print("\nKV-cache + Quantization Inference:")
        print(f"  Avg Time: {total_kv_quantization_time / valid_kv_quantization_count:.4f} seconds")
        print(f"  Avg Tokens/sec: {total_kv_quantization_tps / valid_kv_quantization_count:.4f}")
        print(f"  Avg GPU Memory Allocated: {total_kv_quantization_mem_alloc / valid_kv_quantization_count:.2f} MB")
        print(f"  Avg GPU Memory Reserved: {total_kv_quantization_mem_res / valid_kv_quantization_count:.2f} MB")
    else:
        print("KV-cache + Quantization Inference: No valid data after filtering.")


# 从文件读取多行 prompt
def read_prompts_from_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        prompts = file.readlines()
    return [prompt.strip() for prompt in prompts if prompt.strip()]  # 去除空行和空格

if __name__ == "__main__":
    # 提供文件名
    compare_inference_with_file("data.txt")
