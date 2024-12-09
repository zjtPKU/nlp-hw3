import openai
from datasets import load_dataset
from tqdm import tqdm
import re
import json

# 配置 OpenAI API
openai.api_key = "sk-9e662dcc789a4927bcd4289c1216316a"
openai.api_base = "https://api.deepseek.com"
MODEL = "deepseek-chat"
# 模型调用函数（Reflection Prompt）

def extract_answer(answer_text):

    match = re.search(r"####\s*([\d\.\-eE]+)", answer_text)
    if match:
        return match.group(1)
    return None

def query_model_with_reflection(question, model=MODEL):

    # Step 1: 初始解答
    initial_prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": 
            f"""
            For each question, provide a detailed solution. Then, show the answer started with \"####\" Make sure that the answer is only a number without any other words!\n Question: \n {question}\n
            """
        }
    ]
    
    try:
        initial_response = openai.ChatCompletion.create(
            model=model,
            messages=initial_prompt,
            temperature=0.5
        )
        initial_answer = initial_response['choices'][0]['message']['content']
    except Exception as e:
        print(f"初始解答调用错误: {e}")
        return None

    # Step 2: 反思与修订
    reflection_prompt = [
        {"role": "system", "content": "You are a reflective assistant who verifies and improves math problem solutions."},
        {"role": "user", "content": 
            f"""
            Question: \n {question}\n
            Here is the initial solution I provided:
            {initial_answer}

            Reflect on the solution. Are there any errors or areas for improvement? 
            - If any issues are found, provide a corrected solution. 
            - If the solution is correct, confirm that it is accurate.

            Provide a detailed solution. Then, show the answer started with \"####\" Make sure that the answer is only a number without any other words!
            """
        }
    ]
    
    try:
        reflection_response = openai.ChatCompletion.create(
            model=model,
            messages=reflection_prompt,
            temperature=0.5
        )
        reflection_answer = reflection_response['choices'][0]['message']['content']
    except Exception as e:
        print(f"反思与修订调用错误: {e}")
        return initial_answer  # 如果反思失败，返回初始解答
    
    return reflection_answer

# 正确性判定函数保持不变
def is_correct(predicted, ground_truth):
    """
    判断模型预测是否正确。
    """
    try:
        predicted_value = float(extract_answer(predicted))  # 提取数值
        ground_truth_value = float(extract_answer(ground_truth))  # 提取标准答案
        return abs(predicted_value - ground_truth_value) < 1e-3  # 考虑浮点误差
    except:
        return False

# 加载 GSM8K 数据集
print("加载 GSM8K 数据集...")
dataset = load_dataset("gsm8k", "main", split="test")

# 评估过程
print("开始评估 (Reflection)...")
correct = 0
total = len(dataset)
predictions = []
with open("check_Reflection.json","w") as f:
    for item in tqdm(dataset, desc="Evaluating GSM8K with Reflection"):
        question = item['question']
        ground_truth = item['answer']
        prediction = query_model_with_reflection(question)
        if prediction and is_correct(prediction, ground_truth):
            correct += 1
        else:
            result = {
                    "question": question,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                }
            predictions.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            

# 输出正确率
accuracy = correct / total * 100
print(f"GSM8K (Reflection) 正确率: {accuracy:.2f}% ({correct}/{total})")
