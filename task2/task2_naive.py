import openai
from datasets import load_dataset
from tqdm import tqdm
import re
import json

# 配置 OpenAI API
openai.api_key = "sk-9e662dcc789a4927bcd4289c1216316a"
openai.api_base = "https://api.deepseek.com"
MODEL = "deepseek-chat"

# 答案提取函数
def extract_answer(answer_text):
    """
    从 GSM8K 的答案字段中提取数值答案。
    答案位于连续的 '####' 之后。
    """
    match = re.search(r"####\s*([\d\.\-eE]+)", answer_text)
    if match:
        return match.group(1)
    return None

# 模型调用函数
def query_model(question, model=MODEL):
    """
    使用给定问题调用模型，并要求其输出符合格式。
    """
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"For each question, show the answer started with \"####\" Make sure that the answer is only a number without any other words!\n Question: \n {question}\n"
        }
    ]
        
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=prompt,
            temperature=0.5  # 控制回答的确定性
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"模型调用错误: {e}")
        return None

# 正确性判定函数
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
dataset = load_dataset("gsm8k","main", split="test")

# 评估过程
print("开始评估...")
correct = 0
total = len(dataset)
nums = 0
predictions=[]
with open("check_naive.json","w") as f:
    for item in tqdm(dataset, desc="Evaluating GSM8K"):
        question = item['question']
        ground_truth = item['answer']
        prediction = query_model(question)

        
        if prediction and is_correct(prediction, ground_truth):
            correct += 1
        else:
            result = {
                "index": nums,
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
            }
            predictions.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

# 输出正确率
accuracy = correct / total * 100
print(f"GSM8K 正确率: {accuracy:.2f}% ({correct}/{total})")
