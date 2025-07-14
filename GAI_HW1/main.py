import os
import pandas as pd
from groq import Groq
import random
import re
import csv

# 讀取 CSV 檔案
sample_file = "hw-1-prompt-engineering/mmlu_sample.csv"
submit_file = "hw-1-prompt-engineering/mmlu_submit.csv"

# 讀取並解析 mmlu_sample.csv（作為 prompt 參考）
df_sample = pd.read_csv(sample_file)

# 轉換為字典，按 task 分類題目
task_dict = {}
for _, row in df_sample.iterrows():
    row= row.to_dict()
    task = row["task"]
    if task not in task_dict:
        task_dict[task] = []
    task_dict[task].append(row)  # 儲存為字典，方便取用

# 讀取 mmlu_submit.csv
df_submit = pd.read_csv(submit_file)
# f_submit = df_submit.head(4)# 取前4題做測試

n = 0

# 設定 API Key
api_key = "api_key"//  # 替換為您的 Groq API 金鑰
client = Groq(api_key=api_key)

# 初始化一個列表來存儲結果
results = []

rows_list = df_submit.to_dict(orient="records")

# 逐題回答 submit.csv 中的問題
for i ,row in enumerate(rows_list[n:len(rows_list)]):
    try:
        
   
        task = row["task"]  # 獲取 task 類別
        question_text = row["input"]  # 要回答的題目
        options = f"A: {row['A']}\nB: {row['B']}\nC: {row['C']}\nD: {row['D']}"
        
        # 從 sample.csv 找對應的 prompt 題目
        example_prompts = random.sample(task_dict.get(task, []), min(3, len(task_dict.get(task, []))))
        
        # 建立模型的 prompt 訊息
        messages = [{"role": "system", "content": f"You are a highly knowledgeable expert in {task}, having studied numerous examples and correctly answered many similar questions. Use your expertise to choose the most accurate answer from the options: A, B, C, or D."}]

        # few-shot learning
        for example in example_prompts:
            example_text = f"Question: {example['input']}\nA: {example['A']}\nB: {example['B']}\nC: {example['C']}\nD: {example['D']}\n"
            example_answer = example['target']
            messages.append({"role": "user", "content": f"{example_text}"})
            messages.append({"role": "assistant", "content": f"Answer: {example_answer}"})

        messages.append({"role": "user", "content": f"Question: {question_text}\n{options}\n, Please respond with only a char: A, B, C, or D, and nothing else."})

        # 向 Groq 提交請求（第一次）
        chat_completion_1 = client.chat.completions.create(
            messages=messages,
            model="deepseek-r1-distill-llama-70b",
            temperature=0,
        )

        messages.append(chat_completion_1.choices[0].message)

        messages.append({"role": "user", "content": f"Question: {question_text}\n{options}\nAnswer with only one of the following chars: A, B, C, or D. Do not include any other text."})


        # 向 Groq 提交請求（第二次）
        chat_completion_2 = client.chat.completions.create(
            messages=messages,
            model="deepseek-r1-distill-llama-70b",
            temperature=0,
        )

        # 取得回答
        first_answer = chat_completion_2.choices[0].message.content
        first_answer = re.sub(r"<think>.*?</think>", "", first_answer,flags=re.DOTALL)
        last_char = first_answer[-1]

        print(f"Question {n + i + 1}: First Answer = {first_answer}")

        # 儲存結果
        results.append({"ID": row["Unnamed: 0"],  "target": last_char})
    except Exception as e:
        print(f"Error occurred at question {n + i + 1}: {e}")
        break 
        



file_exists = os.path.isfile("Results_format.csv")
with open("Results_format.csv", mode="a", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["ID", "target"])

    if not file_exists:  
        writer.writeheader()

    writer.writerows(results)  

print(f"Save the results to {"Results_format.csv"}")










