import json
import torch
import os
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Set the checkpoint directory for the LoRA model
peft_model_id = "checkpoints/0327_2/checkpoint-1100"

date_prefix = os.path.basename(os.path.dirname(peft_model_id))
checkpoint_number = os.path.basename(peft_model_id).split("-")[-1]

# Load the PEFT configuration, which contains all the settings for the LoRA model
config = PeftConfig.from_pretrained(peft_model_id)

# Load the base T5 model (T5 is a Seq2Seq architecture), with necessary settings like data type and device configuration
model = AutoModelForSeq2SeqLM.from_pretrained(
    config.base_model_name_or_path, 
    return_dict=True, 
    torch_dtype=torch.float32,
    device_map="auto"
)

# Load the PEFT model, which is the T5 model combined with LoRA parameters
model = PeftModel.from_pretrained(model, peft_model_id)


# Load the tokenizer to convert text into a format the model can understand
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the JSON file and parse its content, where each line represents a paper
def load_json_input(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

# Inference function that generates responses for each paper's abstract
def generate_responses(json_path):
    papers = load_json_input(json_path)
    responses = []
    for index, paper in enumerate(papers):
        paper_id = paper.get("paper_id")
        introduction = paper.get("introduction", "")

        # Print current processing paper_id
        print(f"Processing paper_id: {paper_id} ({index+1}/{len(papers)})", flush=True)

        inputs = tokenizer(introduction, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, min_length=100, max_length=500, early_stopping=True, temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=5.0)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        responses.append({"paper_id": paper_id, "abstract": response})
    return responses

# Save the generated results as JSON, with each JSON object written on a new line
def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"Results have been saved to {output_path}", flush=True)

# File paths
json_file = "data/test.json"
output_file = f"Submission_0415/{date_prefix}_{checkpoint_number}.json"

# Run the process
results = generate_responses(json_file)
save_to_json(results, output_file)
