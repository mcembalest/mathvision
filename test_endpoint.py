import argparse
import httpx
import json
import re

ENDPOINT_URL = "https://cerebella-org--example-sgl-vlm-model-generate.modal.run"
DATASET_URL = "https://raw.githubusercontent.com/mathllm/MATH-V/refs/heads/main/images/"
with open("test.jsonl") as f:
    QUESTION_DATA = [json.loads(x) for x in list(f)]

def generate(image_url: str, question: str):
    payload = {"image_url": image_url, "question": question}
    response = httpx.post(ENDPOINT_URL, json=payload, timeout=200.0)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_num", type=int, required=True)
    args = parser.parse_args()
    image_url = f"{DATASET_URL}/{args.image_num}.jpg" # image URL is 1-indexed
    data = QUESTION_DATA[args.image_num-1] # data file is 0-indexed
    question = data['question']
    question = re.sub(r"\n?<image\d+>", "", question) # remove <imageN> tags

    options = data.get('options', [])
    if options: # question expects a multiple choice answer
        # format options
        labels = ['A', 'B', 'C', 'D', 'E']
        if options == labels[:len(options)]:
            options_str = ", ".join(options)
        else:
            labeled_options = [f"{labels[i]}) {opt}" for i, opt in enumerate(options)]
            options_str = ", ".join(labeled_options)
        question = f"{question}\nOptions: {options_str}"
        prompt = f"Think, and then answer. IMPORTANT: This is multiple choice. Answer with A, B, C, D, or E (e.g. <answer>B</answer>). Place your thinking between <thinking> and </thinking> tags and then answer between <answer> and </answer> tags. {question}"
    else: # question expects a numerical answer
        prompt = f"Think, and then answer. IMPORTANT: Answer with only a single number (e.g. <answer>6</answer>). Place your thinking between <thinking> and </thinking> tags and then answer between <answer> and </answer> tags. {question}"
    print(prompt)
    response = generate(image_url=image_url, question=prompt)
    print("Response: ", response)
    answer = data['answer']
    print(f"Correct answer: {answer}")
