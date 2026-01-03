import asyncio
import httpx
import json
import re
from datetime import datetime

ENDPOINT_URL = "https://cerebella-org--example-sgl-vlm-model-generate.modal.run"
DATASET_URL = "https://raw.githubusercontent.com/mathllm/MATH-V/refs/heads/main/images"

async def generate(client: httpx.AsyncClient, semaphore: asyncio.Semaphore, 
                   image_num: int, question: str, options: list) -> dict:
    image_url = f"{DATASET_URL}/{image_num}.jpg"
    question = re.sub(r"\n?<image\d+>", "", question)

    if options: # question expects a multiple choice answer
        # format options
        labels = ['A', 'B', 'C', 'D', 'E']
        if options == labels[:len(options)]:
            options_str = ", ".join(options)
        else:
            labeled_options = [f"{labels[i]}) {opt}" for i, opt in enumerate(options)]
            options_str = ", ".join(labeled_options)
        question = f"{question}\nOptions: {options_str}"
        text = f"Think, and then answer. IMPORTANT: This is multiple choice. Answer with A, B, C, D, or E (e.g. <answer>B</answer>). Place your thinking between <thinking> and </thinking> tags and then answer between <answer> and </answer> tags. {question}"
    else: # question expects a numerical answer
        prompt = f"Think, and then answer. IMPORTANT: Answer with only a single number (e.g. <answer>6</answer>). Place your thinking between <thinking> and </thinking> tags and then answer between <answer> and </answer> tags. {question}"

    payload = {"image_url": image_url, "text": text}
    
    async with semaphore:
        response = await client.post(ENDPOINT_URL, json=payload, timeout=300.0)
        response.raise_for_status()
        return {"image_num": image_num, "response": response.json()}

async def run_benchmark(start_idx: int, end_idx: int, concurrency: int):
    with open("test.jsonl") as f:
        data = [json.loads(x) for x in f]
    
    if end_idx is None:
        end_idx = len(data)
    
    semaphore = asyncio.Semaphore(concurrency)
    
    async with httpx.AsyncClient() as client:
        tasks = [
            generate(client, semaphore, i, data[i-1]['question'], data[i-1].get('options', []))
            for i in range(start_idx, end_idx + 1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
 
    processed = []
    for i, result in enumerate(results, start=start_idx):
        entry = {
            "image_num": i,
            "question": data[i-1]['question'],
            "expected": data[i-1]['answer'],
        }
        if isinstance(result, Exception):
            entry["error"] = str(result)
            entry["raw_response"] = None
        else:
            entry["raw_response"] = result['response']
        processed.append(entry)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(processed, f, indent=2)
    
    print(f"Saved {len(processed)} results to {filename}")
    return processed

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.start, args.end, args.concurrency))