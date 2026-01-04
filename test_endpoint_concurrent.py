import argparse
import asyncio
import httpx
import json
import re
from datetime import datetime

ENDPOINT_URL = "https://cerebella-org--sgl-vlm-model-generate.modal.run"
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
        text = f"Think, and then answer. IMPORTANT: This is multiple choice. Answer with A, B, C, D, or E (e.g. <answer>B</answer>). Place your thinking between <thinking> and </thinking> tags and then answer between <answer> and </answer> tags. {question}\nOptions: {options_str}"
    else: # question expects a numerical answer
        text = f"Think, and then answer. IMPORTANT: Answer with only a single number (e.g. <answer>6</answer>). Place your thinking between <thinking> and </thinking> tags and then answer between <answer> and </answer> tags. {question}"

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

async def resume_benchmark(resume_file: str, end_idx: int, concurrency: int):
    with open(resume_file) as f:
        existing_results = json.load(f)
    
    failed_entries = [entry for entry in existing_results if "error" in entry]
    
    if not failed_entries:
        print("No entries with errors found in the resume file.")
        return existing_results
    
    failed_image_nums = [entry["image_num"] for entry in failed_entries]
    
    # end_idx limit if provided (acts as a counter)
    if end_idx is not None:
        failed_image_nums = failed_image_nums[:end_idx]
    
    print(f"Found {len(failed_image_nums)} entries with errors to retry")
    
    with open("test.jsonl") as f:
        test_data = [json.loads(x) for x in f]
    
    # test.jsonl is 0-indexed, image_num is 1-indexed
    test_data_map = {i+1: test_data[i] for i in range(len(test_data))}    
    valid_image_nums = [img_num for img_num in failed_image_nums if img_num in test_data_map]
    skipped = len(failed_image_nums) - len(valid_image_nums)
    if skipped > 0:
        print(f"Skipped {skipped} image_nums not found in test.jsonl")
    
    if not valid_image_nums:
        print("No valid image_nums to retry.")
        return existing_results
    
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [
            generate(
                client, 
                semaphore, 
                img_num, 
                test_data_map[img_num]['question'], 
                test_data_map[img_num].get('options', [])
            )
            for img_num in valid_image_nums
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    new_results = []
    for img_num, result in zip(valid_image_nums, results):
        entry = {
            "image_num": img_num,
            "question": test_data_map[img_num]['question'],
            "expected": test_data_map[img_num]['answer'],
        }
        if isinstance(result, Exception):
            entry["error"] = str(result)
            entry["raw_response"] = None
        else:
            entry["raw_response"] = result['response']
        new_results.append(entry)
    
    new_results_map = {entry["image_num"]: entry for entry in new_results}
    for i, entry in enumerate(existing_results):
        if entry["image_num"] in new_results_map:
            existing_results[i] = new_results_map[entry["image_num"]]
    
    with open(resume_file, "w") as f:
        json.dump(existing_results, f, indent=2)
    
    print(f"Updated {len(new_results)} entries in {resume_file}")
    return existing_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--resume_file", type=str, default=None, 
                       help="Path to JSON results file to resume from. Retries entries with errors.")
    args = parser.parse_args()

    if args.resume_file:
        asyncio.run(resume_benchmark(args.resume_file, args.end, args.concurrency))
    else:
        asyncio.run(run_benchmark(args.start, args.end, args.concurrency))