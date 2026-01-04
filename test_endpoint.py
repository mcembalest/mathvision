import argparse
import asyncio
import httpx
import json
import re
from datetime import datetime
from typing import Any

ENDPOINT_URL = "https://cerebella-org--sgl-vlm-model-generate.modal.run"
DATASET_URL = "https://raw.githubusercontent.com/mathllm/MATH-V/refs/heads/main/images"
TEST_FILE = "test.jsonl"


def load_test_data(test_file: str = TEST_FILE) -> list[dict[str, Any]]:
    with open(test_file) as f:
        return [json.loads(line) for line in f]


def clean_question(question: str) -> str:
    # remove <imageN> tags
    return re.sub(r"\n?<image\d+>", "", question)


def format_options(options: list) -> str:
    labels = ["A", "B", "C", "D", "E"]
    if options == labels[: len(options)]:
        return ", ".join(options)
    labeled_options = [f"{labels[i]}) {opt}" for i, opt in enumerate(options)]
    return ", ".join(labeled_options)


def build_text(question: str, options: list) -> str:
    question = clean_question(question)
    if options:
        options_str = format_options(options)
        question = f"{question}\nOptions: {options_str}"
        return (
            "Think, and then answer. IMPORTANT: This is multiple choice. "
            "Answer with A, B, C, D, or E (e.g. <answer>B</answer>). "
            "Place your thinking between <thinking> and </thinking> tags and then "
            f"answer between <answer> and </answer> tags. {question}"
        )
    return (
        "Think, and then answer. IMPORTANT: Answer with only a single number "
        "(e.g. <answer>6</answer>). Place your thinking between <thinking> and </thinking> tags and then "
        f"answer between <answer> and </answer> tags. {question}"
    )


def generate(image_url: str, text: str, timeout_s: float = 300.0):
    payload = {"image_url": image_url, "text": text}
    response = httpx.post(ENDPOINT_URL, json=payload, timeout=timeout_s)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}")


async def generate_async(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    image_num: int,
    question: str,
    options: list,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    image_url = f"{DATASET_URL}/{image_num}.jpg"
    text = build_text(question, options)
    payload = {"image_url": image_url, "text": text}
    async with semaphore:
        response = await client.post(ENDPOINT_URL, json=payload, timeout=timeout_s)
        response.raise_for_status()
        return {"image_num": image_num, "raw_response": response.json()}


def resolve_end_idx(start: int, n: int | None, total: int) -> int:
    if n is None:
        return total
    if n <= 0:
        raise ValueError("--n must be a positive integer or omitted for all")
    return min(start + n - 1, total)


def run_single(image_num: int):
    data = load_test_data()
    if image_num < 1 or image_num > len(data):
        raise ValueError(f"--start must be between 1 and {len(data)} (got {image_num})")

    entry = data[image_num - 1]  # test.jsonl is 0-indexed, image_num is 1-indexed
    image_url = f"{DATASET_URL}/{image_num}.jpg"
    text = build_text(entry["question"], entry.get("options", []))

    print(text)
    response = generate(image_url=image_url, text=text)
    print("Response: ", response)
    print(f"Correct answer: {entry['answer']}")


async def run_benchmark(start: int, n: int | None, concurrency: int) -> list[dict[str, Any]]:
    data = load_test_data()
    end_idx = resolve_end_idx(start=start, n=n, total=len(data))

    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [
            generate_async(
                client,
                semaphore,
                image_num=i,
                question=data[i - 1]["question"],
                options=data[i - 1].get("options", []),
            )
            for i in range(start, end_idx + 1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    processed: list[dict[str, Any]] = []
    for i, result in enumerate(results, start=start):
        entry = {
            "image_num": i,
            "question": data[i - 1]["question"],
            "expected": data[i - 1]["answer"],
        }
        if isinstance(result, Exception):
            entry["error"] = str(result)
            entry["raw_response"] = None
        else:
            entry["raw_response"] = result["raw_response"]
        processed.append(entry)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(processed, f, indent=2)

    print(f"Saved {len(processed)} results to {filename}")
    return processed


async def resume_benchmark(resume_file: str, n: int | None, concurrency: int) -> list[dict[str, Any]]:
    with open(resume_file) as f:
        existing_results = json.load(f)

    failed_entries = [entry for entry in existing_results if "error" in entry]
    if not failed_entries:
        print("No entries with errors found in the resume file.")
        return existing_results

    failed_image_nums = [int(entry["image_num"]) for entry in failed_entries]
    if n is not None:
        if n <= 0:
            raise ValueError("--n must be a positive integer or omitted for all")
        failed_image_nums = failed_image_nums[:n]

    print(f"Found {len(failed_image_nums)} entries with errors to retry")

    test_data = load_test_data()
    test_data_map = {int(entry["id"]): entry for entry in test_data}
    valid_image_nums = [img_num for img_num in failed_image_nums if img_num in test_data_map]
    skipped = len(failed_image_nums) - len(valid_image_nums)
    if skipped > 0:
        print(f"Skipped {skipped} image_nums not found in {TEST_FILE}")
    if not valid_image_nums:
        print("No valid image_nums to retry.")
        return existing_results

    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [
            generate_async(
                client,
                semaphore,
                image_num=img_num,
                question=test_data_map[img_num]["question"],
                options=test_data_map[img_num].get("options", []),
            )
            for img_num in valid_image_nums
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    new_results = []
    for img_num, result in zip(valid_image_nums, results):
        entry = {
            "image_num": img_num,
            "question": test_data_map[img_num]["question"],
            "expected": test_data_map[img_num]["answer"],
        }
        if isinstance(result, Exception):
            entry["error"] = str(result)
            entry["raw_response"] = None
        else:
            entry["raw_response"] = result["raw_response"]
        new_results.append(entry)

    new_results_map = {entry["image_num"]: entry for entry in new_results}
    for i, entry in enumerate(existing_results):
        img_num = int(entry.get("image_num"))
        if img_num in new_results_map:
            existing_results[i] = new_results_map[img_num]

    with open(resume_file, "w") as f:
        json.dump(existing_results, f, indent=2)

    print(f"Updated {len(new_results)} entries in {resume_file}")
    return existing_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Number of images to test (default: all)")
    parser.add_argument("--start", type=int, default=1, help="1-indexed image id to start from")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--resume_file",
        type=str,
        default=None,
        help="Path to JSON results file to resume from. Retries entries with errors.",
    )
    args = parser.parse_args()

    if args.resume_file:
        asyncio.run(resume_benchmark(args.resume_file, args.n, args.concurrency))
    elif args.n == 1:
        run_single(args.start)
    else:
        asyncio.run(run_benchmark(args.start, args.n, args.concurrency))
