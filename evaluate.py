import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def extract_answer(response) -> str | None:
    """Extract content from <answer> tags."""
    if not isinstance(response, str):
        return "None"
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    return match.group(1).strip() if match else "None"


def load_levels(test_file: str) -> dict[int, int]:
    """Load level information from test.jsonl, keyed by image number."""
    levels = {}
    with open(test_file) as f:
        for line in f:
            entry = json.loads(line)
            levels[int(entry["id"])] = entry["level"]
    return levels


def load_options(test_file: str) -> dict[int, list]:
    """Load options information from test.jsonl, keyed by image number."""
    options = {}
    with open(test_file) as f:
        for line in f:
            entry = json.loads(line)
            options[int(entry["id"])] = entry.get("options", [])
    return options


def evaluate(input_file: str, test_file: str = "test.jsonl"):
    with open(input_file) as f:
        results = json.load(f)

    # Load level and options info from test.jsonl
    test_path = Path(input_file).parent / test_file
    levels = load_levels(test_path)
    options_map = load_options(test_path)

    # Track stats by level
    level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    # Track stats by question type (multiple choice vs non-multiple choice)
    mc_stats = {"correct": 0, "total": 0}
    non_mc_stats = {"correct": 0, "total": 0}

    correct = 0
    for entry in results:
        expected = entry["expected"].strip()
        extracted = extract_answer(entry.get("raw_response", ""))
        entry["extracted"] = extracted
        entry["correct"] = extracted in expected or expected in extracted

        # Get level and options for this entry
        image_num = entry.get("image_num")
        level = levels.get(image_num)
        entry["level"] = level
        
        # Determine if multiple choice (non-empty options list)
        options_list = options_map.get(image_num, [])
        is_multiple_choice = len(options_list) > 0
        entry["is_multiple_choice"] = is_multiple_choice

        if level:
            level_stats[level]["total"] += 1
            if entry["correct"]:
                level_stats[level]["correct"] += 1

        # Track stats by question type
        if is_multiple_choice:
            mc_stats["total"] += 1
            if entry["correct"]:
                mc_stats["correct"] += 1
        else:
            non_mc_stats["total"] += 1
            if entry["correct"]:
                non_mc_stats["correct"] += 1

        if entry["correct"]:
            correct += 1

    # Print overall accuracy
    accuracy = correct / len(results) * 100
    print(f"Overall Accuracy: {correct}/{len(results)} ({accuracy:.1f}%)")

    # Print accuracy by level
    print("\nAccuracy by Level:")
    for level in sorted(level_stats.keys()):
        stats = level_stats[level]
        level_acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  Level {level}: {stats['correct']}/{stats['total']} ({level_acc:.1f}%)")
    
    # Print accuracy by question type
    print("\nAccuracy by Question Type:")
    mc_acc = mc_stats["correct"] / mc_stats["total"] * 100 if mc_stats["total"] > 0 else 0
    non_mc_acc = non_mc_stats["correct"] / non_mc_stats["total"] * 100 if non_mc_stats["total"] > 0 else 0
    print(f"  Multiple Choice: {mc_stats['correct']}/{mc_stats['total']} ({mc_acc:.1f}%)")
    print(f"  Non-Multiple Choice: {non_mc_stats['correct']}/{non_mc_stats['total']} ({non_mc_acc:.1f}%)")

    output_file = input_file.replace(".json", "_evaluated.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", help="Path to results JSON file")
    args = parser.parse_args()
    evaluate(args.input_file)
