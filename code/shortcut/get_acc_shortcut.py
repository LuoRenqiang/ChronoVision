import os
import json
from pathlib import Path

BASE_DIR = "/hd/images"

def main():
    base_path = Path(BASE_DIR)
    categories = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.endswith("_images")])
    
    print(f"{'name (Category)':<25} | {'num':<6} | {'correct':<6} | {'acc'}")
    print("-" * 60)

    total_all = 0
    correct_all = 0

    for cat in categories:
        cat_name = cat.name
        prefix = cat_name.replace("_images", "")
        results_file = cat / "subtask1" / f"{prefix}_subtask1_result.json"

        if not results_file.exists():
            continue
        
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total = len(data)
        correct = sum(1 for item in data if item.get("is_correct") is True)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        total_all += total
        correct_all += correct
        
        print(f"{cat.name:<25} | {total:<6} | {correct:<6} | {accuracy:.2f}%")

    print("-" * 60)
    if total_all > 0:
        overall_acc = (correct_all / total_all) * 100
        print(f"{' (Overall)':<25} | {total_all:<6} | {correct_all:<6} | {overall_acc:.2f}%")

if __name__ == "__main__":

    main()
