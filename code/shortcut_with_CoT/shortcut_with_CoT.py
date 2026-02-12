import os
import json
import re
import base64
import requests
import math
import io
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen3-VL-4B-Instruct"
BASE_DIR = "/hd/images"

def encode_image(image_path, max_pixels=2800000):
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            orig_w, orig_h = img.size
            current_pixels = orig_w * orig_h
            target_w, target_h = orig_w, orig_h
            if current_pixels > max_pixels:
                scale = math.sqrt(max_pixels / current_pixels)
                target_w = int(orig_w * scale)
                target_h = int(orig_h * scale)
            target_w = max(28, (target_w // 28) * 28)
            target_h = max(28, (target_h // 28) * 28)
            if (target_w, target_h) != (orig_w, orig_h):
                img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=90)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"图片处理失败 {image_path}: {e}")
        return None

def query_vllm(image_base64_list):
    system_prompt = (
        "You are an expert in historical visual analysis and chronological reasoning. "
        "Your goal is to determine the temporal order of images by identifying intrinsic time anchors.\n\n"
        "You must output a single JSON object with the following keys:\n"
        "1. 'time_anchors': Identified entities for each image.\n"
        "2. 'historical_context': Earliest appearance dates for these entities.\n"
        "3. 'reasoning_path': Step-by-step logical deduction.\n"
        "4. 'bias_mitigation': Self-correction to exclude color bias.\n"
        "5. 'answer': Strictly the string '1' or '2'."
    )
    user_prompt = (
        "Analyze these two images and determine which one appeared EARLIER in history.\n"
        "Follow these steps strictly:\n"
        "Step 1: Identify visual entities.\n"
        "Step 2: List historical appearance times.\n"
        "Step 3: Detail thinking process.\n"
        "Step 4: Perform Bias-Check.\n\n"
        "Output everything in a structured JSON format."
    )

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_list[0]}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64_list[1]}"}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 30000,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        if response.status_code != 200:
            return "Error", {"error": response.text}

        raw_content = response.json()['choices'][0]['message']['content'].strip()
        
        try:
            parsed = json.loads(raw_content)
            ans = str(parsed.get("answer", "")).strip()
            match = re.search(r'[12]', ans)
            clean_ans = match.group(0) if match else ans
            thinking_data = {k: v for k, v in parsed.items() if k != "answer"}
            
            return clean_ans, thinking_data
        except json.JSONDecodeError:
            return "Parse Fail", {"raw_output": raw_content}

    except Exception as e:
        return "Exception", {"exception": str(e)}

def extract_year_from_key(key_str):
    try:
        filename = os.path.basename(key_str)
        idx = filename.find('_')
        if idx != -1:
            year_str = filename[idx + 1 : idx + 5]
            if year_str.isdigit(): return int(year_str)
        return None
    except: return None

def process_category(category_path):
    cat_name = category_path.name
    subtask_dir = category_path / "subtask1"
    ans_path = category_path / "ans.json"
    pic_map_path = category_path / "picture_name_random.json"

    if not (ans_path.exists() and pic_map_path.exists()): return

    with open(ans_path, 'r') as f: ground_truth = json.load(f)
    with open(pic_map_path, 'r') as f: name_mapping = json.load(f)

    prefix = cat_name.replace("_images", "")
    output_path = subtask_dir / f"{prefix}_subtask1_result.json"

    results = []
    correct_count = 0
    total_processed = 0

    all_tests = sorted(ground_truth.keys(), 
                        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

    print(f"\nnow solving: {cat_name}")

    for test_id in tqdm(all_tests, desc=f"Processing {cat_name}"):
        current_test_files = []
        for old_path_key, random_name_val in name_mapping.items():
            if old_path_key.startswith(f"{test_id}/"):
                year = extract_year_from_key(old_path_key)
                full_path = subtask_dir / test_id / random_name_val
                if full_path.exists() and year is not None:
                    current_test_files.append({"year": year, "path": full_path, "orig_key": old_path_key})

        if len(current_test_files) != 2: continue

        current_test_files.sort(key=lambda x: x['year'])
        ans_val = str(ground_truth[test_id])
        input_order = [current_test_files[0], current_test_files[1]] if ans_val == "1" else [current_test_files[1], current_test_files[0]]

        b64_list = [encode_image(item['path']) for item in input_order]
        if None in b64_list: continue

        prediction, thinking_detail = query_vllm(b64_list)

        is_correct = (prediction == ans_val)
        if is_correct: correct_count += 1
        total_processed += 1

        results.append({
            "test_id": test_id,
            "ground_truth": ans_val,
            "prediction": prediction,
            "is_correct": is_correct,
            "thinking_process": thinking_detail,
            "details": {
                "years": [item['year'] for item in input_order],
                "files": [item['orig_key'] for item in input_order]
            }
        })

        if total_processed % 5 == 0:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    accuracy = (correct_count / total_processed * 100) if total_processed > 0 else 0
    final_data = {
        "summary": {
            "category": cat_name,
            "accuracy": f"{accuracy:.2f}%",
            "correct": correct_count,
            "total": total_processed
        },
        "results": results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
    
    print(f" {cat_name} acc: {accuracy:.2f}%")

if __name__ == "__main__":
    base_path = Path(BASE_DIR)
    categories = [d for d in base_path.iterdir() if d.is_dir() and d.name.endswith("_images")]
    for cat in categories:
        process_category(cat)