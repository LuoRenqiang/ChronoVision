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
        print(f"image failed: {image_path}: {e}")
        return None

def query_vllm(image_base64_list):
    system_prompt = (
        "You are an expert in historical visual analysis. "
        "Analyze the provided images for chronological clues such as technology, fashion, architecture, and photo quality. "
        "You must output a single JSON object containing: "
        "1. 'thinking': A detailed step-by-step analysis of which image is older. "
        "2. 'answer': Strictly the number '1' (if Image 1 is earlier) or '2' (if Image 2 is earlier)."
    )
    user_prompt = (
        "Compare these two images and determine which one appeared EARLIER in history.\n"
        "Image 1 is the first image provided.\n"
        "Image 2 is the second image provided.\n"
        "Output your analysis and final choice in JSON format."
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
        "max_tokens": 2048,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        if response.status_code != 200:
            return f"Error {response.status_code}"
            
        res_json = response.json()
        raw_content = res_json['choices'][0]['message']['content'].strip()

        try:
            parsed = json.loads(raw_content)
            final_ans = str(parsed.get("answer", "")).strip()
            if "1" in final_ans: return "1"
            if "2" in final_ans: return "2"
            return final_ans
        except json.JSONDecodeError:
            match = re.search(r'"answer":\s*"(\d)"', raw_content)
            return match.group(1) if match else raw_content
            
    except Exception as e:
        return f"Exception: {e}"

def extract_year_from_key(key_str):
    try:
        filename = os.path.basename(key_str)
        first_underscore_idx = filename.find('_')
        if first_underscore_idx != -1:
            year_str = filename[first_underscore_idx + 1 : first_underscore_idx + 5]
            if year_str.isdigit():
                return int(year_str)
        return None
    except Exception:
        return None

def process_category(category_path):
    cat_name = category_path.name
    subtask_dir = category_path / "subtask1"
    ans_path = category_path / "ans.json"
    pic_map_path = category_path / "test.json"

    if not (ans_path.exists() and pic_map_path.exists()):
        return

    with open(ans_path, 'r') as f:
        ground_truth = json.load(f)
    with open(pic_map_path, 'r') as f:
        name_mapping = json.load(f)

    prefix = cat_name.replace("_images", "")
    output_path = subtask_dir / f"{prefix}_subtask1_result.json"

    results = []
    all_tests = sorted(ground_truth.keys(), 
                        key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)

    for test_id in tqdm(all_tests, desc=f"Processing {cat_name}"):
        current_test_files = []
        for old_path_key, random_name_val in name_mapping.items():
            if old_path_key.startswith(f"{test_id}/"):
                year = extract_year_from_key(old_path_key)
                full_path = subtask_dir / test_id / random_name_val
                if full_path.exists() and year is not None:
                    current_test_files.append({"year": year, "path": full_path, "orig_key": old_path_key})

        if len(current_test_files) != 2:
            continue

        current_test_files.sort(key=lambda x: x['year'])
        file_early = current_test_files[0]
        file_late = current_test_files[1]

        ans_val = str(ground_truth[test_id])
        input_order = [file_early, file_late] if ans_val == "1" else [file_late, file_early]

        b64_list = [encode_image(item['path']) for item in input_order]
        if None in b64_list: continue
            
        prediction = query_vllm(b64_list)

        results.append({
            "test_id": test_id,
            "ground_truth": ans_val,
            "prediction": prediction,
            "is_correct": prediction == ans_val,
            "details": {
                "years": [item['year'] for item in input_order],
                "files": [item['orig_key'] for item in input_order]
            }
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    base_path = Path(BASE_DIR)
    categories = [d for d in base_path.iterdir() if d.is_dir()  and d.name.endswith("_images")]
    for cat in categories:
        process_category(cat)
