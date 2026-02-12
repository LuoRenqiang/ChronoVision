import os
import json
import base64
import requests
import math
import io
from tqdm import tqdm
from PIL import Image
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen3-VL-4B-Instruct"
ROOT_DIR = "/hd/images"

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
        print(f"failed {image_path}: {e}")
        return None

def query_internvl_mmt(image_paths, event_description):
    content = []

    for i, path in enumerate(image_paths, 1):
        b64_data = encode_image(path)
        if b64_data:
            content.append({"type": "text", "text": f"Image {i}:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
            })

    prompt_text = (
        f"\n The images are numbered '1' '2' '3' '4' in the order they were input(e.g. '1' represents the first input image, '4' represents the last input image)"
        f"\nEvent Description: {event_description}\n"
        f"Based on the visual evidence, which image was taken in the same year as the event?"
        f"Respond ONLY with the digit.(e.g. 1)"
    )
    content.append({"type": "text", "text": prompt_text})
    
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

def run_mmt_test_aggregated():
    for folder_name in os.listdir(ROOT_DIR):
        if not folder_name.endswith("_images"):
            continue
        
        abc_prefix = folder_name.replace("_images", "")
        base_path = os.path.join(ROOT_DIR, folder_name)
        mmt_test_root = os.path.join(base_path, "MMT-test")

        json_info_path = os.path.join(base_path, f"{abc_prefix}-MMT-test-information.json")
        if not os.path.exists(json_info_path):
            print(f" {folder_name}: can't find JSON")
            continue


        with open(json_info_path, 'r', encoding='utf-8') as f:
            test_data_list = json.load(f)
            if isinstance(test_data_list, dict):
                test_data_list = [test_data_list]

        print(f"\nnow: {folder_name} (tot cases: {len(test_data_list)} )")

        all_results_for_category = []

        for test_item in tqdm(test_data_list, desc=f"Testing {abc_prefix}"):
            test_id = test_item.get("test_id")
            event_desc = test_item.get("news", {}).get("description", "N/A")
            ground_truth = test_item.get("correct_answer")

            test_folder = os.path.join(mmt_test_root, test_id)
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.JPG', '.JPEG', '.PNG')

            img_paths = [
                next(os.path.join(test_folder, f"{i}{ext}") 
                    for ext in valid_exts 
                    if os.path.exists(os.path.join(test_folder, f"{i}{ext}")))
                for i in range(1, 5)
            ]

            if not all(os.path.exists(p) for p in img_paths):
                print(f" test {test_id}: can't find images")
                continue

            model_ans = query_internvl_mmt(img_paths, event_desc)

            all_results_for_category.append({
                "test_id": test_id,
                "event": event_desc,
                "model_response": model_ans,
                "ground_truth": ground_truth,
                "is_correct": str(ground_truth) in model_ans
            })

        if all_results_for_category:
            summary_file_path = os.path.join(mmt_test_root, f"{abc_prefix}_MMT_result.json")
            os.makedirs(mmt_test_root, exist_ok=True)
            
            with open(summary_file_path, 'w', encoding='utf-8') as f:
                json.dump(all_results_for_category, f, ensure_ascii=False, indent=4)
            
            correct_count = sum(1 for res in all_results_for_category if res["is_correct"])
            accuracy = correct_count / len(all_results_for_category)
            print(f"  >> {folder_name} solved. acc: {accuracy:.2%} (saved at {summary_file_path})")

if __name__ == "__main__":
    run_mmt_test_aggregated()