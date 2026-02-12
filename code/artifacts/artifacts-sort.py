import os
import json
import base64
import requests
import random
import math
import io
from tqdm import tqdm
from PIL import Image
from PIL import Image, ImageFile

INPUT_FILE = "/hd/dynasty_sort_test_en.jsonl"
OUTPUT_FILE = "/hd/images/sort_ans.jsonl"
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen3-VL-4B-Instruct"
BASE_DIR_LINUX = "/hd/Images_dynasty"

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
        print(f"image fail:{image_path}: {e}")
        return None
def get_processed_ids(output_file):
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "id" in data:
                        processed_ids.add(int(data["id"]))
                except:
                    continue
    return processed_ids

def process_sort_test():
    processed_ids = get_processed_ids(OUTPUT_FILE)
    if processed_ids:
        print(f"have solved: {len(processed_ids)}, continue ")

    output_dir = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        all_lines = [line.strip() for line in f if line.strip()]


    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
        for line in tqdm(all_lines, desc="Processing Sort Task"):
            try:
                data = json.loads(line)
                item_id = int(data.get('id', 0))
            except:
                continue

            if not (1 <= item_id <= 1000) or item_id in processed_ids:
                continue

            message_content = [{"type": "text", "text": data['prompt']}]
            img_paths = data.get('images', [])
            img_error = False
            
            for win_path in img_paths:
                linux_path = win_path.replace('E:\\Images_dynasty', BASE_DIR_LINUX).replace('\\', '/')
                if not os.path.exists(linux_path):
                    img_error = True
                    break
                
                try:
                    base64_img = encode_image(linux_path)
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                    })
                except:
                    img_error = True
                    break
            
            if img_error:
                continue

            # --- API ---
            try:
                system_prompt = (
                    "You are a professional historian and artifact expert. "
                    "Analyze the provided images and sort them in chronological order from oldest to newest. "
                    "CRITICAL: You must output ONLY a valid JSON object. Do not include any conversational text or thinking process outside the JSON. "
                    "If you need to think, put your reasoning inside a 'thought' key within the JSON."
                )
                refined_user_prompt = (
                    f"{data['prompt']}\n\n"
                    "Requirement: Sort the images by dynasty. "
                    "Output strictly in this JSON format: {\"ans\": \"the_sorted_indices_here\"}. "
                    "Example: {\"ans\": \"2, 3, 1, 4, 5\"}"
                )

                message_content[0]["text"] = refined_user_prompt

                payload = {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message_content}
                    ],
                    "temperature": 0.1,
                    "response_format": {"type": "json_object"} 
                }

                response = requests.post(API_URL, json=payload, timeout=180)
                response.raise_for_status()

                raw_ans = response.json()['choices'][0]['message']['content'].strip()

                try:
                    parsed_json = json.loads(raw_ans)
                    ans_content = parsed_json.get("ans", raw_ans)
                except:
                    ans_content = raw_ans
                
                result = {
                    "id": item_id,
                    "category": data.get('category', ''),
                    "ans": ans_content,
                    "ground_truth": data.get('ground_truth', [])
                }
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()
                processed_ids.add(item_id)

            except Exception as e:
                print(f"\n[failed] ID {item_id}: {e}")

    print(f"\n[over] saved at: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_sort_test()