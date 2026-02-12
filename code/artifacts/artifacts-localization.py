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

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


BASE_DIR = "/hd/Images_dynasty"
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen3-VL-4B-Instruct"

DYNASTIES = [
    "唐(Tang Dynasty)",
    "宋(Song Dynasty)",
    "元(Yuan Dynasty)",
    "明(Ming Dynasty)",
    "清(Qing Dynasty)"
]

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
        print(f"image failed {image_path}: {e}")
        return None

def get_processed_ids(output_path):
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_ids.add(data.get('id'))
                except:
                    continue
    return processed_ids

def generate_random_prompt():
    shuffled = DYNASTIES.copy()
    random.shuffle(shuffled)
    
    letters = ['A', 'B', 'C', 'D', 'E']
    mapping = dict(zip(letters, shuffled))
    
    options_str = "\n".join([f"{l}. {d}" for l, d in mapping.items()])
    
    prompt = f"""Which historical period does the style of the item in this image belong to, among the following options? 
Options:
{options_str}

Please output the letter and the dynasty name only (e.g., 'A. 唐(Tang Dynasty)')."""
    
    return prompt, mapping

def process_benchmark(file_path):
    file_name = os.path.basename(file_path)
    name_part = file_name.replace('_benchmark.jsonl', '')
    output_name = f"qwen3_4B_{name_part}_ans.jsonl"
    output_path = os.path.join(BASE_DIR, output_name)
    
    processed_ids = get_processed_ids(output_path)

    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())

    print(f"\n[begin] {file_name} | jumped: {len(processed_ids)}")

    with open(file_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'a', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc=f"Processing {name_part}"):
            line = line.strip()
            if not line: continue
                
            try:
                data = json.loads(line)
            except: continue

            if data.get('id') in processed_ids:
                continue

            original_path = data.get('image', '')
            linux_path = original_path.replace('E:\\Images_dynasty', BASE_DIR).replace('\\', '/')
            
            if not os.path.exists(linux_path):
                continue

            current_prompt, current_mapping = generate_random_prompt()

            try:
                base64_image = encode_image(linux_path)
                payload = {
                    "model": MODEL_NAME,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": current_prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                }
                            ]
                        }
                    ],
                    "temperature": 0.1
                }

                response = requests.post(API_URL, json=payload, timeout=60)
                response.raise_for_status()
                ans_content = response.json()['choices'][0]['message']['content'].strip()

                output_item = data.copy()
                output_item['model_response'] = ans_content
                output_item['option_mapping'] = current_mapping
                
                f_out.write(json.dumps(output_item, ensure_ascii=False) + '\n')
                f_out.flush()

            except Exception as e:
                print(f"\n[failed] ID {data.get('id')}: {e}")

def main():
    if not os.path.exists(BASE_DIR):
        return
    benchmark_files = [f for f in os.listdir(BASE_DIR) if f.endswith('_benchmark.jsonl')]
    for bf in benchmark_files:
        process_benchmark(os.path.join(BASE_DIR, bf))
    print("\nover！")

if __name__ == "__main__":
    main()