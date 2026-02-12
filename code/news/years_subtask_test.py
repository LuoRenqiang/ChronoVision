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
PROMPT = "In which year did this image first appear? Respond only with the 4-digit year (e.g., 2000) and nothing else."
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
        print(f"image failed to prepare: {image_path}: {e}")
        return None

def query_internvl(image_path, prompt):
    base64_image = encode_image(image_path)
    
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {str(e)}"

def run_test():
    for folder_name in os.listdir(ROOT_DIR):
        if not folder_name.endswith("_images"):
            continue
        abc_prefix = folder_name.replace("_images", "")
        
        years_dir = os.path.join(ROOT_DIR, folder_name, "years")
        if not os.path.exists(years_dir):
            print(f" {folder_name}: can't find files")
            continue
            
        print(f"\ncategory: {folder_name}")

        image_extensions = ('.jpg', '.jpeg', '.png', '.webp')
        image_files = [f for f in os.listdir(years_dir) if f.lower().endswith(image_extensions)]
        image_files.sort(key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x)

        results = []

        for img_file in tqdm(image_files, desc=f"Testing {folder_name}"):
            img_path = os.path.join(years_dir, img_file)

            answer = query_internvl(img_path, PROMPT)
            
            results.append({
                "image_id": img_file,
                "prompt": PROMPT,
                "model_answer": answer
            })

        output_path = os.path.join(years_dir, f"{abc_prefix}_years_result.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
            
        print(f"saved at: {output_path}")

if __name__ == "__main__":
    run_test()