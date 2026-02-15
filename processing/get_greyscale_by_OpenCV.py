import cv2
import os
import numpy as np
from tqdm import tqdm

def cv2_imread_safe(file_path):
    raw_data = np.fromfile(file_path, dtype=np.uint8)
    img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    return img

def cv2_imwrite_safe(file_path, img):
    file_ext = os.path.splitext(file_path)[1]
    success, enc_data = cv2.imencode(file_ext, img)

    if success:
        enc_data.tofile(file_path)
    else:
        print(f"保存失败: {file_path}")

def convert_specific_folder():
    input_folder = r"your subfolder with color images"
    output_folder = r"your final subfolder with greyscale images"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    file_list = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]

    print(f"tot: {len(file_list)} images")

    for filename in tqdm(file_list, desc="now"):
        src_path = os.path.join(input_folder, filename)
        dst_path = os.path.join(output_folder, filename)
        img = cv2_imread_safe(src_path)

        if img is None:
            print(f"failed to open the image: {filename}")
            continue

        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_bw_3ch = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

            cv2_imwrite_safe(dst_path, img_bw_3ch)

        except Exception as e:
            print(f"failed: {filename}: {e}")

    print("-" * 30)
    print("OK!")


if __name__ == "__main__":
    convert_specific_folder()