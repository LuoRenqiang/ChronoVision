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