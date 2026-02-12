import os
import json
import re

ROOT_DIR = "/hd/images"
MIN_YEAR = 1900
MAX_YEAR = 2025

def extract_year_from_filename(filename):
    if not filename: return None
    match = re.search(r'(\d{4})', filename)
    return int(match.group(1)) if match else None

def extract_year_from_model_answer(answer):
    if not isinstance(answer, str):
        return None

    matches = re.findall(r'(\d{4})', answer)
    for m in matches:
        year = int(m)
        if MIN_YEAR <= year <= MAX_YEAR:
            return year
            
    return None

def analyze_results_detailed():
    overall_y_total_count = 0
    overall_y_correct_count = 0
    overall_y_ae_sum = 0
    overall_y_mae_valid_count = 0
    
    overall_mmt_total = 0
    overall_mmt_correct = 0

    header = f"{'Image_ID':<15} | {'GT Year':<10} | {'Pred Year':<10} | {'AE':<6} | {'Original Filename'}"
    print("=" * 110)
    print(header)
    print("=" * 110)

    for folder_name in sorted(os.listdir(ROOT_DIR)):
        if not folder_name.endswith("_images"):
            continue

        abc_prefix = folder_name.replace("_images", "")
        #print(abc_prefix)
        base_path = os.path.join(ROOT_DIR, folder_name)
        years_dir = os.path.join(base_path, "years")
        years_ans_path = os.path.join(years_dir, f"{abc_prefix}_years_result.json")
        years_map_path = os.path.join(years_dir, "year_name.json")
        
        if os.path.exists(years_ans_path) and os.path.exists(years_map_path):         
            with open(years_ans_path, 'r', encoding='utf-8') as f:
                y_ans = json.load(f)
            with open(years_map_path, 'r', encoding='utf-8') as f:
                y_map = json.load(f)
            y_ans.sort(key=lambda x: int(os.path.splitext(x['image_id'])[0]) if os.path.splitext(x['image_id'])[0].isdigit() else 0)

            f_y_total = 0
            f_y_correct = 0
            f_y_ae_sum = 0
            f_y_mae_valid = 0
            
            for item in y_ans:
                img_id = item['image_id']
                model_reply = item['model_answer']
                
                orig_filename = y_map.get(img_id)
                gt_year = extract_year_from_filename(orig_filename)
                pred_year = extract_year_from_model_answer(model_reply)
                
                f_y_total += 1
                ae_str = "N/A"
                pred_year_str = "Invalid"
                
                if gt_year is not None:
                    if pred_year is not None:
                        ae = abs(gt_year - pred_year)
                        f_y_ae_sum += ae
                        f_y_mae_valid += 1
                        ae_str = str(ae)
                        pred_year_str = str(pred_year)
                        if ae == 0:
                            f_y_correct += 1
                    else:
                        pass

            mmt_ans_path = os.path.join(base_path, "MMT-test", f"{abc_prefix}_MMT_result.json")
            m_acc = 0.0
            if os.path.exists(mmt_ans_path):
                with open(mmt_ans_path, 'r', encoding='utf-8') as f:
                    m_data = json.load(f)
                f_m_total = len(m_data)
                f_m_correct = sum(1 for x in m_data if x.get("is_correct") is True)
                m_acc = f_m_correct / f_m_total if f_m_total > 0 else 0
                overall_mmt_total += f_m_total
                overall_mmt_correct += f_m_correct

            y_acc = f_y_correct / f_y_total if f_y_total > 0 else 0
            y_mae = f_y_ae_sum / f_y_mae_valid if f_y_mae_valid > 0 else 0
            print("-" * 110)
            print(f"summary[{folder_name}]:")
            print(f"      Years Accuracy: {y_acc:.2%} ({f_y_correct}/{f_y_total})")
            print(f"      Years MAE     : {y_mae:.2f} (number: {f_y_mae_valid})")
            print(f"      MMT Accuracy  : {m_acc:.2%}")

            overall_y_total_count += f_y_total
            overall_y_correct_count += f_y_correct
            overall_y_ae_sum += f_y_ae_sum
            overall_y_mae_valid_count += f_y_mae_valid

    print("\n" + "=" * 110)
    print("OVERALL SUMMARY")
    print("=" * 110)
    if overall_y_total_count > 0:
        final_y_acc = overall_y_correct_count / overall_y_total_count
        final_y_mae = overall_y_ae_sum / overall_y_mae_valid_count if overall_y_mae_valid_count > 0 else 0
        print(f"Years Task:  Overall Acc = {final_y_acc:.2%} ({overall_y_correct_count}/{overall_y_total_count})")
        print(f"             Overall MAE = {final_y_mae:.2f} (number: {overall_y_mae_valid_count} )")
    
    if overall_mmt_total > 0:
        final_m_acc = overall_mmt_correct / overall_mmt_total
        print(f"Multimodal Task: Overall Acc = {final_m_acc:.2%} ({overall_mmt_correct}/{overall_mmt_total})")
    print("=" * 110)

if __name__ == "__main__":
    analyze_results_detailed()