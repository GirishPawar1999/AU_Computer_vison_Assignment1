import os
import shutil
import pandas as pd
import torch
from ultralytics import YOLO

def run_task_5_analysis():
    # 1. Configuration
    SOURCE_MODEL = "runs/detect/task4_training/deblur_detector/weights/best.pt"
    RESULTS_DIR = "task5-Results"
    LOCAL_MODEL_NAME = os.path.join(RESULTS_DIR, "best_deblur_detector.pt")
    
    # Define domains for comparison
    DATA_DOMAINS = {
        "Blurred": "data/data/blur/images",
        "Sharp": "data/data/sharp/images",
        "Deblurred": "deblurred_dataset"
    }
    
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # 2. Check for the trained model
    if os.path.exists(SOURCE_MODEL):
        shutil.copy(SOURCE_MODEL, LOCAL_MODEL_NAME)
        model_to_use = LOCAL_MODEL_NAME
        print(f"✅ Using trained model: {LOCAL_MODEL_NAME}")
    else:
        print(f"⚠️ Trained model not found. Using pretrained yolov8n.pt.")
        model_to_use = "yolov8n.pt"

    # 3. Initialize Model
    model = YOLO(model_to_use)
    comparison_data = []

    print("\nStarting Task 5 Analysis...")
    print("-" * 50)

    # 4. Evaluation Loop
    for domain, folder in DATA_DOMAINS.items():
        if not os.path.exists(folder):
            continue
            
        print(f"Analyzing {domain} domain...")
        
        # MEMORY OPTIMIZATION:
        try:
            metrics = model.val(
                data="dataset.yaml", 
                split='test', 
                batch=1,         # Minimal memory usage
                imgsz=320,       # Reduced size for analysis stability
                plots=True, 
                verbose=False, 
                project=RESULTS_DIR, 
                name=domain,
                device=0         # Use GPU (change to 'cpu' if it crashes again)
            )
            
            comparison_data.append({
                "Domain": domain,
                "mAP50": metrics.results_dict.get('metrics/mAP50(B)', 0),
                "Precision": metrics.results_dict.get('metrics/precision(B)', 0),
                "Recall": metrics.results_dict.get('metrics/recall(B)', 0)
            })
        except RuntimeError as e:
            print(f"CUDA error on {domain}. Try restarting your PC to clear VRAM or use device='cpu'.")
            break

    # 5. Save Results
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df.to_csv(os.path.join(RESULTS_DIR, "performance_summary.csv"), index=False)
        print("\n" + "="*40)
        print(df.to_string(index=False))
        print("="*40)

if __name__ == '__main__':
    # Optional: Force clean VRAM before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_task_5_analysis()