import os
import torch
import pandas as pd
from ultralytics import YOLO

def run_task5_analysis():
    # --- CONFIGURATION ---
    MODEL_PATH = "runs/detect/task4_training/deblur_detector13/weights/best.pt"
    SUBSET_DIR = "deblurred_dataset"  
    RESULTS_DIR = "task5-results"
    RUN_NAME = "deblurred_detailed_analysis"

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ Model not found at {MODEL_PATH}")
        return 
    
    # Load model
    model = YOLO(MODEL_PATH)
    print(f"✅ Model Classes: {model.names}") # Let's see what the model actually expects
    
    # 1. Setup YAML (Standard YOLO format)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
    subset_yaml = os.path.join(RESULTS_DIR, "task5_config.yaml")
    abs_subset_path = os.path.abspath(SUBSET_DIR)

    with open(subset_yaml, 'w') as f:
        # We point everything to the same folder for a single-set validation
        f.write(f"path: {abs_subset_path}\n")
        f.write(f"train: .\n") 
        f.write(f"val: .\n")
        f.write(f"names:\n")
        for idx, name in model.names.items():
            f.write(f"  {idx}: {name}\n")

    # 2. Run Validation
    try:
        print(f"🚀 Running validation on {SUBSET_DIR}...")
        results = model.val(
            data=subset_yaml,
            imgsz=320,
            plots=True,     
            project=RESULTS_DIR,
            name=RUN_NAME,
            exist_ok=True,
            conf=0.1,    # Increased confidence to ignore background noise
            iou=0.45,    # Standard NMS threshold
            device=0 if torch.cuda.is_available() else 'cpu'
        )

        # 3. Save Summary CSV
        # Path logic: YOLO creates RESULTS_DIR/RUN_NAME
        final_save_path = os.path.join(RESULTS_DIR, RUN_NAME)
        
        # Extract metrics using the correct keys for YOLOv8/v11
        # result.results_dict contains the main stats
        m = results.results_dict
        stats = {
            "Metric": ["mAP50", "mAP50-95", "Precision", "Recall"],
            "Value": [
                round(m.get('metrics/mAP50(B)', 0), 4),
                round(m.get('metrics/mAP50-95(B)', 0), 4),
                round(m.get('metrics/precision(B)', 0), 4),
                round(m.get('metrics/recall(B)', 0), 4)
            ]
        }
        
        df = pd.DataFrame(stats)
        csv_output = os.path.join(final_save_path, "summary_metrics.csv")
        df.to_csv(csv_output, index=False)
        
        print("-" * 30)
        print(df)
        print("-" * 30)
        print(f"✅ Analysis complete. CSV saved to: {csv_output}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_task5_analysis()