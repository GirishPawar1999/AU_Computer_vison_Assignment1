import os
import random
import shutil
import pandas as pd
from ultralytics import YOLO

def prepare_and_train():
    # 1. Configuration
    DEBLUR_DIR = "deblurred_dataset"
    LABELS_DIR = "deblurred_dataset" # Assuming labels are in the same folder or synced
    DATASET_DIR = "yolo_dataset"
    PROJECT_NAME = "task4_training"
    RUN_NAME = "deblur_detector13"

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2

    # 2. Cleanup and Folder Creation
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

    # 3. Load Images
    valid_exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    images = [f for f in os.listdir(DEBLUR_DIR) if f.lower().endswith(valid_exts)]

    if not images:
        print(f"❌ Error: No images found in {DEBLUR_DIR}")
        return

    random.shuffle(images)

    # 4. Split Logic
    train_split = int(len(images) * TRAIN_RATIO)
    val_split = int(len(images) * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": images[:train_split],
        "val": images[train_split:val_split],
        "test": images[val_split:]
    }

    # 5. File Distribution
    print(f"📂 Moving {len(images)} files...")
    img_count = 0
    lbl_count = 0
    
    for split, files in splits.items():
        for img_name in files:
            # FIX: Match label name EXACTLY to image name (e.g., deblurred_0001.png -> deblurred_0001.txt)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            
            src_img = os.path.join(DEBLUR_DIR, img_name)
            src_lbl = os.path.join(LABELS_DIR, label_name)

            if os.path.exists(src_lbl):
                shutil.copy(src_img, os.path.join(DATASET_DIR, "images", split, img_name))
                shutil.copy(src_lbl, os.path.join(DATASET_DIR, "labels", split, label_name))
                img_count += 1
                lbl_count += 1
            else:
                # Warning if an image has no matching label
                print(f"⚠️ Warning: Missing label for {img_name} (Expected {label_name})")

    print(f"✅ Prepared {img_count} images and {lbl_count} labels.")

    # 6. Training
    if lbl_count > 0:
        # Using yolov11n.pt if you want the newest, otherwise yolov8n.pt is fine
        model = YOLO("yolov8n.pt") 
        
        results = model.train(
            data="dataset.yaml",
            epochs=30,
            imgsz=640,
            batch=16,
            workers=0, # Keep at 0 for Windows stability
            project=PROJECT_NAME,
            name=RUN_NAME,
            exist_ok=True
        )
        
        # 7. Post-Training Analysis
        results_csv = os.path.join(PROJECT_NAME, RUN_NAME, "results.csv")
        if os.path.exists(results_csv):
            print("\n--- Final Metrics Summary ---")
            df = pd.read_csv(results_csv)
            df.columns = df.columns.str.strip()
            final = df.iloc[-1]
            print(f"mAP50: {final.get('metrics/mAP50(B)', 0):.4f}")
            print(f"Precision: {final.get('metrics/precision(B)', 0):.4f}")
            print(f"Recall: {final.get('metrics/recall(B)', 0):.4f}")
    else:
        print("❌ Training aborted: No labels were successfully moved.")

if __name__ == '__main__':
    prepare_and_train()