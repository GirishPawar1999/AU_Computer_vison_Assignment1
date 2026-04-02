# [1] Imports
import os
import shutil
from ultralytics import YOLO
import cv2
import numpy as np
import random
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# [2] Create a new folder from the debuff images richardson_lucy (Only create if it doesn't exist)
src_dir = "results/richardson_lucy"
dst_dir = "results/deblurred_dataset"

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)
    print(f"Created directory: {dst_dir}")
else:
    print(f"Directory {dst_dir} already exists. Skipping creation.")

for file in os.listdir(src_dir):
    if file.endswith((".png", ".jpg", ".jpeg")):
        new_name = f"deblurred_{file}"
        dest_path = os.path.join(dst_dir, new_name)
        
        if not os.path.exists(dest_path):
            shutil.copy(
                os.path.join(src_dir, file),
                dest_path
            )

print("Renaming + copying process complete.")

# [Task-3] :  Object Detection and Analysis 
model = YOLO("yolov8n.pt")
coco_classes = model.names

# Detection Function
def run_detection_subset(image_dir, save_dir, selected_files, prefix=""):
    os.makedirs(save_dir, exist_ok=True)

    results_summary = []

    for file in selected_files:
        actual_file = prefix + file
        path = os.path.join(image_dir, actual_file)

        if not os.path.exists(path):
            print(f"Skipping (not found): {path}")  # DEBUG
            continue

        img = cv2.imread(path)
        if img is None:
            print(f"Failed to load: {path}")
            continue

        results = model(img)[0]

        boxes = results.boxes
        num_detections = len(boxes)

        confidences = []
        class_ids = []

        if num_detections > 0:
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy()

        avg_conf = np.mean(confidences) if len(confidences) > 0 else 0
        class_names = [coco_classes[int(i)] for i in class_ids]

        annotated = results.plot()
        cv2.imwrite(os.path.join(save_dir, actual_file), annotated)

        results_summary.append({
            "image": file,  # IMPORTANT: keep original name
            "detections": num_detections,
            "avg_confidence": avg_conf,
            "classes": class_names
        })

    print(f"Processed {len(results_summary)} images from {image_dir}")
    return results_summary

# Random images selected
blur_dir = "data/data/blur/images"
deblur_dir = "deblurred_dataset"

blur_out = "task3-Results/detect_blur"
deblur_out = "task3-Results/detect_deblur"


blur_files = set(os.listdir(blur_dir))
deblur_files = set([f.replace("deblurred_", "") for f in os.listdir(deblur_dir)])

#print("Blur:",blur_files)
#print("DEBlur:",deblur_files)

common_files = list(blur_files.intersection(deblur_files))

print(f"Total common images: {len(common_files)}")

# Safe sampling
num_samples = min(5, len(common_files))

if num_samples == 0:
    raise ValueError("No matching images found between blur and deblur folders!")

random.seed(42)
sample_files = random.sample(common_files, num_samples)

print("Selected images:", sample_files)

# Running Detection
blur_results = run_detection_subset(
    blur_dir,
    "task3-Results/sample_detect_blur",
    sample_files,
    prefix=""
)

deblur_results = run_detection_subset(
    deblur_dir,
    "task3-Results/sample_detect_deblur",
    sample_files,
    prefix="deblurred_"   
)

# Analysis

df_blur = pd.DataFrame(blur_results)
df_deblur = pd.DataFrame(deblur_results)

print("df_blur columns:", df_blur.columns)
print("df_deblur columns:", df_deblur.columns)

print("df_blur sample:", df_blur.head())
print("df_deblur sample:", df_deblur.head())

required_cols = ["image", "detections", "avg_confidence"]

for col in required_cols:
    if col not in df_blur.columns:
        raise ValueError(f"Missing '{col}' in df_blur")
    if col not in df_deblur.columns:
        raise ValueError(f"Missing '{col}' in df_deblur")

df = df_blur.merge(df_deblur, on="image", suffixes=("_blur", "_deblur"))

df["detection_gain"] = df["detections_deblur"] - df["detections_blur"]
df["confidence_gain"] = df["avg_confidence_deblur"] - df["avg_confidence_blur"]

df.to_csv("task3-Results/comparison_sample.csv", index=False)

print(df)
print(df.describe())

def count_classes(results):
    counter = Counter()
    for r in results:
        counter.update(r["classes"])
    return counter

print("Blurred:", count_classes(blur_results))
print("Deblurred:", count_classes(deblur_results))

plt.figure()
plt.hist(df["confidence_gain"], bins=10)
plt.title("Confidence Gain (5 Sample Images)")
plt.xlabel("Confidence Gain")
plt.ylabel("Frequency")
plt.show()