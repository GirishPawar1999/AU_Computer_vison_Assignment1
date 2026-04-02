
# [1] Imports
import os
import shutil
import time
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from ultralytics import YOLO
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score
)
from scipy.stats import ttest_rel


# [2] Create Deblurred Dataset Folder
src_dir = "results/richardson_lucy"
dst_dir = "deblurred_dataset"

os.makedirs(dst_dir, exist_ok=True)

for file in os.listdir(src_dir):
    if file.endswith((".png", ".jpg", ".jpeg")):
        new_name = f"deblurred_{file}"
        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(dst_dir, new_name)

        if not os.path.exists(dst_path):
            shutil.copy(src_path, dst_path)

print("Renaming and copying complete.")


# [3] Directories
blur_dir = "data/data/blur/images"
sharp_dir = "data/data/sharp/images"
deblur_dir = "deblurred_dataset"

output_dir = "task3-Results"
blur_out = os.path.join(output_dir, "sample_detect_blur")
sharp_out = os.path.join(output_dir, "sample_detect_sharp")
deblur_out = os.path.join(output_dir, "sample_detect_deblur")
failure_out = os.path.join(output_dir, "failure_cases")

os.makedirs(blur_out, exist_ok=True)
os.makedirs(sharp_out, exist_ok=True)
os.makedirs(deblur_out, exist_ok=True)
os.makedirs(failure_out, exist_ok=True)


# [4] Load YOLO Model
model = YOLO("yolov8n.pt")
coco_classes = model.names


# [5] Detection Function

def run_detection_subset(image_dir, save_dir, selected_files, prefix=""):
    os.makedirs(save_dir, exist_ok=True)

    results_summary = []

    for file in selected_files:
        actual_file = prefix + file
        path = os.path.join(image_dir, actual_file)

        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        img = cv2.imread(path)
        if img is None:
            continue

        start_time = time.time()
        results = model(img, verbose=False)[0]
        runtime = time.time() - start_time

        boxes = results.boxes
        num_detections = len(boxes)

        confidences = []
        class_ids = []

        if num_detections > 0:
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)

        avg_conf = np.mean(confidences) if len(confidences) > 0 else 0
        class_names = [coco_classes[int(i)] for i in class_ids]

        annotated = results.plot()
        cv2.imwrite(os.path.join(save_dir, actual_file), annotated)

        results_summary.append({
            "image": file,
            "detections": num_detections,
            "avg_confidence": avg_conf,
            "classes": class_names,
            "class_ids": class_ids.tolist() if len(class_ids) > 0 else [],
            "runtime_sec": runtime
        })

    return results_summary


# [6] Random Sampling
blur_files = set(os.listdir(blur_dir))
sharp_files = set(os.listdir(sharp_dir))
deblur_files = set([f.replace("deblurred_", "") for f in os.listdir(deblur_dir)])

common_files = list(blur_files.intersection(sharp_files).intersection(deblur_files))
print(f"Total common images: {len(common_files)}")

num_samples = min(10, len(common_files))

if num_samples == 0:
    raise ValueError("No matching images found.")

random.seed(42)
sample_files = random.sample(common_files, num_samples)
print("Selected Images:", sample_files)


# [7] Run Detection
blur_results = run_detection_subset(
    blur_dir,
    blur_out,
    sample_files,
    prefix=""
)

sharp_results = run_detection_subset(
    sharp_dir,
    sharp_out,
    sample_files,
    prefix=""
)

deblur_results = run_detection_subset(
    deblur_dir,
    deblur_out,
    sample_files,
    prefix="deblurred_"
)


# [8] Convert to DataFrames

df_blur = pd.DataFrame(blur_results)
df_sharp = pd.DataFrame(sharp_results)
df_deblur = pd.DataFrame(deblur_results)

# Rename columns
for df_name, df_temp in zip(
    ["blur", "sharp", "deblur"],
    [df_blur, df_sharp, df_deblur]
):
    df_temp.rename(columns={
        "detections": f"detections_{df_name}",
        "avg_confidence": f"avg_confidence_{df_name}",
        "classes": f"classes_{df_name}",
        "class_ids": f"class_ids_{df_name}",
        "runtime_sec": f"runtime_sec_{df_name}"
    }, inplace=True)


# [9] Merge Results

df = df_blur.merge(df_sharp, on="image")
df = df.merge(df_deblur, on="image")

# Gains against blur baseline

df["detection_gain_vs_blur"] = df["detections_deblur"] - df["detections_blur"]
df["confidence_gain_vs_blur"] = df["avg_confidence_deblur"] - df["avg_confidence_blur"]

df["detection_difference_vs_sharp"] = df["detections_sharp"] - df["detections_deblur"]
df["confidence_difference_vs_sharp"] = df["avg_confidence_sharp"] - df["avg_confidence_deblur"]


# [10] Precision, Recall, F1 and Approximate mAP

y_true = []
y_pred_blur = []
y_pred_deblur = []

for _, row in df.iterrows():
    gt_classes = row["class_ids_sharp"]
    blur_classes = row["class_ids_blur"]
    deblur_classes = row["class_ids_deblur"]

    all_classes = set(gt_classes + blur_classes + deblur_classes)

    for cls in all_classes:
        y_true.append(1 if cls in gt_classes else 0)
        y_pred_blur.append(1 if cls in blur_classes else 0)
        y_pred_deblur.append(1 if cls in deblur_classes else 0)

precision_blur = precision_score(y_true, y_pred_blur, zero_division=0)
recall_blur = recall_score(y_true, y_pred_blur, zero_division=0)
f1_blur = f1_score(y_true, y_pred_blur, zero_division=0)

precision_deblur = precision_score(y_true, y_pred_deblur, zero_division=0)
recall_deblur = recall_score(y_true, y_pred_deblur, zero_division=0)
f1_deblur = f1_score(y_true, y_pred_deblur, zero_division=0)

map_blur = average_precision_score(y_true, y_pred_blur)
map_deblur = average_precision_score(y_true, y_pred_deblur)

metrics_df = pd.DataFrame({
    "Method": ["Blurred", "Deblurred"],
    "Precision": [precision_blur, precision_deblur],
    "Recall": [recall_blur, recall_deblur],
    "F1 Score": [f1_blur, f1_deblur],
    "mAP": [map_blur, map_deblur]
})

metrics_df.to_csv(os.path.join(output_dir, "detection_metrics.csv"), index=False)
print(metrics_df)


# [11] Per-Class AP Analysis

per_class_results = []

unique_classes = sorted(list(set(y_true)))

for cls_id in range(len(coco_classes)):
    cls_true = []
    cls_pred = []

    for _, row in df.iterrows():
        gt_classes = row["class_ids_sharp"]
        pred_classes = row["class_ids_deblur"]

        cls_true.append(1 if cls_id in gt_classes else 0)
        cls_pred.append(1 if cls_id in pred_classes else 0)

    if sum(cls_true) > 0:
        ap = average_precision_score(cls_true, cls_pred)
        per_class_results.append({
            "class_id": cls_id,
            "class_name": coco_classes[cls_id],
            "AP": ap
        })

per_class_df = pd.DataFrame(per_class_results)
per_class_df.to_csv(os.path.join(output_dir, "per_class_ap.csv"), index=False)


# [12] Precision-Recall Curve

precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_deblur)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Deblurred Images)")
plt.savefig(os.path.join(output_dir, "pr_curve.png"))
plt.close()


# [13] Confusion Matrix

cm = confusion_matrix(y_true, y_pred_deblur)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - Deblurred Detection")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()


# [14] Failure Case Analysis

failure_cases = df[
    (df["detection_gain_vs_blur"] < 0) |
    (df["confidence_gain_vs_blur"] < 0)
]

failure_cases.to_csv(os.path.join(output_dir, "failure_cases.csv"), index=False)

for _, row in failure_cases.iterrows():
    image_name = row["image"]
    src_path = os.path.join(deblur_out, f"deblurred_{image_name}")
    dst_path = os.path.join(failure_out, f"failure_{image_name}")

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)

print(f"Failure cases saved: {len(failure_cases)}")


# [15] Runtime Analysis

runtime_summary = pd.DataFrame({
    "Method": ["Blurred", "Sharp", "Deblurred"],
    "Average Runtime": [
        df["runtime_sec_blur"].mean(),
        df["runtime_sec_sharp"].mean(),
        df["runtime_sec_deblur"].mean()
    ]
})

runtime_summary.to_csv(os.path.join(output_dir, "runtime_summary.csv"), index=False)
print(runtime_summary)

plt.figure(figsize=(8, 5))
plt.bar(runtime_summary["Method"], runtime_summary["Average Runtime"])
plt.ylabel("Runtime (seconds)")
plt.title("Detection Runtime Comparison")
plt.savefig(os.path.join(output_dir, "runtime_comparison.png"))
plt.close()


# [16] Statistical Testing

t_stat_detection, p_detection = ttest_rel(
    df["detections_blur"],
    df["detections_deblur"]
)

t_stat_conf, p_conf = ttest_rel(
    df["avg_confidence_blur"],
    df["avg_confidence_deblur"]
)

stats_df = pd.DataFrame({
    "Metric": ["Detection Count", "Confidence"],
    "t_statistic": [t_stat_detection, t_stat_conf],
    "p_value": [p_detection, p_conf]
})

stats_df.to_csv(os.path.join(output_dir, "statistical_testing.csv"), index=False)
print(stats_df)


# [17] Visualisations

plt.figure(figsize=(8, 5))
plt.hist(df["confidence_gain_vs_blur"], bins=10)
plt.title("Confidence Gain Distribution (10 Samples)")
plt.xlabel("Confidence Gain")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "confidence_gain_histogram.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(metrics_df["Method"], metrics_df["F1 Score"])
plt.ylabel("F1 Score")
plt.title("F1 Score Comparison")
plt.savefig(os.path.join(output_dir, "f1_comparison.png"))
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(per_class_df["class_name"], per_class_df["AP"])
plt.xticks(rotation=90)
plt.ylabel("Average Precision")
plt.title("Per-Class AP Analysis")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "per_class_ap.png"))
plt.close()


# [18] Save Final Combined Results

df.to_csv(os.path.join(output_dir, "comparison_sample.csv"), index=False)

print("\nProcessing Complete.")
print("Saved:")
print("- Detection metrics")
print("- Per-class AP")
print("- PR curve")
print("- Confusion matrix")
print("- Runtime analysis")
print("- Failure case analysis")
print("- Statistical testing")

