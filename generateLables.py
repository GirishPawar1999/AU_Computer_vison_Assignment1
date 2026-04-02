import os
import cv2
from ultralytics import YOLO

# 1. Configuration
MODEL_PATH = "yolov8n.pt"
BLUR_DIR = "data/data/blur/images"
SHARP_DIR = "data/data/sharp/images"
DEBLUR_DIR = "deblurred_dataset" 
LABELS_DIR = "labels"

model = YOLO(MODEL_PATH)

# 2. Sync Directories (Find Common Files)
blur_files = {f for f in os.listdir(BLUR_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
sharp_files = {f for f in os.listdir(SHARP_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}
# Adjusting for the "deblur_" prefix from your previous renaming script
deblur_files = {f.replace("deblurred_", "") for f in os.listdir(DEBLUR_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))}

#print("Blur:",blur_files)
#print("Sharp:",sharp_files)
#print("Deblur:",deblur_files)
common_files = sorted(list(blur_files.intersection(sharp_files).intersection(deblur_files)))

if not os.path.exists(LABELS_DIR):
    os.makedirs(LABELS_DIR)

# 3. Processing
total_images = len(common_files)
images_with_detections = 0

print(f"Total common images detected: {total_images}")
print("-" * 30)

for filename in common_files:
    #SHARP image for label generation (ground truth)
    img_path = os.path.join(SHARP_DIR, filename)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # Run YOLO inference
    results = model(img, verbose=False)[0]
    
    label_filename = os.path.splitext(filename)[0] + ".txt"
    label_path = os.path.join(LABELS_DIR, label_filename)

   
    if len(results.boxes) == 0:
        
        if not os.path.exists(label_path):
            open(label_path, "w").close()
        continue

    detections_in_this_image = 0
    with open(label_path, "w") as f:
        for box in results.boxes:
            conf = float(box.conf[0])
            
            if conf >= 0.2:  
                detections_in_this_image += 1
                cls = int(box.cls[0])
                # Normalized coordinates (x_center, y_center, width, height)
                x, y, w, h = box.xywhn[0]
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    if detections_in_this_image > 0:
        images_with_detections += 1
        if images_with_detections % 50 == 0:
            print(f"Processed {images_with_detections} images with detections...")

print("\n" + "="*20)
print("      SUMMARY")
print("="*20)
print(f"Total common images: {total_images}")
print(f"Images with valid detections: {images_with_detections}")
print(f"Labels saved to: {os.path.abspath(LABELS_DIR)}")