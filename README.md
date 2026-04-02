# 🖼️ AU Computer Vision – Assignment 1

> **Course:** Computer Vision | **Institution:** Athabasca University  
> **Author:** Girish Pawar  
> **Repository:** [AU_Computer_vison_Assignment1](https://github.com/GirishPawar1999/AU_Computer_vison_Assignment1/tree/main)

---

## 📌 Overview

This repository contains the full implementation of **Assignment 1** for the Computer Vision unit. The assignment is divided into **five tasks**, each exploring a distinct area of image processing and computer vision:

| Task | Topic | Key Files |
|------|-------|-----------|
| Task 1 | Image Filtering & Edge Detection | *(see Task 1 section)* |
| Task 2 | Image Deblurring | `deblur.py`, `task2-Results/` |
| Task 3 | Object Detection | `objectDetection.py`, `task3-Results/` |
| Task 4 | Custom Label Generation & YOLO Training | `generateLabels.py`, `TrainingDataPreparation.py`, `runs/`, `task4-Results/` |
| Task 5 | Performance Comparison & Analysis | `performanceCompNAnalysis.py`, `task5-results/` |

> 📁 **AI-Assisted Logs:** A documentation log (`AI_assisted_logs/` or equivalent `.docx`/`.pdf`) is included in the repository. This records all AI tools used during development to assist with implementation, debugging, and understanding of concepts — in accordance with the assignment's academic integrity guidelines.

---

## 📁 Repository Structure

```
AU_Computer_vison_Assignment1/
│
├── deblur.py                        # Task 2 – Image deblurring implementation
├── objectDetection.py               # Task 3 – Object detection pipeline
├── generateLabels.py                # Task 4 – Auto-label generation script
├── TrainingDataPreparation.py       # Task 4 – Dataset preparation for YOLO training
├── performanceCompNAnalysis.py      # Task 5 – Performance comparison & analysis
│
├── runs/                            # Task 4 – YOLO training run outputs (weights, logs)
├── task2-Results/                   # Task 2 – Output images from deblurring
├── task3-Results/                   # Task 3 – Detected object output images
├── task4-Results/                   # Task 4 – Training results & evaluation metrics
├── task5-results/                   # Task 5 – Comparison charts, tables, analysis
│
└── AI_assisted_logs/                # AI assistance documentation log
```

---

## ⚙️ Requirements & Installation

### Prerequisites

- Python **3.8+**
- `pip` (Python package manager)
- GPU recommended for Task 4 (YOLO training), but CPU works

### 1. Clone the Repository

```bash
git clone https://github.com/GirishPawar1999/AU_Computer_vison_Assignment1.git
cd AU_Computer_vison_Assignment1
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If no `requirements.txt` is present, install the core dependencies manually:

```bash
pip install opencv-python numpy matplotlib scikit-image scipy
pip install torch torchvision torchaudio            # For YOLO (Task 4)
pip install ultralytics                              # YOLOv8 (Task 4)
pip install pandas seaborn                           # For Task 5 analysis
```

> 💡 **Tip:** For GPU-accelerated YOLO training, install the CUDA-compatible version of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## 🧪 Task Breakdown

---

### Task 1 – Image Filtering & Edge Detection

This task explores the fundamentals of **spatial image filtering** and **edge detection** using classical computer vision techniques.

#### What Was Implemented

- **Gaussian Smoothing** – Applied to reduce image noise before edge detection, using a configurable kernel size and sigma (σ). The relationship between σ and kernel size is explored to understand blurring strength vs. detail preservation.
- **Sobel Edge Detection** – Computes image gradients in the X and Y directions independently, then combines them to produce edge magnitude maps. Results clearly show horizontal vs. vertical edge sensitivity.
- **Canny Edge Detection** – A multi-stage pipeline (Gaussian blur → gradient computation → non-maximum suppression → double thresholding → edge tracking by hysteresis). Produces clean, thin, well-connected edges.
- **Kernel/Convolution Experimentation** – Custom kernels (sharpening, emboss, Laplacian) were applied to understand how convolution affects image structure.

#### Key Insights

- Larger Gaussian kernels suppress fine noise but also blur legitimate edges — a critical trade-off for downstream detection tasks.
- Sobel is fast but directionally biased; Canny is slower yet far more robust due to hysteresis thresholding.
- The choice of low/high threshold ratio in Canny significantly impacts edge connectivity — a ratio of ~1:2 or ~1:3 works well for most natural images.
- Pre-processing with Gaussian blur before Sobel produces results comparable to Canny on low-noise images.

#### How to Run

```bash
python task1_filtering.py --image <path_to_image>
```

*(Adjust filename to match actual Task 1 script in the repo)*

---

### Task 2 – Image Deblurring

**Script:** `deblur.py`  
**Results:** `task2-Results/`

This task investigates techniques to recover sharp images from blurred inputs, using both classical and frequency-domain approaches.

#### What Was Implemented

- **Wiener Filter** – Inverse filtering with noise regularisation, applied in the frequency domain. Effective when the Point Spread Function (PSF) is known.
- **Richardson-Lucy Deconvolution** – An iterative deblurring method that models the blur as a convolution with a known PSF and progressively refines the estimate.
- **Blind Deblurring** – Attempted recovery without a known PSF, using estimated blur kernels.

#### How to Run

```bash
python deblur.py --input <blurred_image> --output task2-Results/
```

#### Results

Output deblurred images are stored in `task2-Results/`. Each result is compared side-by-side with the original blurred input. Metrics such as **PSNR** (Peak Signal-to-Noise Ratio) and **SSIM** (Structural Similarity Index) are reported to quantify recovery quality.

---

### Task 3 – Object Detection

**Script:** `objectDetection.py`  
**Results:** `task3-Results/`

This task applies pre-trained object detection models to detect and localise objects within images.

#### What Was Implemented

- Loading a **pre-trained detection model** (e.g., YOLOv8 or OpenCV DNN with a pre-trained backbone)
- Running inference on test images
- Drawing **bounding boxes** and **class labels** with confidence scores on detected objects
- Saving annotated images to `task3-Results/`

#### How to Run

```bash
python objectDetection.py --input <image_or_folder> --output task3-Results/
```

#### Results

Annotated output images are in `task3-Results/`, showing detected objects with bounding boxes and confidence percentages.

---

### Task 4 – Custom Label Generation & YOLO Training

**Scripts:** `generateLabels.py`, `TrainingDataPreparation.py`  
**Training Runs:** `runs/`  
**Results:** `task4-Results/`

This task covers the full pipeline of training a custom **YOLOv8** object detector from scratch on a domain-specific dataset.

#### Pipeline

1. **Label Generation** (`generateLabels.py`) – Automatically generates YOLO-format `.txt` annotation files from source data (e.g., using OpenCV contours, existing masks, or a semi-automated labelling approach).

2. **Data Preparation** (`TrainingDataPreparation.py`) – Splits data into `train/val/test` sets, organises the directory structure expected by Ultralytics YOLO, and generates the `data.yaml` configuration file.

3. **Training** – Executed via the Ultralytics YOLO CLI or API:
   ```bash
   yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
   ```

4. **Evaluation** – Training metrics (mAP@0.5, mAP@0.5:0.95, precision, recall) are tracked per epoch and saved under `runs/`.

#### How to Run

```bash
# Step 1 – Generate labels
python generateLabels.py

# Step 2 – Prepare dataset
python TrainingDataPreparation.py

# Step 3 – Train (adjust paths as needed)
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640 project=task4-Results
```

#### Results

Training curves, confusion matrices, and sample predictions are stored in `task4-Results/` and `runs/`. Key metrics reported include:

- **mAP@0.5** and **mAP@0.5:0.95**
- Precision / Recall curves
- Training loss over epochs

---

### Task 5 – Performance Comparison & Analysis

**Script:** `performanceCompNAnalysis.py`  
**Results:** `task5-results/`

This task provides a **quantitative and qualitative comparison** of the models and methods used across all tasks.

#### What Was Implemented

- Comparison of edge detectors (Sobel vs. Canny) on standard metrics
- Deblurring quality comparison across methods using PSNR / SSIM
- Object detection performance benchmarking (pre-trained vs. fine-tuned)
- Visualisation of results as charts, bar graphs, and tables using `matplotlib` and `seaborn`

#### How to Run

```bash
python performanceCompNAnalysis.py
```

Output charts and analysis tables are saved to `task5-results/`.

---

## 📊 Results Summary

| Task | Method | Key Metric | Notes |
|------|--------|-----------|-------|
| Task 1 | Canny Edge Detection | — | Best edge connectivity vs. Sobel |
| Task 2 | Wiener Filter | PSNR / SSIM | Best when PSF known |
| Task 3 | Pre-trained YOLO | mAP | High accuracy on standard objects |
| Task 4 | Custom YOLOv8 Training | mAP@0.5 | Domain-specific accuracy |
| Task 5 | Cross-task Analysis | Multiple | Comparative summary |

---

## 🤖 AI-Assisted Development Log

This assignment made use of AI tools (such as ChatGPT / Claude) to assist with:

- Debugging Python code and resolving OpenCV/PyTorch compatibility issues
- Understanding mathematical concepts (e.g., Wiener filter derivation, YOLO loss functions)
- Suggesting code structure and best practices for dataset preparation
- Reviewing and improving documentation

All AI interactions are documented in the **AI-Assisted Logs** file included in the repository. These logs are provided transparently in accordance with the course's academic integrity policy. All code was reviewed, understood, and validated by the author.

---

## 📚 References

- OpenCV Documentation – https://docs.opencv.org
- Ultralytics YOLOv8 – https://docs.ultralytics.com
- Scikit-Image – https://scikit-image.org
- Canny, J. (1986). *A Computational Approach to Edge Detection.* IEEE TPAMI.
- Richardson, W. H. (1972). *Bayesian-Based Iterative Method of Image Restoration.* JOSA.

---

## 📝 License

This project is submitted as academic coursework for **Athabasca University**. All code is for educational purposes only.
