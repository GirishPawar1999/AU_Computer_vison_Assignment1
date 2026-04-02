# AU Computer Vision Assignment 1

## Overview

This repository contains the implementation of **Assignment 1 for Computer Vision**, covering multiple tasks involving image processing, object detection, dataset preparation, training pipeline creation, and performance evaluation.

The project is structured in a modular way, where each task is implemented as a separate Python script with corresponding output result folders.

---

## Repository Structure


AU_Computer_vison_Assignment1/
│
├── deblur.py # Task 2: Image deblurring
├── task2-Results/ # Outputs for Task 2
│
├── objectDetection.py # Task 3: Object detection
├── task3-Results/ # Outputs for Task 3
│
├── generateLables.py # Task 4: Label generation
├── TraingDataPrepration.py # Task 4: Dataset preparation
├── runs/ # Training outputs / model runs
├── task4-Results/ # Outputs for Task 4
│
├── performanceCompNAnalysis.py # Task 5: Performance comparison & analysis
├── task5-Results/ # Outputs for Task 5
│
├── logs/ # AI-assisted development logs
└── README.md


---

## Task Descriptions

## Task 2 – Image Deblurring

**File:** `deblur.py`

This task focuses on restoring blurred images using image processing techniques. The script processes input images and enhances their quality by reducing blur.

### Outputs:
- Saved deblurred images in `task2-Results/`

### Key Features:
- Image preprocessing
- Blur reduction techniques
- Output visualization

---

## Task 3 – Object Detection

**File:** `objectDetection.py`

This task performs object detection on images using a trained model. It identifies objects and draws bounding boxes around detected objects.

### Outputs:
- Annotated images saved in `task3-Results/`

### Key Features:
- Bounding box detection
- Class prediction
- Visualized detection outputs

---

## Task 4 – Dataset Preparation & Training Pipeline

**Files:**
- `generateLables.py`
- `TraingDataPrepration.py`
- `runs/`

This task involves preparing a dataset for training and organizing it into a structured format suitable for deep learning models.

### Workflow:
1. Generate labels from raw annotations using `generateLables.py`
2. Prepare dataset splits using `TraingDataPrepration.py`
3. Store training outputs in `runs/`

### Outputs:
- Structured dataset
- Training-ready format
- Model training runs

### Key Features:
- Automated label generation
- Dataset splitting (train/val/test)
- Training pipeline preparation

---

## Task 5 – Performance Comparison & Analysis

**File:** `performanceCompNAnalysis.py`

This task evaluates and compares model performance across different experiments.

### Outputs:
- Analysis results in `task5-Results/`

### Key Features:
- Model comparison
- Performance metrics (accuracy, precision, recall, etc.)
- Visual/tabular analysis

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/GirishPawar1999/AU_Computer_vison_Assignment1.git
cd AU_Computer_vison_Assignment1
2. Create Virtual Environment
python -m venv venv

Activate it:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
3. Install Dependencies

If requirements.txt exists:

pip install -r requirements.txt

Otherwise install manually:

pip install opencv-python numpy matplotlib pillow scikit-learn ultralytics
How to Run
Task 2 – Deblurring
python deblur.py
Task 3 – Object Detection
python objectDetection.py
Task 4 – Dataset Preparation
python generateLables.py
python TraingDataPrepration.py
Task 5 – Performance Analysis
python performanceCompNAnalysis.py
Results
Task	Output Folder
Task 2	task2-Results/
Task 3	task3-Results/
Task 4	task4-Results/, runs/
Task 5	task5-Results/

Each folder contains processed outputs, images, logs, or evaluation results.

AI-Assisted Logs

This project includes AI-assisted development logs used during implementation.

These logs document:

Code generation assistance
Debugging support
Optimization suggestions
Documentation support

Stored in:

logs/
Notes
Ensure correct dataset paths before running scripts.
Some scripts may require GPU support for faster execution.
Install all dependencies before running any file.
Output folders are auto-generated after execution.
Author

Girish Pawar
AU Computer Vision Assignment 1

License

This project is created for academic purposes only.


---

If you want next level upgrade, I can also:
- Make it **super professional (GitHub portfolio ready)**
- Add **architecture diagrams**
- Add **badges (Python, OpenCV, YOLO, etc.)**
- Or generate a **project report PDF** from this

Just tell me 👍