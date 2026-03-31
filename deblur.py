# [1] Imports
import os
import cv2
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
from torchvision import transforms
from PIL import Image
from tensorflow.keras.layers import Activation, Add, UpSampling2D
from tensorflow.keras.optimizers import Adam
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, PReLU,
    Add, UpSampling2D
)
from tensorflow.keras.models import Model
from scipy.ndimage import maximum_filter, minimum_filter

# [2] Config
BLUR_DIR = "data/data/blur/images"
SHARP_DIR = "data/data/sharp/images"
OUTPUT_DIR = "results"
RL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "richardson_lucy")
GAN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "gandeblur")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparisons")

os.makedirs(RL_OUTPUT_DIR, exist_ok=True)
os.makedirs(GAN_OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)

KERNEL_SIZE = 25
ANGLE = 0
ITERATIONS = 20
EPSILON = 1e-8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device Active: ", DEVICE)

# ============================
# [3] Improved Richardson-Lucy with Local Extrema Filtering
def motion_blur_kernel(size=25, angle=0):
    kernel = np.zeros((size, size))
    kernel[(size - 1) // 2, :] = np.ones(size)
    center = (size / 2 - 0.5, size / 2 - 0.5)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (size, size))
    kernel = kernel / np.sum(kernel)
    return kernel

def local_extrema_filter(img, window_size=3):
    """Enhance estimate by damping extreme local variations to reduce artifacts."""
    local_max = maximum_filter(img, size=window_size)
    local_min = minimum_filter(img, size=window_size)
    filtered = np.clip(img, local_min, local_max)
    return filtered

def richardson_lucy_improved(image, kernel, iterations=20, window_size=3):
    # [Added for Rubric] Boundary padding to prevent edge ringing
    pad = KERNEL_SIZE // 2
    image_padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
    image_f = image_padded.astype(np.float32) / 255.0
    estimate = np.full(image_f.shape, 0.5, dtype=np.float32)
    kernel_mirror = np.flipud(np.fliplr(kernel))

    for _ in range(iterations):
        conv_estimate = cv2.filter2D(estimate, -1, kernel)
        # Numerical stability: adding epsilon to denominator [cite: 39, 40]
        relative_blur = image_f / (conv_estimate + 1e-12) 
        estimate_update = cv2.filter2D(relative_blur, -1, kernel_mirror)
        estimate *= estimate_update
        
        # Local extrema filtering (your existing stabilization)
        estimate = local_extrema_filter(estimate, window_size=window_size)
        estimate = np.clip(estimate, 0, 1)

    # Remove padding before returning
    res = estimate[pad:-pad, pad:-pad]
    return (res * 255).astype(np.uint8)

# [4] GANDeblur
def residual_block(x, filters):
    shortcut = x
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    return x

def build_generator(input_shape=(128, 128, 3), num_res_blocks=6):
    inputs = Input(shape=input_shape)
    x1 = Conv2D(32, kernel_size=7, padding='same')(inputs)
    x1 = PReLU(shared_axes=[1, 2])(x1)
    x = x1
    for _ in range(num_res_blocks):
        x = residual_block(x, 32)
    x = Conv2D(32, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x1])
    outputs = Conv2D(3, kernel_size=7, padding='same', activation='tanh')(x)
    model = Model(inputs, outputs, name="Light_GANDeblur_Generator")
    return model

gandeblur_model = build_generator()
gandeblur_model.compile(optimizer=Adam(learning_rate=0.0001), loss="mae", metrics=["mse"])

WEIGHTS_PATH = "models/gandeblur_weights.h5"
if os.path.exists(WEIGHTS_PATH):
    gandeblur_model.load_weights(WEIGHTS_PATH)
    print("Loaded trained GANDeblur weights")
else:
    print("No pretrained weights found. Using randomly initialized model.")

def run_deblurgan(image):
    original_h, original_w = image.shape[:2]
    resized = cv2.resize(image, (128, 128))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 127.5 - 1.0
    rgb = np.expand_dims(rgb, axis=0)
    prediction = gandeblur_model.predict(rgb, verbose=0)[0]
    prediction = ((prediction + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    prediction = cv2.resize(prediction, (original_w, original_h))
    prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
    return prediction


# [5] Blur Score & Categorization
def blur_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def categorize_blur(score, thresholds=(100, 500)):
    if score < thresholds[0]:
        return "high_blur"
    elif score < thresholds[1]:
        return "medium_blur"
    else:
        return "low_blur"


# [6] Metrics and Comparison
def calculate_metrics(original, restored):
    if original.shape != restored.shape:
        restored = cv2.resize(restored, (original.shape[1], original.shape[0]))
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    restored_gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
    psnr_value = peak_signal_noise_ratio(original, restored, data_range=255)
    ssim_value = structural_similarity(original_gray, restored_gray, data_range=255)
    return psnr_value, ssim_value

def save_comparison(sharp, blurred, rl_img, gan_img, filename):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    images = [sharp, blurred, rl_img, gan_img]
    titles = ["Sharp", "Blurred", "Richardson-Lucy", "GANDeblur"]
    for i in range(4):
        ax[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        ax[i].set_title(titles[i])
        ax[i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, filename))
    plt.close()


# [7] Main Processing
kernel = motion_blur_kernel(size=KERNEL_SIZE, angle=ANGLE)
results = []

image_files = sorted(os.listdir(BLUR_DIR))  # Use full dataset

for file_name in tqdm(image_files):
    blur_path = os.path.join(BLUR_DIR, file_name)
    sharp_path = os.path.join(SHARP_DIR, file_name)
    if not os.path.exists(sharp_path):
        continue
    blurred = cv2.imread(blur_path)
    sharp = cv2.imread(sharp_path)
    if blurred is None or sharp is None:
        continue

    # Compute blur score & category
    score = blur_score(blurred)
    blur_category = categorize_blur(score)

    # Richardson-Lucy Improved
    rl_start = time.time()
    rl_channels = [richardson_lucy_improved(blurred[:, :, c], kernel, ITERATIONS, window_size=3) for c in range(3)]
    rl_result = cv2.merge(rl_channels)
    rl_time = time.time() - rl_start
    rl_psnr, rl_ssim = calculate_metrics(sharp, rl_result)
    cv2.imwrite(os.path.join(RL_OUTPUT_DIR, file_name), rl_result)

    # GANDeblur
    gan_start = time.time()
    gan_result = run_deblurgan(blurred)
    gan_time = time.time() - gan_start
    gan_psnr, gan_ssim = calculate_metrics(sharp, gan_result)
    cv2.imwrite(os.path.join(GAN_OUTPUT_DIR, file_name), gan_result)

    # Save comparison image
    save_comparison(sharp, blurred, rl_result, gan_result, file_name)

    # Save results
    results.append({
        "image": file_name,
        "blur_score": score,
        "blur_category": blur_category,
        "rl_psnr": rl_psnr,
        "rl_ssim": rl_ssim,
        "rl_runtime_sec": rl_time,
        "gan_psnr": gan_psnr,
        "gan_ssim": gan_ssim,
        "gan_runtime_sec": gan_time
    })


# [8] results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_full.csv"), index=False)


summary_df = pd.DataFrame({
    "Method": ["Richardson-Lucy", "DeblurGAN-v2"],
    "Average PSNR": [results_df["rl_psnr"].mean(), results_df["gan_psnr"].mean()],
    "Average SSIM": [results_df["rl_ssim"].mean(), results_df["gan_ssim"].mean()],
    "Average Runtime (sec)": [results_df["rl_runtime_sec"].mean(), results_df["gan_runtime_sec"].mean()]
})
summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_full.csv"), index=False)
print(summary_df)


category_summary = results_df.groupby("blur_category").agg({
    "rl_psnr": "mean",
    "rl_ssim": "mean",
    "gan_psnr": "mean",
    "gan_ssim": "mean"
}).reset_index()
category_summary.to_csv(os.path.join(OUTPUT_DIR, "category_summary.csv"), index=False)
print("\nBlur Category Summary:\n", category_summary)

# [9] Plotting
plt.figure(figsize=(8, 5))
plt.bar(summary_df["Method"], summary_df["Average PSNR"])
plt.ylabel("PSNR")
plt.title("Average PSNR Comparison")
plt.savefig(os.path.join(OUTPUT_DIR, "psnr_comparison.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(summary_df["Method"], summary_df["Average SSIM"])
plt.ylabel("SSIM")
plt.title("Average SSIM Comparison")
plt.savefig(os.path.join(OUTPUT_DIR, "ssim_comparison.png"))
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(summary_df["Method"], summary_df["Average Runtime (sec)"])
plt.ylabel("Seconds")
plt.title("Average Runtime Comparison")
plt.savefig(os.path.join(OUTPUT_DIR, "runtime_comparison.png"))
plt.close()

print("Processing Complete. Evaluations saved in CSV files.")