# =========================================
# [1] Imports
# =========================================
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
from skimage import img_as_float, io, restoration
from torchvision import transforms
from PIL import Image
from tensorflow.keras.layers import Activation, Add, UpSampling2D, Input, Conv2D, BatchNormalization, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve, maximum_filter, minimum_filter

# =========================================
# [2] Config
# =========================================
BLUR_DIR = "data/data/blur/images"
SHARP_DIR = "data/data/sharp/images"
OUTPUT_DIR = "results"
RL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "richardson_lucy")
GAN_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "gandeblur")
COMPARISON_DIR = os.path.join(OUTPUT_DIR, "comparisons")

os.makedirs(RL_OUTPUT_DIR, exist_ok=True)
os.makedirs(GAN_OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPARISON_DIR, exist_ok=True)

ITERATIONS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device Active: ", DEVICE)

# =========================================
# [3] Gaussian PSF Creation for RL
# =========================================
PSF_SIZE = 21  # Should be odd
PSF_SIGMA = 3

gaussian_kernel_1d = gaussian(PSF_SIZE, std=PSF_SIGMA)
psf = np.outer(gaussian_kernel_1d, gaussian_kernel_1d)
psf /= psf.sum()

# Visualize PSF
plt.figure(figsize=(4,4))
plt.imshow(psf, cmap='gray')
plt.title('Point Spread Function (PSF)')
plt.colorbar()
plt.show()

# =========================================
# [4] Advanced Richardson-Lucy with Edge Handling
# =========================================
def deblur_richardson_lucy_rgb(image, psf, iterations=30):
    """Deblurs an RGB image with edge-padding to prevent ringing artifacts."""
    pad_size = psf.shape[0] // 2  # Symmetric padding
    h, w, c = image.shape
    edge_geometry = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
    img_float = image.astype(np.float64) / 255.0
    padded_img = np.pad(img_float, edge_geometry, mode='symmetric')

    processed_layers = []
    for i in range(c):
        channel = padded_img[:, :, i]
        deconvolved = restoration.richardson_lucy(channel, psf, num_iter=iterations)
        processed_layers.append(deconvolved)

    restored = np.stack(processed_layers, axis=-1)
    restored_cropped = restored[pad_size:-pad_size, pad_size:-pad_size, :]
    return np.clip(restored_cropped * 255, 0, 255).astype(np.uint8)

# =========================================
# [5] GANDeblur Model
# =========================================
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

# =========================================
# [6] Metrics and Comparison Functions
# =========================================
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

def calculate_metrics(original, restored):
    if original.shape != restored.shape:
        restored = cv2.resize(restored, (original.shape[1], original.shape[0]))
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    restored_gray = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
    psnr_value = psnr(original, restored, data_range=255)
    ssim_value = ssim(original_gray, restored_gray, data_range=255)
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

# =========================================
# [7] Main Processing Loop
# =========================================
results = []
image_files = sorted(os.listdir(BLUR_DIR))

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

    # Richardson-Lucy Advanced
    rl_start = time.time()
    rl_result = deblur_richardson_lucy_rgb(blurred, psf, ITERATIONS)
    rl_time = time.time() - rl_start
    rl_psnr, rl_ssim = calculate_metrics(sharp, rl_result)
    cv2.imwrite(os.path.join(RL_OUTPUT_DIR, file_name), rl_result)

    # # GANDeblur
    # gan_start = time.time()
    # gan_result = run_deblurgan(blurred)
    # gan_time = time.time() - gan_start
    # gan_psnr, gan_ssim = calculate_metrics(sharp, gan_result)
    # cv2.imwrite(os.path.join(GAN_OUTPUT_DIR, file_name), gan_result)

    # # Save comparison image
    # save_comparison(sharp, blurred, rl_result, gan_result, file_name)

    # Save results
    results.append({
        "image": file_name,
        "blur_score": score,
        "blur_category": blur_category,
        "rl_psnr": rl_psnr,
        "rl_ssim": rl_ssim,
        "rl_runtime_sec": rl_time,
#         "gan_psnr": gan_psnr,
#         "gan_ssim": gan_ssim,
#         "gan_runtime_sec": gan_time
    })

# # =========================================
# # [8] Save Results
# # =========================================
# results_df = pd.DataFrame(results)
# results_df.to_csv(os.path.join(OUTPUT_DIR, "metrics_full.csv"), index=False)

# summary_df = pd.DataFrame({
#     "Method": ["Richardson-Lucy", "DeblurGAN-v2"],
#     "Average PSNR": [results_df["rl_psnr"].mean(), results_df["gan_psnr"].mean()],
#     "Average SSIM": [results_df["rl_ssim"].mean(), results_df["gan_ssim"].mean()],
#     "Average Runtime (sec)": [results_df["rl_runtime_sec"].mean(), results_df["gan_runtime_sec"].mean()]
# })
# summary_df.to_csv(os.path.join(OUTPUT_DIR, "summary_full.csv"), index=False)
# print(summary_df)

# category_summary = results_df.groupby("blur_category").agg({
#     "rl_psnr": "mean",
#     "rl_ssim": "mean",
#     "gan_psnr": "mean",
#     "gan_ssim": "mean"
# }).reset_index()
# category_summary.to_csv(os.path.join(OUTPUT_DIR, "category_summary.csv"), index=False)
# print("\nBlur Category Summary:\n", category_summary)

# # =========================================
# # [9] Plotting
# # =========================================
# plt.figure(figsize=(8, 5))
# plt.bar(summary_df["Method"], summary_df["Average PSNR"])
# plt.ylabel("PSNR")
# plt.title("Average PSNR Comparison")
# plt.savefig(os.path.join(OUTPUT_DIR, "psnr_comparison.png"))
# plt.close()

# plt.figure(figsize=(8, 5))
# plt.bar(summary_df["Method"], summary_df["Average SSIM"])
# plt.ylabel("SSIM")
# plt.title("Average SSIM Comparison")
# plt.savefig(os.path.join(OUTPUT_DIR, "ssim_comparison.png"))
# plt.close()

# plt.figure(figsize=(8, 5))
# plt.bar(summary_df["Method"], summary_df["Average Runtime (sec)"])
# plt.ylabel("Seconds")
# plt.title("Average Runtime Comparison")
# plt.savefig(os.path.join(OUTPUT_DIR, "runtime_comparison.png"))
# plt.close()

# print("Processing Complete. Evaluations saved in CSV files.")