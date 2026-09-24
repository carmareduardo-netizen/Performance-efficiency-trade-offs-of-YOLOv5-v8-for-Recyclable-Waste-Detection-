import os
import sys
import gc
import time
import pathlib
import platform
import statistics
from pathlib import Path

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
TEST_DIR = BASE_DIR / "test_images"
YOLOV5_DIR = BASE_DIR / "yolov5" / "yolov5-master"

IMG_SIZE = 640
WARMUP = 10
N_REPEATS = 5
CPU_THREADS = 8

MODELS = {
    "YOLOv5m": {
        "family": "v5",
        "path": WEIGHTS_DIR / "best_v5m_4.pt",
    },
    "YOLOv8m": {
        "family": "v8",
        "path": WEIGHTS_DIR / "best_v8m_4.pt",
    },
}

# ============================================================
# CPU CONFIGURATION
# ============================================================

os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)

torch.set_num_threads(CPU_THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# Windows compatibility for YOLOv5 checkpoints saved on Linux
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# ============================================================
# LOAD TEST IMAGES
# ============================================================

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

image_paths = sorted([
    p for p in TEST_DIR.iterdir()
    if p.suffix.lower() in extensions
])

print("=" * 70)
print("REPEAT BENCHMARK — YOLOv5m / YOLOv8m")
print("=" * 70)
print(f"Images detected: {len(image_paths)}")
print(f"CPU threads: {CPU_THREADS}")
print("Device: CPU")
print("Precision: FP32")
print(f"Warm-up: {WARMUP}")
print(f"Repetitions: {N_REPEATS}")

if len(image_paths) != 194:
    raise RuntimeError(
        f"Expected 194 images, but found {len(image_paths)}."
    )

images = []

print("\nLoading 194 images into RAM...")

for path in image_paths:
    img = cv2.imread(str(path))

    if img is None:
        raise RuntimeError(f"Could not read: {path}")

    images.append(img)

print("194 images loaded successfully.")

# ============================================================
# MODEL LOADING
# ============================================================

def load_v5(path):

    if str(YOLOV5_DIR) not in sys.path:
        sys.path.insert(0, str(YOLOV5_DIR))

    model = torch.hub.load(
        str(YOLOV5_DIR),
        "custom",
        path=str(path),
        source="local",
        device="cpu",
        _verbose=False,
    )

    model.conf = 0.25
    model.iou = 0.45

    return model


def load_v8(path):
    return YOLO(str(path))


# ============================================================
# INFERENCE
# ============================================================

def predict_v5(model, image):

    with torch.inference_mode():
        _ = model(
            image,
            size=IMG_SIZE
        )


def predict_v8(model, image):

    with torch.inference_mode():

        _ = model.predict(
            source=image,
            imgsz=IMG_SIZE,
            device="cpu",
            conf=0.25,
            iou=0.45,
            verbose=False,
        )


# ============================================================
# BENCHMARK
# ============================================================

repeat_results = []
raw_results = []

for model_name, config in MODELS.items():

    print("\n" + "=" * 70)
    print(f"BENCHMARKING {model_name}")
    print("=" * 70)

    if config["family"] == "v5":
        model = load_v5(config["path"])
        predict = predict_v5
    else:
        model = load_v8(config["path"])
        predict = predict_v8

    # ------------------ WARM-UP ------------------

    print(f"Running {WARMUP} warm-up inferences...")

    for i in range(WARMUP):
        predict(model, images[i])

    print("Warm-up completed.")

    # ------------------ REPETITIONS ------------------

    for repeat in range(1, N_REPEATS + 1):

        print(
            f"\n{model_name} - Repetition "
            f"{repeat}/{N_REPEATS}"
        )

        latencies = []

        for image_index, image in enumerate(images, start=1):

            start = time.perf_counter()

            predict(model, image)

            end = time.perf_counter()

            latency_ms = (end - start) * 1000

            latencies.append(latency_ms)

            raw_results.append({
                "Model": model_name,
                "Repeat": repeat,
                "Image_Index": image_index,
                "Latency_ms": latency_ms
            })

        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        sd_image_latency = statistics.stdev(latencies)
        fps = 1000 / mean_latency

        repeat_results.append({
            "Model": model_name,
            "Repeat": repeat,
            "Images": len(images),
            "Mean_Latency_ms": mean_latency,
            "Median_Latency_ms": median_latency,
            "SD_Image_Latency_ms": sd_image_latency,
            "FPS": fps,
        })

        print(
            f"Mean: {mean_latency:.2f} ms | "
            f"Median: {median_latency:.2f} ms | "
            f"FPS: {fps:.2f}"
        )

    del model
    gc.collect()


# ============================================================
# SAVE RESULTS
# ============================================================

repeat_df = pd.DataFrame(repeat_results)
raw_df = pd.DataFrame(raw_results)

repeat_df.to_csv(
    BASE_DIR / "CPU_MEDIUM_REPEAT_RESULTS.csv",
    index=False
)

raw_df.to_csv(
    BASE_DIR / "CPU_MEDIUM_REPEAT_RAW.csv",
    index=False
)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(repeat_df.to_string(index=False))

print("\nFiles generated:")
print("CPU_MEDIUM_REPEAT_RESULTS.csv")
print("CPU_MEDIUM_REPEAT_RAW.csv")

print("\nBENCHMARK COMPLETED SUCCESSFULLY.")