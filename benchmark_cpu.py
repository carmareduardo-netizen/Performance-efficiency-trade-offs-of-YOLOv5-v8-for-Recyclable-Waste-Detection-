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
import psutil
import torch
from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
TEST_DIR = BASE_DIR / "test_images"

# Your current YOLOv5 repository location
YOLOV5_DIR = BASE_DIR / "yolov5" / "yolov5-master"

# ============================================================
# BENCHMARK CONFIGURATION
# ============================================================

IMG_SIZE = 640
WARMUP = 10
N_REPEATS = 5
CPU_THREADS = 8

MODELS = {
    "YOLOv5n": {
        "family": "v5",
        "path": WEIGHTS_DIR / "best_v5n_2.pt",
    },
    "YOLOv5s": {
        "family": "v5",
        "path": WEIGHTS_DIR / "best_v5s_3.pt",
    },
    "YOLOv5m": {
        "family": "v5",
        "path": WEIGHTS_DIR / "best_v5m_4.pt",
    },
    "YOLOv8n": {
        "family": "v8",
        "path": WEIGHTS_DIR / "best_v8n_2.pt",
    },
    "YOLOv8s": {
        "family": "v8",
        "path": WEIGHTS_DIR / "best_v8s_3.pt",
    },
    "YOLOv8m": {
        "family": "v8",
        "path": WEIGHTS_DIR / "best_v8m_4.pt",
    },
}

# ============================================================
# CPU CONTROL
# ============================================================

os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)

torch.set_num_threads(CPU_THREADS)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# ============================================================
# WINDOWS FIX FOR YOLOv5 CHECKPOINTS SAVED ON LINUX/COLAB
# ============================================================

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# ============================================================
# SYSTEM INFORMATION
# ============================================================

print("\n" + "=" * 75)
print("CPU BENCHMARK — YOLOv5 vs YOLOv8")
print("=" * 75)

print(f"Processor: {platform.processor()}")
print(f"Physical CPU cores: {psutil.cpu_count(logical=False)}")
print(f"Logical CPU cores: {psutil.cpu_count(logical=True)}")
print(f"PyTorch threads: {torch.get_num_threads()}")
print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")

try:
    import ultralytics
    print(f"Ultralytics: {ultralytics.__version__}")
except Exception:
    pass

print(f"Image size: {IMG_SIZE} x {IMG_SIZE}")
print("Batch size: 1")
print("Device: CPU")
print("Precision: FP32")
print(f"Warm-up inferences/model: {WARMUP}")
print(f"Repetitions/model: {N_REPEATS}")

# ============================================================
# LOAD TEST SET
# ============================================================

extensions = {".jpg", ".jpeg", ".png", ".bmp"}

image_paths = sorted([
    p for p in TEST_DIR.iterdir()
    if p.suffix.lower() in extensions
])

if not image_paths:
    raise RuntimeError(f"No images found in: {TEST_DIR}")

print(f"\nTest images detected: {len(image_paths)}")

if len(image_paths) != 194:
    print(
        f"WARNING: Expected 194 test images, "
        f"but found {len(image_paths)}."
    )
else:
    print("Complete 194-image test set detected.")

# Pre-load images so disk I/O is not included in latency
images = []

print("Loading images into RAM...")

for p in image_paths:
    img = cv2.imread(str(p))

    if img is None:
        raise RuntimeError(f"Could not read image: {p}")

    images.append(img)

print(f"{len(images)} images loaded into RAM.")

# ============================================================
# LOAD MODEL FUNCTIONS
# ============================================================

def load_v5(weight_path):
    if not YOLOV5_DIR.exists():
        raise RuntimeError(
            f"YOLOv5 repository not found: {YOLOV5_DIR}"
        )

    if str(YOLOV5_DIR) not in sys.path:
        sys.path.insert(0, str(YOLOV5_DIR))

    model = torch.hub.load(
        str(YOLOV5_DIR),
        "custom",
        path=str(weight_path),
        source="local",
        device="cpu",
        _verbose=False,
    )

    model.conf = 0.25
    model.iou = 0.45

    return model


def load_v8(weight_path):
    return YOLO(str(weight_path))


# ============================================================
# PREDICTION FUNCTIONS
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

summary_results = []
repeat_results = []
raw_results = []

for model_name, config in MODELS.items():

    print("\n" + "=" * 75)
    print(f"BENCHMARKING {model_name}")
    print("=" * 75)

    weight_path = config["path"]

    if not weight_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weight_path}"
        )

    print(f"Checkpoint: {weight_path.name}")

    if config["family"] == "v5":
        model = load_v5(weight_path)
        predict_fn = predict_v5
    else:
        model = load_v8(weight_path)
        predict_fn = predict_v8

    # --------------------------------------------------------
    # WARM-UP
    # --------------------------------------------------------

    print(f"Running {WARMUP} warm-up inferences...")

    for i in range(WARMUP):
        predict_fn(
            model,
            images[i % len(images)]
        )

    print("Warm-up completed.")

    # --------------------------------------------------------
    # FIVE FULL TEST-SET REPETITIONS
    # --------------------------------------------------------

    repetition_means = []
    all_latencies = []

    for repeat in range(1, N_REPEATS + 1):

        latencies = []

        print(
            f"\nRepetition {repeat}/{N_REPEATS} "
            f"({len(images)} images)"
        )

        repetition_start = time.perf_counter()

        for image_index, image in enumerate(images, start=1):

            start = time.perf_counter()

            predict_fn(model, image)

            end = time.perf_counter()

            latency_ms = (end - start) * 1000.0

            latencies.append(latency_ms)

            raw_results.append({
                "Model": model_name,
                "Repeat": repeat,
                "Image_Index": image_index,
                "Latency_ms": latency_ms,
            })

        repetition_end = time.perf_counter()

        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        repetition_fps = 1000.0 / mean_latency

        repetition_means.append(mean_latency)
        all_latencies.extend(latencies)

        repeat_results.append({
            "Model": model_name,
            "Repeat": repeat,
            "Images": len(images),
            "Mean_Latency_ms": mean_latency,
            "Median_Latency_ms": median_latency,
            "FPS": repetition_fps,
            "Total_Time_s": repetition_end - repetition_start,
        })

        print(
            f"Mean latency: {mean_latency:.2f} ms/image | "
            f"FPS: {repetition_fps:.2f}"
        )

    # --------------------------------------------------------
    # MODEL SUMMARY
    # --------------------------------------------------------

    final_mean = statistics.mean(repetition_means)

    final_sd = (
        statistics.stdev(repetition_means)
        if len(repetition_means) > 1
        else 0.0
    )

    final_median = statistics.median(all_latencies)

    final_fps = 1000.0 / final_mean

    summary_results.append({
        "Model": model_name,
        "Checkpoint": weight_path.name,
        "Images": len(images),
        "Repeats": N_REPEATS,
        "Mean_Latency_ms": final_mean,
        "SD_Latency_ms": final_sd,
        "Median_Latency_ms": final_median,
        "FPS": final_fps,
    })

    print("\nFINAL RESULT")
    print(
        f"{model_name}: "
        f"{final_mean:.2f} ± {final_sd:.2f} ms/image"
    )
    print(f"Median latency: {final_median:.2f} ms/image")
    print(f"FPS: {final_fps:.2f}")

    del model
    gc.collect()

# ============================================================
# SAVE EVERYTHING
# ============================================================

summary_df = pd.DataFrame(summary_results)
repeat_df = pd.DataFrame(repeat_results)
raw_df = pd.DataFrame(raw_results)

summary_path = BASE_DIR / "CPU_BENCHMARK_RESULTS.csv"
repeat_path = BASE_DIR / "CPU_BENCHMARK_REPEATS.csv"
raw_path = BASE_DIR / "CPU_BENCHMARK_RAW.csv"

summary_df.to_csv(summary_path, index=False)
repeat_df.to_csv(repeat_path, index=False)
raw_df.to_csv(raw_path, index=False)

# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n\n" + "=" * 75)
print("FINAL CPU BENCHMARK RESULTS")
print("=" * 75)

print(summary_df.to_string(index=False))

print("\nGenerated files:")
print(summary_path.name)
print(repeat_path.name)
print(raw_path.name)

print("\nBENCHMARK COMPLETED SUCCESSFULLY.")