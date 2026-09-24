import os
import sys
import pathlib
import platform
from pathlib import Path

import torch
import pandas as pd
from ultralytics import YOLO

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
YOLOV5_DIR = BASE_DIR / "yolov5" / "yolov5-master"

MODELS = {
    "YOLOv5n": {
        "family": "v5",
        "path": WEIGHTS_DIR / "best_v5n_2.pt"
    },
    "YOLOv5s": {
        "family": "v5",
        "path": WEIGHTS_DIR / "best_v5s_3.pt"
    },
    "YOLOv5m": {
        "family": "v5",
        "path": WEIGHTS_DIR / "best_v5m_4.pt"
    },
    "YOLOv8n": {
        "family": "v8",
        "path": WEIGHTS_DIR / "best_v8n_2.pt"
    },
    "YOLOv8s": {
        "family": "v8",
        "path": WEIGHTS_DIR / "best_v8s_3.pt"
    },
    "YOLOv8m": {
        "family": "v8",
        "path": WEIGHTS_DIR / "best_v8m_4.pt"
    },
}

# ============================================================
# WINDOWS FIX FOR YOLOv5 CHECKPOINTS SAVED ON LINUX
# ============================================================

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

# Add YOLOv5 repo
if str(YOLOV5_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV5_DIR))

# YOLOv5 utility for FLOPs
from utils.torch_utils import model_info

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


# ============================================================
# RESULTS
# ============================================================

results = []

print("\n" + "=" * 75)
print("MODEL COMPLEXITY ANALYSIS")
print("=" * 75)
print("Input resolution: 640 x 640")

for model_name, config in MODELS.items():

    print("\n" + "=" * 75)
    print(f"ANALYZING {model_name}")
    print("=" * 75)

    weight_path = config["path"]

    if not weight_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {weight_path}"
        )

    size_mb = get_size_mb(weight_path)

    # ========================================================
    # YOLOv5
    # ========================================================

    if config["family"] == "v5":

        hub_model = torch.hub.load(
            str(YOLOV5_DIR),
            "custom",
            path=str(weight_path),
            source="local",
            device="cpu",
            _verbose=False,
        )

        # AutoShape wrapper -> underlying DetectMultiBackend -> model
        try:
            core_model = hub_model.model.model
        except Exception:
            try:
                core_model = hub_model.model
            except Exception:
                core_model = hub_model

        params = count_parameters(core_model)

        # model_info prints the official YOLOv5 summary including GFLOPs.
        # We also try to calculate FLOPs directly using thop.
        try:
            from thop import profile

            dummy = torch.zeros(1, 3, 640, 640)

            core_model.eval()

            with torch.no_grad():
                flops, _ = profile(
                    core_model,
                    inputs=(dummy,),
                    verbose=False
                )

            # THOP returns MACs for many operations.
            # Convert MACs to FLOPs using 2 FLOPs per MAC.
            gflops = (flops * 2) / 1e9

        except Exception as e:
            print(f"Direct GFLOPs calculation failed: {e}")
            gflops = None

    # ========================================================
    # YOLOv8
    # ========================================================

    else:

        yolo = YOLO(str(weight_path))
        core_model = yolo.model

        params = count_parameters(core_model)

        try:
            from thop import profile

            dummy = torch.zeros(1, 3, 640, 640)

            core_model.eval()

            with torch.no_grad():
                flops, _ = profile(
                    core_model,
                    inputs=(dummy,),
                    verbose=False
                )

            gflops = (flops * 2) / 1e9

        except Exception as e:
            print(f"Direct GFLOPs calculation failed: {e}")
            gflops = None

    params_m = params / 1e6

    print(f"Parameters : {params_m:.3f} M")

    if gflops is not None:
        print(f"GFLOPs     : {gflops:.3f}")
    else:
        print("GFLOPs     : ERROR")

    print(f"Size       : {size_mb:.2f} MB")

    results.append({
        "Model": model_name,
        "Checkpoint": weight_path.name,
        "Parameters_M": params_m,
        "GFLOPs_640": gflops,
        "Size_MB": size_mb,
    })


# ============================================================
# SAVE RESULTS
# ============================================================

df = pd.DataFrame(results)

output = BASE_DIR / "MODEL_COMPLEXITY_RESULTS.csv"

df.to_csv(output, index=False)

print("\n\n" + "=" * 75)
print("FINAL MODEL COMPLEXITY RESULTS")
print("=" * 75)

print(df.to_string(index=False))

print("\nSaved to:")
print(output)

print("\nCOMPLETED SUCCESSFULLY.")