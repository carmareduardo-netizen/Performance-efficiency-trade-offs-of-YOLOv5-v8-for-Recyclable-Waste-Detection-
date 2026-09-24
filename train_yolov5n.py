# SECURITY NOTE: set ROBOFLOW_API_KEY as an environment variable before running.
# The credential from the original research notebook was intentionally removed.

# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
#IMPORTAR LIBRERIAS
!git clone https://github.com/ultralytics/yolov5.git
# %cd yolov5
# %pip install -r requirements.txt
# %pip install roboflow thop pandas -q
#IMPORTAR DATASET DE ROBOFLOW
import os
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace(os.environ["ROBOFLOW_WORKSPACE"]).project(os.environ["ROBOFLOW_PROJECT"])
version = project.version(int(os.environ.get("ROBOFLOW_VERSION", "1")))
dataset = version.download("yolov5")


# Desactiva wandb (para evitar el prompt que congela el entrenamiento)
import os
os.environ["WANDB_MODE"] = "disabled"

# EJECUTA ENTRENAMIENTO
!python train.py \
  --img 640 \
  --batch 16 \
  --epochs 50 \
  --optimizer="SGD"\
  --data {dataset.location}/data.yaml \
  --weights yolov5n.pt \
  --name yolov5_residuos

# GRAFICAS
from IPython.display import Image, display
display(Image('runs/train/yolov5_residuos/results.png'))
display(Image('runs/train/yolov5_residuos/confusion_matrix.png'))

# METRICAS POR CADA CATEGORIA
!python val.py --weights runs/train/yolov5_residuos/weights/best.pt --data {dataset.location}/data.yaml --img 640

# ==========================================================
# EFICIENCIA COMPUTACIONAL
# ==========================================================
import torch
import time
import numpy as np
import pandas as pd
from models.experimental import attempt_load
from thop import profile
from google.colab import files

MODEL_NAME = "YOLOv5n"
WEIGHTS = "runs/train/yolov5_residuos/weights/best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = attempt_load(
    WEIGHTS,
    device=device
)

# -------------------------
# PARÁMETROS
# -------------------------
params = sum(p.numel() for p in model.parameters())
params_m = params / 1e6

# -------------------------
# FLOPs
# -------------------------
dummy = torch.randn(1, 3, 640, 640).to(device)

flops, _ = profile(
    model,
    inputs=(dummy,),
    verbose=False
)

flops_g = flops / 1e9

# -------------------------
# TAMAÑO DEL MODELO
# -------------------------
size_mb = os.path.getsize(WEIGHTS) / (1024 * 1024)

# -------------------------
# WARM-UP
# -------------------------
for _ in range(20):
    with torch.no_grad():
        _ = model(dummy)

# -------------------------
# TIEMPO DE INFERENCIA
# -------------------------
times = []

for _ in range(100):

    start = time.perf_counter()

    with torch.no_grad():
        _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    times.append((end - start) * 1000)

avg_time = np.mean(times)
std_time = np.std(times)

fps = 1000 / avg_time

# ==========================================================
# INGRESAR MÉTRICAS DE VAL.PY
# ==========================================================
precision = float(input("Precision: "))
recall = float(input("Recall: "))
map50 = float(input("mAP@50: "))
map5095 = float(input("mAP@50-95: "))

# ==========================================================
# TABLA RESUMEN
# ==========================================================
df = pd.DataFrame({
    "Modelo":[MODEL_NAME],
    "Precision":[round(precision,4)],
    "Recall":[round(recall,4)],
    "mAP50":[round(map50,4)],
    "mAP50_95":[round(map5095,4)],
    "Tiempo_ms":[round(avg_time,2)],
    "FPS":[round(fps,2)],
    "Parametros_M":[round(params_m,2)],
    "FLOPs_GFLOPs":[round(flops_g,2)],
    "Tamano_MB":[round(size_mb,2)]
})

print(df)

# ==========================================================
# EXPORTAR RESULTADOS
# ==========================================================
csv_name = f"{MODEL_NAME}_resumen.csv"

df.to_csv(csv_name, index=False)

# ==========================================================
# DESCARGAR TODO LO IMPORTANTE
# ==========================================================
files.download(csv_name)

files.download("runs/train/yolov5_residuos/results.csv")
files.download("runs/train/yolov5_residuos/results.png")
files.download("runs/train/yolov5_residuos/confusion_matrix.png")
files.download("runs/train/yolov5_residuos/weights/best.pt")