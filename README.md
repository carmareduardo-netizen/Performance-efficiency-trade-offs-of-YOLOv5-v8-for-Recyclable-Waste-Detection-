# Performance–Efficiency Trade-offs of YOLOv5 and YOLOv8 for Recyclable Waste Detection under Mixed Visual Conditions

This repository contains the code and experimental resources associated with the study:

**“Performance–Efficiency Trade-offs of YOLOv5 and YOLOv8 for Recyclable Waste Detection under Mixed Visual Conditions.”**

The study evaluates YOLOv5 and YOLOv8 across three model scales (nano, small, and medium) for recyclable waste detection. The experimental analysis focuses on detection performance, visual-condition sensitivity, model scaling, computational efficiency, run-to-run variability, and model interpretability.

## Models

The following model configurations were evaluated:

- YOLOv5n
- YOLOv5s
- YOLOv5m
- YOLOv8n
- YOLOv8s
- YOLOv8m

All models were initialized from pretrained weights and trained using a fixed 50-epoch training budget with an input resolution of 640 × 640 pixels.

## Repository Structure

```text
YOLO_Waste_Reproducibility/
├── training/
│   ├── train_yolov5.py
│   └── train_yolov8.ipynb
│
├── evaluation/
│   ├── benchmark_cpu.py
│   ├── model_complexity.py
│   └── visual_conditions_evaluation.ipynb
│
├── analysis/
│   ├── partition_leakage_phash.ipynb
│   └── leakage_sensitivity_analysis.ipynb
│
├── interpretability/
│   └── README.md
│
├── dataset_metadata/
│   └── data.yaml
│
├── requirements.txt
└── README.md
