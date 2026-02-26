# Weights

This directory stores all model weights required by the pipeline. Download each component below and place the files in this directory.

---

## Trained Neurosymbolic Seizure Detection Model

Pre-trained weights for the full neurosymbolic model (HyperGCN encoder + Concept Predictors + Logic Layers):

| Dataset | Download |
|---------|----------|
| VSVIG (SAHZU) | [Google Drive](https://drive.google.com/drive/folders/1d-lMPezFwttaNQmMyg89WEzeSIsW-T0D?usp=sharing) |
| IEEE | [Google Drive](https://drive.google.com/drive/folders/1d-lMPezFwttaNQmMyg89WEzeSIsW-T0D?usp=sharing) |

---

## Data Preprocessing Model Weights

These weights are used during the data extraction stages (patient tracking, pose estimation, and activity labelling).

### SAM3 — Patient Segmentation

Download the SAM3 checkpoint and place `sam3.pt` in this directory:

> **[Download sam3.pt (HuggingFace)](https://huggingface.co/facebook/sam3/resolve/main/sam3.pt?download=true)**

### Sapiens 2B — Pose Estimation

Download the **TorchScript** variant for faster inference:

> **[Download Sapiens 2B TorchScript (HuggingFace)](https://huggingface.co/noahcao/sapiens-pose-coco/tree/main/sapiens_lite_host/torchscript/pose/checkpoints/sapiens_2b)**

### Qwen3-VL-8B-Instruct — VLM Activity Classifier

Used by the `vlm_assisted_labeller/` pipeline for frame-level action classification:

> **[Download Qwen3-VL-8B-Instruct (HuggingFace)](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)**

