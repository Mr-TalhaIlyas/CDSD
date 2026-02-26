# Data

This directory contains processed dataset files used for training and evaluation. Below are download links for all required data assets.

---

## Processed Training-Ready Data

The fully processed `.npz` skeleton sequences (windowed, normalized, and split) can be downloaded directly:

> **[Download Processed Data (Google Drive)](https://drive.google.com/drive/folders/1tZEQVVN304sgnBYwWPV6lh482OMd9vSx?usp=sharing)**

Place the downloaded `.npz` files in this directory. These are ready to use with the training scripts in `training_scripts/`.

---

## Raw Extracted Pose Data

If you prefer to run your own preprocessing pipeline, the raw extracted pose keypoints (`.npy` files output by the `patient_tracker/` stage) are available:

> **[Download Raw Pose Data (Google Drive)](https://drive.google.com/drive/folders/1kqV1vkW_iEtXO_c6vxC26F1e6FNeHW-W?usp=sharing)**

These can be fed into `prepare_training_data/` to regenerate the training-ready `.npz` files.

---

## Raw RGB Videos

The original hospital EMU video recordings can be obtained from their respective sources:

| Dataset | Source | Link |
|---------|--------|------|
| VSVIG (SAHZU) | HuggingFace | [WU-SAHZU-EMU-Video](https://huggingface.co/datasets/xuyankun/WU-SAHZU-EMU-Video/tree/main) |
| IEEE | IEEE DataPort | [Seizure Videos of Epilepsy Patients](https://ieee-dataport.org/documents/seizure-videos-epilepsy-patients) |

---

> **Note:** Data will be migrated to a permanent public repository (e.g., HuggingFace or Zenodo) once the anonymization requirement is lifted.