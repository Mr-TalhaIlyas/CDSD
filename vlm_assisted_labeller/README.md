# Seizure Video Labeling Pipeline

A comprehensive pipeline for generating frame-level action labels from hospital seizure monitoring videos using Vision-Language Models (VLM).

## Overview

This pipeline processes epilepsy monitoring unit (EMU) videos to generate action labels for the **normal activity duration** (before seizure onset). It uses a sliding window approach with VLM-based classification to label patient activities.

### Key Features

- **Automated action classification** using Qwen3-VL-8B-Instruct
- **Frame-level labels** at 30 FPS resolution
- **Sliding window approach** with 5-second windows
- **Seizure onset integration** from clinical data (Excel)
- **Comprehensive statistics** and label distribution analysis

## Pipeline Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video File    │ -> │  Extract Frames  │ -> │  VLM Classifier │
│  (.mp4)         │    │  (5s windows)    │    │  (Qwen3-VL)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌──────────────────┐           │
│  Excel Labels   │ -> │  Parse Seizure   │           │
│  (Clinical      │    │  Onset Times     │           │
│   Onset)        │    └──────────────────┘           │
└─────────────────┘              │                    │
                                 v                    v
                       ┌──────────────────────────────────┐
                       │   Generate Frame-Level Labels    │
                       │   (30 labels/sec = FPS)          │
                       └──────────────────────────────────┘
                                       │
                                       v
                       ┌──────────────────────────────────┐
                       │         Output Files             │
                       │  - .npy label arrays             │
                       │  - label_map.json                │
                       │  - statistics.json               │
                       └──────────────────────────────────┘
```

## Installation

### Requirements

```bash
pip install torch torchvision transformers
pip install opencv-python numpy pandas openpyxl
pip install tqdm pillow
```

### Optional (for faster inference)

```bash
pip install flash-attn --no-build-isolation
```

## File Structure

```
seizure_labeling_pipeline/
├── config.py           # Configuration and action classes
├── video_utils.py      # Video processing utilities
├── vlm_classifier.py   # VLM inference module
├── pipeline.py         # Main pipeline orchestration
├── examples.py         # Usage examples
└── README.md           # This file
```

## Action Classes (Labels)

| ID | Label | Description |
|----|-------|-------------|
| 0 | sleeping | Patient is asleep, eyes closed, minimal movement |
| 1 | resting | Awake but lying still, relaxed |
| 2 | reading | Reading a book, magazine, or document |
| 3 | using_phone | Using mobile phone or tablet |
| 4 | watching_tv | Watching TV or looking at screen |
| 5 | eating | Eating food or drinking |
| 6 | talking | Talking to someone (staff/visitor) |
| 7 | sitting_up | Sitting up in bed |
| 8 | adjusting_position | Changing position, adjusting covers |
| 9 | medical_interaction | Medical staff interacting with patient |
| 10 | other_activity | Other normal activity |
| 11 | unclear | Cannot determine (occlusion, blur) |
| 12 | seizure | Seizure activity (post-clinical onset) |

## Usage

### Basic Usage

```python
from pipeline import SeizureLabelingPipeline

# Initialize pipeline
pipeline = SeizureLabelingPipeline(
    video_dir="./videos_raw/vids",
    output_dir="./labels_output",
    excel_path="./dataset_Label_-_Copy.xlsx"
)

# Process all videos
pipeline.process_all_videos()

# Save outputs and print statistics
pipeline.save_combined_output()
pipeline.print_statistics()
```

### Command Line

```bash
# Process all videos
python pipeline.py --video-dir ./videos_raw/vids --output-dir ./labels_output --excel-path ./dataset_Label_-_Copy.xlsx

# With flash attention (faster)
python pipeline.py --flash-attention

# Test mode (no GPU required)
python pipeline.py --mock

# Process single video
python pipeline.py --single-video ./videos_raw/vids/pat01_000_Sz1PG.mp4
```

### Processing Custom Video List

```python
video_files = [
    './videos_raw/vids/pat01_000_Sz1PG.mp4',
    './videos_raw/vids/pat02_002_Sz1PG.mp4',
]

pipeline.process_all_videos(video_paths=video_files)
```

## Output Files

### 1. Label Arrays (`.npy`)

For each video, a numpy array is saved with shape `(total_frames,)`:

```python
import numpy as np

# Load labels
labels = np.load('labels_output/pat01_000_Sz1PG_labels.npy')

print(labels.shape)  # (total_frames,) e.g., (108000,) for 1 hour @ 30fps
print(labels.dtype)  # int32
```

**Important**: Each frame has exactly one label. For a 10-second video at 30 FPS, the array has 300 elements.

### 2. Label Map (`label_map.json`)

```json
{
  "id_to_label": {
    "0": "sleeping",
    "1": "resting",
    ...
    "12": "seizure"
  },
  "label_to_id": {
    "sleeping": 0,
    "resting": 1,
    ...
    "seizure": 12
  },
  "action_classes": ["sleeping", "resting", ...]
}
```

### 3. Statistics (`labeling_statistics.json`)

```json
{
  "summary": {
    "total_videos_processed": 12,
    "total_frames": 1500000,
    "total_duration_min": 833.33
  },
  "label_distribution": {
    "frame_counts": {
      "sleeping": 450000,
      "resting": 300000,
      ...
    },
    "percentages": {
      "sleeping": 30.0,
      "resting": 20.0,
      ...
    }
  }
}
```

## Video Filename Convention

The pipeline expects video filenames in this format:

```
patXX_YYY_ZZZ.mp4
```

Where:
- `patXX`: Patient ID (e.g., `pat01`, `pat02`)
- `YYY`: Video index (e.g., `000`, `001`)
- `ZZZ`: Seizure info or "free"
  - `Sz1PG`: Seizure 1, type PG
  - `Sz2P`: Seizure 2, type P
  - `free`: No seizure in video
  - `no-Sz2P`: Video without seizure

### Examples

| Filename | Patient | Seizure | Type | Has Seizure |
|----------|---------|---------|------|-------------|
| `pat01_000_Sz1PG.mp4` | Pat01 | Sz1 | PG | Yes |
| `pat03_006_free.mp4` | Pat03 | - | - | No |
| `pat04_009_no-Sz2P.mp4` | Pat04 | - | - | No |

## Excel File Format

The Excel file should have these columns:

| PatID | Seizure Type | #Seizure | EEG onset | Clinical Onset |
|-------|--------------|----------|-----------|----------------|
| Pat01 | PG | Sz1 | 00:59:40 | 00:59:51 |
| Pat01 | PG | Sz2 | 00:10:24 | 00:10:40 |

**Clinical Onset** is used as the seizure start marker. All frames after this time are labeled as "seizure".

## VLM Prompt Design

The VLM prompt is carefully designed for hospital bed monitoring:

1. **System Prompt**: Establishes context (hospital EMU, patient monitoring)
2. **Classification Guidelines**: Clear definitions for each action class
3. **Focus Instructions**: Focus on patient only, ignore staff/visitors
4. **Output Format**: Single-word label only

See `config.py` for the full prompts.

## Integration with Training Pipeline

```python
import torch
from torch.utils.data import Dataset
import numpy as np
import json

class SeizureActionDataset(Dataset):
    def __init__(self, video_paths, labels_dir):
        self.video_paths = video_paths
        
        # Load label map
        with open(f'{labels_dir}/label_map.json') as f:
            self.label_info = json.load(f)
        
        # Load all labels
        self.all_labels = {}
        for vp in video_paths:
            name = Path(vp).stem
            self.all_labels[vp] = np.load(f'{labels_dir}/{name}_labels.npy')
    
    def get_label_for_frame(self, video_path, frame_idx):
        return self.all_labels[video_path][frame_idx]
    
    def get_labels_for_window(self, video_path, start_frame, window_size):
        labels = self.all_labels[video_path][start_frame:start_frame + window_size]
        # Return majority label
        unique, counts = np.unique(labels, return_counts=True)
        return unique[np.argmax(counts)]
```

## Testing Without GPU

Use the mock classifier for pipeline testing:

```python
pipeline = SeizureLabelingPipeline(
    ...
    use_mock_vlm=True  # Uses brightness-based pseudo-random labels
)
```

## Performance Tips

1. **Flash Attention**: Use `--flash-attention` for 2-3x speedup
2. **Batch Processing**: The VLM processes one frame per window
3. **GPU Memory**: Qwen3-VL-8B requires ~16GB VRAM

## License

This pipeline is provided for research purposes. Ensure compliance with Qwen model license for VLM usage.
