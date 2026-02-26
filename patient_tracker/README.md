# Seizure Detection Pipeline

A pipeline for preprocessing hospital CCTV videos for seizure detection. Combines:

1. **SAM3 (Segment Anything Model 3)** - Text/concept-based patient tracking
2. **Sapiens 2B** - State-of-the-art pose estimation (82.2 AP on COCO)

## Features

- 🎯 **Text-Prompt Based Tracking**: Track patients using prompts like "patient on bed"
- 🔍 **Patient Isolation**: Segment patient from background with black masking
- 🦴 **17 COCO Keypoints**: Robust pose estimation using Sapiens 2B
- 🌙 **Day/Night Support**: Works with both RGB (daytime) and IR (nighttime) footage
- 📹 **Side-by-Side Visualization**: Full frame + segmentation | Isolated patient + pose
- 💾 **NPY Output**: Keypoints saved as numpy arrays for downstream analysis

## Directory Structure

```
seizure_detection_pipeline/
├── weights/                    # Model weights (you need to add these)
│   ├── sam3.pt                # SAM3 model weights
│   └── sapiens_2b_coco_best_coco_AP_822.pth  # Sapiens model weights
├── output/                     # Output directory (created automatically)
├── pipeline.py                 # Main pipeline code
├── run.py                      # Simple runner script (configure and run)
├── sapiens_pose.py            # Standalone Sapiens pose estimator
├── utils.py                    # Utility functions
├── analyze_keypoints.py       # Keypoint analysis tools
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Download Model Weights

Create a `weights/` directory and download:

1. **SAM3 Model** (`sam3.pt`)
   - Download from: https://huggingface.co/facebook/sam3
   - Requires HuggingFace account with access approval

2. **Sapiens Model** (TorchScript `.pt2` file)
   - **Recommended**: `sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2` (308 keypoints)
   - Download from: https://huggingface.co/facebook/sapiens-pose-1b-torchscript
   - Alternative COCO model: `sapiens_1b_coco_best_coco_AP_821_torchscript.pt2` (17 keypoints)

```bash
mkdir weights
# Place downloaded files in weights/
```

## Quick Start

### Option 1: Configure and Run (Recommended)

Edit `run.py` to set your configuration:

```python
# Input: Path to video file OR directory containing videos
VIDEO_INPUT = "./videos/hospital_room.mp4"

# Output directory
OUTPUT_DIR = "./output"

# Weights directory
WEIGHTS_DIR = "./weights"

# Text prompts for patient detection
PATIENT_PROMPTS = ["patient", "person on bed", "patient on bed"]
```

Then run:
```bash
python run.py
```

### Option 2: Python API

```python
from pipeline import SeizureDetectionPipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    weights_dir="./weights",
    output_dir="./output",
    patient_prompts=["patient", "person on bed"],
    save_video=True,
    save_npy=True,
)

# Initialize and run
pipeline = SeizureDetectionPipeline(config)
result = pipeline.process_video("hospital_video.mp4")

print(f"Keypoints saved to: {result['keypoints_npy']}")
print(f"Video saved to: {result['output_video']}")
```

### Option 3: Process Multiple Videos

```python
from pipeline import SeizureDetectionPipeline, PipelineConfig

config = PipelineConfig(weights_dir="./weights", output_dir="./output")
pipeline = SeizureDetectionPipeline(config)

# Process all videos in a directory
results = pipeline.process_directory("./videos/")

for r in results:
    print(f"{r['video_path']}: {r['frames_with_detection']}/{r['total_frames']} frames")
```

## Output Format

### 1. Keypoints NPY File

```python
import numpy as np

# Load keypoints
data = np.load("output/video_keypoints.npy", allow_pickle=True).item()

# Contents:
keypoints = data['keypoints']      # Shape: (num_frames, 17, 3)
confidences = data['confidences']  # Shape: (num_frames,)
keypoint_names = data['keypoint_names']  # List of 17 names
skeleton = data['skeleton']        # List of bone connections
fps = data['fps']                  # Video frame rate
frame_size = data['frame_size']    # (height, width)

# Each keypoint has [x, y, confidence]
# x, y are in original frame coordinates
# confidence is 0-1 (higher = more confident)
```

### 2. COCO Keypoint Order

| Index | Keypoint | Index | Keypoint |
|-------|----------|-------|----------|
| 0 | nose | 9 | left_wrist |
| 1 | left_eye | 10 | right_wrist |
| 2 | right_eye | 11 | left_hip |
| 3 | left_ear | 12 | right_hip |
| 4 | right_ear | 13 | left_knee |
| 5 | left_shoulder | 14 | right_knee |
| 6 | right_shoulder | 15 | left_ankle |
| 7 | left_elbow | 16 | right_ankle |
| 8 | right_elbow | | |

### 3. Visualization Video

The output video (`*_visualization.mp4`) shows:
- **Left side**: Full frame with green segmentation overlay and pose skeleton
- **Right side**: Zoomed view of isolated patient with pose skeleton

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `weights_dir` | "./weights" | Directory containing model weights |
| `output_dir` | "./output" | Directory for output files |
| `patient_prompts` | ["patient", "person on bed", ...] | Text prompts for SAM3 |
| `sam3_conf` | 0.25 | SAM3 detection confidence threshold |
| `sam3_imgsz` | 640 | SAM3 input image size |
| `device` | auto | "cuda" or "cpu" |
| `save_video` | True | Save visualization video |
| `save_npy` | True | Save keypoints NPY file |
| `keypoint_conf_threshold` | 0.3 | Min confidence to draw keypoint |

## Analyzing Keypoints

Use the analysis script for post-processing:

```bash
# Analyze a keypoints file
python analyze_keypoints.py output/video_keypoints.npy --detect-events

# Plot motion timeline
python analyze_keypoints.py output/video_keypoints.npy --plot-motion

# Export to JSON
python analyze_keypoints.py output/video_keypoints.npy --export-json output.json
```

Python API:
```python
from analyze_keypoints import (
    load_keypoints,
    compute_body_part_motion,
    detect_high_motion_segments,
    compute_statistics,
)

data = load_keypoints("output/video_keypoints.npy")
keypoints = data['keypoints']

# Compute motion for upper body
motion = compute_body_part_motion(keypoints, 'upper_body')

# Detect high motion segments (potential seizure events)
events = detect_high_motion_segments(motion, fps=30.0)
for event in events:
    print(f"Event at {event['start_time']:.2f}s - {event['end_time']:.2f}s")
```

## Troubleshooting

### "SAM3 model not found"
- Download `sam3.pt` from HuggingFace (requires account approval)
- Place in `weights/` directory

### "Sapiens model not found"
- Download from HuggingFace Sapiens collection
- Place in `weights/` directory

### "CUDA out of memory"
- Reduce `sam3_imgsz` (e.g., 480)
- Use `device="cpu"` (slower)

### Low detection rate
- Try different prompts: "person lying down", "patient in bed"
- Adjust `sam3_conf` threshold (lower = more detections)
- Check if camera angle shows patient clearly

### Low pose quality
- Sapiens works best when patient is well-isolated
- Check if segmentation mask is accurate
- IR/night footage may have lower quality

## Edge Cases Handled

- **No patient detected**: Returns zero keypoints for that frame
- **Multiple patients**: Uses largest detection
- **Partial occlusion**: Reduces confidence for occluded keypoints
- **Invalid masks**: Area-based filtering (too small/large rejected)
- **Variable resolution**: Automatic resizing

## License

This pipeline uses:
- **SAM3**: Meta AI Research License
- **Sapiens**: Meta AI Research License
- **Ultralytics**: AGPL-3.0 License

Check individual model licenses for your use case.

## Citation

```bibtex
@article{sam3_2025,
  title={SAM 3: Segment Anything with Concepts},
  author={Meta AI},
  year={2025}
}

@article{sapiens2024,
  title={Sapiens: Foundation for Human Vision Models},
  author={Khirodkar et al.},
  journal={arXiv preprint arXiv:2408.12569},
  year={2024}
}
```
