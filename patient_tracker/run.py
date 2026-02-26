#!/usr/bin/env python3
"""
Seizure Detection Pipeline - Main Runner Script
================================================

Configure the variables below and run this script to process videos.

Outputs:
1. NPY file with pose keypoints: {output_dir}/{video_name}_keypoints.npy
   - Shape: (num_frames, 17, 3) where each keypoint has [x, y, confidence]
   - Contains 17 COCO keypoints per frame
   
2. Visualization MP4: {output_dir}/{video_name}_visualization.mp4
   - Left side: Full frame with green segmentation overlay + pose skeleton
   - Right side: Zoomed isolated patient with pose skeleton

Usage:
    1. Set the configuration variables below
    2. Run: python run.py
"""
#%%
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# CONFIGURATION - MODIFY THESE VARIABLES
# ============================================================================

# Input: Path to video file OR directory containing videos
VIDEO_INPUT = "/data_hdd/talha/miccai_26/seizure_detection_pipeline/videos/S0_0.mp4"  # Can be a single file or directory
# VIDEO_INPUT = "../videos_raw/Seizure"  # Can be a single file or directory

# Output directory for results
OUTPUT_DIR = "../output"
# OUTPUT_DIR = "../output_v2/Seizure"
# Weights directory containing model files
WEIGHTS_DIR = "../weights"

# Model file names (relative to WEIGHTS_DIR)
SAM3_MODEL_NAME = "sam3.pt"
SAPIENS_MODEL_NAME = "sapiens_2b_coco_best_coco_AP_822_torchscript.pt2"

# Text prompts for patient detection (SAM3 will look for these concepts)
    # "patient",
    # "patient on bed", 
    # "person in striped clothes"
PATIENT_PROMPTS = [
    'patient in center of frame',
    "patient on bed in striped clothes"
]

# Device: "cuda" for GPU, "cpu" for CPU, or None for auto-detection
DEVICE = None  # Auto-detect

# SAM3 settings
SAM3_CONFIDENCE = 0.25  # Detection confidence threshold (0.0 - 1.0)
SAM3_IMAGE_SIZE = 644   # Input image size for SAM3

# Output settings
SAVE_VIDEO = True       # Save visualization video
SAVE_NPY = True         # Save keypoints as NPY file

# Keypoint visualization settings
KEYPOINT_CONF_THRESHOLD = 0.3  # Min confidence to draw keypoint
SKELETON_THICKNESS = 2         # Line thickness for skeleton
KEYPOINT_RADIUS = 4            # Circle radius for keypoints


# ============================================================================
# DO NOT MODIFY BELOW THIS LINE
# ============================================================================

def main():
    """Run the seizure detection pipeline."""
    
    import torch
    from pipeline import SeizureDetectionPipeline, PipelineConfig
    
    # Auto-detect device if not specified
    device = DEVICE
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Build full model paths
    sam3_model_path = os.path.join(WEIGHTS_DIR, SAM3_MODEL_NAME)
    sapiens_model_path = os.path.join(WEIGHTS_DIR, SAPIENS_MODEL_NAME)
    
    # Verify weights exist
    if not os.path.exists(sam3_model_path):
        print(f"ERROR: SAM3 model not found at: {sam3_model_path}")
        print(f"Please place {SAM3_MODEL_NAME} in {WEIGHTS_DIR}/")
        sys.exit(1)
    
    if not os.path.exists(sapiens_model_path):
        print(f"ERROR: Sapiens model not found at: {sapiens_model_path}")
        print(f"Please place {SAPIENS_MODEL_NAME} in {WEIGHTS_DIR}/")
        sys.exit(1)
    
    # Create configuration
    config = PipelineConfig(
        weights_dir=WEIGHTS_DIR,
        sam3_model_path=sam3_model_path,
        sapiens_model_path=sapiens_model_path,
        sam3_conf=SAM3_CONFIDENCE,
        sam3_imgsz=SAM3_IMAGE_SIZE,
        patient_prompts=PATIENT_PROMPTS,
        device=device,
        output_dir=OUTPUT_DIR,
        save_video=SAVE_VIDEO,
        save_npy=SAVE_NPY,
        keypoint_conf_threshold=KEYPOINT_CONF_THRESHOLD,
        skeleton_thickness=SKELETON_THICKNESS,
        keypoint_radius=KEYPOINT_RADIUS,
    )
    
    # Print configuration
    print("\n" + "=" * 60)
    print("Seizure Detection Pipeline")
    print("=" * 60)
    print(f"Input:          {VIDEO_INPUT}")
    print(f"Output Dir:     {OUTPUT_DIR}")
    print(f"Weights Dir:    {WEIGHTS_DIR}")
    print(f"Device:         {device}")
    print(f"Prompts:        {PATIENT_PROMPTS}")
    print(f"SAM3 Model:     {SAM3_MODEL_NAME}")
    print(f"Sapiens Model:  {SAPIENS_MODEL_NAME}")
    print(f"Save Video:     {SAVE_VIDEO}")
    print(f"Save NPY:       {SAVE_NPY}")
    print("=" * 60 + "\n")
    
    # Initialize pipeline
    pipeline = SeizureDetectionPipeline(config)
    
    # Check if input is file or directory
    video_input = Path(VIDEO_INPUT)
    
    if video_input.is_file():
        # Process single video
        results = [pipeline.process_video(str(video_input))]
    elif video_input.is_dir():
        # Process all videos in directory
        results = pipeline.process_directory(str(video_input))
    else:
        print(f"ERROR: Input not found: {VIDEO_INPUT}")
        sys.exit(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    
    for result in results:
        if 'error' in result:
            print(f"\n❌ FAILED: {result['video_path']}")
            print(f"   Error: {result['error']}")
        else:
            print(f"\n✓ SUCCESS: {Path(result['video_path']).name}")
            print(f"   Total frames:    {result['total_frames']}")
            print(f"   Detected frames: {result['frames_with_detection']}")
            detection_rate = result['frames_with_detection'] / result['total_frames'] * 100
            print(f"   Detection rate:  {detection_rate:.1f}%")
            if result['keypoints_npy']:
                print(f"   Keypoints NPY:   {result['keypoints_npy']}")
            if result['output_video']:
                print(f"   Visualization:   {result['output_video']}")
    
    print("\n" + "=" * 60)
    print(f"Total videos processed: {len(results)}")
    successful = sum(1 for r in results if 'error' not in r)
    print(f"Successful: {successful}, Failed: {len(results) - successful}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
#%%



#===========================================================================
# ULTRALYTICS TEST
#===========================================================================

# from ultralytics.models.sam import SAM3VideoSemanticPredictor
# from IPython.display import display, clear_output
# from PIL import Image
# import numpy as np

# # Initialize semantic video predictor
# overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=640, model="./weights/sam3.pt", half=True, save=True)
# predictor = SAM3VideoSemanticPredictor(overrides=overrides)

# # Track concepts using text prompts
# results = predictor(source="/data_hdd/talha/miccai_26/seizure_detection_pipeline/videos/S0_0.mp4", text=["patient on bed", "person in striped clothes"], stream=True)

# # Process results
# for r in results:
#     im = r.plot()                 # BGR numpy array
#     im = im[..., ::-1]            # BGR -> RGB
#     clear_output(wait=True)
#     display(Image.fromarray(im))
