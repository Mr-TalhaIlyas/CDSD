#!/usr/bin/env python3
"""
VSVIG Seizure Dataset Preparation
=================================

Specialized script to prepare the VSVIG seizure dataset.
Handles long video sequences by clipping them into 5-second segments.

Key differences from IEEE dataset:
- Files are arbitrarily long (not pre-clipped to 5 sec)
- Labels include seizure mixed with other activities in same file
- Need to clip into 5-second segments during processing

Usage:
    python prepare_vsvig_dataset.py
"""

import sys
import os
import json
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_custom_dataset import (
    NEW_LABEL_MAP, LABEL_CONVERSION,
    load_keypoint_file, load_label_file,
    convert_coco_to_ntu, normalize_skeleton_to_ntu_scale,
    translate_to_origin, frame_normalize, remove_zero_frames
)
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional


# VSVIG dataset specific constants
CLIP_DURATION_SEC = 5.0  # Duration of each clip in seconds
DEFAULT_FPS = 30.0       # Default FPS if not specified


def find_vsvig_files(
    keypoints_dir: str,
    labels_dir: str
) -> List[Tuple[str, str, str]]:
    """
    Find matching keypoint and label files for VSVIG dataset.
    
    Args:
        keypoints_dir: Directory containing keypoint NPY files
        labels_dir: Directory containing label NPY files
        
    Returns:
        List of (sample_name, keypoint_path, label_path) tuples
    """
    matches = []
    labels_dir = Path(labels_dir)
    keypoints_dir = Path(keypoints_dir)
    
    # Get all label files
    label_files = {f.stem.replace('_labels', ''): f for f in labels_dir.glob('*_labels.npy')}
    
    # Find matching keypoint files
    for kp_file in keypoints_dir.glob('*_keypoints.npy'):
        sample_name = kp_file.stem.replace('_keypoints', '')
        
        if sample_name in label_files:
            matches.append((
                sample_name,
                str(kp_file),
                str(label_files[sample_name])
            ))
        else:
            logging.warning(f"No matching label file for: {kp_file.name}")
    
    return matches


def clip_sequence(
    keypoints: np.ndarray,
    labels: np.ndarray,
    fps: float,
    clip_duration: float = CLIP_DURATION_SEC,
    overlap: float = 0.0,
    min_clip_frames: int = 30
) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
    """
    Clip a long sequence into fixed-duration segments.
    
    Args:
        keypoints: Array of shape (T, J, C)
        labels: Array of shape (T,)
        fps: Frames per second
        clip_duration: Duration of each clip in seconds
        overlap: Overlap between clips (0.0 to 1.0)
        min_clip_frames: Minimum frames required for a valid clip
        
    Returns:
        List of (keypoints_clip, labels_clip, start_frame, end_frame) tuples
    """
    total_frames = len(keypoints)
    frames_per_clip = int(clip_duration * fps)
    
    # Calculate stride (with overlap)
    stride = int(frames_per_clip * (1 - overlap))
    if stride < 1:
        stride = 1
    
    clips = []
    start = 0
    
    while start < total_frames:
        end = min(start + frames_per_clip, total_frames)
        
        # Check if clip has enough frames
        if end - start >= min_clip_frames:
            kp_clip = keypoints[start:end]
            label_clip = labels[start:end]
            clips.append((kp_clip, label_clip, start, end))
        
        start += stride
        
        # If remaining frames are less than min_clip_frames, stop
        if total_frames - start < min_clip_frames:
            break
    
    return clips


def process_single_clip(
    keypoints: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True,
    translate: bool = True,
    per_frame_normalize: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Process a single clip: convert to NTU format, normalize, and convert labels.
    
    Args:
        keypoints: Array of shape (T, 17, 3) COCO keypoints
        labels: Array of shape (T,) original labels
        normalize: Whether to normalize to NTU scale
        translate: Whether to translate to origin
        per_frame_normalize: Whether to apply per-frame normalization
        
    Returns:
        processed_keypoints: (T, 25, 3) array with NTU keypoints or None
        processed_labels: (T,) array with converted labels or None
    """
    # Convert COCO 17 to NTU 25
    ntu_keypoints = convert_coco_to_ntu(keypoints, include_confidence=True)
    
    # Remove zero frames
    ntu_keypoints, labels = remove_zero_frames(ntu_keypoints, labels)
    
    if len(ntu_keypoints) == 0:
        return None, None
    
    # Normalize to NTU scale
    if normalize:
        ntu_keypoints = normalize_skeleton_to_ntu_scale(ntu_keypoints)
    
    # Translate to origin
    if translate:
        ntu_keypoints = translate_to_origin(ntu_keypoints)
    
    # Per-frame normalization
    if per_frame_normalize:
        ntu_keypoints = frame_normalize(ntu_keypoints)
    
    # Convert labels to new label map
    converted_labels = np.array([LABEL_CONVERSION.get(int(l), -1) for l in labels])
    
    return ntu_keypoints, converted_labels


def process_vsvig_dataset(
    keypoints_dir: str,
    labels_dir: str,
    output_file: str,
    clip_duration: float = CLIP_DURATION_SEC,
    overlap: float = 0.0,
    normalize: bool = True,
    translate: bool = True,
    per_frame_normalize: bool = False,
    max_frames: int = 300,
    min_clip_frames: int = 30
) -> Dict[str, Any]:
    """
    Process VSVIG dataset with clipping of long sequences.
    
    Args:
        keypoints_dir: Directory containing keypoint NPY files
        labels_dir: Directory containing label NPY files
        output_file: Output NPZ file path
        clip_duration: Duration of each clip in seconds
        overlap: Overlap between clips (0.0 to 1.0)
        normalize: Whether to normalize to NTU scale
        translate: Whether to translate to origin
        per_frame_normalize: Whether to apply per-frame normalization
        max_frames: Maximum number of frames per sample
        min_clip_frames: Minimum frames required for a valid clip
        
    Returns:
        Statistics dictionary
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Find files
    matches = find_vsvig_files(keypoints_dir, labels_dir)
    logging.info(f"Found {len(matches)} video files to process")
    
    if len(matches) == 0:
        raise ValueError("No matching keypoint-label pairs found!")
    
    # Process samples
    all_keypoints = []      # List of (T, 25, 3) arrays
    all_labels = []         # List of (T,) arrays
    sample_names = []       # List of sample names (video_name_clipXXX)
    frame_counts = []       # List of frame counts
    source_videos = []      # List of source video names
    clip_info = []          # List of (start_frame, end_frame) tuples
    
    label_distribution = {i: 0 for i in range(len(NEW_LABEL_MAP))}
    skipped_clips = []
    discarded_frames = 0
    total_frames = 0
    total_clips = 0
    
    for video_name, kp_path, label_path in tqdm(matches, desc="Processing videos"):
        try:
            # Load data
            kp_data = load_keypoint_file(kp_path)
            labels = load_label_file(label_path)
            
            keypoints = kp_data['keypoints']
            fps = kp_data.get('fps', DEFAULT_FPS)
            
            # Verify shapes match
            if len(keypoints) != len(labels):
                logging.warning(f"Shape mismatch for {video_name}: keypoints={len(keypoints)}, labels={len(labels)}")
                # Truncate to minimum length
                min_len = min(len(keypoints), len(labels))
                keypoints = keypoints[:min_len]
                labels = labels[:min_len]
            
            # Clip the sequence
            clips = clip_sequence(
                keypoints, labels, fps, 
                clip_duration=clip_duration,
                overlap=overlap,
                min_clip_frames=min_clip_frames
            )
            
            total_clips += len(clips)
            
            # Process each clip
            for clip_idx, (kp_clip, label_clip, start_frame, end_frame) in enumerate(clips):
                clip_name = f"{video_name}_clip{clip_idx:04d}"
                
                try:
                    # Process clip
                    processed_kps, processed_labels = process_single_clip(
                        kp_clip, label_clip, normalize, translate, per_frame_normalize
                    )
                    
                    if processed_kps is None:
                        skipped_clips.append((clip_name, "Empty after processing"))
                        continue
                    
                    # Filter out discarded labels
                    valid_mask = processed_labels >= 0
                    num_discarded = (~valid_mask).sum()
                    discarded_frames += num_discarded
                    total_frames += len(processed_labels)
                    
                    processed_kps = processed_kps[valid_mask]
                    processed_labels = processed_labels[valid_mask]
                    
                    if len(processed_kps) == 0:
                        skipped_clips.append((clip_name, "No valid labels"))
                        continue
                    
                    # Truncate if needed
                    T = len(processed_kps)
                    if T > max_frames:
                        indices = np.linspace(0, T - 1, max_frames, dtype=int)
                        processed_kps = processed_kps[indices]
                        processed_labels = processed_labels[indices]
                    
                    # Update label distribution
                    for l in processed_labels:
                        label_distribution[int(l)] += 1
                    
                    all_keypoints.append(processed_kps)
                    all_labels.append(processed_labels)
                    sample_names.append(clip_name)
                    frame_counts.append(len(processed_kps))
                    source_videos.append(video_name)
                    clip_info.append((start_frame, end_frame))
                    
                except Exception as e:
                    skipped_clips.append((clip_name, str(e)))
                    logging.warning(f"Error processing clip {clip_name}: {e}")
                    
        except Exception as e:
            logging.warning(f"Error loading video {video_name}: {e}")
    
    logging.info(f"Generated {len(all_keypoints)} clips from {len(matches)} videos")
    logging.info(f"Total clips attempted: {total_clips}")
    
    # Create aligned arrays
    num_samples = len(all_keypoints)
    if num_samples == 0:
        raise ValueError("No valid clips were generated!")
    
    max_T = max(frame_counts)
    
    # Create padded arrays
    aligned_keypoints = np.zeros((num_samples, max_T, 25, 3), dtype=np.float32)
    aligned_labels = np.full((num_samples, max_T), -1, dtype=np.int32)
    
    for i, (kps, labels) in enumerate(zip(all_keypoints, all_labels)):
        T = len(kps)
        aligned_keypoints[i, :T] = kps
        aligned_labels[i, :T] = labels
    
    # Convert to CTR-GCN format: (N, C, T, V, M)
    ctrgcn_keypoints = aligned_keypoints.transpose(0, 3, 1, 2)  # (N, 3, T, 25)
    ctrgcn_keypoints = ctrgcn_keypoints[:, :, :, :, np.newaxis]  # (N, 3, T, 25, 1)
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to NPZ
    np.savez(
        output_file,
        # Main data in CTR-GCN format
        data=ctrgcn_keypoints,              # (N, 3, T, 25, 1)
        labels=aligned_labels,               # (N, T)
        # Alternative format (easier to work with)
        keypoints=aligned_keypoints,         # (N, T, 25, 3)
        # Metadata
        sample_names=np.array(sample_names),
        frame_counts=np.array(frame_counts, dtype=np.int32),
        source_videos=np.array(source_videos),
        clip_info=np.array(clip_info, dtype=np.int32),
        label_map=json.dumps(NEW_LABEL_MAP),
        num_joints=25,
        num_persons=1,
        max_frames=max_T,
        num_classes=len(NEW_LABEL_MAP),
        clip_duration=clip_duration,
        overlap=overlap
    )
    
    # Statistics
    stats = {
        'total_videos': len(matches),
        'total_clips': num_samples,
        'max_frames': max_T,
        'total_processed_frames': sum(frame_counts),
        'total_discarded_frames': discarded_frames,
        'label_distribution': label_distribution,
        'skipped_clips': len(skipped_clips),
        'output_file': str(output_file)
    }
    
    logging.info(f"\nProcessing complete!")
    logging.info(f"Source videos: {len(matches)}")
    logging.info(f"Total clips: {num_samples}")
    logging.info(f"Max frames per clip: {max_T}")
    logging.info(f"Total frames: {sum(frame_counts)}")
    logging.info(f"Discarded frames: {discarded_frames} ({100*discarded_frames/max(1,total_frames):.1f}%)")
    logging.info(f"Output shape: {ctrgcn_keypoints.shape}")
    logging.info(f"\nLabel distribution:")
    for label_name, label_id in NEW_LABEL_MAP.items():
        count = label_distribution[label_id]
        logging.info(f"  {label_name}: {count}")
    
    if skipped_clips:
        logging.info(f"\nSkipped {len(skipped_clips)} clips")
    
    # Save statistics to JSON
    stats_file = output_path.with_suffix('.json')
    with open(stats_file, 'w') as f:
        json.dump({
            'stats': {
                'total_videos': int(len(matches)),
                'total_clips': int(num_samples),
                'max_frames': int(max_T),
                'total_processed_frames': int(stats['total_processed_frames']),
                'discarded_frames': int(stats['total_discarded_frames']),
                'clip_duration_sec': clip_duration,
                'overlap': overlap
            },
            'label_distribution': {
                name: int(label_distribution[id]) for name, id in NEW_LABEL_MAP.items()
            },
            'label_map': NEW_LABEL_MAP,
            'skipped_clips_count': len(skipped_clips)
        }, f, indent=2)
    
    return stats


def prepare_vsvig_dataset(
    overlap: float = 0.0,
    clip_duration: float = CLIP_DURATION_SEC,
    min_clip_frames: int = 30,
    max_frames: int = 300
):
    """
    Prepare the VSVIG seizure detection dataset.

    Args:
        overlap: Overlap between windows (0.0-1.0). 0 = no overlap (original).
                 E.g., 0.5 = 50% overlap, 0.75 = 75% overlap.
        clip_duration: Window duration in seconds (default: 5.0)
        min_clip_frames: Minimum frames per window (default: 30)
        max_frames: Maximum frames per sample (default: 300)
    """
    # Define paths (relative to this script's location)
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent

    keypoints_dir = str(data_dir / 'vsvig' / 'output')
    labels_dir = str(data_dir / 'labels' / 'vsvig_labels')

    # Output filename includes overlap info when overlapping
    if overlap > 0:
        output_file = str(data_dir / 'processed' / f'vsvig_seizure_dataset_overlap{int(overlap*100)}.npz')
    else:
        output_file = str(data_dir / 'processed' / 'vsvig_seizure_dataset.npz')

    print("=" * 60)
    print("VSVIG Seizure Dataset Preparation")
    print("=" * 60)
    print(f"\nKeypoints directory: {keypoints_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Output file: {output_file}")
    print(f"\nClip settings:")
    print(f"  Clip duration: {clip_duration} seconds")
    print(f"  Overlap: {overlap*100:.0f}%")
    print(f"  Min clip frames: {min_clip_frames}")
    print(f"  Max frames per sample: {max_frames}")
    print(f"\nNew label mapping ({len(NEW_LABEL_MAP)} classes):")
    for name, idx in NEW_LABEL_MAP.items():
        print(f"  {idx}: {name}")
    print()

    # Process using VSVIG-specific function
    stats = process_vsvig_dataset(
        keypoints_dir=keypoints_dir,
        labels_dir=labels_dir,
        output_file=output_file,
        clip_duration=clip_duration,
        overlap=overlap,
        normalize=True,
        translate=True,
        per_frame_normalize=False,
        max_frames=max_frames,
        min_clip_frames=min_clip_frames
    )
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    
    # Verify output
    print("\nVerifying output file...")
    data = np.load(output_file, allow_pickle=True)
    print(f"\nOutput file contents:")
    for key in data.files:
        arr = data[key]
        if isinstance(arr, np.ndarray):
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"  {key}: {type(arr)}")
    
    return stats


def load_processed_dataset(filepath: str):
    """
    Load the processed dataset for training.
    
    Args:
        filepath: Path to the NPZ file
        
    Returns:
        Dictionary with data, labels, and metadata
    """
    data = np.load(filepath, allow_pickle=True)
    
    return {
        'data': data['data'],              # (N, 3, T, 25, 1) CTR-GCN format
        'keypoints': data['keypoints'],    # (N, T, 25, 3) easier format
        'labels': data['labels'],          # (N, T) per-frame labels
        'sample_names': data['sample_names'],
        'frame_counts': data['frame_counts'],
        'source_videos': data['source_videos'],
        'clip_info': data['clip_info'],
        'label_map': json.loads(str(data['label_map'])),
        'num_joints': int(data['num_joints']),
        'num_persons': int(data['num_persons']),
        'max_frames': int(data['max_frames']),
        'num_classes': int(data['num_classes']),
        'clip_duration': float(data['clip_duration']),
        'overlap': float(data['overlap'])
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare VSVIG seizure dataset')
    parser.add_argument('--overlap', type=float, default=0.0,
                        help='Overlap between windows (0.0-1.0). Default: 0 (no overlap)')
    parser.add_argument('--clip_duration', type=float, default=CLIP_DURATION_SEC,
                        help=f'Window duration in seconds. Default: {CLIP_DURATION_SEC}')
    parser.add_argument('--min_clip_frames', type=int, default=30,
                        help='Minimum frames per window. Default: 30')
    parser.add_argument('--max_frames', type=int, default=300,
                        help='Maximum frames per sample. Default: 300')
    args = parser.parse_args()

    prepare_vsvig_dataset(
        overlap=args.overlap,
        clip_duration=args.clip_duration,
        min_clip_frames=args.min_clip_frames,
        max_frames=args.max_frames
    )
