#!/usr/bin/env python3
"""
Custom Dataset Preparation Script for CTR-GCN
=============================================

This script converts custom COCO-17 keypoint data to NTU-RGB+D 25 keypoint format
and prepares it for training with CTR-GCN following the same processing pipeline.

Usage:
    python prepare_custom_dataset.py \
        --keypoints_dir ../ieee/Normal ../ieee/Seizure \
        --labels_dir ../labels/ieee \
        --label_map ../labels/ieee/label_map.json \
        --output_file ../processed/custom_dataset.npz
"""

import os
import os.path as osp
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import logging

# ============================================================================
# Label Configuration
# ============================================================================

# Original label mapping (from label_map.json)
ORIGINAL_LABELS = {
    "sleeping": 0,
    "resting": 1,
    "reading": 2,
    "using_phone": 3,
    "watching_tv": 4,
    "eating": 5,
    "talking": 6,
    "sitting_up": 7,
    "adjusting_position": 8,
    "medical_interaction": 9,
    "other_activity": 10,
    "unclear": 11,
    "seizure": 12
}

# New label mapping (9 classes as requested)
# Merging: medical_interaction -> talking
# Discarding: watching_tv, other_activity, unclear
NEW_LABEL_MAP = {
    "sleeping": 0,
    "resting_or_lying_down": 1,  # renamed from "resting"
    "reading": 2,
    "using_phone": 3,
    "eating": 4,
    "talking": 5,  # includes medical_interaction
    "sitting_up": 6,
    "adjusting_position": 7,
    "seizure": 8
}

# Mapping from original label IDs to new label IDs
# -1 means discard
LABEL_CONVERSION = {
    0: 0,   # sleeping -> sleeping
    1: 1,   # resting -> resting_or_lying_down
    2: 2,   # reading -> reading
    3: 3,   # using_phone -> using_phone
    4: -1,  # watching_tv -> discard
    5: 4,   # eating -> eating
    6: 5,   # talking -> talking
    7: 6,   # sitting_up -> sitting_up
    8: 7,   # adjusting_position -> adjusting_position
    9: 5,   # medical_interaction -> talking (merged)
    10: -1, # other_activity -> discard
    11: -1, # unclear -> discard
    12: 8   # seizure -> seizure
}

# NTU RGB+D joint names (25 joints)
NTU_JOINT_NAMES = [
    'base_spine',       # 0
    'middle_spine',     # 1
    'neck',             # 2
    'head',             # 3
    'left_shoulder',    # 4
    'left_elbow',       # 5
    'left_wrist',       # 6
    'left_hand',        # 7
    'right_shoulder',   # 8
    'right_elbow',      # 9
    'right_wrist',      # 10
    'right_hand',       # 11
    'left_hip',         # 12
    'left_knee',        # 13
    'left_ankle',       # 14
    'left_foot',        # 15
    'right_hip',        # 16
    'right_knee',       # 17
    'right_ankle',      # 18
    'right_foot',       # 19
    'spine_shoulder',   # 20
    'left_hand_tip',    # 21
    'left_thumb',       # 22
    'right_hand_tip',   # 23
    'right_thumb'       # 24
]

# COCO keypoint names (17 joints)
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# NTU RGB+D skeleton connections
NTU_SKELETON_CONNECTIONS = [
    (0, 1), (1, 20), (20, 2), (2, 3),           # Spine
    (20, 4), (4, 5), (5, 6), (6, 7),            # Left arm
    (7, 21), (7, 22),                           # Left hand
    (20, 8), (8, 9), (9, 10), (10, 11),         # Right arm
    (11, 23), (11, 24),                         # Right hand
    (0, 12), (12, 13), (13, 14), (14, 15),      # Left leg
    (0, 16), (16, 17), (17, 18), (18, 19)       # Right leg
]


# ============================================================================
# Keypoint Conversion Functions
# ============================================================================

def convert_coco_to_ntu(coco_keypoints: np.ndarray, include_confidence: bool = True) -> np.ndarray:
    """
    Convert 17 COCO keypoints to 25 NTU-RGB+D keypoints.
    
    Args:
        coco_keypoints: Array of shape (N, 17, 3) or (17, 3) with [x, y, confidence]
                       or (N, 17, 2) or (17, 2) with [x, y]
        include_confidence: Whether to include confidence in output
    
    Returns:
        Array of shape (N, 25, 3) or (25, 3) with NTU-RGB+D keypoints
        If include_confidence is False, returns (N, 25, 2) or (25, 2)
    """
    # Handle both single frame and batch inputs
    single_frame = len(coco_keypoints.shape) == 2
    if single_frame:
        coco_keypoints = coco_keypoints[np.newaxis, ...]
    
    batch_size = coco_keypoints.shape[0]
    num_joints = coco_keypoints.shape[1]
    has_conf = coco_keypoints.shape[2] == 3
    
    assert num_joints == 17, f"Expected 17 COCO keypoints, got {num_joints}"
    
    # Ensure we have confidence values
    if not has_conf:
        conf = np.ones((batch_size, 17, 1), dtype=coco_keypoints.dtype)
        coco_keypoints = np.concatenate([coco_keypoints, conf], axis=2)
    
    # Initialize NTU keypoints array
    out_channels = 3 if include_confidence else 2
    ntu_keypoints = np.zeros((batch_size, 25, out_channels), dtype=np.float32)
    
    # COCO keypoint indices
    COCO = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
    }
    
    for i in range(batch_size):
        coco = coco_keypoints[i]
        ntu = np.zeros((25, 3), dtype=np.float32)
        
        # Direct mappings from COCO to NTU
        ntu[3] = coco[COCO['nose']]              # head <- nose
        ntu[4] = coco[COCO['left_shoulder']]     # left_shoulder
        ntu[5] = coco[COCO['left_elbow']]        # left_elbow  
        ntu[6] = coco[COCO['left_wrist']]        # left_wrist
        ntu[8] = coco[COCO['right_shoulder']]    # right_shoulder
        ntu[9] = coco[COCO['right_elbow']]       # right_elbow
        ntu[10] = coco[COCO['right_wrist']]      # right_wrist
        ntu[12] = coco[COCO['left_hip']]         # left_hip
        ntu[13] = coco[COCO['left_knee']]        # left_knee
        ntu[14] = coco[COCO['left_ankle']]       # left_ankle
        ntu[16] = coco[COCO['right_hip']]        # right_hip
        ntu[17] = coco[COCO['right_knee']]       # right_knee
        ntu[18] = coco[COCO['right_ankle']]      # right_ankle
        
        # Derived keypoints (interpolation and extrapolation)
        left_hip = coco[COCO['left_hip']]
        right_hip = coco[COCO['right_hip']]
        left_shoulder = coco[COCO['left_shoulder']]
        right_shoulder = coco[COCO['right_shoulder']]
        nose = coco[COCO['nose']]
        
        # Base of spine (0): midpoint between hips
        if left_hip[2] > 0 and right_hip[2] > 0:
            ntu[0, :2] = (left_hip[:2] + right_hip[:2]) / 2
            ntu[0, 2] = min(left_hip[2], right_hip[2])
        elif left_hip[2] > 0:
            ntu[0] = left_hip
        elif right_hip[2] > 0:
            ntu[0] = right_hip
        
        # Spine at shoulder level (20): midpoint between shoulders  
        if left_shoulder[2] > 0 and right_shoulder[2] > 0:
            ntu[20, :2] = (left_shoulder[:2] + right_shoulder[:2]) / 2
            ntu[20, 2] = min(left_shoulder[2], right_shoulder[2])
        elif left_shoulder[2] > 0:
            ntu[20] = left_shoulder
        elif right_shoulder[2] > 0:
            ntu[20] = right_shoulder
        
        # Middle of spine (1): interpolate between base spine and shoulder spine
        if ntu[0, 2] > 0 and ntu[20, 2] > 0:
            ntu[1, :2] = (ntu[0, :2] + ntu[20, :2]) / 2
            ntu[1, 2] = min(ntu[0, 2], ntu[20, 2])
        
        # Neck (2): interpolate between shoulder spine and head
        if ntu[20, 2] > 0 and nose[2] > 0:
            ntu[2, :2] = ntu[20, :2] * 0.3 + nose[:2] * 0.7  # closer to nose
            ntu[2, 2] = min(ntu[20, 2], nose[2])
        elif ntu[20, 2] > 0:
            ntu[2] = ntu[20]
        
        # Hands (same as wrists initially)
        ntu[7] = ntu[6].copy()    # left_hand <- left_wrist
        ntu[11] = ntu[10].copy()  # right_hand <- right_wrist
        
        # Feet (same as ankles initially)
        ntu[15] = ntu[14].copy()  # left_foot <- left_ankle
        ntu[19] = ntu[18].copy()  # right_foot <- right_ankle
        
        # Hand tips and thumbs (extrapolated from wrist-elbow direction)
        left_wrist = coco[COCO['left_wrist']]
        left_elbow = coco[COCO['left_elbow']]
        right_wrist = coco[COCO['right_wrist']]
        right_elbow = coco[COCO['right_elbow']]
        
        # Left hand tip and thumb
        if left_wrist[2] > 0 and left_elbow[2] > 0:
            direction = left_wrist[:2] - left_elbow[:2]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
                # Hand tip (extend further)
                ntu[21, :2] = left_wrist[:2] + direction * 15
                ntu[21, 2] = left_wrist[2] * 0.8
                # Thumb (extend less, slightly perpendicular)
                perp = np.array([-direction[1], direction[0]])
                ntu[22, :2] = left_wrist[:2] + direction * 8 + perp * 5
                ntu[22, 2] = left_wrist[2] * 0.7
        
        # Right hand tip and thumb  
        if right_wrist[2] > 0 and right_elbow[2] > 0:
            direction = right_wrist[:2] - right_elbow[:2]
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
                # Hand tip
                ntu[23, :2] = right_wrist[:2] + direction * 15
                ntu[23, 2] = right_wrist[2] * 0.8
                # Thumb
                perp = np.array([direction[1], -direction[0]])
                ntu[24, :2] = right_wrist[:2] + direction * 8 + perp * 5
                ntu[24, 2] = right_wrist[2] * 0.7
        
        if include_confidence:
            ntu_keypoints[i] = ntu
        else:
            ntu_keypoints[i] = ntu[:, :2]
    
    # Return single frame if input was single frame
    if single_frame:
        return ntu_keypoints[0]
    return ntu_keypoints


# ============================================================================
# Normalization Functions (following CTR-GCN processing)
# ============================================================================

def normalize_skeleton_to_ntu_scale(keypoints: np.ndarray) -> np.ndarray:
    """
    Normalize skeleton to match NTU RGB+D scale.
    
    NTU RGB+D uses 3D coordinates in meters where:
    - The spine length (base to neck) is approximately 0.5-0.6 meters
    - We normalize using the spine length as reference
    
    Args:
        keypoints: Array of shape (T, 25, C) where C is 2 or 3
        
    Returns:
        Normalized keypoints
    """
    T, J, C = keypoints.shape
    normalized = keypoints.copy()
    
    # Use only x, y for 2D data
    coords = keypoints[:, :, :2]
    
    # Calculate spine length for each frame (base spine to spine shoulder)
    # Joint 0: base_spine, Joint 20: spine_shoulder
    spine_vectors = coords[:, 20, :] - coords[:, 0, :]  # (T, 2)
    spine_lengths = np.linalg.norm(spine_vectors, axis=1)  # (T,)
    
    # Use median spine length to avoid outliers
    valid_lengths = spine_lengths[spine_lengths > 1e-6]
    if len(valid_lengths) > 0:
        ref_length = np.median(valid_lengths)
    else:
        ref_length = 1.0
    
    # Target spine length in NTU scale (approximately 0.5 meters)
    target_spine_length = 0.5
    
    # Scale factor
    if ref_length > 1e-6:
        scale = target_spine_length / ref_length
    else:
        scale = 1.0
    
    # Apply scaling
    normalized[:, :, :2] = coords * scale
    
    return normalized


def translate_to_origin(keypoints: np.ndarray) -> np.ndarray:
    """
    Translate skeleton so that the base of spine (joint 0) of first valid frame
    is at origin. This follows NTU RGB+D preprocessing.
    
    Args:
        keypoints: Array of shape (T, 25, C)
        
    Returns:
        Translated keypoints
    """
    T, J, C = keypoints.shape
    translated = keypoints.copy()
    
    # Find first frame with valid base spine
    origin = None
    for t in range(T):
        if C >= 3:
            if keypoints[t, 0, 2] > 0:  # confidence > 0
                origin = keypoints[t, 0, :2].copy()
                break
        else:
            if np.any(keypoints[t, 0, :2] != 0):
                origin = keypoints[t, 0, :2].copy()
                break
    
    if origin is None:
        origin = np.zeros(2)
    
    # Translate all frames
    translated[:, :, 0] -= origin[0]
    translated[:, :, 1] -= origin[1]
    
    return translated


def frame_normalize(keypoints: np.ndarray) -> np.ndarray:
    """
    Per-frame normalization: normalize by spine length at each frame.
    This is similar to CTR-GCN's frame_translation function.
    
    Args:
        keypoints: Array of shape (T, 25, C)
        
    Returns:
        Normalized keypoints
    """
    T, J, C = keypoints.shape
    normalized = keypoints.copy()
    
    for t in range(T):
        # Get spine length for this frame
        base_spine = keypoints[t, 0, :2]
        spine_shoulder = keypoints[t, 20, :2]
        spine_length = np.linalg.norm(spine_shoulder - base_spine)
        
        if spine_length > 1e-6:
            # Normalize using middle spine as origin
            middle_spine = keypoints[t, 1, :2]
            normalized[t, :, :2] = (keypoints[t, :, :2] - middle_spine) / spine_length + middle_spine
    
    return normalized


def remove_zero_frames(keypoints: np.ndarray, labels: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Remove frames where all joints are zero (missing data).
    
    Args:
        keypoints: Array of shape (T, 25, C)
        labels: Optional array of shape (T,)
        
    Returns:
        Filtered keypoints and labels
    """
    # Sum across joints and coordinates (excluding confidence)
    coords_sum = np.abs(keypoints[:, :, :2]).sum(axis=(1, 2))
    valid_mask = coords_sum > 1e-6
    
    filtered_kps = keypoints[valid_mask]
    filtered_labels = labels[valid_mask] if labels is not None else None
    
    return filtered_kps, filtered_labels


# ============================================================================
# Data Loading and Processing
# ============================================================================

def load_keypoint_file(filepath: str) -> Dict[str, Any]:
    """Load keypoint data from NPY file."""
    data = np.load(filepath, allow_pickle=True).item()
    return data


def load_label_file(filepath: str) -> np.ndarray:
    """Load per-frame labels from NPY file."""
    labels = np.load(filepath)
    return labels


def process_single_sample(
    keypoint_data: Dict[str, Any],
    labels: np.ndarray,
    normalize: bool = True,
    translate: bool = True,
    per_frame_normalize: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Process a single sample: convert to NTU format, normalize, and convert labels.
    
    Args:
        keypoint_data: Dictionary containing keypoints and metadata
        labels: Per-frame labels array
        normalize: Whether to normalize to NTU scale
        translate: Whether to translate to origin
        per_frame_normalize: Whether to apply per-frame normalization
        
    Returns:
        processed_keypoints: (T, 25, 3) array with NTU keypoints
        processed_labels: (T,) array with converted labels
        info: Dictionary with metadata
    """
    # Get keypoints (T, 17, 3) format
    keypoints = keypoint_data['keypoints']
    
    # Handle different input formats
    if len(keypoints.shape) == 3:
        T, J, C = keypoints.shape
    else:
        raise ValueError(f"Unexpected keypoints shape: {keypoints.shape}")
    
    # Convert COCO 17 to NTU 25
    ntu_keypoints = convert_coco_to_ntu(keypoints, include_confidence=True)
    
    # Remove zero frames
    ntu_keypoints, labels = remove_zero_frames(ntu_keypoints, labels)
    
    if len(ntu_keypoints) == 0:
        return None, None, None
    
    # Normalize to NTU scale
    if normalize:
        ntu_keypoints = normalize_skeleton_to_ntu_scale(ntu_keypoints)
    
    # Translate to origin
    if translate:
        ntu_keypoints = translate_to_origin(ntu_keypoints)
    
    # Per-frame normalization
    if per_frame_normalize:
        ntu_keypoints = frame_normalize(ntu_keypoints)
    
    # Convert labels
    converted_labels = np.array([LABEL_CONVERSION.get(int(l), -1) for l in labels])
    
    # Get metadata
    info = {
        'fps': keypoint_data.get('fps', 30.0),
        'frame_size': keypoint_data.get('frame_size', (1080, 1920)),
        'total_frames': len(ntu_keypoints),
        'original_frames': keypoint_data.get('total_frames', len(keypoints)),
    }
    
    return ntu_keypoints, converted_labels, info


def find_matching_files(keypoints_dirs: List[str], labels_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find matching keypoint and label files.
    
    Args:
        keypoints_dirs: List of directories containing keypoint NPY files
        labels_dir: Directory containing label NPY files
        
    Returns:
        List of (sample_name, keypoint_path, label_path) tuples
    """
    matches = []
    labels_dir = Path(labels_dir)
    
    # Get all label files
    label_files = {f.stem.replace('_labels', ''): f for f in labels_dir.glob('*_labels.npy')}
    
    for kp_dir in keypoints_dirs:
        kp_dir = Path(kp_dir)
        for kp_file in kp_dir.glob('*_keypoints.npy'):
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


def process_dataset(
    keypoints_dirs: List[str],
    labels_dir: str,
    output_file: str,
    normalize: bool = True,
    translate: bool = True,
    per_frame_normalize: bool = False,
    max_frames: int = 300
) -> Dict[str, Any]:
    """
    Process entire dataset and save to NPZ file.
    
    Args:
        keypoints_dirs: List of directories containing keypoint NPY files
        labels_dir: Directory containing label NPY files
        output_file: Output NPZ file path
        normalize: Whether to normalize to NTU scale
        translate: Whether to translate to origin
        per_frame_normalize: Whether to apply per-frame normalization
        max_frames: Maximum number of frames per sample
        
    Returns:
        Statistics dictionary
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Find matching files
    matches = find_matching_files(keypoints_dirs, labels_dir)
    logging.info(f"Found {len(matches)} matching sample pairs")
    
    if len(matches) == 0:
        raise ValueError("No matching keypoint-label pairs found!")
    
    # Process samples
    all_keypoints = []      # List of (T, 25, 3) arrays
    all_labels = []         # List of (T,) arrays
    sample_names = []       # List of sample names
    frame_counts = []       # List of frame counts
    label_distribution = {i: 0 for i in range(len(NEW_LABEL_MAP))}
    skipped_samples = []
    discarded_frames = 0
    total_frames = 0
    
    for sample_name, kp_path, label_path in tqdm(matches, desc="Processing samples"):
        try:
            # Load data
            kp_data = load_keypoint_file(kp_path)
            labels = load_label_file(label_path)
            
            # Process
            processed_kps, processed_labels, info = process_single_sample(
                kp_data, labels, normalize, translate, per_frame_normalize
            )
            
            if processed_kps is None:
                skipped_samples.append((sample_name, "Empty after processing"))
                continue
            
            # Filter out discarded labels
            valid_mask = processed_labels >= 0
            num_discarded = (~valid_mask).sum()
            discarded_frames += num_discarded
            total_frames += len(processed_labels)
            
            processed_kps = processed_kps[valid_mask]
            processed_labels = processed_labels[valid_mask]
            
            if len(processed_kps) == 0:
                skipped_samples.append((sample_name, "No valid labels"))
                continue
            
            # Pad or truncate to max_frames
            T = len(processed_kps)
            if T > max_frames:
                # Uniform sampling
                indices = np.linspace(0, T - 1, max_frames, dtype=int)
                processed_kps = processed_kps[indices]
                processed_labels = processed_labels[indices]
            
            # Update label distribution
            for l in processed_labels:
                label_distribution[int(l)] += 1
            
            all_keypoints.append(processed_kps)
            all_labels.append(processed_labels)
            sample_names.append(sample_name)
            frame_counts.append(len(processed_kps))
            
        except Exception as e:
            skipped_samples.append((sample_name, str(e)))
            logging.warning(f"Error processing {sample_name}: {e}")
    
    # Create aligned arrays
    num_samples = len(all_keypoints)
    max_T = max(frame_counts) if frame_counts else max_frames
    
    # Shape: (N, T, 25, 3) - following NTU format
    # But for CTR-GCN, we need (N, C, T, V, M) format
    # C=3 (x,y,conf), T=max_frames, V=25, M=1 (single person)
    
    # Create padded arrays
    aligned_keypoints = np.zeros((num_samples, max_T, 25, 3), dtype=np.float32)
    aligned_labels = np.full((num_samples, max_T), -1, dtype=np.int32)
    
    for i, (kps, labels) in enumerate(zip(all_keypoints, all_labels)):
        T = len(kps)
        aligned_keypoints[i, :T] = kps
        aligned_labels[i, :T] = labels
    
    # Convert to CTR-GCN format: (N, C, T, V, M)
    # Where C=3, T=max_T, V=25, M=1
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
        frame_counts=np.array(frame_counts),
        label_map=json.dumps(NEW_LABEL_MAP),
        num_joints=25,
        num_persons=1,
        max_frames=max_T,
        num_classes=len(NEW_LABEL_MAP)
    )
    
    # Statistics
    stats = {
        'total_samples': num_samples,
        'max_frames': max_T,
        'total_processed_frames': sum(frame_counts),
        'total_discarded_frames': discarded_frames,
        'label_distribution': label_distribution,
        'skipped_samples': skipped_samples,
        'output_file': str(output_file)
    }
    
    logging.info(f"\nProcessing complete!")
    logging.info(f"Total samples: {num_samples}")
    logging.info(f"Max frames: {max_T}")
    logging.info(f"Total frames: {sum(frame_counts)}")
    logging.info(f"Discarded frames: {discarded_frames} ({100*discarded_frames/max(1,total_frames):.1f}%)")
    logging.info(f"Output shape: {ctrgcn_keypoints.shape}")
    logging.info(f"\nLabel distribution:")
    for label_name, label_id in NEW_LABEL_MAP.items():
        count = label_distribution[label_id]
        logging.info(f"  {label_name}: {count}")
    
    if skipped_samples:
        logging.info(f"\nSkipped {len(skipped_samples)} samples")
    
    # Save statistics to JSON
    stats_file = output_path.with_suffix('.json')
    with open(stats_file, 'w') as f:
        json.dump({
            'stats': {
                'total_samples': int(stats['total_samples']),
                'max_frames': int(stats['max_frames']),
                'total_processed_frames': int(stats['total_processed_frames']),
                'discarded_frames': int(stats['total_discarded_frames'])
            },
            'label_distribution': {
                name: int(label_distribution[id]) for name, id in NEW_LABEL_MAP.items()
            },
            'label_map': NEW_LABEL_MAP,
            'skipped_samples': skipped_samples
        }, f, indent=2)
    
    return stats

#%%
# ============================================================================
# Main Entry Point
# ============================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description='Prepare custom dataset for CTR-GCN training',
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Example usage:
#     python prepare_custom_dataset.py \\
#         --keypoints_dir ../ieee/Normal ../ieee/Seizure \\
#         --labels_dir ../labels/ieee \\
#         --output_file ../processed/custom_dataset.npz
#         """
#     )
    
#     parser.add_argument(
#         '--keypoints_dir', '-k',
#         type=str,
#         nargs='+',
#         required=True,
#         help='Directories containing keypoint NPY files'
#     )
    
#     parser.add_argument(
#         '--labels_dir', '-l',
#         type=str,
#         required=True,
#         help='Directory containing label NPY files'
#     )
    
#     parser.add_argument(
#         '--output_file', '-o',
#         type=str,
#         default='./processed/custom_dataset.npz',
#         help='Output NPZ file path'
#     )
    
#     parser.add_argument(
#         '--max_frames', '-m',
#         type=int,
#         default=300,
#         help='Maximum frames per sample (default: 300)'
#     )
    
#     parser.add_argument(
#         '--no_normalize',
#         action='store_true',
#         help='Disable scale normalization'
#     )
    
#     parser.add_argument(
#         '--no_translate',
#         action='store_true',
#         help='Disable translation to origin'
#     )
    
#     parser.add_argument(
#         '--per_frame_normalize',
#         action='store_true',
#         help='Enable per-frame normalization'
#     )
    
#     args = parser.parse_args()
    
    # Process dataset
    # process_dataset(
    #     keypoints_dirs=args.keypoints_dir,
    #     labels_dir=args.labels_dir,
    #     output_file=args.output_file,
    #     normalize=not args.no_normalize,
    #     translate=not args.no_translate,
    #     per_frame_normalize=args.per_frame_normalize,
    #     max_frames=args.max_frames
    # )


# if __name__ == '__main__':
#     main()
