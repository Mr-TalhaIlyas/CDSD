#!/usr/bin/env python3
"""
IEEE Seizure Dataset Preparation
================================

Specialized script to prepare the IEEE seizure dataset.
Handles the case where Seizure folder contains clips without label files
(all frames in Seizure folder are seizure by default).

Usage:
    python prepare_ieee_dataset.py
"""

import sys
import os
import re
import json
import logging
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prepare_custom_dataset import (
    NEW_LABEL_MAP, LABEL_CONVERSION,
    load_keypoint_file, load_label_file,
    convert_coco_to_ntu, normalize_skeleton_to_ntu_scale,
    translate_to_origin, frame_normalize, remove_zero_frames,
    process_single_sample
)
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any, Optional


# IEEE dataset specific constants
SEIZURE_LABEL_NEW = 8          # Seizure label in new label map
SEIZURE_LABEL_ORIGINAL = 12   # Seizure label in original label map
DEFAULT_FPS = 30.0
DEFAULT_CLIP_DURATION = 5.0    # Default window duration in seconds
DEFAULT_MAX_GAP = 5            # Max clip index gap for concatenation


# ============================================================================
# Overlapping Window Extraction Helpers
# ============================================================================

def parse_ieee_filename(sample_name: str) -> Tuple[int, int, int]:
    """
    Parse IEEE filename into (subject, session, clip_index).

    Format: S{subject}_{session}_{clip_index}
    Examples: S0_0_48 -> (0, 0, 48), S15_3_102 -> (15, 3, 102)
    """
    match = re.match(r'S(\d+)_(\d+)_(\d+)', sample_name)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    raise ValueError(f"Cannot parse IEEE filename: {sample_name}")


def clip_sequence(
    keypoints: np.ndarray,
    labels: np.ndarray,
    fps: float,
    clip_duration: float = DEFAULT_CLIP_DURATION,
    overlap: float = 0.0,
    min_clip_frames: int = 30
) -> List[Tuple[np.ndarray, np.ndarray, int, int]]:
    """
    Extract fixed-duration windows from a sequence, optionally with overlap.

    Args:
        keypoints: (T, J, C) array
        labels: (T,) array
        fps: Frames per second
        clip_duration: Duration of each window in seconds
        overlap: Fraction of overlap between windows (0.0 to 1.0)
        min_clip_frames: Minimum frames for a valid window

    Returns:
        List of (keypoints_window, labels_window, start_frame, end_frame)
    """
    total_frames = len(keypoints)
    frames_per_clip = int(clip_duration * fps)
    stride = max(1, int(frames_per_clip * (1 - overlap)))

    clips = []
    start = 0
    while start < total_frames:
        end = min(start + frames_per_clip, total_frames)
        if end - start >= min_clip_frames:
            clips.append((keypoints[start:end], labels[start:end], start, end))
        start += stride
        if total_frames - start < min_clip_frames:
            break
    return clips


def group_and_concatenate_clips(
    clip_data_list: List[Tuple[str, np.ndarray, np.ndarray, float]],
    max_gap: int = DEFAULT_MAX_GAP
) -> List[Tuple[str, np.ndarray, np.ndarray, float]]:
    """
    Group clips by (subject, session), sort by clip_index, and concatenate
    consecutive clips where the index gap <= max_gap.

    Clips with larger gaps are split into separate segments to avoid
    stitching non-contiguous recordings.

    Args:
        clip_data_list: List of (sample_name, keypoints (T,J,C), labels (T,), fps)
        max_gap: Maximum allowed gap in clip indices for concatenation.
                 Set higher (e.g. 25) to merge sparser normal clips.

    Returns:
        List of (segment_name, concatenated_keypoints, concatenated_labels, fps)
    """
    # Group by (subject, session)
    groups = defaultdict(list)
    for name, kps, labels, fps in clip_data_list:
        try:
            subject, session, clip_idx = parse_ieee_filename(name)
            groups[(subject, session)].append((clip_idx, name, kps, labels, fps))
        except ValueError:
            # Cannot parse - treat as standalone segment
            groups[('standalone', name)].append((0, name, kps, labels, fps))

    segments = []
    for group_key, clips in sorted(groups.items()):
        # Sort by clip index within group
        clips.sort(key=lambda x: x[0])

        # Walk through and split on large gaps
        cur_kps = []
        cur_labels = []
        cur_indices = []
        cur_fps = clips[0][4]
        prev_idx = None

        for clip_idx, name, kps, labels, fps in clips:
            if prev_idx is not None and (clip_idx - prev_idx) > max_gap:
                # Gap too large - flush current segment
                if cur_kps:
                    seg_name = _make_segment_name(group_key, cur_indices)
                    segments.append((
                        seg_name,
                        np.concatenate(cur_kps, axis=0),
                        np.concatenate(cur_labels, axis=0),
                        cur_fps
                    ))
                cur_kps, cur_labels, cur_indices = [], [], []

            cur_kps.append(kps)
            cur_labels.append(labels)
            cur_indices.append(clip_idx)
            prev_idx = clip_idx

        # Flush last segment
        if cur_kps:
            seg_name = _make_segment_name(group_key, cur_indices)
            segments.append((
                seg_name,
                np.concatenate(cur_kps, axis=0),
                np.concatenate(cur_labels, axis=0),
                cur_fps
            ))

    return segments


def _make_segment_name(group_key, indices):
    """Create a readable segment name from group key and clip indices."""
    if isinstance(group_key[0], int):
        subject, session = group_key
        return f"S{subject}_{session}_seg{indices[0]}to{indices[-1]}"
    else:
        return f"{group_key[1]}_seg0"


def process_window(
    keypoints: np.ndarray,
    labels: np.ndarray,
    normalize: bool = True,
    translate: bool = True,
    per_frame_normalize: bool = False
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Process a single window: COCO -> NTU conversion, normalization, label mapping.

    Args:
        keypoints: (T, 17, 3) COCO keypoints
        labels: (T,) original labels (mapped via LABEL_CONVERSION)
        normalize: Scale to NTU range
        translate: Translate base spine to origin
        per_frame_normalize: Per-frame spine-length normalization

    Returns:
        (processed_keypoints (T, 25, 3), converted_labels (T,)) or (None, None)
    """
    ntu_keypoints = convert_coco_to_ntu(keypoints, include_confidence=True)
    ntu_keypoints, labels = remove_zero_frames(ntu_keypoints, labels)

    if len(ntu_keypoints) == 0:
        return None, None

    if normalize:
        ntu_keypoints = normalize_skeleton_to_ntu_scale(ntu_keypoints)
    if translate:
        ntu_keypoints = translate_to_origin(ntu_keypoints)
    if per_frame_normalize:
        ntu_keypoints = frame_normalize(ntu_keypoints)

    converted_labels = np.array([LABEL_CONVERSION.get(int(l), -1) for l in labels])
    return ntu_keypoints, converted_labels


# ============================================================================
# File Discovery
# ============================================================================

def find_ieee_files(
    normal_dir: str,
    seizure_dir: str,
    labels_dir: str
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str]]]:
    """
    Find files for IEEE dataset, handling both Normal (with labels) and Seizure (no labels) folders.
    
    Args:
        normal_dir: Directory containing Normal keypoint NPY files
        seizure_dir: Directory containing Seizure keypoint NPY files
        labels_dir: Directory containing label NPY files
        
    Returns:
        Tuple of:
        - List of (sample_name, keypoint_path, label_path) for Normal files
        - List of (sample_name, keypoint_path) for Seizure files (no label files needed)
    """
    normal_matches = []
    seizure_files = []
    
    labels_dir = Path(labels_dir)
    
    # Get all label files
    label_files = {f.stem.replace('_labels', ''): f for f in labels_dir.glob('*_labels.npy')}
    
    # Process Normal directory (has matching label files)
    normal_dir = Path(normal_dir)
    if normal_dir.exists():
        for kp_file in normal_dir.glob('*_keypoints.npy'):
            sample_name = kp_file.stem.replace('_keypoints', '')
            if sample_name in label_files:
                normal_matches.append((
                    sample_name,
                    str(kp_file),
                    str(label_files[sample_name])
                ))
            else:
                logging.warning(f"No matching label file for Normal: {kp_file.name}")
    
    # Process Seizure directory (all files are seizure, no label files needed)
    seizure_dir = Path(seizure_dir)
    if seizure_dir.exists():
        for kp_file in seizure_dir.glob('*_keypoints.npy'):
            sample_name = kp_file.stem.replace('_keypoints', '')
            seizure_files.append((sample_name, str(kp_file)))
    
    return normal_matches, seizure_files


def process_seizure_sample(
    keypoint_data: Dict[str, Any],
    normalize: bool = True,
    translate: bool = True,
    per_frame_normalize: bool = False
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Process a seizure sample where all frames are labeled as seizure.
    
    Args:
        keypoint_data: Dictionary containing keypoints and metadata
        normalize: Whether to normalize to NTU scale
        translate: Whether to translate to origin
        per_frame_normalize: Whether to apply per-frame normalization
        
    Returns:
        processed_keypoints: (T, 25, 3) array with NTU keypoints
        processed_labels: (T,) array with all seizure labels
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
    
    # Create seizure labels for all frames (using NEW label, not original)
    labels = np.full(len(ntu_keypoints), SEIZURE_LABEL_NEW, dtype=np.int32)
    
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
    
    # Labels are already in new format (SEIZURE_LABEL_NEW = 8)
    # No conversion needed
    
    # Get metadata
    info = {
        'fps': keypoint_data.get('fps', 30.0),
        'frame_size': keypoint_data.get('frame_size', (1080, 1920)),
        'total_frames': len(ntu_keypoints),
        'original_frames': keypoint_data.get('total_frames', len(keypoints)),
    }
    
    return ntu_keypoints, labels, info


def _process_ieee_with_overlap(
    normal_dir: str,
    seizure_dir: str,
    labels_dir: str,
    output_file: str,
    overlap: float,
    clip_duration: float,
    max_gap: int,
    min_clip_frames: int,
    normalize: bool,
    translate: bool,
    per_frame_normalize: bool,
    max_frames: int
) -> Dict[str, Any]:
    """
    Process IEEE dataset by concatenating clips into continuous segments,
    then extracting overlapping windows.

    Steps:
    1. Load all clips (Normal with labels, Seizure with auto-labels)
    2. Group by (subject, session) and concatenate consecutive clips
    3. Extract overlapping windows from each concatenated segment
    4. Process each window (COCO -> NTU, normalize, label conversion)
    """
    # Find files
    normal_matches, seizure_files = find_ieee_files(normal_dir, seizure_dir, labels_dir)

    logging.info(f"Found {len(normal_matches)} Normal clips, {len(seizure_files)} Seizure clips")
    logging.info(f"Overlap: {overlap*100:.0f}%, Window: {clip_duration}s, Max gap: {max_gap}")

    total_files = len(normal_matches) + len(seizure_files)
    if total_files == 0:
        raise ValueError("No keypoint files found!")

    # ---- Phase 1: Load all raw clips ----
    all_clip_data = []  # (sample_name, keypoints, labels, fps)
    load_errors = []

    for sample_name, kp_path, label_path in tqdm(normal_matches, desc="Loading Normal clips"):
        try:
            kp_data = load_keypoint_file(kp_path)
            labels = load_label_file(label_path)
            keypoints = kp_data['keypoints']
            fps = kp_data.get('fps', DEFAULT_FPS)

            # Align lengths
            if len(keypoints) != len(labels):
                min_len = min(len(keypoints), len(labels))
                keypoints = keypoints[:min_len]
                labels = labels[:min_len]

            all_clip_data.append((sample_name, keypoints, labels, fps))
        except Exception as e:
            load_errors.append((sample_name, str(e)))
            logging.warning(f"Error loading {sample_name}: {e}")

    for sample_name, kp_path in tqdm(seizure_files, desc="Loading Seizure clips"):
        try:
            kp_data = load_keypoint_file(kp_path)
            keypoints = kp_data['keypoints']
            fps = kp_data.get('fps', DEFAULT_FPS)
            # Use ORIGINAL seizure label so LABEL_CONVERSION maps 12 -> 8 uniformly
            labels = np.full(len(keypoints), SEIZURE_LABEL_ORIGINAL, dtype=np.int32)
            all_clip_data.append((sample_name, keypoints, labels, fps))
        except Exception as e:
            load_errors.append((sample_name, str(e)))
            logging.warning(f"Error loading {sample_name}: {e}")

    logging.info(f"Loaded {len(all_clip_data)} clips ({len(load_errors)} load errors)")

    # ---- Phase 2: Group and concatenate ----
    segments = group_and_concatenate_clips(all_clip_data, max_gap=max_gap)

    logging.info(f"Grouped into {len(segments)} continuous segments:")
    for seg_name, seg_kps, seg_labels, seg_fps in segments:
        duration = len(seg_kps) / seg_fps
        n_seizure_frames = np.sum(seg_labels == SEIZURE_LABEL_ORIGINAL)
        logging.info(f"  {seg_name}: {len(seg_kps)} frames ({duration:.1f}s), "
                     f"seizure: {n_seizure_frames}/{len(seg_kps)}")

    # ---- Phase 3: Extract overlapping windows and process ----
    all_keypoints = []
    all_labels = []
    sample_names = []
    frame_counts = []
    label_distribution = {i: 0 for i in range(len(NEW_LABEL_MAP))}
    skipped_windows = []
    discarded_frames = 0
    total_frames = 0
    total_windows = 0

    for seg_name, seg_kps, seg_labels, seg_fps in tqdm(segments, desc="Extracting windows"):
        windows = clip_sequence(
            seg_kps, seg_labels, seg_fps,
            clip_duration=clip_duration,
            overlap=overlap,
            min_clip_frames=min_clip_frames
        )
        total_windows += len(windows)

        for win_idx, (win_kps, win_labels, start_f, end_f) in enumerate(windows):
            win_name = f"{seg_name}_win{win_idx:04d}"

            try:
                processed_kps, processed_labels = process_window(
                    win_kps, win_labels, normalize, translate, per_frame_normalize
                )

                if processed_kps is None:
                    skipped_windows.append((win_name, "Empty after processing"))
                    continue

                # Filter discarded labels
                valid_mask = processed_labels >= 0
                discarded_frames += (~valid_mask).sum()
                total_frames += len(processed_labels)

                processed_kps = processed_kps[valid_mask]
                processed_labels = processed_labels[valid_mask]

                if len(processed_kps) == 0:
                    skipped_windows.append((win_name, "No valid labels"))
                    continue

                # Truncate if needed
                T = len(processed_kps)
                if T > max_frames:
                    indices = np.linspace(0, T - 1, max_frames, dtype=int)
                    processed_kps = processed_kps[indices]
                    processed_labels = processed_labels[indices]

                for l in processed_labels:
                    label_distribution[int(l)] += 1

                all_keypoints.append(processed_kps)
                all_labels.append(processed_labels)
                sample_names.append(win_name)
                frame_counts.append(len(processed_kps))

            except Exception as e:
                skipped_windows.append((win_name, str(e)))
                logging.warning(f"Error processing {win_name}: {e}")

    logging.info(f"Extracted {len(all_keypoints)} valid windows from {total_windows} total")

    # ---- Phase 4: Create output arrays ----
    num_samples = len(all_keypoints)
    if num_samples == 0:
        raise ValueError("No valid windows generated!")

    max_T = max(frame_counts)

    aligned_keypoints = np.zeros((num_samples, max_T, 25, 3), dtype=np.float32)
    aligned_labels = np.full((num_samples, max_T), -1, dtype=np.int32)

    for i, (kps, labels) in enumerate(zip(all_keypoints, all_labels)):
        T = len(kps)
        aligned_keypoints[i, :T] = kps
        aligned_labels[i, :T] = labels

    ctrgcn_keypoints = aligned_keypoints.transpose(0, 3, 1, 2)  # (N, 3, T, 25)
    ctrgcn_keypoints = ctrgcn_keypoints[:, :, :, :, np.newaxis]  # (N, 3, T, 25, 1)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_file,
        data=ctrgcn_keypoints,
        labels=aligned_labels,
        keypoints=aligned_keypoints,
        sample_names=np.array(sample_names),
        frame_counts=np.array(frame_counts, dtype=np.int32),
        label_map=json.dumps(NEW_LABEL_MAP),
        num_joints=25,
        num_persons=1,
        max_frames=max_T,
        num_classes=len(NEW_LABEL_MAP)
    )

    stats = {
        'total_samples': num_samples,
        'normal_clips': len(normal_matches),
        'seizure_clips': len(seizure_files),
        'segments': len(segments),
        'total_windows_attempted': total_windows,
        'max_frames': max_T,
        'total_processed_frames': sum(frame_counts),
        'total_discarded_frames': discarded_frames,
        'label_distribution': label_distribution,
        'skipped_samples': skipped_windows,
        'output_file': str(output_file)
    }

    logging.info(f"\nProcessing complete!")
    logging.info(f"Input: {len(normal_matches)} normal + {len(seizure_files)} seizure clips")
    logging.info(f"Segments after concatenation: {len(segments)}")
    logging.info(f"Output windows: {num_samples}")
    logging.info(f"Max frames per window: {max_T}")
    logging.info(f"Total frames: {sum(frame_counts)}")
    logging.info(f"Discarded frames: {discarded_frames} ({100*discarded_frames/max(1,total_frames):.1f}%)")
    logging.info(f"Output shape: {ctrgcn_keypoints.shape}")
    logging.info(f"\nLabel distribution:")
    for label_name, label_id in NEW_LABEL_MAP.items():
        count = label_distribution[label_id]
        logging.info(f"  {label_name}: {count}")

    if skipped_windows:
        logging.info(f"\nSkipped {len(skipped_windows)} windows")

    stats_file = output_path.with_suffix('.json')
    with open(stats_file, 'w') as f:
        json.dump({
            'stats': {
                'total_samples': int(num_samples),
                'normal_clips': int(len(normal_matches)),
                'seizure_clips': int(len(seizure_files)),
                'segments': int(len(segments)),
                'overlap': overlap,
                'clip_duration': clip_duration,
                'max_gap': max_gap,
                'max_frames': int(max_T),
                'total_processed_frames': int(sum(frame_counts)),
                'discarded_frames': int(discarded_frames)
            },
            'label_distribution': {
                name: int(label_distribution[id]) for name, id in NEW_LABEL_MAP.items()
            },
            'label_map': NEW_LABEL_MAP,
            'skipped_count': len(skipped_windows)
        }, f, indent=2)

    return stats


def process_ieee_dataset(
    normal_dir: str,
    seizure_dir: str,
    labels_dir: str,
    output_file: str,
    normalize: bool = True,
    translate: bool = True,
    per_frame_normalize: bool = False,
    max_frames: int = 300,
    overlap: float = 0.0,
    clip_duration: float = DEFAULT_CLIP_DURATION,
    max_gap: int = DEFAULT_MAX_GAP,
    min_clip_frames: int = 30
) -> Dict[str, Any]:
    """
    Process IEEE seizure dataset with specialized handling for Seizure folder.

    When overlap > 0, clips from the same (subject, session) are concatenated
    into continuous segments (respecting max_gap), and overlapping windows of
    clip_duration seconds are extracted. When overlap == 0, each clip is
    processed individually (original behavior).

    Args:
        normal_dir: Directory containing Normal keypoint files
        seizure_dir: Directory containing Seizure keypoint files
        labels_dir: Directory containing label NPY files
        output_file: Output NPZ file path
        normalize: Whether to normalize to NTU scale
        translate: Whether to translate to origin
        per_frame_normalize: Whether to apply per-frame normalization
        max_frames: Maximum number of frames per sample
        overlap: Overlap between windows (0.0-1.0). 0 = original per-clip mode.
                 E.g. 0.5 = 50% overlap -> ~2x samples from seizure segments.
        clip_duration: Window duration in seconds (used when overlap > 0)
        max_gap: Max clip index gap to consider clips consecutive for merging.
                 Default 5 works well for seizure clips (consecutive indices).
                 Increase to ~25 if you want to merge sparse normal clips.
        min_clip_frames: Minimum valid frames per window (used when overlap > 0)

    Returns:
        Statistics dictionary
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Dispatch to overlap processing if overlap > 0
    if overlap > 0:
        return _process_ieee_with_overlap(
            normal_dir=normal_dir, seizure_dir=seizure_dir,
            labels_dir=labels_dir, output_file=output_file,
            overlap=overlap, clip_duration=clip_duration,
            max_gap=max_gap, min_clip_frames=min_clip_frames,
            normalize=normalize, translate=translate,
            per_frame_normalize=per_frame_normalize, max_frames=max_frames
        )

    # ---- Original per-clip processing (overlap == 0) ----
    # Find files
    normal_matches, seizure_files = find_ieee_files(normal_dir, seizure_dir, labels_dir)

    logging.info(f"Found {len(normal_matches)} Normal samples with labels")
    logging.info(f"Found {len(seizure_files)} Seizure samples (all labeled as seizure)")
    
    total_files = len(normal_matches) + len(seizure_files)
    if total_files == 0:
        raise ValueError("No keypoint files found!")
    
    # Process samples
    all_keypoints = []      # List of (T, 25, 3) arrays
    all_labels = []         # List of (T,) arrays
    sample_names = []       # List of sample names
    frame_counts = []       # List of frame counts
    label_distribution = {i: 0 for i in range(len(NEW_LABEL_MAP))}
    skipped_samples = []
    discarded_frames = 0
    total_frames = 0
    
    # Process Normal samples (with label files)
    for sample_name, kp_path, label_path in tqdm(normal_matches, desc="Processing Normal samples"):
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
            
            # Truncate if needed (IEEE clips are 5 sec, unlikely to exceed max_frames)
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
            sample_names.append(sample_name)
            frame_counts.append(len(processed_kps))
            
        except Exception as e:
            skipped_samples.append((sample_name, str(e)))
            logging.warning(f"Error processing {sample_name}: {e}")
    
    # Process Seizure samples (all frames are seizure)
    for sample_name, kp_path in tqdm(seizure_files, desc="Processing Seizure samples"):
        try:
            # Load data
            kp_data = load_keypoint_file(kp_path)
            
            # Process (labels created automatically as seizure)
            processed_kps, processed_labels, info = process_seizure_sample(
                kp_data, normalize, translate, per_frame_normalize
            )
            
            if processed_kps is None:
                skipped_samples.append((sample_name, "Empty after processing"))
                continue
            
            total_frames += len(processed_labels)
            
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
            sample_names.append(sample_name)
            frame_counts.append(len(processed_kps))
            
        except Exception as e:
            skipped_samples.append((sample_name, str(e)))
            logging.warning(f"Error processing {sample_name}: {e}")
    
    # Create aligned arrays
    num_samples = len(all_keypoints)
    max_T = max(frame_counts) if frame_counts else max_frames
    
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
        'normal_samples': len(normal_matches),
        'seizure_samples': len(seizure_files),
        'max_frames': max_T,
        'total_processed_frames': sum(frame_counts),
        'total_discarded_frames': discarded_frames,
        'label_distribution': label_distribution,
        'skipped_samples': skipped_samples,
        'output_file': str(output_file)
    }
    
    logging.info(f"\nProcessing complete!")
    logging.info(f"Total samples: {num_samples}")
    logging.info(f"  - Normal samples: {len(normal_matches) - len([s for s in skipped_samples if s[0] in [m[0] for m in normal_matches]])}")
    logging.info(f"  - Seizure samples: {len(seizure_files) - len([s for s in skipped_samples if s[0] in [f[0] for f in seizure_files]])}")
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
                'normal_samples': int(stats['normal_samples']),
                'seizure_samples': int(stats['seizure_samples']),
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


def prepare_ieee_dataset(
    overlap: float = 0.0,
    clip_duration: float = DEFAULT_CLIP_DURATION,
    max_gap: int = DEFAULT_MAX_GAP,
    min_clip_frames: int = 30,
    max_frames: int = 300
):
    """
    Prepare the IEEE seizure detection dataset.

    Args:
        overlap: Overlap between windows (0.0-1.0). 0 = no overlap (original).
                 E.g., 0.5 = 50% overlap, 0.75 = 75% overlap.
        clip_duration: Window duration in seconds (default: 5.0)
        max_gap: Max clip index gap for concatenation (default: 5).
                 Seizure clips are typically consecutive (gap=1).
                 Normal clips are sparser; increase to ~25 to merge them.
        min_clip_frames: Minimum frames per window (default: 30)
        max_frames: Maximum frames per sample (default: 300)
    """
    # Define paths (relative to this script's location)
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent

    normal_dir = str(data_dir / 'ieee2_clean' / 'Normal')
    seizure_dir = str(data_dir / 'ieee2_clean' / 'Seizure')
    labels_dir = str(data_dir / 'labels' / 'ieee')

    # Output filename includes overlap info when overlapping
    if overlap > 0:
        output_file = str(data_dir / 'processed' / f'ieee_seizure_dataset_overlap{int(overlap*100)}.npz')
    else:
        output_file = str(data_dir / 'processed' / 'ieee_seizure_dataset.npz')

    print("=" * 60)
    print("IEEE Seizure Dataset Preparation")
    print("=" * 60)
    print(f"\nNormal directory: {normal_dir}")
    print(f"Seizure directory: {seizure_dir}")
    print(f"Labels directory: {labels_dir}")
    print(f"Output file: {output_file}")
    print(f"\nWindow settings:")
    print(f"  Clip duration: {clip_duration}s")
    print(f"  Overlap: {overlap*100:.0f}%")
    if overlap > 0:
        print(f"  Max gap for concatenation: {max_gap}")
        print(f"  Min clip frames: {min_clip_frames}")
    print(f"  Max frames per sample: {max_frames}")
    print(f"\nNew label mapping ({len(NEW_LABEL_MAP)} classes):")
    for name, idx in NEW_LABEL_MAP.items():
        print(f"  {idx}: {name}")
    print(f"\nNote: All files in Seizure folder will be labeled as 'seizure' (label {SEIZURE_LABEL_NEW})")
    print()

    # Process using IEEE-specific function
    stats = process_ieee_dataset(
        normal_dir=normal_dir,
        seizure_dir=seizure_dir,
        labels_dir=labels_dir,
        output_file=output_file,
        normalize=True,
        translate=True,
        per_frame_normalize=False,
        max_frames=max_frames,
        overlap=overlap,
        clip_duration=clip_duration,
        max_gap=max_gap,
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
        'label_map': json.loads(str(data['label_map'])),
        'num_joints': int(data['num_joints']),
        'num_persons': int(data['num_persons']),
        'max_frames': int(data['max_frames']),
        'num_classes': int(data['num_classes'])
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare IEEE seizure dataset')
    parser.add_argument('--overlap', type=float, default=0.0,
                        help='Overlap between windows (0.0-1.0). Default: 0 (no overlap)')
    parser.add_argument('--clip_duration', type=float, default=DEFAULT_CLIP_DURATION,
                        help=f'Window duration in seconds. Default: {DEFAULT_CLIP_DURATION}')
    parser.add_argument('--max_gap', type=int, default=DEFAULT_MAX_GAP,
                        help=f'Max clip index gap for concatenation. Default: {DEFAULT_MAX_GAP}')
    parser.add_argument('--min_clip_frames', type=int, default=30,
                        help='Minimum frames per window. Default: 30')
    parser.add_argument('--max_frames', type=int, default=300,
                        help='Maximum frames per sample. Default: 300')
    args = parser.parse_args()

    prepare_ieee_dataset(
        overlap=args.overlap,
        clip_duration=args.clip_duration,
        max_gap=args.max_gap,
        min_clip_frames=args.min_clip_frames,
        max_frames=args.max_frames
    )
