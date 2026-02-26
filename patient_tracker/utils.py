#!/usr/bin/env python3
"""
Utility functions for the Seizure Detection Pipeline.
Handles edge cases, video I/O, and data validation.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ============================================================================
# Video Utilities
# ============================================================================

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get comprehensive video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        'path': video_path,
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'duration_seconds': 0.0,
    }
    
    if info['fps'] > 0:
        info['duration_seconds'] = info['frame_count'] / info['fps']
    
    cap.release()
    return info


def is_infrared_frame(frame: np.ndarray, threshold: float = 0.15) -> bool:
    """
    Detect if a frame is infrared (grayscale/near-grayscale).
    
    Args:
        frame: BGR image
        threshold: Maximum color variance threshold
        
    Returns:
        True if frame appears to be infrared
    """
    if frame is None or frame.size == 0:
        return False
    
    if len(frame.shape) == 2:
        return True  # Already grayscale
    
    # Check color variance
    b, g, r = cv2.split(frame.astype(np.float32))
    
    # Calculate per-pixel color deviation
    mean_intensity = (b + g + r) / 3
    color_variance = np.mean(np.abs(b - mean_intensity) + 
                            np.abs(g - mean_intensity) + 
                            np.abs(r - mean_intensity))
    
    # Normalize by intensity
    mean_val = np.mean(mean_intensity)
    if mean_val > 0:
        normalized_variance = color_variance / mean_val
    else:
        normalized_variance = 0
    
    return normalized_variance < threshold


def enhance_infrared_frame(frame: np.ndarray) -> np.ndarray:
    """
    Enhance infrared frame for better detection.
    
    Args:
        frame: Input frame (BGR or grayscale)
        
    Returns:
        Enhanced BGR frame
    """
    if frame is None or frame.size == 0:
        return frame
    
    # Convert to grayscale if needed
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Convert back to BGR
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced_bgr


# ============================================================================
# Mask Utilities
# ============================================================================

def clean_mask(
    mask: np.ndarray,
    min_area: int = 1000,
    kernel_size: int = 5
) -> np.ndarray:
    """
    Clean up a binary mask by removing small regions and filling holes.
    
    Args:
        mask: Binary mask (0/1)
        min_area: Minimum area to keep
        kernel_size: Morphological kernel size
        
    Returns:
        Cleaned mask
    """
    if mask is None or mask.size == 0:
        return mask
    
    mask = mask.astype(np.uint8)
    
    # Morphological closing to fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Keep only large components
    cleaned = np.zeros_like(mask)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 1
    
    return cleaned


def get_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Get the largest connected component from a binary mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Mask with only the largest component
    """
    if mask is None or mask.size == 0:
        return mask
    
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        return mask
    
    # Find largest component (excluding background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = np.argmax(areas) + 1
    
    result = np.zeros_like(mask)
    result[labels == largest_idx] = 1
    
    return result


def mask_to_bbox(
    mask: np.ndarray,
    padding: int = 0,
    padding_ratio: float = 0.0
) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert binary mask to bounding box.
    
    Args:
        mask: Binary mask
        padding: Absolute padding in pixels
        padding_ratio: Padding as ratio of bbox size
        
    Returns:
        Bounding box (x1, y1, x2, y2) or None
    """
    if mask is None or mask.size == 0:
        return None
    
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Add padding
    h, w = mask.shape
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    
    pad_x = padding + int(bbox_w * padding_ratio)
    pad_y = padding + int(bbox_h * padding_ratio)
    
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)
    
    return (x1, y1, x2, y2)


def crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 20,
    black_background: bool = True
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Crop image to mask region with optional black background.
    
    Args:
        image: Input image
        mask: Binary mask
        padding: Padding around mask
        black_background: If True, set non-mask pixels to black
        
    Returns:
        Tuple of (cropped image, bbox) or (None, None)
    """
    if image is None or mask is None:
        return None, None
    
    bbox = mask_to_bbox(mask, padding=padding)
    if bbox is None:
        return None, None
    
    x1, y1, x2, y2 = bbox
    
    # Crop image and mask
    cropped_image = image[y1:y2, x1:x2].copy()
    cropped_mask = mask[y1:y2, x1:x2]
    
    if black_background:
        # Apply mask
        if len(cropped_image.shape) == 3:
            mask_3ch = np.stack([cropped_mask] * 3, axis=-1)
            cropped_image = cropped_image * mask_3ch
        else:
            cropped_image = cropped_image * cropped_mask
    
    return cropped_image, bbox


# ============================================================================
# Keypoint Utilities
# ============================================================================

def validate_keypoints(
    keypoints: np.ndarray,
    image_size: Tuple[int, int],
    confidence_threshold: float = 0.1
) -> np.ndarray:
    """
    Validate and clean keypoint predictions.
    
    Args:
        keypoints: Keypoints array (K, 3) with [x, y, conf]
        image_size: Image size (height, width)
        confidence_threshold: Minimum confidence
        
    Returns:
        Validated keypoints
    """
    if keypoints is None or keypoints.size == 0:
        return keypoints
    
    h, w = image_size
    validated = keypoints.copy()
    
    for i in range(len(validated)):
        x, y, conf = validated[i]
        
        # Check confidence
        if conf < confidence_threshold:
            validated[i] = [0, 0, 0]
            continue
        
        # Clamp to image bounds
        validated[i, 0] = np.clip(x, 0, w - 1)
        validated[i, 1] = np.clip(y, 0, h - 1)
    
    return validated


def transform_keypoints(
    keypoints: np.ndarray,
    bbox: Tuple[int, int, int, int],
    to_original: bool = True
) -> np.ndarray:
    """
    Transform keypoints between cropped and original coordinates.
    
    Args:
        keypoints: Keypoints array (K, 3)
        bbox: Crop bounding box (x1, y1, x2, y2)
        to_original: If True, transform crop->original; else original->crop
        
    Returns:
        Transformed keypoints
    """
    if keypoints is None or keypoints.size == 0:
        return keypoints
    
    transformed = keypoints.copy()
    x1, y1, x2, y2 = bbox
    
    if to_original:
        # Add offset
        transformed[:, 0] += x1
        transformed[:, 1] += y1
    else:
        # Remove offset
        transformed[:, 0] -= x1
        transformed[:, 1] -= y1
    
    return transformed


def interpolate_missing_keypoints(
    keypoints_sequence: np.ndarray,
    confidence_threshold: float = 0.1,
    max_gap: int = 30
) -> np.ndarray:
    """
    Interpolate missing keypoints in a temporal sequence.
    
    Args:
        keypoints_sequence: Array of shape (T, K, 3)
        confidence_threshold: Threshold for valid keypoints
        max_gap: Maximum frames to interpolate across
        
    Returns:
        Interpolated keypoints sequence
    """
    T, K, _ = keypoints_sequence.shape
    interpolated = keypoints_sequence.copy()
    
    for k in range(K):
        # Find valid frames for this keypoint
        valid = keypoints_sequence[:, k, 2] >= confidence_threshold
        valid_indices = np.where(valid)[0]
        
        if len(valid_indices) < 2:
            continue
        
        # Interpolate between valid frames
        for i in range(len(valid_indices) - 1):
            start_idx = valid_indices[i]
            end_idx = valid_indices[i + 1]
            gap = end_idx - start_idx
            
            if gap > 1 and gap <= max_gap:
                # Linear interpolation
                for t in range(start_idx + 1, end_idx):
                    alpha = (t - start_idx) / gap
                    interpolated[t, k, 0] = (1 - alpha) * keypoints_sequence[start_idx, k, 0] + \
                                            alpha * keypoints_sequence[end_idx, k, 0]
                    interpolated[t, k, 1] = (1 - alpha) * keypoints_sequence[start_idx, k, 1] + \
                                            alpha * keypoints_sequence[end_idx, k, 1]
                    interpolated[t, k, 2] = min(
                        keypoints_sequence[start_idx, k, 2],
                        keypoints_sequence[end_idx, k, 2]
                    ) * 0.8  # Reduce confidence for interpolated
    
    return interpolated


def smooth_keypoints_temporal(
    keypoints_sequence: np.ndarray,
    window_size: int = 5,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Apply temporal smoothing to keypoints sequence.
    
    Args:
        keypoints_sequence: Array of shape (T, K, 3)
        window_size: Smoothing window size
        sigma: Gaussian sigma
        
    Returns:
        Smoothed keypoints sequence
    """
    from scipy.ndimage import gaussian_filter1d
    
    T, K, _ = keypoints_sequence.shape
    smoothed = keypoints_sequence.copy()
    
    # Only smooth x and y coordinates, not confidence
    for k in range(K):
        # Get confidence mask
        conf = keypoints_sequence[:, k, 2]
        valid = conf > 0.1
        
        if np.sum(valid) < window_size:
            continue
        
        # Smooth valid portions
        for dim in range(2):  # x and y
            values = keypoints_sequence[:, k, dim].copy()
            values[~valid] = np.nan
            
            # Forward fill NaN
            for i in range(1, T):
                if np.isnan(values[i]) and not np.isnan(values[i-1]):
                    values[i] = values[i-1]
            
            # Backward fill NaN
            for i in range(T-2, -1, -1):
                if np.isnan(values[i]) and not np.isnan(values[i+1]):
                    values[i] = values[i+1]
            
            # Apply Gaussian filter
            if not np.any(np.isnan(values)):
                smoothed[:, k, dim] = gaussian_filter1d(values, sigma)
    
    return smoothed


# ============================================================================
# Output Utilities
# ============================================================================

def save_keypoints_npy(
    output_path: str,
    keypoints: np.ndarray,
    confidences: np.ndarray,
    metadata: Dict[str, Any]
) -> str:
    """
    Save keypoints to NPY file with metadata.
    
    Args:
        output_path: Output file path
        keypoints: Keypoints array (T, K, 3)
        confidences: Frame confidences (T,)
        metadata: Additional metadata
        
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'keypoints': keypoints,
        'confidences': confidences,
        **metadata
    }
    
    np.save(str(output_path), data, allow_pickle=True)
    return str(output_path)


def load_keypoints_npy(file_path: str) -> Dict[str, Any]:
    """
    Load keypoints from NPY file.
    
    Args:
        file_path: Path to NPY file
        
    Returns:
        Dictionary with keypoints and metadata
    """
    data = np.load(file_path, allow_pickle=True).item()
    return data


class VideoWriter:
    """Context manager for video writing with automatic cleanup."""
    
    def __init__(
        self,
        output_path: str,
        fps: float,
        frame_size: Tuple[int, int],
        codec: str = 'mp4v'
    ):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.writer = None
    
    def __enter__(self):
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            self.frame_size
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.release()
    
    def write(self, frame: np.ndarray):
        """Write a frame to the video."""
        if self.writer is not None:
            # Ensure correct size
            if frame.shape[:2][::-1] != self.frame_size:
                frame = cv2.resize(frame, self.frame_size)
            self.writer.write(frame)


# ============================================================================
# Edge Case Handlers
# ============================================================================

def handle_empty_detection(
    frame: np.ndarray,
    last_valid_mask: Optional[np.ndarray] = None,
    last_valid_bbox: Optional[Tuple[int, int, int, int]] = None,
    use_last_valid: bool = True
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """
    Handle frames where no patient is detected.
    
    Args:
        frame: Current frame
        last_valid_mask: Last valid detection mask
        last_valid_bbox: Last valid bounding box
        use_last_valid: Whether to use last valid detection
        
    Returns:
        Tuple of (mask, bbox) or (None, None)
    """
    if not use_last_valid or last_valid_mask is None:
        return None, None
    
    # Return last valid detection (with reduced confidence)
    logger.debug("Using last valid detection for empty frame")
    return last_valid_mask, last_valid_bbox


def handle_multiple_detections(
    masks: List[np.ndarray],
    bboxes: List[np.ndarray],
    confidences: List[float],
    strategy: str = 'largest'
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Handle frames with multiple patient detections.
    
    Args:
        masks: List of detection masks
        bboxes: List of bounding boxes
        confidences: List of confidence scores
        strategy: Selection strategy ('largest', 'highest_conf', 'center')
        
    Returns:
        Tuple of (selected_mask, selected_bbox, confidence)
    """
    if not masks:
        return None, None, 0.0
    
    if len(masks) == 1:
        return masks[0], bboxes[0], confidences[0]
    
    if strategy == 'largest':
        areas = [np.sum(m) for m in masks]
        idx = np.argmax(areas)
    elif strategy == 'highest_conf':
        idx = np.argmax(confidences)
    elif strategy == 'center':
        # Select detection closest to frame center
        centers = []
        for bbox in bboxes:
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            centers.append((cx, cy))
        
        frame_center = (masks[0].shape[1] / 2, masks[0].shape[0] / 2)
        distances = [
            np.sqrt((c[0] - frame_center[0])**2 + (c[1] - frame_center[1])**2)
            for c in centers
        ]
        idx = np.argmin(distances)
    else:
        idx = 0
    
    return masks[idx], bboxes[idx], confidences[idx]


def handle_occluded_keypoints(
    keypoints: np.ndarray,
    mask: np.ndarray,
    confidence_penalty: float = 0.5
) -> np.ndarray:
    """
    Reduce confidence for keypoints outside the mask region.
    
    Args:
        keypoints: Keypoints array (K, 3)
        mask: Patient mask
        confidence_penalty: Penalty multiplier for outside keypoints
        
    Returns:
        Updated keypoints
    """
    if keypoints is None or mask is None:
        return keypoints
    
    h, w = mask.shape
    updated = keypoints.copy()
    
    for i in range(len(updated)):
        x, y, conf = updated[i]
        
        if conf <= 0:
            continue
        
        # Check if keypoint is inside mask
        xi, yi = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
        
        if mask[yi, xi] == 0:
            updated[i, 2] *= confidence_penalty
    
    return updated
