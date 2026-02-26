"""
Video Processing Utilities for Seizure Labeling Pipeline
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from config import FPS, WINDOW_SIZE_SEC, WINDOW_SIZE_FRAMES


@dataclass
class VideoInfo:
    """Stores video metadata and seizure timing information"""
    filepath: str
    filename: str
    patient_id: str
    video_index: int
    seizure_number: Optional[str]  # e.g., "Sz1", "Sz2", None for free videos
    seizure_type: Optional[str]    # e.g., "P", "PG", None
    is_seizure_video: bool
    clinical_onset_sec: Optional[float]  # Seizure onset in seconds
    total_frames: int
    duration_sec: float
    normal_duration_sec: float     # Duration before seizure onset


def parse_time_to_seconds(time_str: str) -> float:
    """Convert time string (HH:MM:SS) to seconds"""
    if pd.isna(time_str) or time_str is None:
        return 0.0
    
    # Handle different time formats
    time_str = str(time_str).strip()
    
    # If it's already a timedelta or datetime
    if hasattr(time_str, 'total_seconds'):
        return time_str.total_seconds()
    
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    else:
        return float(time_str)


def parse_video_filename(filename: str) -> Dict:
    """
    Parse video filename to extract metadata.
    
    Examples:
        pat01_000_Sz1PG.mp4 -> {patient: Pat01, seizure: Sz1, type: PG}
        pat03_006_free.mp4 -> {patient: Pat03, seizure: None, type: None}
        pat04_009_no-Sz2P.mp4 -> {patient: Pat04, seizure: None, type: None}
    """
    # Remove extension
    name = Path(filename).stem
    
    # Pattern: patXX_YYY_ZZZ
    pattern = r'pat(\d+)_(\d+)_(.+)'
    match = re.match(pattern, name, re.IGNORECASE)
    
    if not match:
        return None
    
    patient_num = match.group(1)
    video_idx = int(match.group(2))
    suffix = match.group(3)
    
    result = {
        'patient_id': f'Pat{patient_num.zfill(2)}',
        'video_index': video_idx,
        'seizure_number': None,
        'seizure_type': None,
        'is_seizure_video': False
    }
    
    # Check if it's a seizure video (not "free" or "no-")
    if 'free' in suffix.lower() or suffix.lower().startswith('no-'):
        return result
    
    # Parse seizure info: Sz1PG, Sz2P, etc.
    sz_pattern = r'Sz(\d+)(P|PG)?'
    sz_match = re.match(sz_pattern, suffix, re.IGNORECASE)
    
    if sz_match:
        result['seizure_number'] = f'Sz{sz_match.group(1)}'
        result['seizure_type'] = sz_match.group(2) if sz_match.group(2) else ''
        result['is_seizure_video'] = True
    
    return result


def load_seizure_labels(excel_path: str) -> pd.DataFrame:
    """Load seizure onset times from Excel file"""
    df = pd.read_excel(excel_path)
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Convert Clinical Onset to seconds
    df['clinical_onset_sec'] = df['Clinical Onset'].apply(parse_time_to_seconds)
    
    # Standardize patient ID format
    df['PatID'] = df['PatID'].str.strip()
    
    return df


def get_video_info(video_path: str, labels_df: pd.DataFrame) -> VideoInfo:
    """Get complete video information including seizure timing"""
    filename = os.path.basename(video_path)
    parsed = parse_video_filename(filename)
    
    if parsed is None:
        raise ValueError(f"Could not parse filename: {filename}")
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / fps if fps > 0 else 0
    cap.release()
    
    clinical_onset_sec = None
    normal_duration_sec = duration_sec  # Default: entire video is normal
    
    # If it's a seizure video, find the clinical onset time
    if parsed['is_seizure_video']:
        # Match with labels dataframe
        mask = (
            (labels_df['PatID'].str.lower() == parsed['patient_id'].lower()) &
            (labels_df['#Seizure'] == parsed['seizure_number'])
        )
        
        matching_rows = labels_df[mask]
        
        if len(matching_rows) > 0:
            clinical_onset_sec = matching_rows.iloc[0]['clinical_onset_sec']
            normal_duration_sec = clinical_onset_sec
    
    return VideoInfo(
        filepath=video_path,
        filename=filename,
        patient_id=parsed['patient_id'],
        video_index=parsed['video_index'],
        seizure_number=parsed['seizure_number'],
        seizure_type=parsed['seizure_type'],
        is_seizure_video=parsed['is_seizure_video'],
        clinical_onset_sec=clinical_onset_sec,
        total_frames=total_frames,
        duration_sec=duration_sec,
        normal_duration_sec=normal_duration_sec
    )


def extract_frame_from_window(
    video_path: str,
    window_start_frame: int,
    window_size_frames: int = WINDOW_SIZE_FRAMES,
    sample_position: str = 'middle'
) -> np.ndarray:
    """
    Extract a single frame from a window for VLM analysis.
    
    Args:
        video_path: Path to video file
        window_start_frame: Starting frame of the window
        window_size_frames: Size of window in frames
        sample_position: 'start', 'middle', or 'end'
    
    Returns:
        Frame as numpy array (RGB)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Calculate sample frame position
    if sample_position == 'start':
        target_frame = window_start_frame
    elif sample_position == 'end':
        target_frame = window_start_frame + window_size_frames - 1
    else:  # middle
        target_frame = window_start_frame + window_size_frames // 2
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return None
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def generate_sliding_windows(
    total_frames: int,
    normal_end_frame: int,
    window_size_frames: int = WINDOW_SIZE_FRAMES,
    stride_frames: Optional[int] = None
) -> List[Tuple[int, int]]:
    """
    Generate sliding window positions for normal duration of video.
    
    Args:
        total_frames: Total frames in video
        normal_end_frame: Last frame of normal activity (before seizure)
        window_size_frames: Size of each window
        stride_frames: Stride between windows (default: window_size for non-overlapping)
    
    Returns:
        List of (start_frame, end_frame) tuples
    """
    if stride_frames is None:
        stride_frames = window_size_frames
    
    windows = []
    start_frame = 0
    
    while start_frame + window_size_frames <= normal_end_frame:
        end_frame = start_frame + window_size_frames
        windows.append((start_frame, end_frame))
        start_frame += stride_frames
    
    # Handle partial last window if there's remaining normal duration
    if start_frame < normal_end_frame and normal_end_frame - start_frame >= window_size_frames // 2:
        # Only include if at least half a window remains
        windows.append((start_frame, min(start_frame + window_size_frames, normal_end_frame)))
    
    return windows


def create_frame_level_labels(
    window_labels: List[Tuple[int, int, int]],  # (start, end, label_id)
    total_frames: int,
    seizure_start_frame: Optional[int] = None,
    seizure_label_id: int = 12
) -> np.ndarray:
    """
    Create frame-level label array from window-level labels.
    
    Args:
        window_labels: List of (start_frame, end_frame, label_id)
        total_frames: Total frames in video
        seizure_start_frame: Frame where seizure begins (None if no seizure)
        seizure_label_id: Label ID for seizure class
    
    Returns:
        Numpy array of shape (total_frames,) with label IDs
    """
    # Initialize with -1 (unlabeled) or use a default label
    labels = np.full(total_frames, -1, dtype=np.int32)
    
    # Fill in window labels
    for start, end, label_id in window_labels:
        labels[start:end] = label_id
    
    # Mark seizure frames
    if seizure_start_frame is not None:
        labels[seizure_start_frame:] = seizure_label_id
    
    return labels


def save_labels(
    labels: np.ndarray,
    output_path: str,
    video_info: VideoInfo,
    label_map: Dict[int, str]
):
    """Save labels and metadata"""
    np.save(output_path, labels)
    
    # Also save metadata
    metadata = {
        'video_filename': video_info.filename,
        'patient_id': video_info.patient_id,
        'total_frames': video_info.total_frames,
        'duration_sec': video_info.duration_sec,
        'clinical_onset_sec': video_info.clinical_onset_sec,
        'fps': FPS,
        'label_map': label_map
    }
    
    metadata_path = output_path.replace('.npy', '_metadata.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
