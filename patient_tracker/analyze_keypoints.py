#!/usr/bin/env python3
"""
Keypoint Analysis Utilities
===========================

Tools for loading, analyzing, and visualizing the extracted pose keypoints.
Useful for downstream seizure detection analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json


# COCO keypoint definitions
COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# Body part groups for analysis
BODY_PARTS = {
    'head': [0, 1, 2, 3, 4],
    'upper_body': [5, 6, 7, 8, 9, 10],
    'torso': [5, 6, 11, 12],
    'lower_body': [11, 12, 13, 14, 15, 16],
    'left_arm': [5, 7, 9],
    'right_arm': [6, 8, 10],
    'left_leg': [11, 13, 15],
    'right_leg': [12, 14, 16],
}


def load_keypoints(npy_path: str) -> Dict[str, Any]:
    """
    Load keypoints from NPY file.
    
    Args:
        npy_path: Path to the NPY file
        
    Returns:
        Dictionary with keypoints and metadata
    """
    data = np.load(npy_path, allow_pickle=True).item()
    return data


def compute_velocities(
    keypoints: np.ndarray,
    fps: float = 30.0,
    confidence_threshold: float = 0.3
) -> np.ndarray:
    """
    Compute velocities for each keypoint.
    
    Args:
        keypoints: Array of shape (T, K, 3) with [x, y, conf]
        fps: Video frame rate
        confidence_threshold: Min confidence for valid keypoints
        
    Returns:
        Velocities array of shape (T-1, K, 2) with [vx, vy]
    """
    T, K, _ = keypoints.shape
    velocities = np.zeros((T - 1, K, 2), dtype=np.float32)
    
    dt = 1.0 / fps
    
    for t in range(T - 1):
        for k in range(K):
            if (keypoints[t, k, 2] > confidence_threshold and
                keypoints[t + 1, k, 2] > confidence_threshold):
                
                dx = keypoints[t + 1, k, 0] - keypoints[t, k, 0]
                dy = keypoints[t + 1, k, 1] - keypoints[t, k, 1]
                
                velocities[t, k, 0] = dx / dt
                velocities[t, k, 1] = dy / dt
    
    return velocities


def compute_accelerations(
    velocities: np.ndarray,
    fps: float = 30.0
) -> np.ndarray:
    """
    Compute accelerations from velocities.
    
    Args:
        velocities: Array of shape (T, K, 2)
        fps: Video frame rate
        
    Returns:
        Accelerations array of shape (T-1, K, 2)
    """
    T, K, _ = velocities.shape
    accelerations = np.zeros((T - 1, K, 2), dtype=np.float32)
    
    dt = 1.0 / fps
    
    for t in range(T - 1):
        accelerations[t] = (velocities[t + 1] - velocities[t]) / dt
    
    return accelerations


def compute_body_part_motion(
    keypoints: np.ndarray,
    body_part: str,
    confidence_threshold: float = 0.3
) -> np.ndarray:
    """
    Compute aggregate motion for a body part.
    
    Args:
        keypoints: Array of shape (T, K, 3)
        body_part: Name of body part ('head', 'upper_body', etc.)
        confidence_threshold: Min confidence threshold
        
    Returns:
        Motion magnitude per frame, shape (T-1,)
    """
    if body_part not in BODY_PARTS:
        raise ValueError(f"Unknown body part: {body_part}")
    
    indices = BODY_PARTS[body_part]
    T = keypoints.shape[0]
    motion = np.zeros(T - 1, dtype=np.float32)
    
    for t in range(T - 1):
        valid_motions = []
        
        for k in indices:
            if (keypoints[t, k, 2] > confidence_threshold and
                keypoints[t + 1, k, 2] > confidence_threshold):
                
                dx = keypoints[t + 1, k, 0] - keypoints[t, k, 0]
                dy = keypoints[t + 1, k, 1] - keypoints[t, k, 1]
                magnitude = np.sqrt(dx**2 + dy**2)
                valid_motions.append(magnitude)
        
        if valid_motions:
            motion[t] = np.mean(valid_motions)
    
    return motion


def compute_joint_angles(
    keypoints: np.ndarray,
    confidence_threshold: float = 0.3
) -> Dict[str, np.ndarray]:
    """
    Compute joint angles from keypoints.
    
    Args:
        keypoints: Array of shape (T, K, 3)
        confidence_threshold: Min confidence threshold
        
    Returns:
        Dictionary with angle arrays for each joint
    """
    T = keypoints.shape[0]
    
    # Joint definitions: (parent, joint, child)
    joints = {
        'left_elbow': (5, 7, 9),   # shoulder -> elbow -> wrist
        'right_elbow': (6, 8, 10),
        'left_knee': (11, 13, 15),  # hip -> knee -> ankle
        'right_knee': (12, 14, 16),
        'left_shoulder': (6, 5, 7),  # right shoulder -> left shoulder -> elbow
        'right_shoulder': (5, 6, 8),
        'left_hip': (12, 11, 13),   # right hip -> left hip -> knee
        'right_hip': (11, 12, 14),
    }
    
    angles = {}
    
    for name, (p, j, c) in joints.items():
        joint_angles = np.full(T, np.nan, dtype=np.float32)
        
        for t in range(T):
            if (keypoints[t, p, 2] > confidence_threshold and
                keypoints[t, j, 2] > confidence_threshold and
                keypoints[t, c, 2] > confidence_threshold):
                
                # Vectors
                v1 = keypoints[t, p, :2] - keypoints[t, j, :2]
                v2 = keypoints[t, c, :2] - keypoints[t, j, :2]
                
                # Angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle) * 180 / np.pi
                
                joint_angles[t] = angle
        
        angles[name] = joint_angles
    
    return angles


def detect_high_motion_segments(
    motion: np.ndarray,
    threshold_multiplier: float = 2.0,
    min_duration_frames: int = 5,
    fps: float = 30.0
) -> List[Dict[str, Any]]:
    """
    Detect segments with unusually high motion (potential seizure events).
    
    Args:
        motion: Motion magnitude per frame
        threshold_multiplier: Multiple of median to use as threshold
        min_duration_frames: Minimum duration to count as a segment
        fps: Video frame rate
        
    Returns:
        List of detected segments with start/end times
    """
    # Compute threshold
    valid_motion = motion[motion > 0]
    if len(valid_motion) == 0:
        return []
    
    threshold = np.median(valid_motion) * threshold_multiplier
    
    # Find segments above threshold
    above = motion > threshold
    segments = []
    
    start = None
    for i, is_above in enumerate(above):
        if is_above and start is None:
            start = i
        elif not is_above and start is not None:
            if i - start >= min_duration_frames:
                segments.append({
                    'start_frame': start,
                    'end_frame': i,
                    'start_time': start / fps,
                    'end_time': i / fps,
                    'duration': (i - start) / fps,
                    'max_motion': float(np.max(motion[start:i])),
                    'mean_motion': float(np.mean(motion[start:i])),
                })
            start = None
    
    # Handle segment at end
    if start is not None and len(motion) - start >= min_duration_frames:
        segments.append({
            'start_frame': start,
            'end_frame': len(motion),
            'start_time': start / fps,
            'end_time': len(motion) / fps,
            'duration': (len(motion) - start) / fps,
            'max_motion': float(np.max(motion[start:])),
            'mean_motion': float(np.mean(motion[start:])),
        })
    
    return segments


def compute_statistics(keypoints: np.ndarray) -> Dict[str, Any]:
    """
    Compute summary statistics for the keypoint sequence.
    
    Args:
        keypoints: Array of shape (T, K, 3)
        
    Returns:
        Dictionary with various statistics
    """
    T, K, _ = keypoints.shape
    
    # Valid keypoint ratio per frame
    valid_per_frame = np.sum(keypoints[:, :, 2] > 0.3, axis=1) / K
    
    # Motion per body part
    body_part_motion = {}
    for part in BODY_PARTS:
        motion = compute_body_part_motion(keypoints, part)
        body_part_motion[part] = {
            'mean': float(np.mean(motion)),
            'std': float(np.std(motion)),
            'max': float(np.max(motion)),
        }
    
    # Joint angle statistics
    angles = compute_joint_angles(keypoints)
    angle_stats = {}
    for name, angle_array in angles.items():
        valid_angles = angle_array[~np.isnan(angle_array)]
        if len(valid_angles) > 0:
            angle_stats[name] = {
                'mean': float(np.mean(valid_angles)),
                'std': float(np.std(valid_angles)),
                'min': float(np.min(valid_angles)),
                'max': float(np.max(valid_angles)),
            }
    
    return {
        'num_frames': T,
        'valid_frame_ratio': float(np.mean(valid_per_frame)),
        'body_part_motion': body_part_motion,
        'joint_angles': angle_stats,
    }


def plot_motion_timeline(
    keypoints: np.ndarray,
    fps: float = 30.0,
    save_path: Optional[str] = None
) -> None:
    """
    Plot motion timeline for all body parts.
    
    Args:
        keypoints: Array of shape (T, K, 3)
        fps: Video frame rate
        save_path: Optional path to save figure
    """
    T = keypoints.shape[0]
    time = np.arange(T - 1) / fps
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    fig.suptitle('Body Part Motion Over Time', fontsize=14)
    
    parts = list(BODY_PARTS.keys())
    
    for idx, part in enumerate(parts):
        ax = axes[idx // 2, idx % 2]
        motion = compute_body_part_motion(keypoints, part)
        
        ax.plot(time, motion, linewidth=0.5)
        ax.set_ylabel('Motion (px/frame)')
        ax.set_title(part.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # Mark high motion segments
        segments = detect_high_motion_segments(motion, fps=fps)
        for seg in segments:
            ax.axvspan(seg['start_time'], seg['end_time'], 
                      alpha=0.3, color='red', label='High motion')
    
    axes[-1, 0].set_xlabel('Time (seconds)')
    axes[-1, 1].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


def plot_joint_angles(
    keypoints: np.ndarray,
    fps: float = 30.0,
    save_path: Optional[str] = None
) -> None:
    """
    Plot joint angles over time.
    
    Args:
        keypoints: Array of shape (T, K, 3)
        fps: Video frame rate
        save_path: Optional path to save figure
    """
    angles = compute_joint_angles(keypoints)
    T = keypoints.shape[0]
    time = np.arange(T) / fps
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    fig.suptitle('Joint Angles Over Time', fontsize=14)
    
    joint_names = list(angles.keys())
    
    for idx, name in enumerate(joint_names):
        ax = axes[idx // 2, idx % 2]
        angle_data = angles[name]
        
        ax.plot(time, angle_data, linewidth=0.5)
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 180)
    
    axes[-1, 0].set_xlabel('Time (seconds)')
    axes[-1, 1].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


def export_to_json(
    npy_path: str,
    output_path: str
) -> None:
    """
    Export keypoints to JSON format for web visualization.
    
    Args:
        npy_path: Path to NPY file
        output_path: Path to output JSON file
    """
    data = load_keypoints(npy_path)
    
    export_data = {
        'video_path': data.get('video_path', ''),
        'fps': float(data.get('fps', 30.0)),
        'total_frames': int(data.get('total_frames', 0)),
        'keypoint_names': COCO_KEYPOINTS,
        'skeleton': COCO_SKELETON,
        'frames': []
    }
    
    keypoints = data['keypoints']
    confidences = data.get('confidences', np.ones(len(keypoints)))
    
    for i in range(len(keypoints)):
        frame_data = {
            'frame_idx': i,
            'detection_confidence': float(confidences[i]),
            'keypoints': keypoints[i].tolist(),
        }
        export_data['frames'].append(frame_data)
    
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Exported to: {output_path}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze keypoint data')
    parser.add_argument('npy_file', help='Path to keypoints NPY file')
    parser.add_argument('--plot-motion', action='store_true', help='Plot motion timeline')
    parser.add_argument('--plot-angles', action='store_true', help='Plot joint angles')
    parser.add_argument('--export-json', type=str, help='Export to JSON file')
    parser.add_argument('--detect-events', action='store_true', help='Detect high motion events')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading: {args.npy_file}")
    data = load_keypoints(args.npy_file)
    
    keypoints = data['keypoints']
    fps = data.get('fps', 30.0)
    
    print(f"\nData summary:")
    print(f"  Frames: {keypoints.shape[0]}")
    print(f"  Keypoints: {keypoints.shape[1]}")
    print(f"  FPS: {fps}")
    
    # Compute statistics
    stats = compute_statistics(keypoints)
    print(f"\n  Valid frame ratio: {stats['valid_frame_ratio']:.2%}")
    
    # Detect events
    if args.detect_events:
        print("\nDetecting high motion segments...")
        total_motion = compute_body_part_motion(keypoints, 'upper_body')
        events = detect_high_motion_segments(total_motion, fps=fps)
        
        if events:
            print(f"Found {len(events)} potential events:")
            for i, event in enumerate(events):
                print(f"  Event {i+1}: {event['start_time']:.2f}s - {event['end_time']:.2f}s "
                      f"(duration: {event['duration']:.2f}s, max motion: {event['max_motion']:.1f})")
        else:
            print("No high motion segments detected")
    
    # Plot
    if args.plot_motion:
        plot_motion_timeline(keypoints, fps=fps)
    
    if args.plot_angles:
        plot_joint_angles(keypoints, fps=fps)
    
    # Export
    if args.export_json:
        export_to_json(args.npy_file, args.export_json)


if __name__ == '__main__':
    main()
