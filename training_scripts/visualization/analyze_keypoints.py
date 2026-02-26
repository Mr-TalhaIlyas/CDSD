#!/usr/bin/env python3
"""
Keypoint Analysis Utilities
===========================

Tools for loading, analyzing, and visualizing the extracted pose keypoints.
Useful for downstream seizure detection analysis.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from custom_data_utils import (
    NEW_LABEL_MAP, LABEL_CONVERSION,
    load_keypoint_file, load_label_file,
    convert_coco_to_ntu, normalize_skeleton_to_ntu_scale,
    translate_to_origin, frame_normalize, remove_zero_frames
)

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

JOINT_TO_BODY_PART = {
    0: 'hip',    # base of spine
    1: 'hip',    # middle of spine
    2: 'hip',    # neck
    3: 'head',   # head
    4: 'arm',    # left shoulder
    5: 'arm',    # left elbow
    6: 'arm',    # left wrist
    7: 'hand',   # left hand
    8: 'arm',    # right shoulder
    9: 'arm',    # right elbow
    10: 'arm',   # right wrist
    11: 'hand',  # right hand
    12: 'leg',   # left hip
    13: 'leg',   # left knee
    14: 'leg',   # left ankle
    15: 'foot',  # left foot
    16: 'leg',   # right hip
    17: 'leg',   # right knee
    18: 'leg',   # right ankle
    19: 'foot',  # right foot
    20: 'hip',   # spine (shoulder level)
    21: 'hand',  # left hand tip
    22: 'hand',  # left thumb
    23: 'hand',  # right hand tip
    24: 'hand',  # right thumb
}

# Standard NTU RGB+D skeleton connections (25 joints)
# Based on the official NTU RGB+D skeleton structure
NTU_SKELETON_CONNECTIONS = [
    # Spine
    (0, 1),    # base of spine -> middle of spine
    (1, 20),   # middle of spine -> spine (shoulder level)
    (20, 2),   # spine (shoulder level) -> neck
    (2, 3),    # neck -> head
    # Left arm
    (20, 4),   # spine -> left shoulder
    (4, 5),    # left shoulder -> left elbow
    (5, 6),    # left elbow -> left wrist
    (6, 7),    # left wrist -> left hand
    (7, 21),   # left hand -> left hand tip
    (7, 22),   # left hand -> left thumb
    # Right arm
    (20, 8),   # spine -> right shoulder
    (8, 9),    # right shoulder -> right elbow
    (9, 10),   # right elbow -> right wrist
    (10, 11),  # right wrist -> right hand
    (11, 23),  # right hand -> right hand tip
    (11, 24),  # right hand -> right thumb
    # Left leg
    (0, 12),   # base of spine -> left hip
    (12, 13),  # left hip -> left knee
    (13, 14),  # left knee -> left ankle
    (14, 15),  # left ankle -> left foot
    # Right leg
    (0, 16),   # base of spine -> right hip
    (16, 17),  # right hip -> right knee
    (17, 18),  # right knee -> right ankle
    (18, 19),  # right ankle -> right foot
]

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


def draw_normalized_pose(
    keypoints: np.ndarray,
    ax: Optional[plt.Axes] = None,
    conf_threshold: float = 0.1,
    skeleton_color: str = 'blue',
    keypoint_color: str = 'red',
    linewidth: float = 2.0,
    markersize: float = 8,
    skeleton_connections: Optional[List[Tuple[int, int]]] = None,
    title: str = '',
    show_joint_indices: bool = False,
    invert_y: bool = True,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Axes:
    """
    Draw normalized pose using matplotlib (works with normalized coordinates).
    
    Args:
        keypoints: Keypoints array (K, 3) with [x, y, conf] in normalized coords
        ax: Matplotlib axis (creates new figure if None)
        conf_threshold: Min confidence to draw
        skeleton_color: Color for skeleton lines
        keypoint_color: Color for keypoints
        linewidth: Line width for skeleton
        markersize: Size for keypoint markers
        skeleton_connections: List of (i, j) joint connections
        title: Plot title
        show_joint_indices: Whether to show joint index numbers
        invert_y: Invert Y axis (for image-like coords where Y increases downward)
        figsize: Figure size if creating new figure
        
    Returns:
        Matplotlib axis with pose drawn
    """
    if skeleton_connections is None:
        # Default to NTU skeleton if 25 joints, else COCO
        if len(keypoints) == 25:
            skeleton_connections = NTU_SKELETON_CONNECTIONS
        else:
            skeleton_connections = COCO_SKELETON
    
    # Create figure if no axis provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Extract coordinates
    x = keypoints[:, 0]
    y = keypoints[:, 1]
    conf = keypoints[:, 2] if keypoints.shape[1] > 2 else np.ones(len(keypoints))
    
    # Draw skeleton connections
    for i, j in skeleton_connections:
        if (i < len(keypoints) and j < len(keypoints) and
            conf[i] > conf_threshold and conf[j] > conf_threshold):
            ax.plot([x[i], x[j]], [y[i], y[j]], 
                   color=skeleton_color, linewidth=linewidth, 
                   solid_capstyle='round', zorder=1)
    
    # Draw keypoints
    valid_mask = conf > conf_threshold
    ax.scatter(x[valid_mask], y[valid_mask], 
              c=keypoint_color, s=markersize**2, 
              edgecolors='white', linewidths=0.5, zorder=2)
    
    # Show joint indices if requested
    if show_joint_indices:
        for idx in range(len(keypoints)):
            if conf[idx] > conf_threshold:
                ax.annotate(str(idx), (x[idx], y[idx]), 
                           fontsize=7, ha='center', va='bottom',
                           color='black', zorder=3)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Invert Y axis if needed (normalized coords often have Y pointing up)
    if invert_y:
        ax.invert_yaxis()
    
    # Add grid for reference
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5, alpha=0.5)
    
    if title:
        ax.set_title(title)
    
    ax.set_xlabel('X (normalized)')
    ax.set_ylabel('Y (normalized)')
    
    return ax


def visualize_normalized_sequence(
    keypoints: np.ndarray,
    n_frames: int = 8,
    conf_threshold: float = 0.1,
    skeleton_connections: Optional[List[Tuple[int, int]]] = None,
    figsize: Tuple[int, int] = (20, 4),
    title: str = '',
    show_joint_indices: bool = False,
    invert_y: bool = True
) -> plt.Figure:
    """
    Visualize multiple frames from a normalized pose sequence.
    
    Args:
        keypoints: Keypoints array (T, K, 3) with [x, y, conf]
        n_frames: Number of frames to display (equidistant sampling)
        conf_threshold: Min confidence to draw
        skeleton_connections: List of (i, j) joint connections
        figsize: Figure size
        title: Overall figure title
        show_joint_indices: Whether to show joint index numbers
        invert_y: Invert Y axis
        
    Returns:
        Matplotlib figure
    """
    T = keypoints.shape[0]
    frame_indices = np.linspace(0, T - 1, n_frames, dtype=int)
    
    fig, axes = plt.subplots(1, n_frames, figsize=figsize)
    if n_frames == 1:
        axes = [axes]
    
    # Find global bounds for consistent scaling
    valid_mask = keypoints[:, :, 2] > conf_threshold if keypoints.shape[2] > 2 else np.ones((T, keypoints.shape[1]), dtype=bool)
    x_all = keypoints[:, :, 0][valid_mask]
    y_all = keypoints[:, :, 1][valid_mask]
    
    if len(x_all) > 0:
        x_min, x_max = x_all.min(), x_all.max()
        y_min, y_max = y_all.min(), y_all.max()
        
        # Add padding
        x_pad = (x_max - x_min) * 0.15 + 0.05
        y_pad = (y_max - y_min) * 0.15 + 0.05
    else:
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1
        x_pad, y_pad = 0.1, 0.1
    
    for i, frame_idx in enumerate(frame_indices):
        ax = axes[i]
        frame_kps = keypoints[frame_idx]
        
        draw_normalized_pose(
            frame_kps, ax=ax,
            conf_threshold=conf_threshold,
            skeleton_connections=skeleton_connections,
            title=f'Frame {frame_idx}',
            show_joint_indices=show_joint_indices,
            invert_y=invert_y
        )
        
        # Set consistent limits across all frames
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        if invert_y:
            ax.set_ylim(y_max + y_pad, y_min - y_pad)
        else:
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        
        # Only show y-label on first plot
        if i > 0:
            ax.set_ylabel('')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def visualize_ctrgcn_format(
    data: np.ndarray,
    n_frames: int = 8,
    person_idx: int = 0,
    skeleton_connections: Optional[List[Tuple[int, int]]] = None,
    figsize: Tuple[int, int] = (20, 4),
    title: str = '',
    show_joint_indices: bool = False
) -> plt.Figure:
    """
    Visualize poses from CTR-GCN format (C, T, V, M).
    
    Args:
        data: Skeleton data in CTR-GCN format (3, T, 25, M) or (C, T, V, M)
        n_frames: Number of frames to display
        person_idx: Which person to visualize (M dimension)
        skeleton_connections: List of (i, j) joint connections
        figsize: Figure size
        title: Overall figure title
        show_joint_indices: Whether to show joint index numbers
        
    Returns:
        Matplotlib figure
    """
    C, T, V, M = data.shape
    
    if skeleton_connections is None:
        skeleton_connections = NTU_SKELETON_CONNECTIONS
    
    # Convert to (T, V, C) format for visualization
    # X, Y are in channels 0, 1
    keypoints = np.zeros((T, V, 3))
    keypoints[:, :, 0] = data[0, :, :, person_idx]  # X
    keypoints[:, :, 1] = data[1, :, :, person_idx]  # Y
    
    # Use Z or magnitude as confidence proxy (or just set to 1)
    if C > 2:
        # Use absolute Z value as confidence indicator
        keypoints[:, :, 2] = np.abs(data[0, :, :, person_idx]) + np.abs(data[1, :, :, person_idx])
        keypoints[:, :, 2] = (keypoints[:, :, 2] > 0.001).astype(float)
    else:
        keypoints[:, :, 2] = 1.0
    
    return visualize_normalized_sequence(
        keypoints,
        n_frames=n_frames,
        skeleton_connections=skeleton_connections,
        figsize=figsize,
        title=title,
        show_joint_indices=show_joint_indices,
        invert_y=True  # Usually Y is inverted in image coords
    )


def convert_coco_to_ntu_old(coco_keypoints: np.ndarray) -> np.ndarray:
    """
    Convert 17 COCO keypoints to 25 NTU-RGB+D keypoints.
    
    Args:
        coco_keypoints: Array of shape (N, 17, 3) or (17, 3) with [x, y, confidence]
                       COCO format: ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                                   'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                                   'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                                   'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    
    Returns:
        Array of shape (N, 25, 3) or (25, 3) with NTU-RGB+D keypoints:
        [0: base_spine, 1: middle_spine, 2: neck, 3: head, 
         4: left_shoulder, 5: left_elbow, 6: left_wrist, 7: left_hand,
         8: right_shoulder, 9: right_elbow, 10: right_wrist, 11: right_hand,
         12: left_hip, 13: left_knee, 14: left_ankle, 15: left_foot,
         16: right_hip, 17: right_knee, 18: right_ankle, 19: right_foot,
         20: spine_shoulder, 21: left_hand_tip, 22: left_thumb, 
         23: right_hand_tip, 24: right_thumb]
    """
    # Handle both single frame and batch inputs
    single_frame = len(coco_keypoints.shape) == 2
    if single_frame:
        coco_keypoints = coco_keypoints[np.newaxis, ...]
    
    batch_size, num_joints, coords = coco_keypoints.shape
    assert num_joints == 17, f"Expected 17 COCO keypoints, got {num_joints}"
    assert coords == 3, f"Expected 3 coordinates [x, y, conf], got {coords}"
    
    # Initialize NTU keypoints array
    ntu_keypoints = np.zeros((batch_size, 25, 3), dtype=coco_keypoints.dtype)
    
    # COCO keypoint indices
    coco_indices = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
    }
    
    for i in range(batch_size):
        coco = coco_keypoints[i]
        ntu = ntu_keypoints[i]
        
        # Direct mappings from COCO to NTU
        ntu[3] = coco[coco_indices['nose']]           # head <- nose
        ntu[4] = coco[coco_indices['left_shoulder']]  # left_shoulder
        ntu[5] = coco[coco_indices['left_elbow']]     # left_elbow  
        ntu[6] = coco[coco_indices['left_wrist']]     # left_wrist
        ntu[8] = coco[coco_indices['right_shoulder']] # right_shoulder
        ntu[9] = coco[coco_indices['right_elbow']]    # right_elbow
        ntu[10] = coco[coco_indices['right_wrist']]   # right_wrist
        ntu[12] = coco[coco_indices['left_hip']]      # left_hip
        ntu[13] = coco[coco_indices['left_knee']]     # left_knee
        ntu[14] = coco[coco_indices['left_ankle']]    # left_ankle
        ntu[16] = coco[coco_indices['right_hip']]     # right_hip
        ntu[17] = coco[coco_indices['right_knee']]    # right_knee
        ntu[18] = coco[coco_indices['right_ankle']]   # right_ankle
        
        # Derived keypoints (interpolation and extrapolation)
        left_hip = coco[coco_indices['left_hip']]
        right_hip = coco[coco_indices['right_hip']]
        left_shoulder = coco[coco_indices['left_shoulder']]
        right_shoulder = coco[coco_indices['right_shoulder']]
        nose = coco[coco_indices['nose']]
        
        # Base of spine (0): midpoint between hips
        if left_hip[2] > 0 and right_hip[2] > 0:
            ntu[0, :2] = (left_hip[:2] + right_hip[:2]) / 2
            ntu[0, 2] = min(left_hip[2], right_hip[2])
        
        # Spine at shoulder level (20): midpoint between shoulders  
        if left_shoulder[2] > 0 and right_shoulder[2] > 0:
            ntu[20, :2] = (left_shoulder[:2] + right_shoulder[:2]) / 2
            ntu[20, 2] = min(left_shoulder[2], right_shoulder[2])
        
        # Middle of spine (1): interpolate between base spine and shoulder spine
        if ntu[0, 2] > 0 and ntu[20, 2] > 0:
            ntu[1, :2] = (ntu[0, :2] + ntu[20, :2]) / 2
            ntu[1, 2] = min(ntu[0, 2], ntu[20, 2])
        
        # Neck (2): interpolate between shoulder spine and head
        if ntu[20, 2] > 0 and nose[2] > 0:
            ntu[2, :2] = (ntu[20, :2] + nose[:2]) / 2
            ntu[2, 2] = min(ntu[20, 2], nose[2])
        
        # Hands (initially same as wrists)
        ntu[7] = ntu[6]   # left_hand <- left_wrist
        ntu[11] = ntu[10] # right_hand <- right_wrist
        
        # Feet (initially same as ankles)
        ntu[15] = ntu[14] # left_foot <- left_ankle
        ntu[19] = ntu[18] # right_foot <- right_ankle
        
        # Hand tips and thumbs (extrapolated from wrist-elbow direction)
        left_wrist = coco[coco_indices['left_wrist']]
        left_elbow = coco[coco_indices['left_elbow']]
        right_wrist = coco[coco_indices['right_wrist']]
        right_elbow = coco[coco_indices['right_elbow']]
        
        # Left hand tip and thumb
        if left_wrist[2] > 0 and left_elbow[2] > 0:
            wrist_to_elbow = left_elbow[:2] - left_wrist[:2]
            hand_direction = -wrist_to_elbow  # opposite direction
            hand_direction = hand_direction / (np.linalg.norm(hand_direction) + 1e-8)
            
            # Hand tip (extend further)
            ntu[21, :2] = left_wrist[:2] + hand_direction * 30
            ntu[21, 2] = left_wrist[2] * 0.8  # lower confidence
            
            # Thumb (extend less, slightly perpendicular)
            thumb_direction = hand_direction + np.array([-hand_direction[1], hand_direction[0]]) * 0.3
            thumb_direction = thumb_direction / (np.linalg.norm(thumb_direction) + 1e-8)
            ntu[22, :2] = left_wrist[:2] + thumb_direction * 15
            ntu[22, 2] = left_wrist[2] * 0.7
        
        # Right hand tip and thumb  
        if right_wrist[2] > 0 and right_elbow[2] > 0:
            wrist_to_elbow = right_elbow[:2] - right_wrist[:2]
            hand_direction = -wrist_to_elbow
            hand_direction = hand_direction / (np.linalg.norm(hand_direction) + 1e-8)
            
            # Hand tip
            ntu[23, :2] = right_wrist[:2] + hand_direction * 30
            ntu[23, 2] = right_wrist[2] * 0.8
            
            # Thumb
            thumb_direction = hand_direction + np.array([hand_direction[1], -hand_direction[0]]) * 0.3
            thumb_direction = thumb_direction / (np.linalg.norm(thumb_direction) + 1e-8)
            ntu[24, :2] = right_wrist[:2] + thumb_direction * 15
            ntu[24, 2] = right_wrist[2] * 0.7
    
    # Return single frame if input was single frame
    if single_frame:
        return ntu_keypoints[0]
    else:
        return ntu_keypoints


def draw_pose(
    image: np.ndarray,
    keypoints: np.ndarray,
    conf_threshold: float = 0.3,
    skeleton_color: Tuple[int, int, int] = (0, 255, 0),
    keypoint_color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
    radius: int = 4,
    draw_skeleton: bool = True,
    skeleton_connections: Optional[List[Tuple[int, int]]] = COCO_SKELETON
) -> np.ndarray:
    """
    Draw pose on image.
    
    Args:
        image: Input image (will be copied)
        keypoints: Keypoints array (K, 3) with [x, y, conf]
        conf_threshold: Min confidence to draw
        skeleton_color: BGR color for skeleton
        keypoint_color: BGR color for keypoints
        thickness: Line thickness
        radius: Keypoint radius
        draw_skeleton: Whether to draw skeleton connections
        
    Returns:
        Image with pose overlay
    """
    result = image.copy()
    
    # Only draw skeleton for COCO-compatible keypoints
    if draw_skeleton and len(keypoints) >= 17:
        for i, j in skeleton_connections:
            if (i < len(keypoints) and j < len(keypoints) and
                keypoints[i, 2] > conf_threshold and
                keypoints[j, 2] > conf_threshold):
                
                pt1 = tuple(keypoints[i, :2].astype(int))
                pt2 = tuple(keypoints[j, :2].astype(int))
                cv2.line(result, pt1, pt2, skeleton_color, thickness)
    
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > conf_threshold:
            cv2.circle(result, (int(x), int(y)), radius, keypoint_color, -1)
            cv2.circle(result, (int(x), int(y)), radius, (255, 255, 255), 1)
    
    return result
#%%

# Load data
# npy_file = Path('D:/miccai_26/DATA/ieee/Normal/S0_0_0_keypoints.npy')
# print(f"Loading: {npy_file}")
# data = load_keypoints(npy_file)

# keypoints = data['keypoints']
# fps = data.get('fps', 30.0)

# print(f"\nData summary:")
# print(keypoints.shape)
# print(f"  Frames: {keypoints.shape[0]}")
# print(f"  Keypoints: {keypoints.shape[1]}")
# print(f"  FPS: {fps}")

# # blank frame for visualization
# img_dim = data['frame_size'] # (height, width)
# image = np.zeros((img_dim[0], img_dim[1], 3), dtype=np.uint8)

# # Visualize some frames with keypoints
# for frame_idx in range(0, keypoints.shape[0]):
#     frame_kps = keypoints[frame_idx]
#     vis_image = draw_pose(image, frame_kps,
#                           skeleton_connections=COCO_SKELETON)
#     clear_output(wait=True)
#     plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
#     plt.title(f'Frame {frame_idx}')
#     plt.axis('off')
#     plt.show()
#     plt.pause(0.05)

#%% Visualize normalized poses
# This cell shows how to visualize poses after normalization

# Load raw keypoints
npy_file = Path('D:/miccai_26/DATA/ieee/Normal/S0_0_0_keypoints.npy')
data = load_keypoints(npy_file)
keypoints = data['keypoints']

print(f"Original keypoints shape: {keypoints.shape}")
print(f"Original coordinate range: X=[{keypoints[:,:,0].min():.1f}, {keypoints[:,:,0].max():.1f}], Y=[{keypoints[:,:,1].min():.1f}, {keypoints[:,:,1].max():.1f}]")

# Convert and normalize
ntu_kps = convert_coco_to_ntu(keypoints, include_confidence=True)
ntu_kps, _ = remove_zero_frames(ntu_kps, None)
ntu_kps_normalized = normalize_skeleton_to_ntu_scale(ntu_kps.copy())
ntu_kps_translated = translate_to_origin(ntu_kps_normalized.copy())

print(f"\nAfter normalization:")
print(f"  Shape: {ntu_kps_normalized.shape}")
print(f"  Coordinate range: X=[{ntu_kps_normalized[:,:,0].min():.3f}, {ntu_kps_normalized[:,:,0].max():.3f}], Y=[{ntu_kps_normalized[:,:,1].min():.3f}, {ntu_kps_normalized[:,:,1].max():.3f}]")

print(f"\nAfter translation to origin:")
print(f"  Coordinate range: X=[{ntu_kps_translated[:,:,0].min():.3f}, {ntu_kps_translated[:,:,0].max():.3f}], Y=[{ntu_kps_translated[:,:,1].min():.3f}, {ntu_kps_translated[:,:,1].max():.3f}]")

#%% Visualize single normalized frame
frame_idx = 50
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original (scaled down for comparison)
orig_scaled = ntu_kps[frame_idx].copy()
orig_scaled[:, :2] = orig_scaled[:, :2] / 1000  # Scale to similar range
draw_normalized_pose(orig_scaled, ax=axes[0], 
                    skeleton_connections=NTU_SKELETON_CONNECTIONS,
                    title=f'Original (scaled /1000)', 
                    show_joint_indices=True)

# Normalized
draw_normalized_pose(ntu_kps_normalized[frame_idx], ax=axes[1],
                    skeleton_connections=NTU_SKELETON_CONNECTIONS,
                    title='After Scale Normalization',
                    show_joint_indices=True)

# Translated
draw_normalized_pose(ntu_kps_translated[frame_idx], ax=axes[2],
                    skeleton_connections=NTU_SKELETON_CONNECTIONS,
                    title='After Translation to Origin',
                    show_joint_indices=True)

plt.tight_layout()
plt.show()

#%% Visualize sequence of normalized frames
fig = visualize_normalized_sequence(
    ntu_kps_translated,
    n_frames=8,
    skeleton_connections=NTU_SKELETON_CONNECTIONS,
    figsize=(20, 4),
    title='Normalized & Translated Pose Sequence',
    show_joint_indices=False
)
plt.show()

#%% Load and visualize processed dataset (CTR-GCN format)
processed_file = Path('/mnt/ssd/Talha/reason/data/vsvig_seizure_dataset.npz')
if processed_file.exists():
    processed_data = np.load(processed_file, allow_pickle=True)
    
    print("Processed dataset keys:", processed_data.files)
    print(f"Data shape (CTR-GCN format): {processed_data['data'].shape}")
    
    # Visualize first sample
    sample_idx = 1870
    sample_data = processed_data['data'][sample_idx]  # (C, T, V, M)
    sample_name = processed_data['sample_names'][sample_idx]
    label = processed_data['labels'][sample_idx] #<-- list of ints
    # get most common label
    label = max(set(label), key=list(label).count)
    # get name from NEW_LABEL_MAP dict where values is label
    label_name = [k for k, v in NEW_LABEL_MAP.items() if v == label]
    fig = visualize_ctrgcn_format(
        sample_data,
        n_frames=8,
        person_idx=0,
        skeleton_connections=NTU_SKELETON_CONNECTIONS,
        figsize=(20, 4),
        title=f'Sample: {sample_name} Label: {label_name}',
        show_joint_indices=False
    )
    plt.show()
else:
    print(f"Processed file not found: {processed_file}")
# %%
