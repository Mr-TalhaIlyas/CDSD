#!/usr/bin/env python3
"""
Sapiens Pose Estimation Module (Lite Version)
==============================================

Uses TorchScript (.pt2) models from Sapiens-Lite for pose estimation.
Supports both COCO (17 keypoints) and Goliath (308 keypoints) models.

Model files:
- COCO 17 keypoints: sapiens_1b_coco_best_coco_AP_821_torchscript.pt2
- Goliath 308 keypoints: sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2

Download from: https://huggingface.co/facebook/sapiens-pose-1b-torchscript

Usage:
    from sapiens_pose import SapiensPoseEstimator
    
    estimator = SapiensPoseEstimator("./weights/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2")
    keypoints, scores = estimator.estimate(image)  # image in BGR format
"""

import os
import math
import logging
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

import numpy as np
import cv2
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Model Configuration
# ============================================================================

class SapiensModelType(Enum):
    """Sapiens model types and their keypoint counts."""
    COCO_17 = 17        # Standard COCO body keypoints
    COCO_133 = 133      # COCO whole-body keypoints
    GOLIATH_308 = 308   # Full body + face + hands + feet


# Sapiens model input configuration
SAPIENS_INPUT_SIZE = (1024, 768)  # (height, width)
SAPIENS_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAPIENS_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# ============================================================================
# COCO 17 Keypoint Definitions
# ============================================================================

COCO_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

# Goliath 308 keypoint names (first 17 match COCO approximately)
GOLIATH_BODY_KEYPOINTS = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
    # INDEX 41 is 'right_wrist'
    # INDEX 62 is 'left_wrist'
    # ... 291 more keypoints for face, hands, feet
]

GOLIATH_TO_COCO_MAPPING = {
    0: 0,   # nose
    1: 1,   # left_eye
    2: 2,   # right_eye
    3: 3,   # left_ear
    4: 4,   # right_ear
    5: 5,   # left_shoulder
    6: 6,   # right_shoulder
    7: 7,   # left_elbow
    8: 8,   # right_elbow
    9: 62,  # left_wrist (Goliath index 62)
    10: 41, # right_wrist (Goliath index 41)
    11: 9,  # left_hip
    12: 10, # right_hip
    13: 11, # left_knee
    14: 12, # right_knee
    15: 13, # left_ankle
    16: 14, # right_ankle
}

# Goliath skeleton for body keypoints (using Goliath indices)
GOLIATH_SKELETON = [
    # Head connections
    (0, 1), (0, 2), (1, 3), (2, 4),
    
    # Torso
    (5, 6),      # shoulders
    (5, 9),      # left shoulder to left hip
    (6, 10),     # right shoulder to right hip
    (9, 10),     # hips
    
    # Left arm
    (5, 7),      # left shoulder to left elbow
    (7, 62),     # left elbow to left wrist (index 62)
    
    # Right arm
    (6, 8),      # right shoulder to right elbow
    (8, 41),     # right elbow to right wrist (index 41)
    
    # Left leg
    (9, 11),     # left hip to left knee
    (11, 13),    # left knee to left ankle
    
    # Right leg
    (10, 12),    # right hip to right knee
    (12, 14),    # right knee to right ankle
]


# ============================================================================
# UDP Decoding Functions (from pose_utils.py)
# ============================================================================

def gaussian_blur(heatmaps: np.ndarray, kernel: int = 11) -> np.ndarray:
    """
    Modulate heatmap distribution with Gaussian.
    
    Args:
        heatmaps (np.ndarray[K, H, W]): Model predicted heatmaps.
        kernel (int): Gaussian kernel size for modulation.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    K, H, W = heatmaps.shape

    for k in range(K):
        origin_max = np.max(heatmaps[k])
        dr = np.zeros((H + 2 * border, W + 2 * border), dtype=np.float32)
        dr[border:-border, border:-border] = heatmaps[k].copy()
        dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
        heatmaps[k] = dr[border:-border, border:-border].copy()
        if np.max(heatmaps[k]) > 0:
            heatmaps[k] *= origin_max / np.max(heatmaps[k])
    return heatmaps


def get_heatmap_maximum(heatmaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get maximum response location and value from heatmaps.

    Args:
        heatmaps (np.ndarray): Heatmaps in shape (K, H, W) or (B, K, H, W)

    Returns:
        tuple:
        - locs (np.ndarray): locations of maximum heatmap responses
        - vals (np.ndarray): values of maximum heatmap responses
    """
    assert isinstance(heatmaps, np.ndarray), 'heatmaps should be numpy.ndarray'
    assert heatmaps.ndim == 3 or heatmaps.ndim == 4, f'Invalid shape {heatmaps.shape}'

    if heatmaps.ndim == 3:
        K, H, W = heatmaps.shape
        B = None
        heatmaps_flatten = heatmaps.reshape(K, -1)
    else:
        B, K, H, W = heatmaps.shape
        heatmaps_flatten = heatmaps.reshape(B * K, -1)

    y_locs, x_locs = np.unravel_index(
        np.argmax(heatmaps_flatten, axis=1), shape=(H, W))
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    vals = np.amax(heatmaps_flatten, axis=1)
    locs[vals <= 0.] = -1

    if B:
        locs = locs.reshape(B, K, 2)
        vals = vals.reshape(B, K)

    return locs, vals


def refine_keypoints_dark_udp(
    keypoints: np.ndarray, 
    heatmaps: np.ndarray,
    blur_kernel_size: int
) -> np.ndarray:
    """
    Refine keypoint predictions using distribution aware coordinate decoding
    for UDP (Unbiased Data Processing).

    Args:
        keypoints (np.ndarray): The keypoint coordinates in shape (N, K, D)
        heatmaps (np.ndarray): The heatmaps in shape (K, H, W)
        blur_kernel_size (int): The Gaussian blur kernel size

    Returns:
        np.ndarray: Refined keypoint coordinates in shape (N, K, D)
    """
    N, K = keypoints.shape[:2]
    H, W = heatmaps.shape[1:]

    # Modulate heatmaps
    heatmaps = gaussian_blur(heatmaps, blur_kernel_size)
    np.clip(heatmaps, 1e-3, 50., heatmaps)
    np.log(heatmaps, heatmaps)

    heatmaps_pad = np.pad(
        heatmaps, ((0, 0), (1, 1), (1, 1)), mode='edge').flatten()

    for n in range(N):
        index = keypoints[n, :, 0] + 1 + (keypoints[n, :, 1] + 1) * (W + 2)
        index += (W + 2) * (H + 2) * np.arange(0, K)
        index = index.astype(int).reshape(-1, 1)
        
        i_ = heatmaps_pad[index]
        ix1 = heatmaps_pad[index + 1]
        iy1 = heatmaps_pad[index + W + 2]
        ix1y1 = heatmaps_pad[index + W + 3]
        ix1_y1_ = heatmaps_pad[index - W - 3]
        ix1_ = heatmaps_pad[index - 1]
        iy1_ = heatmaps_pad[index - 2 - W]

        dx = 0.5 * (ix1 - ix1_)
        dy = 0.5 * (iy1 - iy1_)
        derivative = np.concatenate([dx, dy], axis=1)
        derivative = derivative.reshape(K, 2, 1)

        dxx = ix1 - 2 * i_ + ix1_
        dyy = iy1 - 2 * i_ + iy1_
        dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
        hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
        hessian = hessian.reshape(K, 2, 2)
        hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
        keypoints[n] -= np.einsum('imn,ink->imk', hessian, derivative).squeeze()

    return keypoints


def udp_decode(
    heatmaps: np.ndarray, 
    input_size: Tuple[int, int],
    heatmap_size: Tuple[int, int],
    blur_kernel_size: int = 11
) -> Tuple[np.ndarray, np.ndarray]:
    """
    UDP decoding for keypoint location refinement.

    Args:
        heatmaps (np.ndarray[K, H, W]): Model predicted heatmaps.
        input_size (tuple): Input image size (H, W)
        heatmap_size (tuple): Heatmap size (H, W)
        blur_kernel_size (int): Gaussian kernel size for modulation.

    Returns:
        tuple: (keypoints, scores)
            - keypoints: shape (1, K, 2) with [x, y] coordinates
            - scores: shape (1, K) with confidence scores
    """
    keypoints, scores = get_heatmap_maximum(heatmaps)
    
    # Unsqueeze the instance dimension for single-instance results
    keypoints = keypoints[None]
    scores = scores[None]
    
    # Refine keypoints using UDP
    keypoints = refine_keypoints_dark_udp(
        keypoints, heatmaps, blur_kernel_size=blur_kernel_size)

    # Scale keypoints from heatmap coordinates to input image coordinates
    H, W = heatmap_size
    keypoints = (keypoints / [W - 1, H - 1]) * [input_size[1], input_size[0]]
    
    return keypoints, scores


# ============================================================================
# Sapiens Pose Estimator Class
# ============================================================================

class SapiensPoseEstimator:
    """
    Sapiens pose estimation using TorchScript lite models.
    
    Supports:
    - COCO models (17 keypoints): sapiens_*_coco_*_torchscript.pt2
    - Goliath models (308 keypoints): sapiens_*_goliath_*_torchscript.pt2
    
    Usage:
        estimator = SapiensPoseEstimator("./weights/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2")
        keypoints, scores = estimator.estimate(image)  # BGR image
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize Sapiens pose estimator.
        
        Args:
            model_path: Path to TorchScript model (.pt2 file)
            device: Device to use ('cuda' or 'cpu', auto-detected if None)
            dtype: Data type for inference (torch.float32 recommended)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        self.dtype = dtype
        self.input_size = SAPIENS_INPUT_SIZE  # (H, W) = (1024, 768)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Detect model type based on output keypoints
        self.num_keypoints = self._detect_num_keypoints()
        self.model_type = self._get_model_type()
        
        logger.info(f"Sapiens estimator initialized on {self.device}")
        logger.info(f"Model type: {self.model_type.name} ({self.num_keypoints} keypoints)")
    
    def _load_model(self, model_path: str) -> torch.jit.ScriptModule:
        """Load TorchScript model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"Loading Sapiens model from: {model_path}")
        
        try:
            # Load TorchScript model
            model = torch.jit.load(model_path, map_location='cpu')
            model = model.eval()
            model = model.to(self.device).to(self.dtype)
            logger.info("TorchScript model loaded successfully")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _detect_num_keypoints(self) -> int:
        """Detect number of keypoints by running a dummy forward pass."""
        try:
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, *self.input_size, 
                                         device=self.device, dtype=self.dtype)
                output = self.model(dummy_input)
                
                # Output shape is (B, num_keypoints, H/4, W/4)
                if output.dim() == 4:
                    return output.shape[1]
                else:
                    logger.warning(f"Unexpected output shape: {output.shape}")
                    return 17  # Default to COCO
        except Exception as e:
            logger.warning(f"Could not detect keypoints, defaulting to 17: {e}")
            return 17
    
    def _get_model_type(self) -> SapiensModelType:
        """Determine model type based on number of keypoints."""
        if self.num_keypoints == 17:
            return SapiensModelType.COCO_17
        elif self.num_keypoints == 133:
            return SapiensModelType.COCO_133
        elif self.num_keypoints >= 308:
            return SapiensModelType.GOLIATH_308
        else:
            logger.warning(f"Unknown keypoint count: {self.num_keypoints}")
            return SapiensModelType.COCO_17
    
    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Preprocess image for Sapiens model.
        
        Args:
            image: Input BGR image (H, W, C)
            
        Returns:
            Tuple of (tensor, transform_info)
        """
        orig_h, orig_w = image.shape[:2]
        target_h, target_w = self.input_size
        
        # Resize to model input size (1024, 768)
        # Note: cv2.resize takes (width, height)
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to CHW format
        img = resized.transpose(2, 0, 1).astype(np.float32)
        
        # BGR to RGB (swap channels) - Sapiens expects BGR in specific order
        # Actually, looking at the code, it expects BGR input but swaps to RGB internally
        # The preprocessing does: img[[2,1,0], ...] which converts BGR to RGB
        img = img[[2, 1, 0], ...]  # BGR -> RGB
        
        # Normalize
        mean = SAPIENS_MEAN.reshape(3, 1, 1)
        std = SAPIENS_STD.reshape(3, 1, 1)
        img = (img - mean) / std
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(img).unsqueeze(0)
        tensor = tensor.to(self.device).to(self.dtype)
        
        transform_info = {
            'orig_size': (orig_h, orig_w),
            'input_size': (target_h, target_w),
            'scale_x': orig_w / target_w,
            'scale_y': orig_h / target_h,
        }
        
        return tensor, transform_info
    
    def postprocess(
        self,
        heatmaps: torch.Tensor,
        transform_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert heatmaps to keypoint coordinates using UDP decoding.
        
        Args:
            heatmaps: Model output heatmaps (B, K, H, W)
            transform_info: Preprocessing transform information
            
        Returns:
            Tuple of (keypoints, scores)
            - keypoints: shape (K, 3) with [x, y, confidence] in original image coords
            - scores: shape (K,) with raw confidence scores
        """
        # Convert to numpy
        heatmaps_np = heatmaps[0].cpu().float().numpy()  # (K, H, W)
        
        # Get heatmap size
        hm_h, hm_w = heatmaps_np.shape[1:]
        input_h, input_w = transform_info['input_size']
        
        # UDP decode to get keypoints in input image coordinates
        keypoints, scores = udp_decode(
            heatmaps_np,
            input_size=(input_h, input_w),
            heatmap_size=(hm_h, hm_w),
            blur_kernel_size=11
        )
        
        # Remove batch dimension
        keypoints = keypoints[0]  # (K, 2)
        scores = scores[0]  # (K,)
        
        # Scale keypoints to original image coordinates
        scale_x = transform_info['scale_x']
        scale_y = transform_info['scale_y']
        
        keypoints[:, 0] *= scale_x
        keypoints[:, 1] *= scale_y
        
        # Combine into (K, 3) format with [x, y, confidence]
        keypoints_with_conf = np.concatenate([
            keypoints,
            scores[:, np.newaxis]
        ], axis=1)
        
        return keypoints_with_conf, scores
    
    @torch.no_grad()
    def estimate(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate pose keypoints from image.
        
        Args:
            image: Input BGR image (H, W, C)
            
        Returns:
            Tuple of (keypoints, scores)
            - keypoints: shape (K, 3) with [x, y, confidence]
            - scores: shape (K,) raw confidence scores
        """
        if image is None or image.size == 0:
            return (np.zeros((self.num_keypoints, 3), dtype=np.float32),
                    np.zeros(self.num_keypoints, dtype=np.float32))
        
        try:
            # Preprocess
            tensor, transform_info = self.preprocess(image)
            
            # Inference
            with torch.inference_mode():
                heatmaps = self.model(tensor)
            
            # Postprocess
            keypoints, scores = self.postprocess(heatmaps, transform_info)
            
            return keypoints, scores
            
        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return (np.zeros((self.num_keypoints, 3), dtype=np.float32),
                    np.zeros(self.num_keypoints, dtype=np.float32))
    
    def get_coco_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract COCO 17 keypoints from any model output.
        
        For Goliath models, extracts the first 17 body keypoints.
        For COCO models, returns as-is.
        
        Args:
            keypoints: Full keypoint array (K, 3)
            
        Returns:
            COCO keypoints (17, 3)
        """
        if self.model_type == SapiensModelType.COCO_17:
            return keypoints
        
        # For Goliath, first 17 keypoints roughly correspond to COCO
        # Note: The mapping may not be exact, but body keypoints are first
        return keypoints[:17]
    
    def get_goliath_keypoints(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract COCO 17 keypoints from any model output.
        
        For Goliath models, maps keypoints to COCO format using proper indices.
        For COCO models, returns as-is.
        
        Args:
            keypoints: Full keypoint array (K, 3)
            
        Returns:
            COCO keypoints (17, 3)
        """
        if self.model_type == SapiensModelType.COCO_17:
            return keypoints
        
        # For Goliath, extract keypoints using the correct mapping
        coco_keypoints = np.zeros((17, 3), dtype=keypoints.dtype)
        for coco_idx in range(17):
            goliath_idx = GOLIATH_TO_COCO_MAPPING[coco_idx]
            if goliath_idx < len(keypoints):
                coco_keypoints[coco_idx] = keypoints[goliath_idx]
            else:
                # If index out of bounds, set confidence to 0
                coco_keypoints[coco_idx] = [0, 0, 0]
        
        return coco_keypoints
    
    def draw_pose(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        conf_threshold: float = 0.3,
        skeleton_color: Tuple[int, int, int] = (0, 255, 0),
        keypoint_color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2,
        radius: int = 4,
        draw_skeleton: bool = True,
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
            for i, j in COCO_SKELETON:
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


# ============================================================================
# Standalone Usage
# ============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python sapiens_pose.py <model_path> <image_path>")
        print("\nSupported models (TorchScript .pt2 files):")
        print("  - sapiens_1b_coco_best_coco_AP_821_torchscript.pt2 (17 keypoints)")
        print("  - sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2 (308 keypoints)")
        print("\nDownload from: https://huggingface.co/facebook/sapiens-pose-1b-torchscript")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        sys.exit(1)
    
    # Initialize estimator
    print(f"Loading model from: {model_path}")
    estimator = SapiensPoseEstimator(model_path)
    
    # Estimate pose
    print("Estimating pose...")
    keypoints, scores = estimator.estimate(image)
    
    # Print results
    print(f"\nDetected {len(keypoints)} keypoints")
    print("\nFirst 17 keypoints (COCO format):")
    coco_kpts = estimator.get_coco_keypoints(keypoints)
    for i, name in enumerate(COCO_KEYPOINTS):
        x, y, conf = coco_kpts[i]
        print(f"  {name:15s}: ({x:7.1f}, {y:7.1f}) conf={conf:.3f}")
    
    # Draw and save
    result = estimator.draw_pose(image, coco_kpts)
    output_path = "pose_result.jpg"
    cv2.imwrite(output_path, result)
    print(f"\nSaved result to: {output_path}")
