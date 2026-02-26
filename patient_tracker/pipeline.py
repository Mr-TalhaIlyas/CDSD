#!/usr/bin/env python3
"""
Seizure Detection Pipeline: Patient Isolation + Pose Estimation
================================================================
This pipeline combines:
1. SAM3 (Segment Anything Model 3) for text-based patient tracking in hospital videos
2. Sapiens (TorchScript lite) for robust pose estimation on isolated patient frames

Designed for hospital CCTV footage (RGB daytime + IR nighttime).

Author: Talha
Date: 2025
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Generator
from dataclasses import dataclass, field
import warnings

import numpy as np
import cv2
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
# Import Sapiens pose estimator from our module
from sapiens_pose import SapiensPoseEstimator, COCO_KEYPOINTS, COCO_SKELETON

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress some warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the seizure detection pipeline."""
    
    # Model weights paths (local)
    weights_dir: str = "./weights"
    sam3_model_path: str = None  # Will be set to weights_dir/sam3.pt if None
    sapiens_model_path: str = None  # Will be set to weights_dir/sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2 if None
    
    # SAM3 configuration
    sam3_conf: float = 0.25  # Confidence threshold
    sam3_imgsz: int = 640  # Input image size for SAM3
    patient_prompts: List[str] = field(default_factory=lambda: ['patient in center of frame',
                                                                "patient on bed in striped clothes"])
    
    # Sapiens configuration  
    sapiens_dtype: torch.dtype = torch.float32
    
    # Processing configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 1
    
    # Output configuration
    output_dir: str = "./output"
    save_npy: bool = True
    save_video: bool = True
    video_fps: float = 30.0
    
    # Edge case handling
    min_patient_area_ratio: float = 0.01  # Minimum patient area as ratio of frame
    max_patient_area_ratio: float = 0.95  # Maximum patient area as ratio of frame
    keypoint_conf_threshold: float = 0.3  # Minimum keypoint confidence
    
    # Visualization
    skeleton_thickness: int = 2
    keypoint_radius: int = 4
    
    def __post_init__(self):
        """Set default model paths based on weights_dir."""
        if self.sam3_model_path is None:
            self.sam3_model_path = os.path.join(self.weights_dir, "sam3.pt")
        if self.sapiens_model_path is None:
            # Try common model file names
            possible_names = [
                "sapiens_1b_goliath_best_goliath_AP_639_torchscript.pt2",
                "sapiens_1b_coco_best_coco_AP_821_torchscript.pt2",
                "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2",
                "sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2",
            ]
            for name in possible_names:
                path = os.path.join(self.weights_dir, name)
                if os.path.exists(path):
                    self.sapiens_model_path = path
                    break
            else:
                # Default to first option even if not found
                self.sapiens_model_path = os.path.join(self.weights_dir, possible_names[0])


# Colors for visualization (BGR format)
KEYPOINT_COLORS = {
    'head': (0, 255, 255),      # Yellow
    'arm_left': (0, 165, 255),   # Orange
    'arm_right': (0, 255, 0),    # Green
    'torso': (255, 0, 0),        # Blue
    'leg_left': (255, 0, 255),   # Magenta
    'leg_right': (255, 255, 0),  # Cyan
}

def get_keypoint_color(idx: int) -> Tuple[int, int, int]:
    """Get color for a keypoint based on body part."""
    if idx <= 4:
        return KEYPOINT_COLORS['head']
    elif idx in [5, 7, 9]:
        return KEYPOINT_COLORS['arm_left']
    elif idx in [6, 8, 10]:
        return KEYPOINT_COLORS['arm_right']
    elif idx in [11, 12]:
        return KEYPOINT_COLORS['torso']
    elif idx in [13, 15]:
        return KEYPOINT_COLORS['leg_left']
    else:
        return KEYPOINT_COLORS['leg_right']


# ============================================================================
# SAM3 Patient Tracker
# ============================================================================

class SAM3PatientTracker:
    """
    SAM3-based patient tracking using text prompts.
    Uses Ultralytics integration for semantic video segmentation.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.predictor = None
        self._setup_model()
    
    def _setup_model(self):
        """Initialize SAM3 video semantic predictor."""
        # Verify model file exists
        if not os.path.exists(self.config.sam3_model_path):
            raise FileNotFoundError(
                f"SAM3 model not found at: {self.config.sam3_model_path}\n"
                f"Please place sam3.pt in the weights directory."
            )
        
        try:
            from ultralytics.models.sam import SAM3VideoSemanticPredictor
            
            overrides = dict(
                conf=self.config.sam3_conf,
                task="segment",
                mode="predict",
                imgsz=self.config.sam3_imgsz,
                model=self.config.sam3_model_path,
                half=True if self.config.device == "cuda" else False,
                save=False,
                verbose=False,
            )
            
            self.predictor = SAM3VideoSemanticPredictor(overrides=overrides)
            logger.info(f"SAM3 model loaded from: {self.config.sam3_model_path}")
            logger.info(f"Running on device: {self.config.device}")
            
        except ImportError as e:
            logger.error(f"Failed to import SAM3: {e}")
            logger.error("Please install ultralytics >= 8.3.237: pip install -U ultralytics")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize SAM3: {e}")
            raise
    
    def track_patient_in_video(
        self,
        video_path: str,
        stream: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Track patient in video using text prompts.
        
        Args:
            video_path: Path to input video
            stream: Whether to stream results
            
        Yields:
            Dictionary containing:
                - frame_idx: Frame index
                - frame: Original frame (numpy array)
                - mask: Binary segmentation mask (numpy array or None)
                - bbox: Bounding box [x1, y1, x2, y2] or None
                - confidence: Tracking confidence
                - isolated_patient: Cropped patient on black background
        """
        logger.info(f"Starting patient tracking in: {video_path}")
        logger.info(f"Using prompts: {self.config.patient_prompts}")
        
        try:
            # Reinitialize predictor for each video to avoid state issues
            self._setup_model()
            # Run SAM3 with text prompts
            results = self.predictor(
                source=video_path,
                text=self.config.patient_prompts,
                stream=stream
            )
            
            frame_idx = 0
            for result in results:
                output = {
                    'frame_idx': frame_idx,
                    'frame': result.orig_img,
                    'mask': None,
                    'bbox': None,
                    'confidence': 0.0,
                    'isolated_patient': None,
                }
                
                # Extract mask and bbox if detection exists
                if result.masks is not None and len(result.masks) > 0:
                    # Get the best mask (highest confidence for patient)
                    mask_data = result.masks.data.cpu().numpy()
                    
                    if len(mask_data) > 0:
                        # Merge all masks using logical OR (union of all detections)
                        mask = np.zeros_like(mask_data[0], dtype=np.float32)
                        for single_mask in mask_data:
                            # Resize each mask to original frame size if needed
                            if single_mask.shape != result.orig_img.shape[:2]:
                                single_mask = cv2.resize(
                                    single_mask.astype(np.float32),
                                    (result.orig_img.shape[1], result.orig_img.shape[0]),
                                    interpolation=cv2.INTER_NEAREST
                                )
                            # Merge using maximum (union)
                            mask = np.maximum(mask, single_mask)
                        # Use first mask (best match for patient prompt)
                        # mask = mask_data[0]
                        # # Resize mask to original frame size if needed
                        # if mask.shape != result.orig_img.shape[:2]:
                        #     mask = cv2.resize(
                        #         mask.astype(np.float32),
                        #         (result.orig_img.shape[1], result.orig_img.shape[0]),
                        #         interpolation=cv2.INTER_NEAREST
                        #     )
                        
                        mask = (mask > 0.5).astype(np.uint8)
                        
                        # Validate mask area
                        frame_area = mask.shape[0] * mask.shape[1]
                        mask_area = np.sum(mask)
                        area_ratio = mask_area / frame_area
                        
                        if (self.config.min_patient_area_ratio <= area_ratio <= 
                            self.config.max_patient_area_ratio):
                            
                            output['mask'] = mask
                            
                            # Get bounding box
                            if result.boxes is not None and len(result.boxes) > 0:
                                bbox = result.boxes.xyxy[0].cpu().numpy()
                                output['bbox'] = bbox.astype(int)
                                output['confidence'] = float(result.boxes.conf[0].cpu())
                            else:
                                # Compute bbox from mask
                                output['bbox'] = self._mask_to_bbox(mask)
                                output['confidence'] = 1.0
                            
                            # Create isolated patient (black background)
                            output['isolated_patient'] = self._isolate_patient(
                                result.orig_img, mask, output['bbox']
                            )
                        else:
                            logger.debug(
                                f"Frame {frame_idx}: Mask area ratio {area_ratio:.3f} "
                                f"outside valid range"
                            )
                
                yield output
                frame_idx += 1
                
        except Exception as e:
            logger.error(f"Error during patient tracking: {e}")
            raise
    
    def _mask_to_bbox(self, mask: np.ndarray) -> np.ndarray:
        """Convert binary mask to bounding box."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return np.array([0, 0, mask.shape[1], mask.shape[0]])
        
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        return np.array([x_min, y_min, x_max, y_max])
    
    def _isolate_patient(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        bbox: np.ndarray,
        padding: int = 20
    ) -> np.ndarray:
        """
        Isolate patient against black background with padding.
        
        Args:
            frame: Original frame
            mask: Binary segmentation mask
            bbox: Bounding box [x1, y1, x2, y2]
            padding: Padding around bbox
            
        Returns:
            Isolated patient image (cropped region with mask applied)
        """
        h, w = frame.shape[:2]
        
        # Expand bbox with padding
        x1 = max(0, int(bbox[0]) - padding)
        y1 = max(0, int(bbox[1]) - padding)
        x2 = min(w, int(bbox[2]) + padding)
        y2 = min(h, int(bbox[3]) + padding)
        
        # Crop frame and mask to bbox region
        cropped_frame = frame[y1:y2, x1:x2].copy()
        cropped_mask = mask[y1:y2, x1:x2]
        
        # Apply mask (black background)
        mask_3ch = np.stack([cropped_mask] * 3, axis=-1)
        isolated = cropped_frame * mask_3ch
        
        return isolated


# ============================================================================
# Visualization
# ============================================================================

class Visualizer:
    """Visualization utilities for the pipeline."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def draw_segmentation_overlay(
        self,
        frame: np.ndarray,
        mask: Optional[np.ndarray],
        alpha: float = 0.5,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Draw segmentation mask overlay on frame."""
        result = frame.copy()
        
        if mask is not None:
            overlay = result.copy()
            overlay[mask > 0] = color
            result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
            
            # Draw contour
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, color, 2)
        
        return result
    
    def draw_pose(
        self,
        image: np.ndarray,
        keypoints: Optional[np.ndarray],
        draw_skeleton: bool = True
    ) -> np.ndarray:
        """Draw pose keypoints and skeleton on image."""
        result = image.copy()
        
        if keypoints is None or len(keypoints) == 0:
            return result
        
        # Only use first 17 keypoints for skeleton (COCO format)
        num_kpts = min(len(keypoints), 17)
        
        # Draw skeleton first (so keypoints are on top)
        if draw_skeleton:
            for i, j in COCO_SKELETON:
                if i < num_kpts and j < num_kpts:
                    if (keypoints[i, 2] > self.config.keypoint_conf_threshold and
                        keypoints[j, 2] > self.config.keypoint_conf_threshold):
                        
                        pt1 = tuple(keypoints[i, :2].astype(int))
                        pt2 = tuple(keypoints[j, :2].astype(int))
                        
                        color = get_keypoint_color(i)
                        cv2.line(
                            result, pt1, pt2, color,
                            self.config.skeleton_thickness
                        )
        
        # Draw keypoints (first 17 for visualization)
        for i in range(num_kpts):
            x, y, conf = keypoints[i]
            if conf > self.config.keypoint_conf_threshold:
                color = get_keypoint_color(i)
                cv2.circle(
                    result,
                    (int(x), int(y)),
                    self.config.keypoint_radius,
                    color,
                    -1
                )
                # Draw outline
                cv2.circle(
                    result,
                    (int(x), int(y)),
                    self.config.keypoint_radius,
                    (255, 255, 255),
                    1
                )
        
        return result
    
    def create_side_by_side(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        target_height: int = 720
    ) -> np.ndarray:
        """
        Create side-by-side visualization.
        Left: Full frame with segmentation overlay
        Right: Zoomed isolated patient with pose overlay
        """
        # Resize left image to target height
        h_left, w_left = left_image.shape[:2]
        scale_left = target_height / h_left
        new_w_left = int(w_left * scale_left)
        left_resized = cv2.resize(left_image, (new_w_left, target_height))
        
        # Resize right image to target height (zoomed view)
        if right_image is not None and right_image.size > 0:
            h_right, w_right = right_image.shape[:2]
            if h_right > 0 and w_right > 0:
                scale_right = target_height / h_right
                new_w_right = int(w_right * scale_right)
                # Limit width to reasonable size
                max_right_width = int(new_w_left * 0.6)
                if new_w_right > max_right_width:
                    new_w_right = max_right_width
                right_resized = cv2.resize(right_image, (new_w_right, target_height))
            else:
                right_resized = None
        else:
            right_resized = None
        
        if right_resized is None:
            # Create placeholder
            new_w_right = int(new_w_left * 0.4)
            right_resized = np.zeros((target_height, new_w_right, 3), dtype=np.uint8)
            cv2.putText(
                right_resized,
                "No Patient Detected",
                (10, target_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (128, 128, 128),
                2
            )
        
        # Add border between images
        border = np.ones((target_height, 5, 3), dtype=np.uint8) * 128
        
        # Concatenate
        combined = np.concatenate([left_resized, border, right_resized], axis=1)
        
        # Add labels
        cv2.putText(
            combined, "Full Frame + Segmentation",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        cv2.putText(
            combined, "Isolated Patient + Pose",
            (new_w_left + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        
        return combined


# ============================================================================
# Main Pipeline
# ============================================================================

class SeizureDetectionPipeline:
    """
    Main pipeline for seizure detection preprocessing.
    Combines SAM3 tracking and Sapiens pose estimation.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        logger.info("Initializing SAM3 patient tracker...")
        self.tracker = SAM3PatientTracker(config)
        
        logger.info("Initializing Sapiens pose estimator...")
        self.pose_estimator = SapiensPoseEstimator(
            model_path=config.sapiens_model_path,
            device=config.device,
            dtype=config.sapiens_dtype,
        )
        
        logger.info("Initializing visualizer...")
        self.visualizer = Visualizer(config)
        
        logger.info("Pipeline initialized successfully!")
    
    def process_video(
        self,
        video_path: str,
        output_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a video file through the full pipeline.
        
        Args:
            video_path: Path to input video
            output_name: Base name for output files (default: input video name)
            
        Returns:
            Dictionary with processing results and output paths
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        if output_name is None:
            output_name = video_path.stem
        
        logger.info(f"Processing video: {video_path}")
        
        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.config.video_fps
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS, {frame_width}x{frame_height}")
        
        # Prepare outputs
        all_keypoints = []
        all_confidences = []
        video_writer = None
        output_video_path = None
        
        # Get number of keypoints from pose estimator
        num_keypoints = self.pose_estimator.num_keypoints
        
        # Process frames
        pbar = tqdm(
            self.tracker.track_patient_in_video(str(video_path)),
            total=total_frames,
            desc="Processing"
        )
        
        for result in pbar:
            frame_idx = result['frame_idx']
            frame = result['frame']
            mask = result['mask']
            bbox = result['bbox']
            isolated = result['isolated_patient']
            
            # Estimate pose on isolated patient
            keypoints = None
            if isolated is not None and isolated.size > 0:
                keypoints, scores = self.pose_estimator.estimate(isolated)
                
                # Get COCO keypoints (first 17) for compatibility
                keypoints = self.pose_estimator.get_coco_keypoints(keypoints)
                
                # Transform keypoints back to original frame coordinates
                if keypoints is not None and bbox is not None:
                    # Add bbox offset to keypoints (with padding consideration)
                    padding = 20
                    x_offset = max(0, int(bbox[0]) - padding)
                    y_offset = max(0, int(bbox[1]) - padding)
                    keypoints[:, 0] += x_offset
                    keypoints[:, 1] += y_offset
            
            # Store keypoints
            if keypoints is not None:
                all_keypoints.append(keypoints)
                all_confidences.append(result['confidence'])
            else:
                # Append empty keypoints for this frame
                all_keypoints.append(np.zeros((17, 3), dtype=np.float32))
                all_confidences.append(0.0)
            
            # Create visualization
            if self.config.save_video:
                # Left: Full frame with segmentation overlay
                left_vis = self.visualizer.draw_segmentation_overlay(frame, mask)
                
                # Also draw pose on full frame
                if keypoints is not None:
                    left_vis = self.visualizer.draw_pose(left_vis, keypoints)
                
                # Right: Isolated patient with pose overlay (in cropped coordinates)
                if isolated is not None and isolated.size > 0:
                    right_vis = isolated.copy()
                    if keypoints is not None:
                        # Create keypoints in isolated image coordinates
                        isolated_keypoints = keypoints.copy()
                        padding = 20
                        x_offset = max(0, int(bbox[0]) - padding)
                        y_offset = max(0, int(bbox[1]) - padding)
                        isolated_keypoints[:, 0] -= x_offset
                        isolated_keypoints[:, 1] -= y_offset
                        right_vis = self.visualizer.draw_pose(right_vis, isolated_keypoints)
                else:
                    right_vis = None
                
                # Combine
                combined = self.visualizer.create_side_by_side(left_vis, right_vis)
                combined = cv2.resize(combined, (1920, 720))
                plt.imshow(combined[..., ::-1])  # Convert BGR to RGB for matplotlib
                # Initialize video writer if needed
                if video_writer is None:
                    output_video_path = self.output_dir / f"{output_name}_visualization.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        str(output_video_path),
                        fourcc,
                        fps,
                        (combined.shape[1], combined.shape[0])
                    )
                
                video_writer.write(combined)
            
            # Update progress bar
            pbar.set_postfix({
                'detected': mask is not None,
                'conf': f"{result['confidence']:.2f}"
            })
        
        # Close video writer
        if video_writer is not None:
            video_writer.release()
            logger.info(f"Saved visualization video: {output_video_path}")
        
        # Save keypoints as NPY
        npy_path = None
        if self.config.save_npy:
            keypoints_array = np.array(all_keypoints)  # Shape: (N, 17, 3)
            confidences_array = np.array(all_confidences)  # Shape: (N,)
            
            npy_path = self.output_dir / f"{output_name}_keypoints.npy"
            np.save(str(npy_path), {
                'keypoints': keypoints_array,
                'confidences': confidences_array,
                'keypoint_names': COCO_KEYPOINTS,
                'skeleton': COCO_SKELETON,
                'video_path': str(video_path),
                'fps': fps,
                'total_frames': len(all_keypoints),
                'frame_size': (frame_height, frame_width),
            }, allow_pickle=True)
            logger.info(f"Saved keypoints: {npy_path}")
            logger.info(f"Keypoints shape: {keypoints_array.shape}")
        
        return {
            'video_path': str(video_path),
            'output_video': str(output_video_path) if output_video_path else None,
            'keypoints_npy': str(npy_path) if npy_path else None,
            'total_frames': len(all_keypoints),
            'frames_with_detection': sum(1 for c in all_confidences if c > 0),
        }
    
    def process_directory(
        self,
        video_dir: str,
        video_extensions: List[str] = ['.mp4', '.avi', '.mov', '.mkv']
    ) -> List[Dict[str, Any]]:
        """
        Process all videos in a directory.
        
        Args:
            video_dir: Directory containing video files
            video_extensions: List of video file extensions to process
            
        Returns:
            List of results for each video
        """
        video_dir = Path(video_dir)
        if not video_dir.exists():
            raise FileNotFoundError(f"Directory not found: {video_dir}")
        
        # Find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f"*{ext}"))
            video_files.extend(video_dir.glob(f"*{ext.upper()}"))
        
        video_files = sorted(set(video_files))
        
        if not video_files:
            logger.warning(f"No video files found in: {video_dir}")
            return []
        
        logger.info(f"Found {len(video_files)} videos to process")
        
        results = []
        for video_path in video_files:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {video_path.name}")
            logger.info(f"{'='*60}")
            
            try:
                result = self.process_video(str(video_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {video_path.name}: {e}")
                results.append({
                    'video_path': str(video_path),
                    'error': str(e),
                })
        
        return results


# ============================================================================
# Main entry point
# ============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <video_path_or_directory>")
        print("\nExample:")
        print("  python pipeline.py video.mp4")
        print("  python pipeline.py ./videos/")
        sys.exit(1)
    
    video_input = sys.argv[1]
    
    # Create configuration
    config = PipelineConfig(
        weights_dir="./weights",
        output_dir="./output",
    )
    
    # Initialize pipeline
    pipeline = SeizureDetectionPipeline(config)
    
    # Process
    video_input = Path(video_input)
    if video_input.is_file():
        results = [pipeline.process_video(str(video_input))]
    elif video_input.is_dir():
        results = pipeline.process_directory(str(video_input))
    else:
        print(f"Error: Input not found: {video_input}")
        sys.exit(1)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    
    for result in results:
        if 'error' in result:
            print(f"\n❌ {result['video_path']}: {result['error']}")
        else:
            print(f"\n✓ {result['video_path']}")
            print(f"  Frames: {result['total_frames']}")
            print(f"  Detected: {result['frames_with_detection']}")
            if result['keypoints_npy']:
                print(f"  Keypoints: {result['keypoints_npy']}")
            if result['output_video']:
                print(f"  Video: {result['output_video']}")
