import os
import cv2
import numpy as np

def random_colors(N, bright=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    colors = np.random.rand(N, 3)
    if bright:
        colors = np.clip(colors * 0.8 + 0.2, 0, 1)  # avoid very dark colors
    return colors

def draw_contours(image, masks, color=(255,255,255), thickness=2):
    img = image.copy()
    for m in masks:
        cnts, _ = cv2.findContours(
            m.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, cnts, -1, color, thickness)
    return img

def overlay_masks_rcnn(image, masks, alpha=0.5, seed=42):
    """
    image: (H, W, 3) RGB uint8 or float
    masks: (N, H, W) bool or {0,1}
    """
    img = image.copy().astype(float) / 255.0
    N = masks.shape[0]
    colors = random_colors(N, seed=seed)

    for i in range(N):
        mask = masks[i].astype(bool)
        color = colors[i]

        for c in range(3):
            img[..., c] = np.where(
                mask,
                img[..., c] * (1 - alpha) + alpha * color[c],
                img[..., c]
            )

    return (img * 255).astype("uint8")


class VideoFrameExtractor:
    def __init__(self, video_path):
        """
        Initializes the video capture object and loads metadata.
        """
        self.video_path = video_path
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"File not found: {video_path}")

        # Initialize Video Capture
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video file.")

        # Get total frame count efficiently from metadata
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self._print_video_info()

    def _print_video_info(self):
        """Helper to display video stats upon loading."""
        print("\n" + "="*30)
        print(f" VIDEO LOADED SUCCESSFULLY")
        print(f" Total Frames: {self.total_frames}")
        print(f" FPS:          {self.fps:.2f}")
        print(f" Valid Index:  0 to {self.total_frames - 1}")
        print("="*30 + "\n")

    def extract_frame(self, frame_index, output_name=None):
        """
        Seeks to the specific frame index and saves it to disk.
        """
        # Validate index
        if frame_index < 0 or frame_index >= self.total_frames:
            print(f"[Error] Index {frame_index} is out of bounds (0-{self.total_frames - 1}).")
            return False

        # SEEK: Jump directly to the frame index
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        # READ: Decode the specific frame
        success, frame = self.cap.read()

        if success:
            if output_name is None:
                output_name = f"frame_{frame_index}.jpg"
            
            # cv2.imwrite(output_name, frame)
            # print(f"[Success] Frame {frame_index} saved as '{output_name}'")
            return frame
        else:
            print(f"[Error] Could not read frame at index {frame_index}.")
            return False

    def close(self):
        """Release video resources."""
        if self.cap:
            self.cap.release()
            print("Video resources released.")

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

def convert_coco_wholebody_to_ntu(coco_wb_keypoints: np.ndarray, include_confidence: bool = True) -> np.ndarray:
    """
    Convert 133 COCO WholeBody keypoints to 25 NTU-RGB+D keypoints.
    
    COCO WholeBody format (133 keypoints):
    - 0-16: Body (same as COCO 17)
    - 17-22: Feet (left_big_toe, left_small_toe, left_heel, right_big_toe, right_small_toe, right_heel)
    - 23-90: Face (68 landmarks)
    - 91-111: Left hand (21 landmarks, 91=wrist/root)
    - 112-132: Right hand (21 landmarks, 112=wrist/root)
    
    Args:
        coco_wb_keypoints: Array of shape (N, 133, 3) or (133, 3) with [x, y, confidence]
                          or (N, 133, 2) or (133, 2) with [x, y]
        include_confidence: Whether to include confidence in output
    
    Returns:
        Array of shape (N, 25, 3) or (25, 3) with NTU-RGB+D keypoints
        If include_confidence is False, returns (N, 25, 2) or (25, 2)
    """
    # Handle both single frame and batch inputs
    single_frame = len(coco_wb_keypoints.shape) == 2
    if single_frame:
        coco_wb_keypoints = coco_wb_keypoints[np.newaxis, ...]
    
    batch_size = coco_wb_keypoints.shape[0]
    num_joints = coco_wb_keypoints.shape[1]
    has_conf = coco_wb_keypoints.shape[2] == 3
    
    assert num_joints == 133, f"Expected 133 COCO WholeBody keypoints, got {num_joints}"
    
    # Ensure we have confidence values
    if not has_conf:
        conf = np.ones((batch_size, 133, 1), dtype=coco_wb_keypoints.dtype)
        coco_wb_keypoints = np.concatenate([coco_wb_keypoints, conf], axis=2)
    
    # Initialize NTU keypoints array
    out_channels = 3 if include_confidence else 2
    ntu_keypoints = np.zeros((batch_size, 25, out_channels), dtype=np.float32)
    
    # COCO WholeBody keypoint indices
    COCO_WB = {
        # Body (0-16)
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16,
        # Feet (17-22)
        'left_big_toe': 17, 'left_small_toe': 18, 'left_heel': 19,
        'right_big_toe': 20, 'right_small_toe': 21, 'right_heel': 22,
        # Face starts at 23 (68 landmarks)
        # Left hand (91-111): 91 is wrist/root
        'left_hand_root': 91,
        'left_thumb1': 92, 'left_thumb2': 93, 'left_thumb3': 94, 'left_thumb4': 95,
        'left_forefinger1': 96, 'left_forefinger2': 97, 'left_forefinger3': 98, 'left_forefinger4': 99,
        'left_middle_finger1': 100, 'left_middle_finger2': 101, 'left_middle_finger3': 102, 'left_middle_finger4': 103,
        'left_ring_finger1': 104, 'left_ring_finger2': 105, 'left_ring_finger3': 106, 'left_ring_finger4': 107,
        'left_pinky_finger1': 108, 'left_pinky_finger2': 109, 'left_pinky_finger3': 110, 'left_pinky_finger4': 111,
        # Right hand (112-132): 112 is wrist/root
        'right_hand_root': 112,
        'right_thumb1': 113, 'right_thumb2': 114, 'right_thumb3': 115, 'right_thumb4': 116,
        'right_forefinger1': 117, 'right_forefinger2': 118, 'right_forefinger3': 119, 'right_forefinger4': 120,
        'right_middle_finger1': 121, 'right_middle_finger2': 122, 'right_middle_finger3': 123, 'right_middle_finger4': 124,
        'right_ring_finger1': 125, 'right_ring_finger2': 126, 'right_ring_finger3': 127, 'right_ring_finger4': 128,
        'right_pinky_finger1': 129, 'right_pinky_finger2': 130, 'right_pinky_finger3': 131, 'right_pinky_finger4': 132,
    }
    
    for i in range(batch_size):
        coco = coco_wb_keypoints[i]
        ntu = np.zeros((25, 3), dtype=np.float32)
        
        # Direct mappings from COCO WholeBody to NTU
        ntu[3] = coco[COCO_WB['nose']]              # head <- nose
        ntu[4] = coco[COCO_WB['left_shoulder']]     # left_shoulder
        ntu[5] = coco[COCO_WB['left_elbow']]        # left_elbow  
        ntu[6] = coco[COCO_WB['left_wrist']]        # left_wrist
        ntu[8] = coco[COCO_WB['right_shoulder']]    # right_shoulder
        ntu[9] = coco[COCO_WB['right_elbow']]       # right_elbow
        ntu[10] = coco[COCO_WB['right_wrist']]      # right_wrist
        ntu[12] = coco[COCO_WB['left_hip']]         # left_hip
        ntu[13] = coco[COCO_WB['left_knee']]        # left_knee
        ntu[14] = coco[COCO_WB['left_ankle']]       # left_ankle
        ntu[16] = coco[COCO_WB['right_hip']]        # right_hip
        ntu[17] = coco[COCO_WB['right_knee']]       # right_knee
        ntu[18] = coco[COCO_WB['right_ankle']]      # right_ankle
        
        # Derived keypoints for spine
        left_hip = coco[COCO_WB['left_hip']]
        right_hip = coco[COCO_WB['right_hip']]
        left_shoulder = coco[COCO_WB['left_shoulder']]
        right_shoulder = coco[COCO_WB['right_shoulder']]
        nose = coco[COCO_WB['nose']]
        
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
        
        # Left hand (7): Use left hand root or average of finger bases
        left_hand_root = coco[COCO_WB['left_hand_root']]
        if left_hand_root[2] > 0:
            # Average of finger base joints for better hand center
            finger_bases = [
                coco[COCO_WB['left_thumb1']],
                coco[COCO_WB['left_forefinger1']],
                coco[COCO_WB['left_middle_finger1']],
                coco[COCO_WB['left_ring_finger1']],
                coco[COCO_WB['left_pinky_finger1']]
            ]
            valid_bases = [fb for fb in finger_bases if fb[2] > 0]
            if len(valid_bases) > 0:
                ntu[7, :2] = np.mean([fb[:2] for fb in valid_bases], axis=0)
                ntu[7, 2] = np.mean([fb[2] for fb in valid_bases])
            else:
                ntu[7] = left_hand_root
        else:
            ntu[7] = ntu[6].copy()  # fallback to wrist
        
        # Right hand (11): Use right hand root or average of finger bases
        right_hand_root = coco[COCO_WB['right_hand_root']]
        if right_hand_root[2] > 0:
            finger_bases = [
                coco[COCO_WB['right_thumb1']],
                coco[COCO_WB['right_forefinger1']],
                coco[COCO_WB['right_middle_finger1']],
                coco[COCO_WB['right_ring_finger1']],
                coco[COCO_WB['right_pinky_finger1']]
            ]
            valid_bases = [fb for fb in finger_bases if fb[2] > 0]
            if len(valid_bases) > 0:
                ntu[11, :2] = np.mean([fb[:2] for fb in valid_bases], axis=0)
                ntu[11, 2] = np.mean([fb[2] for fb in valid_bases])
            else:
                ntu[11] = right_hand_root
        else:
            ntu[11] = ntu[10].copy()  # fallback to wrist
        
        # Left foot (15): Use foot keypoints - average of big_toe, small_toe, heel
        left_big_toe = coco[COCO_WB['left_big_toe']]
        left_small_toe = coco[COCO_WB['left_small_toe']]
        left_heel = coco[COCO_WB['left_heel']]
        
        foot_pts = [left_big_toe, left_small_toe, left_heel]
        valid_foot = [pt for pt in foot_pts if pt[2] > 0]
        if len(valid_foot) > 0:
            ntu[15, :2] = np.mean([pt[:2] for pt in valid_foot], axis=0)
            ntu[15, 2] = np.mean([pt[2] for pt in valid_foot])
        else:
            ntu[15] = ntu[14].copy()  # fallback to ankle
        
        # Right foot (19): Use foot keypoints
        right_big_toe = coco[COCO_WB['right_big_toe']]
        right_small_toe = coco[COCO_WB['right_small_toe']]
        right_heel = coco[COCO_WB['right_heel']]
        
        foot_pts = [right_big_toe, right_small_toe, right_heel]
        valid_foot = [pt for pt in foot_pts if pt[2] > 0]
        if len(valid_foot) > 0:
            ntu[19, :2] = np.mean([pt[:2] for pt in valid_foot], axis=0)
            ntu[19, 2] = np.mean([pt[2] for pt in valid_foot])
        else:
            ntu[19] = ntu[18].copy()  # fallback to ankle
        
        # Left hand tip (21): Use middle finger tip (most extended)
        left_middle_tip = coco[COCO_WB['left_middle_finger4']]
        if left_middle_tip[2] > 0:
            ntu[21] = left_middle_tip
        elif coco[COCO_WB['left_forefinger4']][2] > 0:
            ntu[21] = coco[COCO_WB['left_forefinger4']]
        else:
            # Extrapolate from wrist-elbow direction
            left_wrist = coco[COCO_WB['left_wrist']]
            left_elbow = coco[COCO_WB['left_elbow']]
            if left_wrist[2] > 0 and left_elbow[2] > 0:
                direction = left_wrist[:2] - left_elbow[:2]
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction = direction / norm
                    ntu[21, :2] = left_wrist[:2] + direction * 15
                    ntu[21, 2] = left_wrist[2] * 0.8
        
        # Left thumb (22): Use thumb tip
        left_thumb_tip = coco[COCO_WB['left_thumb4']]
        if left_thumb_tip[2] > 0:
            ntu[22] = left_thumb_tip
        else:
            # Extrapolate
            left_wrist = coco[COCO_WB['left_wrist']]
            left_elbow = coco[COCO_WB['left_elbow']]
            if left_wrist[2] > 0 and left_elbow[2] > 0:
                direction = left_wrist[:2] - left_elbow[:2]
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction = direction / norm
                    perp = np.array([-direction[1], direction[0]])
                    ntu[22, :2] = left_wrist[:2] + direction * 8 + perp * 5
                    ntu[22, 2] = left_wrist[2] * 0.7
        
        # Right hand tip (23): Use middle finger tip
        right_middle_tip = coco[COCO_WB['right_middle_finger4']]
        if right_middle_tip[2] > 0:
            ntu[23] = right_middle_tip
        elif coco[COCO_WB['right_forefinger4']][2] > 0:
            ntu[23] = coco[COCO_WB['right_forefinger4']]
        else:
            right_wrist = coco[COCO_WB['right_wrist']]
            right_elbow = coco[COCO_WB['right_elbow']]
            if right_wrist[2] > 0 and right_elbow[2] > 0:
                direction = right_wrist[:2] - right_elbow[:2]
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction = direction / norm
                    ntu[23, :2] = right_wrist[:2] + direction * 15
                    ntu[23, 2] = right_wrist[2] * 0.8
        
        # Right thumb (24): Use thumb tip
        right_thumb_tip = coco[COCO_WB['right_thumb4']]
        if right_thumb_tip[2] > 0:
            ntu[24] = right_thumb_tip
        else:
            right_wrist = coco[COCO_WB['right_wrist']]
            right_elbow = coco[COCO_WB['right_elbow']]
            if right_wrist[2] > 0 and right_elbow[2] > 0:
                direction = right_wrist[:2] - right_elbow[:2]
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction = direction / norm
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


def draw_pose_ntu(
    image: np.ndarray,
    keypoints: np.ndarray,
    conf_threshold: float = 0.3,
    skeleton_color = (255, 191, 0),
    keypoint_color = (255, 85, 51),
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
        for i, j in NTU_SKELETON_CONNECTIONS:
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