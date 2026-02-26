#%%
import os

from matplotlib import image
import matplotlib
os.chdir('/data_hdd/talha/miccai_26/seizure_detection_pipeline/tracker')

import cv2
from ultralytics.models.sam import SAM3SemanticPredictor
from sapiens_pose_tests import SapiensPoseEstimator, COCO_KEYPOINTS

from IPython.display import display, clear_output
from PIL import Image
import numpy as np
from fmutils import fmutils as fmu
import matplotlib.pyplot as plt
from paper_viz.utils import (random_colors, draw_contours, overlay_masks_rcnn,
                            VideoFrameExtractor, draw_pose_ntu,
                            convert_coco_wholebody_to_ntu, convert_coco_to_ntu,
                            is_infrared_frame, enhance_infrared_frame)

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 500


# Initialize semantic video predictor
overrides = dict(conf=0.2, task="segment", mode="predict", imgsz=644,
                 model="../weights/sam3.pt", half=True, save=False)
model_path = '../weights/sapiens_2b_coco_wholebody_best_coco_wholebody_AP_745_torchscript.pt2'
estimator = SapiensPoseEstimator(model_path)

#%%

# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
video_path = '/data_hdd/talha/miccai_26/seizure_detection_pipeline/videos_raw/vids/pat02_002_Sz1PG.mp4'
extractor = VideoFrameExtractor(video_path)


img = extractor.extract_frame(1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

#%%
if is_infrared_frame(img):
    img = enhance_infrared_frame(img)
    print("Enhanced infrared frame")
#%%
predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image(img)

results = predictor(text=[
                        'patient in center of frame',
                        "patient on bed in striped clothes",
                        'legs of patient on bed',
                        'hands of patient on bed',
                        # 'patient on bed only'
                        ]
                    )


r = results[0]
masks = r.masks.data.cpu().numpy()  # (N, H, W)
# merge masks into one mask
if masks.shape[0] > 1:
    masks = np.any(masks, axis=0).astype(np.uint8)  # (H, W)
    # add channel dimension
    masks = np.expand_dims(masks, axis=0)  # (1, H, W)
vis = overlay_masks_rcnn(img, masks, alpha=0.5)
vis = draw_contours(vis, masks)

# plt.figure(figsize=(8, 8))
# plt.imshow(vis)
# plt.axis("off")
# plt.show()

# result = estimator.draw_pose_wholebody(img, coco_kpts,thickness = 6,
#                                         radius= 8)
# plt.imshow(result)

# center crop image x-axis 500-1500, y-axis/height full
img = img[0:img.shape[0], 500:1500]
vis = vis[0:vis.shape[0], 500:1500]
print("Estimating pose...")
coco_kpts, scores = estimator.estimate(img)

ntu_kpts = convert_coco_wholebody_to_ntu(coco_kpts)
result_ntu = draw_pose_ntu(vis, ntu_kpts, thickness = 6,
                            radius= 8, conf_threshold=0.3)
plt.imshow(result_ntu)
plt.axis("off")
# %%
# =============================================================================
# YOLO Pose Detection (COCO 17 -> NTU) for Comparison
# =============================================================================
from ultralytics import YOLO

result_sapiens = result_ntu.copy()
ntu_kpts_sapiens = ntu_kpts.copy()  # Save for comparison
img_crop = img.copy()
vis_crop = vis.copy()
# Load YOLO pose model (uses 17 COCO keypoints)
print("Loading YOLO pose model...")
yolo_model = YOLO("yolo26x-pose.pt")  # Using YOLO26 pose model

# Run YOLO pose detection on cropped image
print("Estimating pose with YOLO...")
# YOLO expects BGR format
img_crop_bgr = cv2.cvtColor(img_crop, cv2.COLOR_RGB2BGR)
yolo_results = yolo_model(img_crop_bgr, verbose=False)

# Extract keypoints from YOLO results
if len(yolo_results) > 0 and yolo_results[0].keypoints is not None:
    # Get keypoints: shape (num_persons, 17, 3) where 3 = [x, y, confidence]
    yolo_kpts = yolo_results[0].keypoints.data.cpu().numpy()
    
    if len(yolo_kpts) > 0:
        # Take the first detected person (or highest confidence)
        if len(yolo_kpts) > 1:
            # Select person with highest average keypoint confidence
            avg_confs = [kpts[:, 2].mean() for kpts in yolo_kpts]
            best_idx = np.argmax(avg_confs)
            yolo_kpts_single = yolo_kpts[best_idx]
        else:
            yolo_kpts_single = yolo_kpts[0]
        
        print(f"YOLO detected keypoints shape: {yolo_kpts_single.shape}")
        print(f"YOLO average confidence: {yolo_kpts_single[:, 2].mean():.3f}")
        
        # Convert COCO 17 to NTU 25
        ntu_kpts_yolo = convert_coco_to_ntu(yolo_kpts_single, include_confidence=True)
        
        # Draw YOLO pose in NTU format
        result_yolo = draw_pose_ntu(vis_crop.copy(), ntu_kpts_yolo, 
                                     thickness=6, radius=8, conf_threshold=0.3)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(result_yolo)
        plt.title("YOLO Pose (COCO-17 → NTU)")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        
        # Print keypoint comparison
        # print("\n" + "="*50)
        # print("Keypoint Confidence Comparison:")
        # print("="*50)
        # print(f"{'Joint':<20} {'Sapiens':<12} {'YOLO':<12}")
        # print("-"*50)
        
        from paper_viz.utils import NTU_JOINT_NAMES
        for i, joint_name in enumerate(NTU_JOINT_NAMES):
            sapiens_conf = ntu_kpts_sapiens[i, 2] if i < len(ntu_kpts_sapiens) else 0
            yolo_conf = ntu_kpts_yolo[i, 2] if i < len(ntu_kpts_yolo) else 0
            print(f"{joint_name:<20} {sapiens_conf:>6.3f}      {yolo_conf:>6.3f}")
        
        print("-"*50)
        print(f"{'Average':<20} {ntu_kpts_sapiens[:, 2].mean():>6.3f}      {ntu_kpts_yolo[:, 2].mean():>6.3f}")
        print("="*50)
        
    else:
        print("YOLO: No person detected in the image")
        result_yolo = vis_crop.copy()
else:
    print("YOLO: No keypoints detected")
    result_yolo = vis_crop.copy()

#%%
# =============================================================================
# Side-by-Side Comparison
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(20, 10))

axes[0].imshow(result_sapiens)
axes[0].set_title("Sapiens Pose\n(WholeBody 133 → NTU 25)", fontsize=14, fontweight='bold')
axes[0].axis("off")

axes[1].imshow(result_yolo)
axes[1].set_title("YOLO Pose\n(COCO-17 → NTU 25)", fontsize=14, fontweight='bold')
axes[1].axis("off")

plt.tight_layout()
plt.show()

#%%
# =============================================================================
# Overlay Comparison (Optional - both skeletons on same image)
# =============================================================================
# Draw both poses on the same image with different colors
result_combined = vis_crop.copy()

# Draw Sapiens in blue
if ntu_kpts_sapiens is not None:
    result_combined = draw_pose_ntu(result_combined, ntu_kpts_sapiens, 
                                    thickness=4, radius=6, conf_threshold=0.3,
                                    skeleton_color=(0, 100, 255),  # Blue
                                    keypoint_color=(0, 150, 255))

# Draw YOLO in orange
if 'ntu_kpts_yolo' in locals() and ntu_kpts_yolo is not None:
    result_combined = draw_pose_ntu(result_combined, ntu_kpts_yolo, 
                                    thickness=4, radius=6, conf_threshold=0.3,
                                    skeleton_color=(255, 165, 0),  # Orange
                                    keypoint_color=(255, 100, 0))

plt.figure(figsize=(12, 10))
plt.imshow(result_combined)
plt.title("Combined: Sapiens (Blue) vs YOLO (Orange)", fontsize=14, fontweight='bold')
plt.axis("off")
plt.tight_layout()
plt.show()