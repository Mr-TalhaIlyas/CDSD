#%%
import os
import cv2
from ultralytics.models.sam import SAM3VideoSemanticPredictor, SAM3SemanticPredictor
from IPython.display import display, clear_output
from PIL import Image
import numpy as np
from fmutils import fmutils as fmu
import matplotlib.pyplot as plt
# Initialize semantic video predictor
overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=640,
                 model="../weights/sam3.pt", half=True, save=False)
#%%
vid = '/data_hdd/talha/miccai_26/seizure_detection_pipeline/videos_raw/Normal/S8_2_208.mp4'
predictor = SAM3VideoSemanticPredictor(overrides=overrides)
# Track concepts using text prompts
results = predictor(source=vid,
                    text=[
                        'patient in center of frame'
                        # "patient on bed",
                        "patient on bed in striped clothes"
                        #   ,"person on bed"
                        ], stream=True)
'''
'patient in center of frame',
"patient on bed in striped clothes"
'''
# PLAY VIDEO INLINE results
for r in results:
    im = r.plot()                 # BGR numpy array
    im = im[..., ::-1]            # BGR -> RGB
    clear_output(wait=True)
    display(Image.fromarray(im))
    break
#%%    
all_vids = fmu.get_all_files("./videos_raw/", sort=True)
all_dets = []
for vid in all_vids:
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    # Track concepts using text prompts
    results = predictor(source=vid,
                        text=[
                            'patient in center of frame',
                            "patient on bed in striped clothes"
                            ], stream=True)

    # only get frames
    for r in results:
        im = r.plot()                 # BGR numpy array
        im = im[..., ::-1]            # BGR -> RGB
        all_dets.append(im)
        break

# PLOT ALL DETECTIONS totalling 843  so make grids 10x10 grid

def save_image_grids(
    images,
    rows,
    cols,
    out_dir="../dets",
    base_name="grid",
    dpi=600
):
    os.makedirs(out_dir, exist_ok=True)
    
    per_fig = rows * cols
    n_figs = (len(images) + per_fig - 1) // per_fig
    idx = 0
    
    for fig_i in range(n_figs):
        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * 1.5, rows * 1.5),
            constrained_layout=True
        )
        
        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                if idx < len(images):
                    ax.imshow(images[idx])
                    ax.axis('off')
                else:
                    ax.set_visible(False)
                idx += 1
        
        filename = f"{base_name}_{fig_i+1:02d}.png"
        path = os.path.join(out_dir, filename)
        
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        
        print(f"Saved: {path}")

save_image_grids(all_dets, rows=10, cols=10)
#%%
'''
IMAGE TESTING FUNCTIONS

'''

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


# %%


img_path = '/data_hdd/talha/miccai_26/seizure_detection_pipeline/tracker/viz.png'

overrides = dict(conf=0.2, task="segment", mode="predict", imgsz=640,
                 model="../weights/sam3.pt", half=True, save=False)
predictor = SAM3SemanticPredictor(overrides=overrides)
predictor.set_image(img_path)

results = predictor(text=[
                        # 'person',
                        # 'human',
                        # 'blanket on person'
                        'patient in center of frame',
                        "patient on bed in striped clothes"
                        ]
                    )

r = results[0]
masks = r.masks.data.cpu().numpy()  # (N, H, W)

# masks = masks.numpy()   # (N, H, W)

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

vis = overlay_masks_rcnn(img, masks, alpha=0.5)
vis = draw_contours(vis, masks)

plt.figure(figsize=(8, 8))
plt.imshow(vis)
plt.axis("off")
plt.show()

# cv2.imwrite('/data_hdd/talha/miccai_26/seizure_detection_pipeline/tracker/viz_output.png', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))




# %%
