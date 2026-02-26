#%%
import cv2
import matplotlib.pyplot as plt
from vlm_classifier import get_classifier

vlm_kwargs = {}
classifier = get_classifier(use_mock=False, **vlm_kwargs)
#%%

frame = cv2.imread("/data_hdd/talha/miccai_26/seizure_detection_pipeline/vlm_labeller/test.png")
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
label_name, label_id, raw_output = classifier.classify_frame(frame)

#%%


plt.imshow(frame)
plt.title(f"Predicted: {label_name} (ID: {label_id})")
plt.axis('off')
plt.show()
#%%