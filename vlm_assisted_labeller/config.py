"""
Configuration for Seizure Video Labeling Pipeline
"""

# Video settings
FPS = 30
WINDOW_SIZE_SEC = 5
WINDOW_SIZE_FRAMES = FPS * WINDOW_SIZE_SEC  # 150 frames per window

# Paths (update these for your setup)
VIDEO_DIR = "/data_hdd/talha/miccai_26/seizure_detection_pipeline/videos_raw/vids/"
OUTPUT_DIR = "/data_hdd/talha/miccai_26/seizure_detection_pipeline/vlm_labeller/vsvig_labels/"
EXCEL_PATH = "/data_hdd/talha/miccai_26/seizure_detection_pipeline/videos_raw/dataset_Label.xlsx"

# Action classes for hospital bed monitoring
# These are typical activities observed before seizure onset
ACTION_CLASSES = [
    "sleeping",           # 0 - Patient is asleep, eyes closed, minimal movement
    "resting",            # 1 - Patient is awake but lying still, relaxed
    "reading",            # 2 - Patient is reading a book, magazine, or document
    "using_phone",        # 3 - Patient is using mobile phone or tablet
    "watching_tv",        # 4 - Patient is watching TV or looking at screen
    "eating",             # 5 - Patient is eating food or drinking
    "talking",            # 6 - Patient is talking to someone (staff/visitor)
    "sitting_up",         # 7 - Patient is sitting up in bed
    "adjusting_position", # 8 - Patient is changing position, adjusting covers
    "medical_interaction",# 9 - Medical staff interacting with patient
    "other_activity",     # 10 - Other normal activity not covered above
    "unclear",            # 11 - Cannot determine activity (occlusion, blur, etc.)
    "seizure",            # 12 - Seizure activity (for frames after clinical onset)
]

# Create label map
LABEL_MAP = {i: label for i, label in enumerate(ACTION_CLASSES)}
LABEL_TO_ID = {label: i for i, label in enumerate(ACTION_CLASSES)}

# VLM settings
VLM_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
MAX_NEW_TOKENS = 64

# VLM Prompt for action classification
VLM_SYSTEM_PROMPT = """You are an expert medical video analyst specializing in patient monitoring in hospital environments. Your task is to classify the activity of a patient lying on a hospital bed from a single video frame.

CONTEXT:
- This is a hospital room with epilepsy monitoring equipment
- The patient is typically on a bed, often wearing striped hospital clothing
- There may be medical staff or visitors present - FOCUS ONLY ON THE PATIENT
- The video captures normal daily activities BEFORE any seizure event

OUTPUT FORMAT:
You must respond with EXACTLY ONE of these action labels (no other text):
- sleeping
- resting
- reading
- using_phone
- watching_tv
- eating
- talking
- sitting_up
- adjusting_position
- medical_interaction
- other_activity
- unclear

CLASSIFICATION GUIDELINES:
- "sleeping": Eyes closed, no voluntary movement, relaxed posture
- "resting": Awake but lying still, may have eyes open, minimal activity
- "reading": Holding/looking at book, magazine, papers, documents
- "using_phone": Holding/interacting with mobile device or tablet
- "watching_tv": Looking toward TV/monitor direction, attentive posture
- "eating": Food/drink visible, eating or drinking motion
- "talking": Mouth moving, gesturing, clearly engaged with another person
- "sitting_up": Upper body raised, not lying flat
- "adjusting_position": Actively moving, shifting position, adjusting bedding
- "medical_interaction": Staff performing checks, adjusting equipment on patient
- "other_activity": Any other identifiable normal activity
- "unclear": Cannot determine due to occlusion, blur, or ambiguity

IMPORTANT: Focus ONLY on the patient (usually on the bed). Ignore activities of other people in the room."""

VLM_USER_PROMPT = """Analyze this hospital room image and classify the patient's current activity.

Look at the patient on the bed and determine what they are doing. Consider:
1. Body posture (lying flat, sitting up, moving)
2. Eye state if visible (open/closed)
3. Hand position and any objects being held
4. Interaction with others or equipment

Respond with ONLY the action label, nothing else."""
