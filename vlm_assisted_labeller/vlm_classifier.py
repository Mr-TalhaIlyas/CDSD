"""
VLM Inference Module for Action Classification
Uses Qwen3-VL-8B-Instruct for classifying patient activities
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple
from config import (
    VLM_MODEL_NAME, 
    VLM_SYSTEM_PROMPT, 
    VLM_USER_PROMPT,
    MAX_NEW_TOKENS,
    LABEL_TO_ID,
    ACTION_CLASSES
)


class VLMActionClassifier:
    """
    VLM-based action classifier for hospital patient monitoring.
    Uses Qwen3-VL-8B-Instruct to classify patient activities from video frames.
    """
    
    def __init__(
        self,
        model_name: str = VLM_MODEL_NAME,
        device_map: str = "auto",
        use_flash_attention: bool = False,
        dtype: str = "auto"
    ):
        """
        Initialize the VLM classifier.
        
        Args:
            model_name: HuggingFace model name
            device_map: Device placement strategy
            use_flash_attention: Use flash attention 2 for efficiency
            dtype: Model dtype ("auto", torch.bfloat16, etc.)
        """
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        
        print(f"Loading VLM model: {model_name}")
        
        if use_flash_attention:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                device_map=device_map
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map
            )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Valid action labels for parsing
        self.valid_labels = set(ACTION_CLASSES[:-1])  # Exclude 'seizure' from VLM outputs
        
        print("VLM model loaded successfully!")
    
    def _prepare_messages(self, image: Image.Image) -> List[dict]:
        """Prepare message format for Qwen3-VL"""
        messages = [
            {
                "role": "system",
                "content": [
                {"type": "text", "text": VLM_SYSTEM_PROMPT}
            ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": VLM_USER_PROMPT}
                ]
            }
        ]
        return messages
    
    def _parse_vlm_output(self, output_text: str) -> Tuple[str, int]:
        """
        Parse VLM output to extract action label.
        
        Returns:
            Tuple of (label_name, label_id)
        """
        # Clean and lowercase the output
        output_clean = output_text.strip().lower()
        
        # Try to find exact match first
        for label in self.valid_labels:
            if label in output_clean:
                return label, LABEL_TO_ID[label]
        
        # Fallback mappings for common variations
        fallback_map = {
            'sleep': 'sleeping',
            'rest': 'resting',
            'read': 'reading',
            'phone': 'using_phone',
            'mobile': 'using_phone',
            'tablet': 'using_phone',
            'tv': 'watching_tv',
            'television': 'watching_tv',
            'screen': 'watching_tv',
            'eat': 'eating',
            'drink': 'eating',
            'food': 'eating',
            'talk': 'talking',
            'speak': 'talking',
            'conversation': 'talking',
            'sit': 'sitting_up',
            'upright': 'sitting_up',
            'move': 'adjusting_position',
            'adjust': 'adjusting_position',
            'shift': 'adjusting_position',
            'medical': 'medical_interaction',
            'nurse': 'medical_interaction',
            'doctor': 'medical_interaction',
            'staff': 'medical_interaction',
            'other': 'other_activity',
            'unknown': 'unclear',
            'cannot': 'unclear',
            'occlud': 'unclear',
            'blur': 'unclear',
        }
        
        for key, label in fallback_map.items():
            if key in output_clean:
                return label, LABEL_TO_ID[label]
        
        # Default to unclear if no match
        return 'unclear', LABEL_TO_ID['unclear']
    
    @torch.no_grad()
    def classify_frame(self, frame: np.ndarray) -> Tuple[str, int, str]:
        """
        Classify action in a single frame.
        
        Args:
            frame: RGB image as numpy array (H, W, 3)
        
        Returns:
            Tuple of (label_name, label_id, raw_vlm_output)
        """
        # Convert numpy array to PIL Image
        if isinstance(frame, np.ndarray):
            image = Image.fromarray(frame)
        else:
            image = frame
        
        # Prepare messages
        messages = self._prepare_messages(image)
        
        # Process and generate
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # Generate
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # Deterministic output
            temperature=None,
            top_p=None,
        )
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Parse output
        label_name, label_id = self._parse_vlm_output(output_text)
        
        return label_name, label_id, output_text
    
    @torch.no_grad()
    def classify_frames_batch(
        self, 
        frames: List[np.ndarray],
        batch_size: int = 4
    ) -> List[Tuple[str, int, str]]:
        """
        Classify multiple frames (currently processes sequentially).
        
        Args:
            frames: List of RGB frames
            batch_size: Not used currently (for future batch processing)
        
        Returns:
            List of (label_name, label_id, raw_output) tuples
        """
        results = []
        for frame in frames:
            result = self.classify_frame(frame)
            results.append(result)
        return results


class MockVLMClassifier:
    """
    Mock classifier for testing without GPU/model.
    Returns random but consistent labels based on frame content.
    """
    
    def __init__(self):
        self.valid_labels = list(LABEL_TO_ID.keys())[:-1]  # Exclude seizure
        print("Using MockVLMClassifier for testing")
    
    def classify_frame(self, frame: np.ndarray) -> Tuple[str, int, str]:
        """Return a mock classification based on frame statistics"""
        # Use frame statistics for pseudo-random but reproducible results
        mean_val = np.mean(frame)
        
        # Simple heuristics based on brightness
        if mean_val < 50:
            label = 'sleeping'
        elif mean_val < 80:
            label = 'resting'
        elif mean_val < 120:
            label = 'reading'
        elif mean_val < 150:
            label = 'using_phone'
        else:
            # Random from remaining based on pixel variance
            var_val = np.var(frame)
            idx = int(var_val) % len(self.valid_labels)
            label = self.valid_labels[idx]
        
        return label, LABEL_TO_ID[label], f"Mock output: {label}"
    
    def classify_frames_batch(
        self, 
        frames: List[np.ndarray],
        batch_size: int = 4
    ) -> List[Tuple[str, int, str]]:
        """Process multiple frames"""
        return [self.classify_frame(f) for f in frames]


def get_classifier(use_mock: bool = False, **kwargs) -> VLMActionClassifier:
    """
    Factory function to get appropriate classifier.
    
    Args:
        use_mock: If True, return mock classifier for testing
        **kwargs: Arguments passed to VLMActionClassifier
    
    Returns:
        Classifier instance
    """
    if use_mock:
        return MockVLMClassifier()
    return VLMActionClassifier(**kwargs)
