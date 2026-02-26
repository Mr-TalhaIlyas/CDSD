#%%
"""
Prompt Tools for Skeleton Action Concept Learning (ACL)

Generates dynamic, CLIP-compatible prompts from concept bank activations.
Supports both:
1. Per-concept prompts for contrastive alignment
2. Full action prompts for semantic grounding

Author: Talha
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd


# =============================================================================
# CONCEPT TO NATURAL LANGUAGE MAPPINGS
# =============================================================================

# These mappings convert concept names to natural, CLIP-friendly descriptions
CONCEPT_TO_TEXT = {
    # HEAD concepts
    'head_nod': 'head nodding up and down',
    'head_shake': 'head shaking side to side',
    'head_tilt_down': 'head tilted downward',
    'head_tilt_up': 'head tilted upward',
    'head_rotate': 'head rotating to the side',
    'head_static': 'head remaining still',
    'head_hyperextend': 'head tilted far backward',
    'head_rhythmic_jerk': 'head jerking rhythmically',
    'head_versive_deviation': 'head forced to one side',
    'head_flaccid': 'head hanging limply',
    
    # HAND concepts
    'hand_grasp': 'hand grasping an object',
    'hand_release': 'hand releasing an object',
    'hand_wave': 'hand waving',
    'hand_point': 'hand pointing',
    'hand_clap': 'hands clapping together',
    'hand_manipulate': 'hand manipulating an object',
    'hand_fist': 'hand forming a fist',
    'hand_open': 'hand open with fingers spread',
    'hand_touch': 'hand touching something',
    'hand_lift_to_face': 'hand raised to face',
    'hand_circular': 'hand making circular motion',
    'hand_bilateral_fist': 'both hands clenched into fists',
    'hand_rhythmic_clench': 'hands clenching rhythmically',
    'hand_dystonic_posture': 'hand in abnormal twisted posture',
    'hand_flaccid': 'hands completely relaxed',
    
    # ARM concepts
    'arm_raise': 'arm raised upward',
    'arm_lower': 'arm lowered downward',
    'arm_extend': 'arm extended outward',
    'arm_retract': 'arm pulled back',
    'arm_circular': 'arm making circular motion',
    'arm_swing': 'arm swinging',
    'arm_cross': 'arms crossed',
    'arm_hold': 'arm holding position',
    'arm_alternating': 'arms moving alternately',
    'arm_overhead': 'arm raised overhead',
    'arm_lateral': 'arm moving to the side',
    'arm_bilateral_symmetric': 'both arms moving identically',
    'arm_tonic_posture': 'arm held stiffly',
    'arm_rhythmic_jerk': 'arm jerking rhythmically',
    'arm_flaccid': 'arm hanging limply',
    
    # HIP concepts
    'hip_bend': 'hip bending forward',
    'hip_extend': 'hip extending',
    'hip_sit': 'hip in sitting position',
    'hip_squat': 'hip in squat position',
    'hip_rotate': 'hip rotating',
    'hip_static_stand': 'hip stationary while standing',
    'hip_static_sit': 'hip stationary while sitting',
    'hip_sway': 'hip swaying side to side',
    
    # TORSO concepts
    'torso_rigid': 'torso held rigidly',
    'torso_opisthotonus': 'back arched severely backward',
    'torso_flaccid': 'torso completely limp',
    
    # LEG concepts
    'leg_step': 'leg stepping',
    'leg_kick': 'leg kicking',
    'leg_jump': 'legs jumping',
    'leg_squat': 'legs squatting',
    'leg_lift': 'leg lifted',
    'leg_static_stand': 'legs stationary while standing',
    'leg_static_sit': 'legs stationary while sitting',
    'leg_alternating': 'legs moving alternately',
    'leg_balance': 'leg balancing',
    'leg_bilateral_symmetric': 'both legs moving identically',
    'leg_tonic_extension': 'legs held stiffly straight',
    'leg_rhythmic_jerk': 'legs jerking rhythmically',
    'leg_flaccid': 'legs completely limp',
    
    # FOOT concepts
    'foot_step': 'foot stepping',
    'foot_kick': 'foot kicking',
    'foot_jump': 'foot jumping',
    'foot_pivot': 'foot pivoting',
    'foot_static': 'foot stationary',
    'foot_tap': 'foot tapping',
    'foot_slide': 'foot sliding',
    'foot_bilateral_plantar': 'both feet pointing downward',
    'foot_rhythmic_movement': 'feet moving rhythmically',
    'foot_flaccid': 'feet completely relaxed',
    
    # INTERACTION concepts
    'interaction_two_person': 'interacting with another person',
    
    # TEMPORAL DIRECTION concepts
    'temporal_direction_motion_upward': 'with upward movement',
    'temporal_direction_motion_downward': 'with downward movement',
    'temporal_direction_motion_forward': 'with forward movement',
    'temporal_direction_motion_backward': 'with backward movement',
    'temporal_direction_motion_converging': 'with converging movement',
    'temporal_direction_motion_diverging': 'with diverging movement',
    'temporal_direction_motion_reversible': 'with reversible movement',
    
    # TEMPORAL SEQUENCE concepts
    'temporal_sequence_seq_hands_first': 'hands moving first',
    'temporal_sequence_seq_legs_first': 'legs moving first',
    'temporal_sequence_seq_body_first': 'body moving first',
    'temporal_sequence_seq_simultaneous': 'all parts moving together',
    'temporal_sequence_seq_alternating': 'parts moving alternately',
    'temporal_sequence_seq_cascading': 'movement cascading through body',
    
    # TEMPORAL DYNAMICS concepts
    'temporal_dynamics_speed_slow': 'performed slowly',
    'temporal_dynamics_speed_fast': 'performed quickly',
    'temporal_dynamics_speed_accelerating': 'with accelerating speed',
    'temporal_dynamics_speed_decelerating': 'with decelerating speed',
    'temporal_dynamics_rhythm_regular': 'with regular rhythm',
    'temporal_dynamics_rhythm_irregular': 'with irregular rhythm',
    'temporal_dynamics_duration_brief': 'brief in duration',
    'temporal_dynamics_duration_sustained': 'sustained over time',
    'temporal_dynamics_bilateral_sync': 'with bilateral synchronization',
    'temporal_dynamics_frequency_decreasing': 'with decreasing frequency',
}

# Body part groupings for organizing concepts
BODY_PARTS = ['head', 'hand', 'arm', 'hip', 'torso', 'leg', 'foot']
TEMPORAL_PARTS = ['temporal_direction', 'temporal_sequence', 'temporal_dynamics']
INTERACTION_PARTS = ['interaction']

# Short descriptions for per-concept contrastive learning
CONCEPT_SHORT_DESCRIPTIONS = {
    # HEAD
    'head_nod': 'a person with head nodding',
    'head_shake': 'a person with head shaking',
    'head_tilt_down': 'a person with head tilted down',
    'head_tilt_up': 'a person with head tilted up',
    'head_rotate': 'a person with head turned',
    'head_static': 'a person with head still',
    'head_hyperextend': 'a person with head tilted far back',
    'head_rhythmic_jerk': 'a person with head jerking',
    'head_versive_deviation': 'a person with head forced to side',
    'head_flaccid': 'a person with head limp',
    
    # HAND
    'hand_grasp': 'a person grasping with hand',
    'hand_release': 'a person releasing from hand',
    'hand_wave': 'a person waving hand',
    'hand_point': 'a person pointing with hand',
    'hand_clap': 'a person clapping hands',
    'hand_manipulate': 'a person manipulating with hands',
    'hand_fist': 'a person making a fist',
    'hand_open': 'a person with open hand',
    'hand_touch': 'a person touching with hand',
    'hand_lift_to_face': 'a person with hand at face',
    'hand_circular': 'a person with hand moving circularly',
    'hand_bilateral_fist': 'a person with both fists clenched',
    'hand_rhythmic_clench': 'a person with hands clenching rhythmically',
    'hand_dystonic_posture': 'a person with hand in twisted posture',
    'hand_flaccid': 'a person with hands limp',
    
    # ARM
    'arm_raise': 'a person raising arm',
    'arm_lower': 'a person lowering arm',
    'arm_extend': 'a person extending arm',
    'arm_retract': 'a person retracting arm',
    'arm_circular': 'a person with arm circling',
    'arm_swing': 'a person swinging arm',
    'arm_cross': 'a person with arms crossed',
    'arm_hold': 'a person holding arm still',
    'arm_alternating': 'a person with arms alternating',
    'arm_overhead': 'a person with arm overhead',
    'arm_lateral': 'a person with arm to side',
    'arm_bilateral_symmetric': 'a person with both arms moving together',
    'arm_tonic_posture': 'a person with arm held stiffly',
    'arm_rhythmic_jerk': 'a person with arm jerking',
    'arm_flaccid': 'a person with arm limp',
    
    # HIP
    'hip_bend': 'a person bending at hip',
    'hip_extend': 'a person extending hip',
    'hip_sit': 'a person sitting',
    'hip_squat': 'a person squatting',
    'hip_rotate': 'a person rotating hip',
    'hip_static_stand': 'a person standing still',
    'hip_static_sit': 'a person sitting still',
    'hip_sway': 'a person swaying hips',
    
    # TORSO
    'torso_rigid': 'a person with rigid torso',
    'torso_opisthotonus': 'a person with back arched',
    'torso_flaccid': 'a person with limp torso',
    
    # LEG
    'leg_step': 'a person stepping',
    'leg_kick': 'a person kicking',
    'leg_jump': 'a person jumping',
    'leg_squat': 'a person squatting',
    'leg_lift': 'a person lifting leg',
    'leg_static_stand': 'a person with legs still standing',
    'leg_static_sit': 'a person with legs still sitting',
    'leg_alternating': 'a person with legs alternating',
    'leg_balance': 'a person balancing on leg',
    'leg_bilateral_symmetric': 'a person with both legs moving together',
    'leg_tonic_extension': 'a person with legs stiff',
    'leg_rhythmic_jerk': 'a person with legs jerking',
    'leg_flaccid': 'a person with legs limp',
    
    # FOOT
    'foot_step': 'a person stepping with foot',
    'foot_kick': 'a person kicking with foot',
    'foot_jump': 'a person jumping',
    'foot_pivot': 'a person pivoting on foot',
    'foot_static': 'a person with feet still',
    'foot_tap': 'a person tapping foot',
    'foot_slide': 'a person sliding foot',
    'foot_bilateral_plantar': 'a person with feet pointed down',
    'foot_rhythmic_movement': 'a person with feet moving rhythmically',
    'foot_flaccid': 'a person with feet relaxed',
    
    # INTERACTION
    'interaction_two_person': 'two people interacting',
    
    # TEMPORAL - these are modifiers, shorter forms
    'temporal_direction_motion_upward': 'moving upward',
    'temporal_direction_motion_downward': 'moving downward',
    'temporal_direction_motion_forward': 'moving forward',
    'temporal_direction_motion_backward': 'moving backward',
    'temporal_direction_motion_converging': 'converging motion',
    'temporal_direction_motion_diverging': 'diverging motion',
    'temporal_direction_motion_reversible': 'reversible motion',
    'temporal_sequence_seq_hands_first': 'hands leading',
    'temporal_sequence_seq_legs_first': 'legs leading',
    'temporal_sequence_seq_body_first': 'body leading',
    'temporal_sequence_seq_simultaneous': 'simultaneous motion',
    'temporal_sequence_seq_alternating': 'alternating motion',
    'temporal_sequence_seq_cascading': 'cascading motion',
    'temporal_dynamics_speed_slow': 'slow movement',
    'temporal_dynamics_speed_fast': 'fast movement',
    'temporal_dynamics_speed_accelerating': 'accelerating',
    'temporal_dynamics_speed_decelerating': 'decelerating',
    'temporal_dynamics_rhythm_regular': 'regular rhythm',
    'temporal_dynamics_rhythm_irregular': 'irregular rhythm',
    'temporal_dynamics_duration_brief': 'brief action',
    'temporal_dynamics_duration_sustained': 'sustained action',
    'temporal_dynamics_bilateral_sync': 'bilateral sync',
    'temporal_dynamics_frequency_decreasing': 'decreasing frequency',
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_body_part(concept_name: str) -> str:
    """Extract body part from concept name."""
    if concept_name.startswith('temporal_'):
        # Handle temporal concepts specially
        if 'direction' in concept_name:
            return 'temporal_direction'
        elif 'sequence' in concept_name:
            return 'temporal_sequence'
        elif 'dynamics' in concept_name:
            return 'temporal_dynamics'
    elif concept_name.startswith('interaction_'):
        return 'interaction'
    else:
        # Standard body part concepts
        return concept_name.split('_')[0]


def split_vector(concept_vector: np.ndarray, 
                 ordered_concept_dict: Dict[str, int]) -> Dict[str, np.ndarray]:
    """
    Split a concept vector into body-part-specific vectors.
    
    Args:
        concept_vector: Binary concept activation vector [n_concepts]
        ordered_concept_dict: Dict mapping body parts to concept counts
        
    Returns:
        Dict mapping body parts to their concept vectors
    """
    result = {}
    idx = 0
    
    # ordered_concept_dict should have actual counts per part
    # e.g., {'head': 10, 'hand': 15, 'arm': 15, ...}
    for part, count in ordered_concept_dict.items():
        result[part] = concept_vector[idx:idx + count]
        idx += count
    
    # Create aggregated spatial and temporal vectors
    spatial_parts = ['head', 'hand', 'arm', 'hip', 'torso', 'leg', 'foot', 'interaction']
    temporal_parts = ['temporal_direction', 'temporal_sequence', 'temporal_dynamics']
    
    spatial_vectors = [result[p] for p in spatial_parts if p in result and len(result[p]) > 0]
    temporal_vectors = [result[p] for p in temporal_parts if p in result and len(result[p]) > 0]
    
    if spatial_vectors:
        result['full_body'] = np.concatenate(spatial_vectors)
    if temporal_vectors:
        result['temporal'] = np.concatenate(temporal_vectors)
    
    return result


def split_prompts_by_part(concept_df: pd.DataFrame,
                          ordered_concept_dict: Dict[str, int],
                          use_short_descriptions: bool = True) -> Dict[str, List[str]]:
    """
    Generate prompts organized by body part for contrastive learning.
    
    This creates a list of concept description prompts for each body part,
    which can be used for per-concept contrastive alignment.
    
    Args:
        concept_df: DataFrame with concept columns
        ordered_concept_dict: Dict mapping body parts to concept counts
        use_short_descriptions: If True, use shorter prompts for CLIP efficiency
        
    Returns:
        Dict mapping body parts to lists of concept prompts
    """
    vocab = concept_df.columns.tolist()[1:]  # Skip 'action_class' column
    
    prompts_by_part = {
        'head': [], 'hand': [], 'arm': [], 'hip': [], 'torso': [],
        'leg': [], 'foot': [], 'interaction': [],
        'temporal_direction': [], 'temporal_sequence': [], 'temporal_dynamics': [],
        'full_body': [], 'temporal': []
    }
    
    description_dict = CONCEPT_SHORT_DESCRIPTIONS if use_short_descriptions else CONCEPT_TO_TEXT
    
    for concept_name in vocab:
        part = get_body_part(concept_name)
        
        # Get description, with fallback
        if concept_name in description_dict:
            desc = description_dict[concept_name]
        else:
            # Fallback: convert concept name to readable text
            desc = f"a person with {concept_name.replace('_', ' ')}"
        
        if part in prompts_by_part:
            prompts_by_part[part].append(desc)
    
    # Create aggregated prompt lists
    spatial_parts = ['head', 'hand', 'arm', 'hip', 'torso', 'leg', 'foot', 'interaction']
    temporal_parts = ['temporal_direction', 'temporal_sequence', 'temporal_dynamics']
    
    for part in spatial_parts:
        prompts_by_part['full_body'].extend(prompts_by_part[part])
    
    for part in temporal_parts:
        prompts_by_part['temporal'].extend(prompts_by_part[part])
    
    return prompts_by_part


# =============================================================================
# DYNAMIC ACTION PROMPT GENERATION
# =============================================================================

def format_action_name(action_name: str) -> str:
    """Convert action class name to readable format."""
    # Replace underscores with spaces and handle special cases
    formatted = action_name.replace('_', ' ')
    
    # Handle common patterns
    replacements = {
        'phone tablet': 'phone or tablet',
        'hat cap': 'hat or cap',
        'gtc': 'generalized tonic-clonic',
        'pnes': 'psychogenic non-epileptic seizure',
    }
    
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    
    return formatted


def generate_action_prompt(action_name: str,
                           concept_vector: np.ndarray,
                           vocab: List[str],
                           include_temporal: bool = False,
                           max_concepts: int = 6,
                           template: str = 'detailed') -> str:
    """
    Generate a dynamic prompt for a specific action based on active concepts.
    
    Args:
        action_name: Name of the action class
        concept_vector: Binary concept activation vector
        vocab: List of concept names (same order as vector)
        include_temporal: Whether to include temporal concepts
        max_concepts: Maximum number of concepts to include (for token limit)
        template: 'detailed', 'simple', or 'minimal'
        
    Returns:
        CLIP-compatible prompt string (guaranteed <= 77 tokens)
    """
    formatted_action = format_action_name(action_name)
    
    # Get active concepts
    active_indices = np.where(concept_vector == 1)[0]
    active_concepts = [vocab[i] for i in active_indices]
    
    # Separate spatial and temporal concepts
    spatial_concepts = []
    temporal_concepts = []
    
    for concept in active_concepts:
        if concept.startswith('temporal_'):
            temporal_concepts.append(concept)
        else:
            spatial_concepts.append(concept)
    
    # Prioritize body part diversity - select representative concepts
    selected_spatial = _select_diverse_concepts(spatial_concepts, max_concepts)
    
    if include_temporal and temporal_concepts:
        selected_temporal = temporal_concepts[:1]  # Max 1 temporal concept
    else:
        selected_temporal = []
    
    # Build prompt based on template
    if template == 'minimal':
        return f"a person {formatted_action}"
    
    elif template == 'simple':
        concept_texts = [CONCEPT_TO_TEXT.get(c, c.replace('_', ' ')) 
                        for c in selected_spatial[:3]]
        if concept_texts:
            return f"a person {formatted_action}, {', '.join(concept_texts)}"
        return f"a person {formatted_action}"
    
    else:  # detailed
        # Build compact body part descriptions
        part_descriptions = []
        concepts_by_part = {}
        
        for concept in selected_spatial:
            part = get_body_part(concept)
            if part not in concepts_by_part:
                concepts_by_part[part] = []
            concepts_by_part[part].append(concept)
        
        # Map to readable part names
        part_display = {
            'head': 'head', 'hand': 'hands', 'arm': 'arms', 
            'hip': 'hips', 'torso': 'torso', 'leg': 'legs', 'foot': 'feet'
        }
        
        # Create compact descriptions per part
        for part in ['head', 'hand', 'arm', 'torso', 'hip', 'leg', 'foot']:
            if part in concepts_by_part and len(part_descriptions) < 4:
                concept = concepts_by_part[part][0]
                # Get short action description
                if concept in CONCEPT_TO_TEXT:
                    action_text = CONCEPT_TO_TEXT[concept]
                    # Extract just the action part (remove "the X is")
                    part_descriptions.append(action_text)
        
        # Build final prompt
        if part_descriptions:
            body_desc = ', '.join(part_descriptions[:4])  # Limit to 4 parts
            prompt = f"a person {formatted_action}, with {body_desc}"
        else:
            prompt = f"a person {formatted_action}"
        
        # Add temporal modifier if requested and fits
        if selected_temporal and estimate_tokens(prompt) < 60:
            temporal_text = CONCEPT_TO_TEXT.get(selected_temporal[0], '')
            if temporal_text:
                prompt += f", {temporal_text}"
        
        # Ensure we're under token limit
        prompt = truncate_prompt_to_tokens(prompt, max_tokens=77)
        
        return prompt


def _select_diverse_concepts(concepts: List[str], max_count: int) -> List[str]:
    """Select concepts ensuring body part diversity."""
    if len(concepts) <= max_count:
        return concepts
    
    # Group by body part
    by_part = {}
    for concept in concepts:
        part = get_body_part(concept)
        if part not in by_part:
            by_part[part] = []
        by_part[part].append(concept)
    
    # Select one from each part first, then fill remaining
    selected = []
    for part, part_concepts in by_part.items():
        if part_concepts and len(selected) < max_count:
            selected.append(part_concepts[0])
    
    # Fill remaining slots
    for concept in concepts:
        if concept not in selected and len(selected) < max_count:
            selected.append(concept)
    
    return selected


# =============================================================================
# BATCH PROMPT GENERATION FOR TRAINING
# =============================================================================

def generate_batch_prompts(concept_df: pd.DataFrame,
                           labels: np.ndarray,
                           include_temporal: bool = True,
                           mode: str = 'per_concept') -> Dict[str, List[str]]:
    """
    Generate prompts for a batch of samples.
    
    Args:
        concept_df: DataFrame with action_class and concept columns
        labels: Array of action class indices for the batch
        include_temporal: Whether to include temporal concepts
        mode: 'per_concept' for contrastive learning, 'per_action' for action prompts
        
    Returns:
        Dict with prompts organized by body part or action
    """
    vocab = concept_df.columns.tolist()[1:]
    action_names = concept_df['action_class'].values
    concept_matrix = concept_df.values[:, 1:].astype(np.uint8)
    
    if mode == 'per_concept':
        # Return per-concept prompts (same for all samples in batch)
        ordered_dict = _get_ordered_concept_dict(vocab)
        return split_prompts_by_part(concept_df, ordered_dict)
    
    elif mode == 'per_action':
        # Generate action-specific prompts for each sample
        prompts = []
        for label in labels:
            action_name = action_names[label]
            concept_vector = concept_matrix[label]
            prompt = generate_action_prompt(
                action_name, concept_vector, vocab, 
                include_temporal=include_temporal
            )
            prompts.append(prompt)
        return {'action_prompts': prompts}
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _get_ordered_concept_dict(vocab: List[str]) -> Dict[str, int]:
    """
    Build ordered concept dict from vocabulary.
    Returns dict with count of concepts per body part, preserving order.
    """
    from collections import OrderedDict
    
    ordered_dict = OrderedDict()
    
    for concept in vocab:
        part = get_body_part(concept)
        if part in ordered_dict:
            ordered_dict[part] += 1
        else:
            ordered_dict[part] = 1
    
    return ordered_dict


# =============================================================================
# CLIP TOKEN ESTIMATION
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Rough estimation of CLIP tokens.
    CLIP uses BPE tokenization, roughly 1 token per 4 characters.
    """
    return len(text) // 4 + len(text.split())


def truncate_prompt_to_tokens(prompt: str, max_tokens: int = 77) -> str:
    """
    Truncate prompt to fit within CLIP's token limit.
    Preserves the beginning (action description) and truncates concepts.
    """
    if estimate_tokens(prompt) <= max_tokens:
        return prompt
    
    # Strategy 1: Remove ", with ..." clause progressively
    if ', with ' in prompt:
        parts = prompt.split(', with ')
        action_part = parts[0]
        
        if estimate_tokens(action_part) <= max_tokens:
            concept_part = parts[1] if len(parts) > 1 else ''
            
            # Progressively remove concept descriptions from the end
            concept_items = concept_part.split(', ')
            while estimate_tokens(f"{action_part}, with {', '.join(concept_items)}") > max_tokens and len(concept_items) > 1:
                concept_items = concept_items[:-1]
            
            if concept_items:
                return f"{action_part}, with {', '.join(concept_items)}"
            return action_part
    
    # Strategy 2: Try removing ", where ..." clause
    if ', where ' in prompt:
        parts = prompt.split(', where ')
        action_part = parts[0]
        if estimate_tokens(action_part) <= max_tokens:
            return action_part
    
    # Strategy 3: Hard truncate by words
    words = prompt.split()
    while estimate_tokens(' '.join(words)) > max_tokens and len(words) > 3:
        words = words[:-1]
    
    return ' '.join(words)


# =============================================================================
# CONVENIENCE FUNCTIONS FOR DATALOADER
# =============================================================================

def get_concept_info(concept_df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str], Dict]:
    """
    Extract all concept information from DataFrame.
    
    Returns:
        concept_matrix: Binary activation matrix [n_actions, n_concepts]
        action_names: List of action class names
        vocab: List of concept names
        ordered_concept_dict: Dict mapping parts to counts
    """
    concept_matrix = concept_df.values[:, 1:].astype(np.uint8)
    action_names = concept_df['action_class'].values.tolist()
    vocab = concept_df.columns.tolist()[1:]
    ordered_concept_dict = _get_ordered_concept_dict(vocab)
    
    return concept_matrix, action_names, vocab, ordered_concept_dict


def create_concept_prompt_cache(concept_df: pd.DataFrame) -> Dict[str, str]:
    """
    Pre-compute prompts for all concepts for efficiency.
    
    Returns:
        Dict mapping concept names to their prompt strings
    """
    cache = {}
    vocab = concept_df.columns.tolist()[1:]
    
    for concept in vocab:
        if concept in CONCEPT_SHORT_DESCRIPTIONS:
            cache[concept] = CONCEPT_SHORT_DESCRIPTIONS[concept]
        else:
            cache[concept] = f"a person with {concept.replace('_', ' ')}"
    
    return cache


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    # Test with sample concept bank
    import pandas as pd
    
    # Create mini test DataFrame
    test_data = {
        'action_class': ['drink_water', 'jumping'],
        'head_nod': [0, 0],
        'head_tilt_up': [1, 1],
        'hand_grasp': [1, 0],
        'hand_fist': [0, 1],
        'arm_raise': [1, 1],
        'temporal_dynamics_speed_slow': [1, 0],
        'temporal_dynamics_duration_sustained': [0, 1],
    }
    test_df = pd.DataFrame(test_data)
    
    vocab = test_df.columns.tolist()[1:]
    ordered_dict = _get_ordered_concept_dict(vocab)
    
    print("=" * 60)
    print("PROMPT TOOLS TEST")
    print("=" * 60)
    
    # Test per-concept prompts
    prompts_by_part = split_prompts_by_part(test_df, ordered_dict)
    print("\nPer-concept prompts:")
    for part, prompts in prompts_by_part.items():
        if prompts:
            print(f"  {part}: {prompts}")
    
    # Test action prompt generation
    for idx in range(len(test_df)):
        action_name = test_df['action_class'].iloc[idx]
        concept_vector = test_df.iloc[idx, 1:].values.astype(np.uint8)
        
        prompt = generate_action_prompt(
            action_name, concept_vector, vocab, 
            include_temporal=True, template='detailed',
            max_concepts=20
        )
        print(f"\n{action_name}:")
        print(f"  Prompt: {prompt}")
        print(f"  Est. tokens: {estimate_tokens(prompt)}")
#%%