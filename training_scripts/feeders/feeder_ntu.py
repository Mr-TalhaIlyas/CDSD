"""
NTU Dataset Feeder with Concept-Based Prompt Generation for CLIP Alignment

This feeder:
1. Loads NTU skeleton data
2. Loads concept bank (CSV)
3. Generates dynamic prompts for CLIP-based contrastive learning
4. Supports both per-concept and per-action prompt modes

Author: Talha
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import json
import random

from feeders import tools
from feeders.prompt_tools import (
    split_prompts_by_part, 
    split_vector,
    generate_action_prompt,
    get_concept_info,
    create_concept_prompt_cache,
    truncate_prompt_to_tokens,
    _get_ordered_concept_dict
)


def custom_collate_fn(batch):
    """
    Custom collate function that handles:
    1. Stacking skeleton tensors
    2. Batching concept vectors by body part
    3. Collecting prompts (NOT tokenized - tokenization happens in forward pass)
    
    Returns:
        data_numpy: [B, C, T, V, M] skeleton tensor
        label: [B] label tensor
        concept_vectors_by_part: Dict of [B, n_concepts_part] tensors
        prompts_by_part: Dict of prompt lists (same for all samples - for contrastive)
    """
    data_numpy = torch.stack([torch.from_numpy(item[0]) for item in batch])
    label = torch.tensor([item[1] for item in batch])
    
    # Concept vectors grouped by body part - collate into dict of tensors
    concept_vectors_by_part = {}
    for key in batch[0][2].keys():
        concept_vectors_by_part[key] = torch.stack([
            torch.from_numpy(item[2][key]) for item in batch
        ])
    
    # Prompts by part - these are the SAME for all samples (per-concept prompts)
    # We only need one copy, not one per sample
    prompts_by_part = batch[0][3]  # Take from first sample
    
    return data_numpy, label, concept_vectors_by_part, prompts_by_part


def custom_collate_fn_with_action_prompts(batch):
    """
    Alternative collate function that also includes per-action prompts.
    Use this if you want action-specific prompts in addition to concept prompts.
    
    Returns:
        data_numpy: [B, C, T, V, M] skeleton tensor
        label: [B] label tensor  
        concept_vectors_by_part: Dict of [B, n_concepts_part] tensors
        prompts_by_part: Dict of prompt lists (per-concept)
        action_prompts: List of action-specific prompts [B]
    """
    data_numpy = torch.stack([torch.from_numpy(item[0]) for item in batch])
    label = torch.tensor([item[1] for item in batch])
    
    # Concept vectors grouped by body part
    concept_vectors_by_part = {}
    for key in batch[0][2].keys():
        concept_vectors_by_part[key] = torch.stack([
            torch.from_numpy(item[2][key]) for item in batch
        ])
    
    # Per-concept prompts (same for all samples)
    prompts_by_part = batch[0][3]
    
    # Per-action prompts (different for each sample)
    action_prompts = [item[4] for item in batch]
    
    return data_numpy, label, concept_vectors_by_part, prompts_by_part, action_prompts


class Feeder(Dataset):
    """
    NTU Dataset Feeder with efficient prompt generation for CLIP alignment.
    
    Supports two prompt modes:
    1. per_concept: Generate prompts for each concept (for contrastive alignment)
    2. per_action: Generate action-specific prompts with active concepts
    
    Args:
        data_path: Path to NTU skeleton data (.npz)
        concepts_csv: Path to concept bank CSV file
        prompt_mode: 'per_concept' or 'per_action'
        include_temporal_in_prompts: Whether to include temporal concepts in prompts
        use_short_prompts: Use shorter prompts for CLIP efficiency
    """
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', 
                 random_choose=False, random_shift=False, random_move=False, 
                 random_rot=False, window_size=-1, normalization=False, 
                 debug=False, use_mmap=False, bone=False, vel=False, 
                 concepts_csv=None, remove_null_concept=True,
                 prompt_mode='per_concept', include_temporal_in_prompts=True,
                 use_short_prompts=True, return_action_prompts=False):
        
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.concepts_csv = concepts_csv
        self.remove_null_concept = remove_null_concept
        
        # Prompt generation settings
        self.prompt_mode = prompt_mode
        self.include_temporal_in_prompts = include_temporal_in_prompts
        self.use_short_prompts = use_short_prompts
        self.return_action_prompts = return_action_prompts
        
        # Augmentation probabilities
        self.aug_prob_rot = 0.0
        self.aug_prob_move = 0.0
        
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        """Load skeleton data and concept bank."""
        # Load skeleton data
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
        
        # Load concept bank
        if self.concepts_csv is not None:
            self.concept_df = pd.read_csv(self.concepts_csv)
        else:
            raise ValueError("concepts_csv must be provided...")
        
        self._setup_concepts()

    def _setup_concepts(self):
        """Setup concept matrix, vocabulary, and pre-compute prompts."""
        # Extract concept information
        self.concept_matrix_np, self.action_names, self.vocab, self.ordered_concept_dict = \
            get_concept_info(self.concept_df)
        
        # Concept counts per body part (for reference)
        concepts_list = [v.split('_', 1)[0] if not v.startswith('temporal') 
                        else '_'.join(v.split('_')[:2]) for v in self.vocab]
        unique_parts, counts = np.unique(concepts_list, return_counts=True)
        self.concept_counts = dict(zip(unique_parts, counts))
        
        # Pre-compute per-concept prompts (same for all samples)
        self.prompts_by_part = split_prompts_by_part(
            self.concept_df, 
            self.ordered_concept_dict,
            use_short_descriptions=self.use_short_prompts
        )
        
        # Pre-compute action prompts if needed
        if self.return_action_prompts:
            self._precompute_action_prompts()
        
        # Verify dimensions
        assert len(self.vocab) == self.concept_matrix_np.shape[1], \
            f"Vocab size {len(self.vocab)} != matrix cols {self.concept_matrix_np.shape[1]}"
        # get spationa dn temporal concept counts to use in main.py
        self.n_spatial_concepts = sum(1 for v in self.vocab if not v.startswith('temporal'))
        self.n_temporal_concepts = sum(1 for v in self.vocab if v.startswith('temporal'))
        print(f"Loaded concept bank:")
        print(f"  - {len(self.action_names)} action classes")
        print(f"  - {len(self.vocab)} concepts")
        print(f"  - Spatial concepts: {self.n_spatial_concepts}")
        print(f"  - Temporal concepts: {self.n_temporal_concepts}")

    def _precompute_action_prompts(self):
        """Pre-compute action prompts for all classes."""
        self.action_prompts_cache = {}
        
        for label_idx, action_name in enumerate(self.action_names):
            concept_vector = self.concept_matrix_np[label_idx]
            prompt = generate_action_prompt(
                action_name,
                concept_vector,
                self.vocab,
                include_temporal=self.include_temporal_in_prompts,
                max_concepts=8,
                template='detailed'
            )
            # Ensure prompt fits CLIP token limit
            prompt = truncate_prompt_to_tokens(prompt, max_tokens=77)
            self.action_prompts_cache[label_idx] = prompt

    def get_mean_map(self):
        """Compute mean map for normalization."""
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        """
        Get a single sample with concept vectors and prompts.
        
        Returns:
            data_numpy: Skeleton data (C, T, V, M)
            label: Action class label (int)
            concept_vectors_by_part: Dict of concept vectors by body part
            prompts_by_part: Dict of prompt lists by body part
            [optional] action_prompt: Action-specific prompt string
        """
        data_numpy = self.data[index]
        label = self.label[index]
        
        # Get concept vector for the label and split by body part
        concept_vector = self.concept_matrix_np[label]
        concept_vectors_by_part = split_vector(concept_vector, self.ordered_concept_dict)
        
        # Process skeleton data
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        
        # Apply augmentations during training
        if self.split == 'train':
            aug_type = self._sample_augmentation()
            
            if aug_type == 'rotation' and self.random_rot:
                data_numpy = tools.random_rot(data_numpy, theta=0.3).numpy()
            elif aug_type == 'move' and self.random_move:
                data_numpy = tools.random_move(
                    data_numpy,
                    angle_candidate=[-5., 5.],
                    scale_candidate=[0.5, 0.9, 1.1, 1.5],
                    transform_candidate=[-0.2, -0.1, 0.1, 0.2],
                    move_time_candidate=[1]
                )
        
        # Bone features
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        elif not self.vel:  # Add centering for consistency
            trajectory = data_numpy[:, :, 20]
            data_numpy = data_numpy - data_numpy[:, :, 20:21]
            data_numpy[:, :, 20] = trajectory
        # Velocity features
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0
        
        # Return based on mode
        if self.return_action_prompts:
            action_prompt = self.action_prompts_cache[label]
            return data_numpy, label, concept_vectors_by_part, self.prompts_by_part, action_prompt
        else:
            return data_numpy, label, concept_vectors_by_part, self.prompts_by_part

    def _sample_augmentation(self):
        """Sample augmentation type based on probabilities."""
        rand_val = random.random()
        
        if rand_val < self.aug_prob_rot:
            return 'rotation'
        elif rand_val < self.aug_prob_rot + self.aug_prob_move:
            return 'move'
        else:
            return 'none'

    def top_k(self, score, top_k):
        """Compute top-k accuracy."""
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def get_concept_info_for_model(self):
        """
        Return concept information needed by the model.
        Useful for model initialization.
        """
        return {
            'n_concepts': len(self.vocab),
            'n_spatial_concepts': sum(1 for v in self.vocab if not v.startswith('temporal')),
            'n_temporal_concepts': sum(1 for v in self.vocab if v.startswith('temporal')),
            'n_actions': len(self.action_names),
            'vocab': self.vocab,
            'action_names': self.action_names,
            'concept_counts': self.concept_counts,
        }


# =============================================================================
# UTILITY FUNCTIONS FOR PROMPT HANDLING IN MODEL
# =============================================================================

def parse_prompts_for_clip(prompts_by_part, include_temporal=True):
    """
    Parse prompts from dataloader output into flat list for CLIP encoding.
    
    Args:
        prompts_by_part: Dict from dataloader
        include_temporal: Whether to include temporal concept prompts
        
    Returns:
        List of prompt strings ready for CLIP tokenization
    """
    all_prompts = prompts_by_part['full_body'].copy()
    
    if include_temporal:
        all_prompts.extend(prompts_by_part['temporal'])
    
    return all_prompts


def create_concept_to_prompt_mapping(prompts_by_part, vocab):
    """
    Create mapping from concept index to its prompt.
    Useful for selective prompt encoding.
    
    Args:
        prompts_by_part: Dict from dataloader
        vocab: List of concept names
        
    Returns:
        Dict mapping concept index to prompt string
    """
    all_prompts = prompts_by_part['full_body'] + prompts_by_part['temporal']
    
    assert len(all_prompts) == len(vocab), \
        f"Prompt count {len(all_prompts)} != vocab size {len(vocab)}"
    
    return {i: prompt for i, prompt in enumerate(all_prompts)}


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("Feeder module loaded successfully")
    print("Use custom_collate_fn for DataLoader")
    print("Use custom_collate_fn_with_action_prompts if you need per-action prompts")