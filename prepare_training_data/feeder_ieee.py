"""
IEEE Seizure Dataset Feeder with Concept-Based Prompt Generation for CLIP Alignment

This feeder:
1. Loads IEEE seizure skeleton data (processed .npz format)
2. Loads concept bank (CSV)
3. Generates dynamic prompts for CLIP-based contrastive learning
4. Supports custom train/test splits with stratified sampling
5. Supports K-fold cross-validation

Key differences from NTU feeder:
- Per-frame labels -> uses majority label for sample-level classification
- Custom train/test splitting (stratified by Normal vs Seizure)
- Supports percentage-based split and K-fold CV

Author: Talha
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import StratifiedKFold, train_test_split

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


# Label map for IEEE dataset (9 classes)
LABEL_MAP = {
    "sleeping": 0,
    "resting_or_lying_down": 1,
    "reading": 2,
    "using_phone": 3,
    "eating": 4,
    "talking": 5,
    "sitting_up": 6,
    "adjusting_position": 7,
    "seizure": 8
}

# Reverse mapping
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

# Seizure label for binary stratification
SEIZURE_LABEL = 8


def get_majority_label(labels: np.ndarray) -> int:
    """
    Get the majority label from per-frame labels.
    
    Args:
        labels: Array of per-frame labels (T,)
        
    Returns:
        Most frequent label in the sequence
    """
    # Filter out padding (typically -1 or values > num_classes)
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return 0  # Default to first class if no valid labels
    
    unique, counts = np.unique(valid_labels, return_counts=True)
    return int(unique[np.argmax(counts)])


def create_stratified_split(
    labels: np.ndarray,
    test_ratio: float = 0.2,
    random_state: int = 42,
    binary_stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create stratified train/test split indices.
    
    Args:
        labels: Sample-level labels (N,)
        test_ratio: Ratio of test samples
        random_state: Random seed for reproducibility
        binary_stratify: If True, stratify by seizure vs non-seizure
                        If False, stratify by all classes
        
    Returns:
        train_indices, test_indices
    """
    n_samples = len(labels)
    indices = np.arange(n_samples)
    
    if binary_stratify:
        # Binary stratification: seizure vs non-seizure
        binary_labels = (labels == SEIZURE_LABEL).astype(int)
        stratify_labels = binary_labels
    else:
        stratify_labels = labels
    
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=random_state,
        stratify=stratify_labels
    )
    
    return train_idx, test_idx


def create_kfold_splits(
    labels: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
    binary_stratify: bool = True
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create K-fold cross-validation splits.
    
    Args:
        labels: Sample-level labels (N,)
        n_folds: Number of folds
        random_state: Random seed
        binary_stratify: If True, stratify by seizure vs non-seizure
        
    Returns:
        List of (train_indices, test_indices) tuples for each fold
    """
    n_samples = len(labels)
    indices = np.arange(n_samples)
    
    if binary_stratify:
        stratify_labels = (labels == SEIZURE_LABEL).astype(int)
    else:
        stratify_labels = labels
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    for train_idx, test_idx in skf.split(indices, stratify_labels):
        folds.append((train_idx, test_idx))
    
    return folds


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
    prompts_by_part = batch[0][3]
    
    return data_numpy, label, concept_vectors_by_part, prompts_by_part


def custom_collate_fn_with_action_prompts(batch):
    """
    Alternative collate function that also includes per-action prompts.
    """
    data_numpy = torch.stack([torch.from_numpy(item[0]) for item in batch])
    label = torch.tensor([item[1] for item in batch])
    
    concept_vectors_by_part = {}
    for key in batch[0][2].keys():
        concept_vectors_by_part[key] = torch.stack([
            torch.from_numpy(item[2][key]) for item in batch
        ])
    
    prompts_by_part = batch[0][3]
    action_prompts = [item[4] for item in batch]
    
    return data_numpy, label, concept_vectors_by_part, prompts_by_part, action_prompts


class Feeder(Dataset):
    """
    IEEE Seizure Dataset Feeder with concept-based prompt generation.
    
    Handles per-frame labels by computing majority label for each sample.
    Supports stratified train/test splitting and K-fold cross-validation.
    
    Args:
        data_path: Path to processed IEEE dataset (.npz)
        concepts_csv: Path to concept bank CSV file
        split: 'train' or 'test'
        test_ratio: Ratio of test samples (default: 0.2)
        fold: Fold index for K-fold CV (0 to n_folds-1), None for simple split
        n_folds: Number of folds for K-fold CV (default: 5)
        random_state: Random seed for split reproducibility
        binary_stratify: Stratify by seizure vs non-seizure (default: True)
        window_size: Target window size for temporal cropping (-1 for no crop)
        p_interval: Interval for valid_crop_resize
        random_rot: Apply random rotation augmentation
        random_move: Apply random move augmentation
        bone: Use bone features instead of joint features
        vel: Use velocity features
        normalization: Apply mean/std normalization
        prompt_mode: 'per_concept' or 'per_action'
        include_temporal_in_prompts: Include temporal concepts in prompts
        use_short_prompts: Use shorter prompts for CLIP efficiency
        return_action_prompts: Return per-action prompts in __getitem__
    """
    
    def __init__(
        self,
        data_path: str,
        concepts_csv: str,
        split: str = 'train',
        test_ratio: float = 0.2,
        fold: Optional[int] = None,
        n_folds: int = 5,
        random_state: int = 42,
        binary_stratify: bool = True,
        window_size: int = -1,
        p_interval: int = 1,
        random_choose: bool = False,
        random_shift: bool = False,
        random_move: bool = False,
        random_rot: bool = False,
        bone: bool = False,
        vel: bool = False,
        normalization: bool = False,
        debug: bool = False,
        prompt_mode: str = 'per_concept',
        include_temporal_in_prompts: bool = True,
        use_short_prompts: bool = True,
        return_action_prompts: bool = False,
        label_path: str = None,  # For compatibility, not used
        use_mmap: bool = False,  # For compatibility
    ):
        self.data_path = data_path
        self.concepts_csv = concepts_csv
        self.split = split
        self.test_ratio = test_ratio
        self.fold = fold
        self.n_folds = n_folds
        self.random_state = random_state
        self.binary_stratify = binary_stratify
        self.window_size = window_size
        self.p_interval = p_interval
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.normalization = normalization
        self.debug = debug
        self.prompt_mode = prompt_mode
        self.include_temporal_in_prompts = include_temporal_in_prompts
        self.use_short_prompts = use_short_prompts
        self.return_action_prompts = return_action_prompts
        
        # Augmentation probabilities
        self.aug_prob_rot = 0.1
        self.aug_prob_move = 0.0
        
        # Load data
        self.load_data()
        
        if normalization:
            self.get_mean_map()
    
    def load_data(self):
        """Load IEEE skeleton data and create train/test split."""
        # Load processed NPZ file
        npz_data = np.load(self.data_path, allow_pickle=True)
        
        # Get data arrays
        # Data is already in (N, C, T, V, M) format = (N, 3, T, 25, 1)
        all_data = npz_data['data']
        all_frame_labels = npz_data['labels']  # (N, T) per-frame labels
        self.sample_names_all = npz_data['sample_names']
        self.frame_counts = npz_data['frame_counts']
        
        # Load metadata
        self.label_map = json.loads(str(npz_data['label_map']))
        self.num_classes = int(npz_data['num_classes'])
        self.max_frames = int(npz_data['max_frames'])
        
        # Compute majority labels for each sample
        all_labels = np.array([
            get_majority_label(all_frame_labels[i, :self.frame_counts[i]])
            for i in range(len(all_data))
        ])
        
        # Create train/test split
        if self.fold is not None:
            # K-fold cross-validation
            folds = create_kfold_splits(
                all_labels, 
                n_folds=self.n_folds,
                random_state=self.random_state,
                binary_stratify=self.binary_stratify
            )
            train_idx, test_idx = folds[self.fold]
        else:
            # Simple stratified split
            train_idx, test_idx = create_stratified_split(
                all_labels,
                test_ratio=self.test_ratio,
                random_state=self.random_state,
                binary_stratify=self.binary_stratify
            )
        
        # Select data based on split
        if self.split == 'train':
            self.indices = train_idx
        elif self.split == 'test':
            self.indices = test_idx
        else:
            raise ValueError(f"split must be 'train' or 'test', got {self.split}")
        
        self.data = all_data[self.indices]
        self.label = all_labels[self.indices]
        self.frame_labels = all_frame_labels[self.indices]
        self.sample_name = [self.sample_names_all[i] for i in self.indices]
        
        # Print split statistics
        print(f"IEEE Dataset - {self.split} split:")
        print(f"  Total samples: {len(self.data)}")
        print(f"  Data shape: {self.data.shape}")
        
        # Label distribution
        unique, counts = np.unique(self.label, return_counts=True)
        print(f"  Label distribution:")
        for label_id, count in zip(unique, counts):
            label_name = LABEL_NAMES.get(label_id, f"unknown_{label_id}")
            print(f"    {label_name}: {count}")
        
        # Seizure vs non-seizure
        n_seizure = np.sum(self.label == SEIZURE_LABEL)
        n_normal = len(self.label) - n_seizure
        print(f"  Seizure: {n_seizure}, Non-seizure: {n_normal}")
        
        # Load concept bank
        if self.concepts_csv is not None:
            self.concept_df = pd.read_csv(self.concepts_csv)
            self._setup_concepts()
        else:
            raise ValueError("concepts_csv must be provided")
    
    def _setup_concepts(self):
        """Setup concept matrix, vocabulary, and pre-compute prompts."""
        # Extract concept information
        self.concept_matrix_np, self.action_names, self.vocab, self.ordered_concept_dict = \
            get_concept_info(self.concept_df)
        
        # Concept counts per body part
        concepts_list = [v.split('_', 1)[0] if not v.startswith('temporal') 
                        else '_'.join(v.split('_')[:2]) for v in self.vocab]
        unique_parts, counts = np.unique(concepts_list, return_counts=True)
        self.concept_counts = dict(zip(unique_parts, counts))
        
        # Pre-compute per-concept prompts
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
        
        # Concept counts for model
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
        data_numpy = self.data[index].copy()
        label = int(self.label[index])
        
        # Get concept vector for the label and split by body part
        concept_vector = self.concept_matrix_np[label]
        concept_vectors_by_part = split_vector(concept_vector, self.ordered_concept_dict)
        
        # Process skeleton data
        data_numpy = np.array(data_numpy)
        
        # Valid frame handling
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        if valid_frame_num == 0:
            valid_frame_num = data_numpy.shape[1]  # Use all frames if detection fails
        
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
        elif not self.vel:
            # Add centering for consistency
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
        """Return concept information needed by the model."""
        return {
            'n_concepts': len(self.vocab),
            'n_spatial_concepts': self.n_spatial_concepts,
            'n_temporal_concepts': self.n_temporal_concepts,
            'n_actions': len(self.action_names),
            'vocab': self.vocab,
            'action_names': self.action_names,
            'concept_counts': self.concept_counts,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_split_info(data_path: str, test_ratio: float = 0.2, n_folds: int = 5, 
                   random_state: int = 42, binary_stratify: bool = True) -> Dict:
    """
    Get information about train/test splits without loading full data.
    Useful for planning experiments.
    """
    npz_data = np.load(data_path, allow_pickle=True)
    all_frame_labels = npz_data['labels']
    frame_counts = npz_data['frame_counts']
    
    all_labels = np.array([
        get_majority_label(all_frame_labels[i, :frame_counts[i]])
        for i in range(len(all_frame_labels))
    ])
    
    # Simple split info
    train_idx, test_idx = create_stratified_split(
        all_labels, test_ratio, random_state, binary_stratify
    )
    
    # K-fold info
    folds = create_kfold_splits(all_labels, n_folds, random_state, binary_stratify)
    
    return {
        'total_samples': len(all_labels),
        'simple_split': {
            'train': len(train_idx),
            'test': len(test_idx)
        },
        'kfold_splits': [
            {'train': len(f[0]), 'test': len(f[1])} for f in folds
        ],
        'label_distribution': dict(zip(*np.unique(all_labels, return_counts=True)))
    }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("IEEE Feeder module loaded successfully")
    print("Use custom_collate_fn for DataLoader")
    print("Use custom_collate_fn_with_action_prompts if you need per-action prompts")
    
    # Example usage:
    # feeder = Feeder(
    #     data_path='../processed/ieee_seizure_dataset.npz',
    #     concepts_csv='../concepts/seizure_concepts.csv',
    #     split='train',
    #     test_ratio=0.2,
    #     fold=None,  # or 0-4 for K-fold
    #     binary_stratify=True
    # )
