#!/usr/bin/env python
"""
Qualitative Evaluation Script for IEEE Seizure Detection Dataset

Generates CVPR-quality visualizations showing:
- GT vs Predicted action segments for patient episodes
- Full timeline visualization per patient
- Per-patient performance analysis

Key differences from VSVIG evaluator:
- IEEE dataset uses naming: S{patient}_{episode}_{clip_start_frame}
- No source_videos field - extract from sample names
- Groups by patient (all episodes) or patient+episode

Author: Talha
"""
#%%
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#%%
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Sequence, Union
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects

from feeders.feeder_ieee import Feeder, custom_collate_fn, LABEL_NAMES, LABEL_MAP, SEIZURE_LABEL
from model.model_sup_logic import SkeletonACL_CLIP_Logic  # Same model as VSVIG (9 classes)


# =============================================================================
# COLOR PALETTE (CVPR-quality, your specified colors)
# =============================================================================
SEIZURE_PALETTE = {
    'seizure': '#F69EA7',           # Pink for seizure
    'play_with_phone_tablet': '#70B28C',          # Sage green
    'resting_or_lying_down': '#A6DCD3',  # Teal/mint
    'eat_mean': '#8ba0a4',#'#92BDCC',           # Steel blue  
    'sitting_up': '#89A8D9',  # Soft blue
    'adjusting_position': '#C5DCE6',          # Light blue-gray
    'sleeping': '#F2D6A2',           # Warm beige/gold
    'talking': '#FFEDB8',        # Light yellow
    'reading': '#BFA6A7', # Dusty rose/mauve
    'background': '#E8E8E8',        # Light gray for background/gaps
}

# Binary palette
BINARY_PALETTE = {
    'seizure': '#F69EA7',           # Pink for seizure
    'normal': '#89A8D9',            # Blue for normal
    'background': '#E8E8E8',
}


def init_seed(seed=1):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_ieee_sample_name(name: str) -> Tuple[str, int, int]:
    """
    Parse IEEE sample name: S{patient}_{episode}_{clip_start_frame}
    
    Returns:
        patient_id: e.g., 'S0'
        episode: e.g., 0
        clip_start: e.g., 0, 20, 40
    """
    parts = name.split('_')
    patient_id = parts[0]
    episode = int(parts[1])
    clip_start = int(parts[2])
    return patient_id, episode, clip_start


# =============================================================================
# SEGMENTATION UTILITIES
# =============================================================================
def frame_labels_to_segments(labels: Sequence[int]) -> List[Tuple[int, int, int]]:
    """Run-length encode per-frame labels into segments [start, end, label]."""
    labels = np.asarray(labels)
    if labels.size == 0:
        return []
    segments = []
    start = 0
    prev = int(labels[0])
    for i in range(1, len(labels)):
        if int(labels[i]) != prev:
            segments.append((start, i, prev))
            start = i
            prev = int(labels[i])
    segments.append((start, len(labels), prev))
    return segments


def merge_short_segments(
    segments: List[Tuple[int, int, int]],
    min_len: int,
    merge_with_neighbors: bool = True
) -> List[Tuple[int, int, int]]:
    """Merge segments shorter than min_len with neighbors."""
    if min_len <= 1 or len(segments) <= 1:
        return segments
    
    result = []
    for seg in segments:
        if result and seg[2] == result[-1][2]:
            # Merge with previous if same label
            result[-1] = (result[-1][0], seg[1], seg[2])
        else:
            result.append(seg)
    
    # Remove very short segments by merging with neighbors
    if merge_with_neighbors:
        changed = True
        while changed:
            changed = False
            new_result = []
            for i, seg in enumerate(result):
                seg_len = seg[1] - seg[0]
                if seg_len < min_len:
                    if new_result:
                        # Merge with previous
                        prev = new_result[-1]
                        new_result[-1] = (prev[0], seg[1], prev[2])
                        changed = True
                    elif i + 1 < len(result):
                        # Merge with next
                        result[i+1] = (seg[0], result[i+1][1], result[i+1][2])
                        changed = True
                else:
                    new_result.append(seg)
            result = new_result
    
    return result


# =============================================================================
# CVPR-QUALITY PLOTTING
# =============================================================================
def plot_gt_vs_pred_bars(
    gt_segments: List[Tuple[float, float, str]],
    pred_segments: List[Tuple[float, float, str]],
    total_duration_s: float,
    class_names: List[str],
    palette: Dict[str, str],
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 2.0),
    show_legend: bool = True,
    show_time_axis: bool = True,
    show_boundaries: bool = True,
    boundary_color: str = '#666666',
    boundary_alpha: float = 0.3,
    savepath: Optional[str] = None,
    dpi: int = 300,
    return_fig: bool = False,
):
    """
    Create CVPR-quality GT vs Prediction bar visualization.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    bar_height = 0.35
    gap = 0.15
    y_gt = 0.5 + gap/2
    y_pred = 0.5 - gap/2 - bar_height
    
    def draw_segments(segments, y, label_text):
        for start, end, class_name in segments:
            color = palette.get(class_name, '#CCCCCC')
            rect = Rectangle(
                (start, y), end - start, bar_height,
                facecolor=color, edgecolor='white', linewidth=0.5, zorder=2
            )
            ax.add_patch(rect)
        ax.text(-total_duration_s * 0.015, y + bar_height/2, label_text,
               ha='right', va='center', fontsize=9, fontweight='bold', color='#333333')
    
    draw_segments(gt_segments, y_gt, 'GT')
    draw_segments(pred_segments, y_pred, 'Pred')
    
    if show_boundaries:
        for start, end, _ in gt_segments:
            ax.axvline(start, color=boundary_color, alpha=boundary_alpha, 
                      linewidth=0.5, linestyle='--', zorder=1)
    
    ax.set_xlim(0, total_duration_s)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    
    if show_time_axis:
        if total_duration_s > 120:
            tick_interval = 60 if total_duration_s <= 600 else 120
            tick_positions = np.arange(0, total_duration_s + 1, tick_interval)
            tick_labels = [f"{int(t//60)}:{int(t%60):02d}" for t in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
        else:
            tick_interval = 30 if total_duration_s > 60 else 10
            ax.set_xticks(np.arange(0, total_duration_s + 1, tick_interval))
    else:
        ax.set_xticks([])
    
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    if subtitle:
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
               ha='center', va='bottom', fontsize=8, color='#666666')
    
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.grid(axis='x', color='#EEEEEE', linewidth=0.5, zorder=0)
    
    if show_legend:
        classes_in_plots = set()
        for _, _, c in gt_segments:
            classes_in_plots.add(c)
        for _, _, c in pred_segments:
            classes_in_plots.add(c)
        
        legend_classes = sorted(
            classes_in_plots,
            key=lambda x: (x == 'seizure', class_names.index(x) if x in class_names else 999)
        )
        
        handles = [
            Patch(facecolor=palette.get(c, '#CCCCCC'), edgecolor='white',
                  linewidth=0.5, label=c.replace('_', ' ').title())
            for c in legend_classes
        ]
        
        ax.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(len(handles), 6),
            frameon=False,
            columnspacing=1.0,
            handlelength=1.5,
            handleheight=0.8,
        )
    
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✅ Saved: {savepath}")
    
    if return_fig:
        return fig
    else:
        plt.close(fig)


def plot_multi_video_comparison(
    video_results: List[Dict],
    class_names: List[str],
    palette: Dict[str, str],
    title: str = "Action Segmentation: Ground Truth vs Predictions",
    figsize_per_video: Tuple[float, float] = (14, 1.2),
    max_videos: int = 6,
    savepath: Optional[str] = None,
    dpi: int = 300,
):
    """Create a multi-panel figure showing GT vs Pred for multiple videos."""
    n_videos = min(len(video_results), max_videos)
    
    fig_height = figsize_per_video[1] * n_videos + 1.5
    fig, axes = plt.subplots(
        n_videos, 1,
        figsize=(figsize_per_video[0], fig_height),
        constrained_layout=True
    )
    
    if n_videos == 1:
        axes = [axes]
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'pdf.fonttype': 42,
    })
    
    bar_height = 0.35
    gap = 0.1
    
    for idx, (ax, result) in enumerate(zip(axes, video_results[:max_videos])):
        gt_segments = result['gt_segments']
        pred_segments = result['pred_segments']
        duration = result['duration_s']
        video_name = result['video_name']
        metrics = result.get('metrics', {})
        
        y_gt = 0.5 + gap/2
        y_pred = 0.5 - gap/2 - bar_height
        
        for start, end, class_name in gt_segments:
            color = palette.get(class_name, '#CCCCCC')
            rect = Rectangle(
                (start, y_gt), end - start, bar_height,
                facecolor=color, edgecolor='white', linewidth=0.4
            )
            ax.add_patch(rect)
        
        for start, end, class_name in pred_segments:
            color = palette.get(class_name, '#CCCCCC')
            rect = Rectangle(
                (start, y_pred), end - start, bar_height,
                facecolor=color, edgecolor='white', linewidth=0.4
            )
            ax.add_patch(rect)
        
        ax.text(-duration * 0.01, y_gt + bar_height/2, 'GT',
               ha='right', va='center', fontsize=7, fontweight='bold', color='#333333')
        ax.text(-duration * 0.01, y_pred + bar_height/2, 'Pred',
               ha='right', va='center', fontsize=7, fontweight='bold', color='#333333')
        
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([])
        
        acc = metrics.get('accuracy', 0)
        ax.set_title(f"{video_name} | Acc: {acc*100:.1f}%", fontsize=9, fontweight='bold', pad=3)
        
        if idx == n_videos - 1:
            tick_interval = 60 if duration > 120 else 30
            ax.set_xticks(np.arange(0, duration + 1, tick_interval))
        else:
            ax.set_xticks([])
        
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.spines['bottom'].set_linewidth(0.5)
    
    classes_in_plots = set()
    for result in video_results[:max_videos]:
        for _, _, c in result['gt_segments']:
            classes_in_plots.add(c)
        for _, _, c in result['pred_segments']:
            classes_in_plots.add(c)
    
    legend_classes = sorted(
        classes_in_plots,
        key=lambda x: (x == 'seizure', class_names.index(x) if x in class_names else 999)
    )
    
    handles = [
        Patch(facecolor=palette.get(c, '#CCCCCC'), edgecolor='white',
              linewidth=0.5, label=c.replace('_', ' ').title())
        for c in legend_classes
    ]
    
    fig.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(len(handles), 6),
        frameon=False,
        fontsize=8,
    )
    
    if savepath:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"✅ Saved: {savepath}")
    
    plt.close(fig)


# =============================================================================
# QUALITATIVE EVALUATOR CLASS FOR IEEE DATASET
# =============================================================================
class QualitativeEvaluatorIEEE:
    """Generate qualitative visualizations for IEEE seizure dataset (zero-shot from VSVIG model)."""
    
    def __init__(
        self, 
        config_path: str, 
        checkpoint_path: str = None, 
        binary_mode: bool = False,
        group_by: str = 'patient',  # 'patient' or 'episode'
        ieee_data_path: str = None,  # Override data path for IEEE
        concepts_csv: str = None,    # Override concepts CSV
    ):
        """
        Args:
            config_path: Path to YAML config file (use VSVIG config for model)
            checkpoint_path: Path to model checkpoint (use VSVIG checkpoint)
            binary_mode: If True, use binary (seizure vs normal) classification
            group_by: 'patient' to group all episodes per patient,
                      'episode' to show each patient-episode separately
            ieee_data_path: Path to IEEE dataset NPZ file (overrides config)
            concepts_csv: Path to concepts CSV (overrides config)
        """
        with open(config_path, 'r') as f:
            self.args = yaml.safe_load(f)
        
        # Override data paths for IEEE zero-shot testing
        if ieee_data_path:
            self.args['test_feeder_args']['data_path'] = ieee_data_path
        if concepts_csv:
            self.args['test_feeder_args']['concepts_csv'] = concepts_csv
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        init_seed(self.args.get('seed', 1))
        
        self.n_classes = self.args['model_args']['num_class']
        self.binary_mode = binary_mode
        self.group_by = group_by
        
        # Setup class names and palette
        if binary_mode:
            self.class_names = ['normal', 'seizure']
            self.palette = BINARY_PALETTE
            self.id_to_name = {0: 'normal', 1: 'seizure'}
        else:
            self.class_names = [LABEL_NAMES.get(i, f"class_{i}") for i in range(self.n_classes)]
            self.palette = SEIZURE_PALETTE
            self.id_to_name = LABEL_NAMES
        
        # Window parameters
        self.window_size = self.args['train_feeder_args'].get('window_size', 64)
        self.fps = 30.0  # Assuming 30 FPS
        
        # Output directory - use separate dir for IEEE zero-shot
        self.output_dir = Path(self.args['work_dir']) / 'qualitative_ieee_zeroshot'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data and model
        self._load_data()
        self._load_model(checkpoint_path)
        
        print(f"\n{'='*60}")
        print("✅ IEEE QUALITATIVE EVALUATOR (ZERO-SHOT) INITIALIZED")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.n_classes}")
        print(f"Binary mode: {self.binary_mode}")
        print(f"Group by: {self.group_by}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Output dir: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _load_data(self):
        """Load test dataset and parse sample names."""
        # Filter out VSVIG-specific args that IEEE feeder doesn't support
        feeder_args = self.args['test_feeder_args'].copy()
        vsvig_only_args = ['video_aware_split']
        for arg in vsvig_only_args:
            feeder_args.pop(arg, None)
        
        self.test_dataset = Feeder(**feeder_args)
        
        # Create dataloader WITHOUT shuffling
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            collate_fn=custom_collate_fn,
        )
        
        # Parse sample names to extract patient/episode/clip info
        self.sample_info = []
        for name in self.test_dataset.sample_name:
            patient_id, episode, clip_start = parse_ieee_sample_name(name)
            self.sample_info.append({
                'name': name,
                'patient': patient_id,
                'episode': episode,
                'clip_start': clip_start,
            })
        
        # Group samples by patient (or patient+episode)
        self.patient_episodes = defaultdict(lambda: defaultdict(list))
        for idx, info in enumerate(self.sample_info):
            self.patient_episodes[info['patient']][info['episode']].append((
                info['clip_start'],  # for sorting
                idx,                 # dataset index
                info['name'],        # sample name
            ))
        
        # Sort clips within each episode by clip_start
        for patient in self.patient_episodes:
            for episode in self.patient_episodes[patient]:
                self.patient_episodes[patient][episode].sort(key=lambda x: x[0])
        
        print(f"Found {len(self.patient_episodes)} patients in test set")
        for patient in sorted(self.patient_episodes.keys(), key=lambda x: int(x[1:])):
            n_eps = len(self.patient_episodes[patient])
            n_clips = sum(len(self.patient_episodes[patient][e]) for e in self.patient_episodes[patient])
            print(f"  {patient}: {n_eps} episodes, {n_clips} clips")
    
    def _load_model(self, checkpoint_path: str = None):
        """Load model with checkpoint."""
        self.model = SkeletonACL_CLIP_Logic(self.args, self.device).to(self.device)
        
        if checkpoint_path is None:
            checkpoint_path = Path(self.args['work_dir']) / 'best_model.pth'
        
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
        else:
            print(f"⚠️  No checkpoint found at {checkpoint_path}")
        
        self.model.eval()
    
    @torch.no_grad()
    def collect_predictions(self) -> Dict[int, Dict]:
        """
        Collect predictions for all samples indexed by dataset index.
        
        Returns:
            Dict mapping sample_idx to {
                'gt_label': int,
                'pred_label': int,
                'pred_probs': np.array,
                'confidence': float,
            }
        """
        print("Collecting predictions...")
        
        predictions = {}
        
        for idx, (batch_data, batch_label, batch_concept_vecs, batch_prompts) in enumerate(tqdm(self.test_loader)):
            batch_data = batch_data.to(self.device)
            batch_label = batch_label.to(self.device)
            
            # Prepare concepts_gt for model_sup_logic forward signature
            concepts_gt = torch.cat(
                (batch_concept_vecs['full_body'],
                 batch_concept_vecs['temporal']),
                dim=1
            ).float().to(self.device)
            
            # Get prediction using model_sup_logic forward
            action_probs, concept_probs = self.model(batch_data, batch_label, concepts_gt)
            
            probs = action_probs
            pred_class = action_probs.argmax(dim=1).item()
            confidence = probs.max().item()
            
            gt_label = batch_label.item()
            
            # Convert to binary if needed
            if self.binary_mode:
                gt_label = 1 if gt_label == SEIZURE_LABEL else 0
                pred_class = 1 if pred_class == SEIZURE_LABEL else 0
            
            predictions[idx] = {
                'gt_label': gt_label,
                'pred_label': pred_class,
                'pred_probs': probs.cpu().numpy()[0],
                'confidence': confidence,
            }
        
        return predictions
    
    def create_video_segments(
        self,
        labels: List[int],
        clip_positions: List[int],
        clip_duration_frames: int = 127,  # IEEE uses 127-128 frames
    ) -> Tuple[List[Tuple[float, float, str]], float]:
        """
        Convert clip-level labels to time segments based on clip positions.
        
        IEEE clips have explicit start positions, so we use those to reconstruct timeline.
        
        Args:
            labels: List of per-clip labels
            clip_positions: List of clip start positions (frame indices)
            clip_duration_frames: Duration of each clip in frames
            
        Returns:
            segments: List of (start_s, end_s, class_name)
            total_duration_s: Total video duration in seconds
        """
        if not clip_positions:
            return [], 0.0
        
        # Determine total duration from clips
        max_end = max(pos + clip_duration_frames for pos in clip_positions)
        total_frames = max_end
        total_duration_s = total_frames / self.fps
        
        # Create frame-level labels by assigning each clip's label to its frames
        frame_labels = np.zeros(total_frames, dtype=int)
        frame_counts = np.zeros(total_frames, dtype=int)
        
        for label, start in zip(labels, clip_positions):
            end = min(start + clip_duration_frames, total_frames)
            for f in range(start, end):
                # Voting: last clip wins for overlapping regions
                frame_labels[f] = label
                frame_counts[f] += 1
        
        # Convert to segments
        raw_segments = frame_labels_to_segments(frame_labels)
        
        # Merge short segments (less than 1 second = 30 frames)
        min_segment_frames = int(1.0 * self.fps)
        segments = merge_short_segments(raw_segments, min_segment_frames)
        
        # Convert to seconds and class names
        result_segments = []
        for start_f, end_f, label_id in segments:
            start_s = start_f / self.fps
            end_s = end_f / self.fps
            class_name = self.id_to_name.get(label_id, f"class_{label_id}")
            result_segments.append((start_s, end_s, class_name))
        
        return result_segments, total_duration_s
    
    def compute_segment_metrics(
        self,
        gt_labels: List[int],
        pred_labels: List[int],
    ) -> Dict:
        """Compute per-video metrics."""
        gt = np.array(gt_labels)
        pred = np.array(pred_labels)
        
        accuracy = (gt == pred).mean()
        
        if self.binary_mode:
            tp = ((gt == 1) & (pred == 1)).sum()
            tn = ((gt == 0) & (pred == 0)).sum()
            fp = ((gt == 0) & (pred == 1)).sum()
            fn = ((gt == 1) & (pred == 0)).sum()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            }
        else:
            return {'accuracy': accuracy}
    
    def generate_patient_plot(
        self,
        patient_id: str,
        predictions: Dict[int, Dict],
        savepath: Optional[str] = None,
    ) -> Dict:
        """
        Generate visualization for a single patient (all episodes combined).
        
        Creates a multi-row figure with one row per episode.
        """
        episodes = self.patient_episodes[patient_id]
        n_episodes = len(episodes)
        
        if n_episodes == 0:
            return None
        
        # Collect data for each episode
        episode_results = []
        
        for episode_id in sorted(episodes.keys()):
            clips = episodes[episode_id]
            
            gt_labels = []
            pred_labels = []
            clip_positions = []
            
            for clip_start, dataset_idx, name in clips:
                if dataset_idx in predictions:
                    pred = predictions[dataset_idx]
                    gt_labels.append(pred['gt_label'])
                    pred_labels.append(pred['pred_label'])
                    clip_positions.append(clip_start)
            
            if not gt_labels:
                continue
            
            gt_segments, duration = self.create_video_segments(gt_labels, clip_positions)
            pred_segments, _ = self.create_video_segments(pred_labels, clip_positions)
            metrics = self.compute_segment_metrics(gt_labels, pred_labels)
            
            episode_results.append({
                'episode': episode_id,
                'gt_segments': gt_segments,
                'pred_segments': pred_segments,
                'duration_s': duration,
                'metrics': metrics,
                'n_clips': len(gt_labels),
            })
        
        if not episode_results:
            return None
        
        # Single episode: simple plot
        if len(episode_results) == 1:
            return self._plot_single_episode(patient_id, episode_results[0], savepath)
        
        # Multiple episodes: multi-row figure
        return self._plot_multi_episode(patient_id, episode_results, savepath)
    
    def _plot_single_episode(
        self,
        patient_id: str,
        episode_data: Dict,
        savepath: Optional[str] = None,
    ) -> Dict:
        """Plot a single episode for a patient."""
        gt_segments = episode_data['gt_segments']
        pred_segments = episode_data['pred_segments']
        duration = episode_data['duration_s']
        metrics = episode_data['metrics']
        n_clips = episode_data['n_clips']
        episode_id = episode_data['episode']
        
        # Build subtitle
        duration_str = f"{int(duration//60)}:{int(duration%60):02d}"
        if self.binary_mode:
            subtitle = f"Episode {episode_id} | Duration: {duration_str} | {n_clips} clips | Acc: {metrics['accuracy']*100:.1f}% | Sens: {metrics.get('sensitivity', 0)*100:.1f}%"
        else:
            subtitle = f"Episode {episode_id} | Duration: {duration_str} | {n_clips} clips | Accuracy: {metrics['accuracy']*100:.1f}%"
        
        fig_width = max(14, min(24, 14 + (duration / 300) * 6))
        
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
            'font.size': 9,
            'pdf.fonttype': 42,
        })
        
        fig, ax = plt.subplots(figsize=(fig_width, 2.2), constrained_layout=True)
        
        bar_height = 0.35
        gap = 0.15
        y_gt = 0.5 + gap/2
        y_pred = 0.5 - gap/2 - bar_height
        
        # Draw segments
        for start, end, class_name in gt_segments:
            color = self.palette.get(class_name, '#CCCCCC')
            rect = Rectangle(
                (start, y_gt), end - start, bar_height,
                facecolor=color, edgecolor='white', linewidth=0.5, zorder=2
            )
            ax.add_patch(rect)
        
        for start, end, class_name in pred_segments:
            color = self.palette.get(class_name, '#CCCCCC')
            rect = Rectangle(
                (start, y_pred), end - start, bar_height,
                facecolor=color, edgecolor='white', linewidth=0.5, zorder=2
            )
            ax.add_patch(rect)
        
        # Labels
        ax.text(-duration * 0.015, y_gt + bar_height/2, 'GT',
               ha='right', va='center', fontsize=9, fontweight='bold', color='#333333')
        ax.text(-duration * 0.015, y_pred + bar_height/2, 'Pred',
               ha='right', va='center', fontsize=9, fontweight='bold', color='#333333')
        
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([])
        
        # Time axis
        if duration > 120:
            tick_interval = 60 if duration <= 600 else 120
            tick_positions = np.arange(0, duration + 1, tick_interval)
            tick_labels = [f"{int(t//60)}:{int(t%60):02d}" for t in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
        else:
            tick_interval = 30 if duration > 60 else 10
            ax.set_xticks(np.arange(0, duration + 1, tick_interval))
        
        ax.set_title(f"Patient: {patient_id}", fontsize=10, fontweight='bold', pad=10)
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
               ha='center', va='bottom', fontsize=8, color='#666666')
        
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.spines['bottom'].set_linewidth(0.5)
        ax.grid(axis='x', color='#EEEEEE', linewidth=0.5, zorder=0)
        
        # Legend
        classes_in_plot = set()
        for _, _, c in gt_segments:
            classes_in_plot.add(c)
        for _, _, c in pred_segments:
            classes_in_plot.add(c)
        
        legend_classes = sorted(
            classes_in_plot,
            key=lambda x: (x == 'seizure', self.class_names.index(x) if x in self.class_names else 999)
        )
        
        handles = [
            Patch(facecolor=self.palette.get(c, '#CCCCCC'), edgecolor='white',
                  linewidth=0.5, label=c.replace('_', ' ').title())
            for c in legend_classes
        ]
        
        ax.legend(
            handles=handles,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(len(handles), 6),
            frameon=False,
        )
        
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"✅ Saved: {savepath}")
        
        plt.close(fig)
        
        return {
            'patient': patient_id,
            'video_name': f"{patient_id}_ep{episode_id}",
            'gt_segments': gt_segments,
            'pred_segments': pred_segments,
            'duration_s': duration,
            'metrics': metrics,
        }
    
    def _plot_multi_episode(
        self,
        patient_id: str,
        episode_results: List[Dict],
        savepath: Optional[str] = None,
    ) -> Dict:
        """Plot multiple episodes for a patient in a multi-row figure."""
        n_episodes = len(episode_results)
        
        fig_height = 1.8 * n_episodes + 1.5
        fig, axes = plt.subplots(
            n_episodes, 1,
            figsize=(14, fig_height),
            constrained_layout=True
        )
        
        if n_episodes == 1:
            axes = [axes]
        
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 9,
            'pdf.fonttype': 42,
        })
        
        bar_height = 0.35
        gap = 0.12
        
        # Compute overall metrics
        all_gt = []
        all_pred = []
        
        for idx, ep_data in enumerate(episode_results):
            ax = axes[idx]
            
            gt_segments = ep_data['gt_segments']
            pred_segments = ep_data['pred_segments']
            duration = ep_data['duration_s']
            metrics = ep_data['metrics']
            episode_id = ep_data['episode']
            n_clips = ep_data['n_clips']
            
            y_gt = 0.5 + gap/2
            y_pred = 0.5 - gap/2 - bar_height
            
            # Draw segments
            for start, end, class_name in gt_segments:
                color = self.palette.get(class_name, '#CCCCCC')
                rect = Rectangle(
                    (start, y_gt), end - start, bar_height,
                    facecolor=color, edgecolor='white', linewidth=0.4
                )
                ax.add_patch(rect)
            
            for start, end, class_name in pred_segments:
                color = self.palette.get(class_name, '#CCCCCC')
                rect = Rectangle(
                    (start, y_pred), end - start, bar_height,
                    facecolor=color, edgecolor='white', linewidth=0.4
                )
                ax.add_patch(rect)
            
            # Row labels
            ax.text(-duration * 0.015, y_gt + bar_height/2, 'GT',
                   ha='right', va='center', fontsize=8, fontweight='bold', color='#333333')
            ax.text(-duration * 0.015, y_pred + bar_height/2, 'Pred',
                   ha='right', va='center', fontsize=8, fontweight='bold', color='#333333')
            
            # Episode label on right
            duration_str = f"{int(duration//60)}:{int(duration%60):02d}"
            ax.text(1.01, 0.5, f"Ep {episode_id}\n{duration_str}\nAcc: {metrics['accuracy']*100:.0f}%",
                   transform=ax.transAxes, ha='left', va='center', fontsize=7, color='#666666')
            
            ax.set_xlim(0, duration)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([])
            
            # X-axis only on bottom
            if idx == n_episodes - 1:
                tick_interval = 60 if duration > 120 else 30
                ax.set_xticks(np.arange(0, duration + 1, tick_interval))
            else:
                ax.set_xticks([])
            
            for spine in ['top', 'right', 'left']:
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#DDDDDD')
            ax.spines['bottom'].set_linewidth(0.5)
        
        # Compute overall accuracy from all episodes
        for ep_data in episode_results:
            # We need original labels, not segments
            pass  # Already computed per-episode metrics
        
        avg_acc = np.mean([ep['metrics']['accuracy'] for ep in episode_results])
        
        fig.suptitle(
            f"Patient: {patient_id} | {n_episodes} episodes | Avg Accuracy: {avg_acc*100:.1f}%",
            fontsize=10, fontweight='bold', y=1.0
        )
        
        # Legend
        classes_in_plot = set()
        for ep in episode_results:
            for _, _, c in ep['gt_segments']:
                classes_in_plot.add(c)
            for _, _, c in ep['pred_segments']:
                classes_in_plot.add(c)
        
        legend_classes = sorted(
            classes_in_plot,
            key=lambda x: (x == 'seizure', self.class_names.index(x) if x in self.class_names else 999)
        )
        
        handles = [
            Patch(facecolor=self.palette.get(c, '#CCCCCC'), edgecolor='white',
                  linewidth=0.5, label=c.replace('_', ' ').title())
            for c in legend_classes
        ]
        
        fig.legend(
            handles=handles,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(handles), 6),
            frameon=False,
            fontsize=8,
        )
        
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"✅ Saved: {savepath}")
        
        plt.close(fig)
        
        # Combine all segments for return value
        total_duration = sum(ep['duration_s'] for ep in episode_results)
        
        return {
            'patient': patient_id,
            'video_name': patient_id,
            'gt_segments': episode_results[0]['gt_segments'],  # First episode
            'pred_segments': episode_results[0]['pred_segments'],
            'duration_s': total_duration,
            'n_episodes': n_episodes,
            'metrics': {'accuracy': avg_acc},
        }
    
    def generate_all_visualizations(
        self,
        max_patients: int = 100,
        generate_individual: bool = True,
        generate_combined: bool = True,
    ):
        """
        Generate all qualitative visualizations.
        
        Args:
            max_patients: Maximum number of patients to visualize
            generate_individual: Whether to save individual patient plots
            generate_combined: Whether to generate combined comparison plot
        """
        print("\n" + "="*60)
        print("🎨 GENERATING IEEE QUALITATIVE VISUALIZATIONS")
        print("="*60 + "\n")
        
        # Collect predictions
        predictions = self.collect_predictions()
        
        # Sort patients by number of clips
        sorted_patients = sorted(
            self.patient_episodes.keys(),
            key=lambda p: sum(
                len(self.patient_episodes[p][e]) 
                for e in self.patient_episodes[p]
            ),
            reverse=True
        )
        
        all_results = []
        
        for patient_id in tqdm(sorted_patients[:max_patients], desc="Processing patients"):
            savepath = self.output_dir / f"patient_{patient_id}.png" if generate_individual else None
            result = self.generate_patient_plot(patient_id, predictions, savepath)
            if result:
                all_results.append(result)
        
        # Generate combined multi-patient plot
        if generate_combined and len(all_results) > 1:
            plot_multi_video_comparison(
                video_results=all_results[:6],
                class_names=self.class_names,
                palette=self.palette,
                title="IEEE Dataset: Ground Truth vs Predictions by Patient",
                savepath=str(self.output_dir / "multi_patient_comparison.png"),
            )
        
        # Save summary
        summary = {
            'n_patients': len(all_results),
            'avg_accuracy': np.mean([r['metrics']['accuracy'] for r in all_results]),
            'per_patient': [
                {
                    'patient': r.get('patient', r['video_name']),
                    'duration_s': r['duration_s'],
                    'metrics': r['metrics'],
                }
                for r in all_results
            ]
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Generated visualizations for {len(all_results)} patients")
        print(f"   Average accuracy: {summary['avg_accuracy']*100:.1f}%")
        print(f"   Saved to: {self.output_dir}")
        
        return all_results


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Use VSVIG config for model architecture (same 9-class model)
    # but we override test_feeder_args to use IEEE data
    config_path = '/mnt/ssd/Talha/reason/config/szr/config_skeleton_vsvig.yaml'
    checkpoint = '/mnt/ssd/Talha/reason/work_dir/vsvig3_f1_logic2/best_model.pth'
    
    # IEEE data paths to override
    ieee_data_path = '/mnt/ssd/Talha/reason/data/ieee_seizure_dataset2.npz'
    concepts_csv = '/mnt/ssd/Talha/reason/concepts/szr_cbm.csv'
    
    max_patients = 100
    
    evaluator = QualitativeEvaluatorIEEE(
        config_path,
        checkpoint_path=checkpoint,
        binary_mode=False,
        group_by='patient',  # 'patient' shows all episodes per patient
        ieee_data_path=ieee_data_path,  # Override data path for IEEE
        concepts_csv=concepts_csv,
    )
    
    evaluator.generate_all_visualizations(
        max_patients=max_patients,
        generate_individual=True,
        generate_combined=True,
    )


if __name__ == '__main__':
    main()
#%%
