#!/usr/bin/env python
"""
Qualitative Evaluation Script for Seizure Detection

Generates CVPR-quality visualizations showing:
- GT vs Predicted action segments for video sequences
- Sliding window aggregation visualization
- Per-video performance analysis

Author: Talha
"""
#%%
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as path_effects

from feeders.feeder_vsvig import Feeder, custom_collate_fn, LABEL_NAMES, LABEL_MAP, SEIZURE_LABEL
from model.model_sup_logic import SkeletonACL_CLIP_Logic


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


def aggregate_sliding_windows(
    window_preds: np.ndarray,
    window_size: int,
    stride: int,
    total_length: int,
    n_classes: int,
    method: str = "vote"
) -> np.ndarray:
    """
    Aggregate sliding window predictions to per-frame predictions.
    
    Args:
        window_preds: (N,) class predictions or (N, C) probabilities
        window_size: Size of each window in frames
        stride: Stride between windows in frames
        total_length: Total number of frames
        n_classes: Number of classes
        method: "vote" for hard voting, "prob" for probability averaging
    
    Returns:
        Per-frame predictions (total_length,)
    """
    if method == "prob" and window_preds.ndim == 2:
        # Probability averaging
        prob_sum = np.zeros((total_length, n_classes), dtype=np.float32)
        count = np.zeros(total_length, dtype=np.float32)
        
        for i, probs in enumerate(window_preds):
            start = i * stride
            end = min(start + window_size, total_length)
            prob_sum[start:end] += probs
            count[start:end] += 1
        
        # Avoid division by zero
        count = np.maximum(count, 1)
        avg_probs = prob_sum / count[:, np.newaxis]
        return np.argmax(avg_probs, axis=1)
    
    else:
        # Hard voting
        votes = np.zeros((total_length, n_classes), dtype=np.int32)
        
        for i, pred in enumerate(window_preds):
            start = i * stride
            end = min(start + window_size, total_length)
            pred_class = int(pred) if np.isscalar(pred) or pred.ndim == 0 else int(pred)
            votes[start:end, pred_class] += 1
        
        # Handle frames with no votes
        frame_preds = np.argmax(votes, axis=1)
        no_votes = votes.sum(axis=1) == 0
        frame_preds[no_votes] = 0  # Default to class 0
        
        return frame_preds


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
            i = 0
            while i < len(result):
                s, e, lab = result[i]
                if (e - s) < min_len and len(result) > 1:
                    # Merge with neighbor
                    if i > 0 and (i == len(result) - 1 or 
                                  (result[i-1][1] - result[i-1][0]) >= (result[i+1][1] - result[i+1][0] if i+1 < len(result) else 0)):
                        # Merge with previous
                        if new_result:
                            new_result[-1] = (new_result[-1][0], e, new_result[-1][2])
                            changed = True
                            i += 1
                            continue
                    elif i < len(result) - 1:
                        # Merge with next
                        result[i+1] = (s, result[i+1][1], result[i+1][2])
                        changed = True
                        i += 1
                        continue
                new_result.append(result[i])
                i += 1
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
    
    Args:
        gt_segments: List of (start_s, end_s, class_name) for ground truth
        pred_segments: List of (start_s, end_s, class_name) for predictions
        total_duration_s: Total duration in seconds
        class_names: List of all class names
        palette: Dict mapping class names to hex colors
        title: Main title
        subtitle: Subtitle (e.g., metrics)
        figsize: Figure size
        show_legend: Whether to show legend
        show_time_axis: Whether to show time axis
        show_boundaries: Whether to show GT boundary lines on pred row
        savepath: Path to save figure
        dpi: Resolution for saving
        return_fig: Whether to return figure object
    """
    # CVPR style settings
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'pdf.fonttype': 42,  # TrueType fonts for PDF
        'ps.fonttype': 42,
    })
    
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
    # Bar parameters
    bar_height = 0.35
    gap = 0.15
    y_gt = 0.5 + gap/2
    y_pred = 0.5 - gap/2 - bar_height
    
    def draw_segments(segments, y, label_text):
        """Draw colored segments as a horizontal bar."""
        for start, end, class_name in segments:
            color = palette.get(class_name, '#CCCCCC')
            width = end - start
            rect = Rectangle(
                (start, y), width, bar_height,
                facecolor=color,
                edgecolor='white',
                linewidth=0.5,
                zorder=2
            )
            ax.add_patch(rect)
        
        # Row label
        ax.text(
            -total_duration_s * 0.02, y + bar_height/2,
            label_text,
            ha='right', va='center',
            fontsize=9, fontweight='bold',
            color='#333333'
        )
    
    # Draw GT and Pred bars
    draw_segments(gt_segments, y_gt, 'GT')
    draw_segments(pred_segments, y_pred, 'Pred')
    
    # Draw GT boundary lines on prediction row (for alignment reference)
    if show_boundaries:
        for start, end, _ in gt_segments:
            ax.axvline(start, color=boundary_color, alpha=boundary_alpha, 
                      linewidth=0.8, linestyle='--', zorder=1)
    
    # Configure axes
    ax.set_xlim(0, total_duration_s)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    
    if show_time_axis:
        # ax.set_xlabel('Time (seconds)', fontsize=9)
        # Smart tick spacing
        if total_duration_s <= 60:
            tick_interval = 10
        elif total_duration_s <= 300:
            tick_interval = 30
        else:
            tick_interval = 60
        ax.set_xticks(np.arange(0, total_duration_s + 1, tick_interval))
    else:
        ax.set_xticks([])
    
    # Title and subtitle
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    if subtitle:
        ax.text(
            0.5, 1.02, subtitle,
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=8, color='#666666'
        )
    
    # Minimal spines
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['bottom'].set_linewidth(0.5)
    
    # Light grid
    ax.grid(axis='x', color='#EEEEEE', linewidth=0.5, zorder=0)
    
    # Legend
    if show_legend:
        # Get unique classes that appear in either GT or pred
        classes_in_plot = set()
        for _, _, c in gt_segments:
            classes_in_plot.add(c)
        for _, _, c in pred_segments:
            classes_in_plot.add(c)
        
        # Sort by class order, seizure last for emphasis
        legend_classes = sorted(
            classes_in_plot,
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
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(len(handles), 5),
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
        return fig, ax
    else:
        plt.close(fig)
        return None


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
    """
    Create a multi-panel figure showing GT vs Pred for multiple videos.
    
    Args:
        video_results: List of dicts with keys:
            - 'video_name': str
            - 'gt_segments': List[(start_s, end_s, class_name)]
            - 'pred_segments': List[(start_s, end_s, class_name)]
            - 'duration_s': float
            - 'metrics': Optional dict with accuracy, etc.
        class_names: List of all class names
        palette: Color palette
        title: Main figure title
        figsize_per_video: Size per video panel
        max_videos: Maximum number of videos to show
        savepath: Path to save figure
        dpi: Resolution
    """
    n_videos = min(len(video_results), max_videos)
    
    fig_height = figsize_per_video[1] * n_videos + 1.5  # Extra for legend
    fig, axes = plt.subplots(
        n_videos, 1,
        figsize=(figsize_per_video[0], fig_height),
        constrained_layout=True
    )
    
    if n_videos == 1:
        axes = [axes]
    
    # Style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 9,
        'pdf.fonttype': 42,
    })
    
    bar_height = 0.35
    gap = 0.1
    
    for idx, (ax, result) in enumerate(zip(axes, video_results[:max_videos])):
        video_name = result['video_name']
        gt_segments = result['gt_segments']
        pred_segments = result['pred_segments']
        duration = result['duration_s']
        metrics = result.get('metrics', {})
        
        y_gt = 0.5 + gap/2
        y_pred = 0.5 - gap/2 - bar_height
        
        # Draw segments
        for start, end, class_name in gt_segments:
            color = palette.get(class_name, '#CCCCCC')
            rect = Rectangle(
                (start, y_gt), end - start, bar_height,
                facecolor=color, edgecolor='white', linewidth=0.3
            )
            ax.add_patch(rect)
        
        for start, end, class_name in pred_segments:
            color = palette.get(class_name, '#CCCCCC')
            rect = Rectangle(
                (start, y_pred), end - start, bar_height,
                facecolor=color, edgecolor='white', linewidth=0.3
            )
            ax.add_patch(rect)
        
        # Row labels
        ax.text(-duration * 0.015, y_gt + bar_height/2, 'GT',
               ha='right', va='center', fontsize=8, fontweight='bold')
        ax.text(-duration * 0.015, y_pred + bar_height/2, 'Pred',
               ha='right', va='center', fontsize=8, fontweight='bold')
        
        # Video name and metrics
        video_label = video_name[:30] + '...' if len(video_name) > 30 else video_name
        if metrics:
            acc = metrics.get('accuracy', 0) * 100
            video_label += f" (Acc: {acc:.1f}%)"
        
        ax.text(0.5, 1.05, video_label, transform=ax.transAxes,
               ha='center', va='bottom', fontsize=8, color='#555555')
        
        # Axes
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([])
        
        if idx == n_videos - 1:
            # ax.set_xlabel('Time (seconds)', fontsize=9)
            pass
        else:
            ax.set_xticks([])
        
        # Minimal spines
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.spines['bottom'].set_linewidth(0.5)
        ax.grid(axis='x', color='#F0F0F0', linewidth=0.5, zorder=0)
    
    # Title
    # fig.suptitle(title, fontsize=11, fontweight='bold', y=1.0)
    
    # Legend at bottom
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
# QUALITATIVE EVALUATOR CLASS
# =============================================================================
class QualitativeEvaluator:
    """Generate qualitative visualizations for action segmentation."""
    
    def __init__(self, config_path: str, checkpoint_path: str = None, binary_mode: bool = False):
        with open(config_path, 'r') as f:
            self.args = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        init_seed(self.args.get('seed', 1))
        
        self.n_classes = self.args['model_args']['num_class']
        self.binary_mode = binary_mode
        
        # Setup class names and palette
        if binary_mode:
            self.class_names = ['normal', 'seizure']
            self.palette = BINARY_PALETTE
            self.id_to_name = {0: 'normal', 1: 'seizure'}
        else:
            self.class_names = [LABEL_NAMES.get(i, f"class_{i}") for i in range(self.n_classes)]
            self.palette = SEIZURE_PALETTE
            self.id_to_name = LABEL_NAMES
        
        # Window parameters (from your setup)
        self.window_size = self.args['train_feeder_args'].get('window_size', 64)
        self.fps = 30.0  # Assuming 30 FPS, adjust if different
        self.stride = 1  # Stride in frames (1 frame = ~1/30 second, but clips overlap)
        
        # Output directory
        self.output_dir = Path(self.args['work_dir']) / 'qualitative'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data and model
        self._load_data()
        self._load_model(checkpoint_path)
        
        print(f"\n{'='*60}")
        print("✅ QUALITATIVE EVALUATOR INITIALIZED")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.n_classes}")
        print(f"Binary mode: {self.binary_mode}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Output dir: {self.output_dir}")
        print(f"{'='*60}\n")
    
    def _load_data(self):
        """Load test dataset with video information preserved."""
        self.test_dataset = Feeder(**self.args['test_feeder_args'])
        
        # Create dataloader WITHOUT shuffling to preserve video order
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            batch_size=1,  # Process one clip at a time for visualization
            shuffle=False,
            num_workers=0,  # Single thread for ordered processing
            drop_last=False,
            collate_fn=custom_collate_fn,
        )
        
        # Get source video info
        self.source_videos = self.test_dataset.source_videos
        self.unique_videos = np.unique(self.source_videos)
        print(f"Found {len(self.unique_videos)} unique videos in test set")
    
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
    def collect_video_predictions(self) -> Dict[str, Dict]:
        """
        Collect predictions organized by source video.
        
        Returns:
            Dict mapping video_name to {
                'clip_indices': list of clip indices,
                'gt_labels': list of ground truth labels,
                'pred_labels': list of predicted labels,
                'pred_probs': list of prediction probabilities,
                'confidences': list of max confidence values,
            }
        """
        print("Collecting predictions by video...")
        
        video_data = defaultdict(lambda: {
            'clip_indices': [],
            'gt_labels': [],
            'pred_labels': [],
            'pred_probs': [],
            'confidences': [],
        })
        
        for idx, (batch_data, batch_label, batch_concept_vecs, _) in enumerate(tqdm(self.test_loader)):
            batch_data = batch_data.to(self.device)
            batch_label = batch_label.to(self.device)
            concepts_gt = torch.cat(
                            (batch_concept_vecs['full_body'],
                            batch_concept_vecs['temporal']),
                            dim=1).float().to(self.device)
            # Get prediction
            y_pred, _ = self.model(batch_data, batch_label, concepts_gt)
            probs = y_pred #F.softmax(y_pred, dim=1)
            pred_class = y_pred.argmax(dim=1).item()
            confidence = probs.max().item()
            
            # Get ground truth
            gt_label = batch_label.item()
            
            # Convert to binary if needed
            if self.binary_mode:
                gt_label = 1 if gt_label == SEIZURE_LABEL else 0
                pred_class = 1 if pred_class == SEIZURE_LABEL else 0
            
            # Get source video
            video_name = str(self.source_videos[idx])
            
            video_data[video_name]['clip_indices'].append(idx)
            video_data[video_name]['gt_labels'].append(gt_label)
            video_data[video_name]['pred_labels'].append(pred_class)
            video_data[video_name]['pred_probs'].append(probs.cpu().numpy()[0])
            video_data[video_name]['confidences'].append(confidence)
        
        return dict(video_data)
    
    def create_video_segments(
        self,
        labels: List[int],
        clip_duration_s: float = 5.0,
        stride_s: float = 1.0,
    ) -> Tuple[List[Tuple[float, float, str]], float]:
        """
        Convert clip-level labels to time segments using sliding window aggregation.
        
        Args:
            labels: List of per-clip labels
            clip_duration_s: Duration of each clip in seconds
            stride_s: Stride between clips in seconds
            
        Returns:
            segments: List of (start_s, end_s, class_name)
            total_duration_s: Total video duration in seconds
        """
        n_clips = len(labels)
        
        # Handle stride=0 case: treat clips as consecutive (non-overlapping)
        if stride_s <= 0:
            stride_s = clip_duration_s  # Each clip follows the previous
        
        # Total duration: (n_clips - 1) * stride + clip_duration
        total_duration_s = (n_clips - 1) * stride_s + clip_duration_s
        total_frames = int(total_duration_s * self.fps)
        
        # Convert clip predictions to frame predictions via voting
        frame_labels = np.zeros(total_frames, dtype=int)
        frame_counts = np.zeros(total_frames, dtype=int)
        
        clip_frames = int(clip_duration_s * self.fps)
        stride_frames = int(stride_s * self.fps)
        
        for i, label in enumerate(labels):
            start_frame = i * stride_frames
            end_frame = min(start_frame + clip_frames, total_frames)
            
            # Vote for this label
            for f in range(start_frame, end_frame):
                # Simple last-write wins (or you can do voting)
                frame_labels[f] = label
                frame_counts[f] += 1
        
        # Convert to segments
        raw_segments = frame_labels_to_segments(frame_labels)
        
        # Merge short segments (less than 1 second)
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
        
        # Binary metrics if in binary mode
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
    
    def generate_single_video_plot(
        self,
        video_name: str,
        video_data: Dict,
        clip_duration_s: float = 5.0,
        stride_s: float = 1.0,
        savepath: Optional[str] = None,
    ):
        """
        Generate visualization for a single video as ONE continuous plot.
        
        Handles any video length with adaptive figure width and time formatting.
        """
        gt_segments, duration = self.create_video_segments(
            video_data['gt_labels'], clip_duration_s, stride_s
        )
        pred_segments, _ = self.create_video_segments(
            video_data['pred_labels'], clip_duration_s, stride_s
        )
        
        metrics = self.compute_segment_metrics(
            video_data['gt_labels'], video_data['pred_labels']
        )
        
        # Build subtitle with metrics
        n_clips = len(video_data['gt_labels'])
        duration_str = f"{int(duration//60)}:{int(duration%60):02d}"
        
        if self.binary_mode:
            subtitle = f"Duration: {duration_str} | {n_clips} clips | Acc: {metrics['accuracy']*100:.1f}% | Sens: {metrics['sensitivity']*100:.1f}% | Spec: {metrics['specificity']*100:.1f}%"
        else:
            subtitle = f"Duration: {duration_str} | {n_clips} clips | Accuracy: {metrics['accuracy']*100:.1f}%"
        
        if savepath is None:
            savepath = self.output_dir / f"video_{video_name}.png"
        
        # Adaptive figure width based on duration (min 14, scale with length)
        fig_width = max(14, min(24, 14 + (duration / 300) * 6))  # Scale up for longer videos
        
        # CVPR style settings
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
        
        fig, ax = plt.subplots(figsize=(fig_width, 2.2), constrained_layout=True)
        
        bar_height = 0.35
        gap = 0.15
        y_gt = 0.5 + gap/2
        y_pred = 0.5 - gap/2 - bar_height
        
        # Draw GT segments
        for start, end, class_name in gt_segments:
            color = self.palette.get(class_name, '#CCCCCC')
            rect = Rectangle(
                (start, y_gt), end - start, bar_height,
                facecolor=color, edgecolor='white', linewidth=0.5, zorder=2
            )
            ax.add_patch(rect)
        
        # Draw Pred segments
        for start, end, class_name in pred_segments:
            color = self.palette.get(class_name, '#CCCCCC')
            rect = Rectangle(
                (start, y_pred), end - start, bar_height,
                facecolor=color, edgecolor='white', linewidth=0.5, zorder=2
            )
            ax.add_patch(rect)
        
        # Row labels
        ax.text(-duration * 0.015, y_gt + bar_height/2, 'GT',
               ha='right', va='center', fontsize=9, fontweight='bold', color='#333333')
        ax.text(-duration * 0.015, y_pred + bar_height/2, 'Pred',
               ha='right', va='center', fontsize=9, fontweight='bold', color='#333333')
        
        # Configure axes
        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([])
        
        # Smart time axis formatting (MM:SS for videos > 2 min)
        if duration > 120:
            # Use minute:second format
            tick_interval = 60 if duration <= 600 else 120  # 1 min or 2 min intervals
            tick_positions = np.arange(0, duration + 1, tick_interval)
            tick_labels = [f"{int(t//60)}:{int(t%60):02d}" for t in tick_positions]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            # ax.set_xlabel('Time (min:sec)', fontsize=9)
        else:
            tick_interval = 30 if duration > 60 else 10
            ax.set_xticks(np.arange(0, duration + 1, tick_interval))
            # ax.set_xlabel('Time (seconds)', fontsize=9)
        
        # Title and subtitle
        ax.set_title(f"Patient: {video_name}", fontsize=10, fontweight='bold', pad=10)
        ax.text(0.5, 1.02, subtitle, transform=ax.transAxes,
               ha='center', va='bottom', fontsize=8, color='#666666')
        
        # Minimal spines
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
            columnspacing=1.0,
            handlelength=1.5,
            handleheight=0.8,
        )
        
        if savepath:
            fig.savefig(savepath, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"✅ Saved: {savepath}")
        
        plt.close(fig)
        
        return {
            'video_name': video_name,
            'gt_segments': gt_segments,
            'pred_segments': pred_segments,
            'duration_s': duration,
            'metrics': metrics,
        }
    
    def generate_chunked_video_plot(
        self,
        video_name: str,
        video_data: Dict,
        clip_duration_s: float = 5.0,
        stride_s: float = 1.0,
        chunk_duration_s: float = 120.0,  # 2 minutes per chunk
        savepath: Optional[str] = None,
    ):
        """
        Generate visualization for a video split into time chunks.
        
        Each chunk is shown as a separate row, making long videos more readable.
        """
        # Get full video segments first
        gt_segments, total_duration = self.create_video_segments(
            video_data['gt_labels'], clip_duration_s, stride_s
        )
        pred_segments, _ = self.create_video_segments(
            video_data['pred_labels'], clip_duration_s, stride_s
        )
        
        # Compute overall metrics
        metrics = self.compute_segment_metrics(
            video_data['gt_labels'], video_data['pred_labels']
        )
        
        # Calculate number of chunks needed
        n_chunks = int(np.ceil(total_duration / chunk_duration_s))
        
        if n_chunks <= 1:
            # Video is short enough, use regular plot
            return self.generate_single_video_plot(
                video_name, video_data, clip_duration_s, stride_s, savepath
            )
        
        # Create figure with multiple rows (one per chunk)
        fig_height = 1.8 * n_chunks + 1.2  # Height per chunk + space for legend/title
        fig, axes = plt.subplots(
            n_chunks, 1,
            figsize=(14, fig_height),
            constrained_layout=True
        )
        
        if n_chunks == 1:
            axes = [axes]
        
        # CVPR style
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
            'font.size': 9,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })
        
        bar_height = 0.35
        gap = 0.12
        
        def filter_segments_for_chunk(segments, chunk_start, chunk_end):
            """Filter and clip segments to fit within a chunk."""
            chunk_segs = []
            for start, end, label in segments:
                if end <= chunk_start or start >= chunk_end:
                    continue  # Segment outside chunk
                # Clip to chunk boundaries
                clipped_start = max(start, chunk_start) - chunk_start
                clipped_end = min(end, chunk_end) - chunk_start
                chunk_segs.append((clipped_start, clipped_end, label))
            return chunk_segs
        
        for chunk_idx in range(n_chunks):
            ax = axes[chunk_idx]
            chunk_start = chunk_idx * chunk_duration_s
            chunk_end = min((chunk_idx + 1) * chunk_duration_s, total_duration)
            chunk_len = chunk_end - chunk_start
            
            # Filter segments for this chunk
            gt_chunk = filter_segments_for_chunk(gt_segments, chunk_start, chunk_end)
            pred_chunk = filter_segments_for_chunk(pred_segments, chunk_start, chunk_end)
            
            y_gt = 0.5 + gap/2
            y_pred = 0.5 - gap/2 - bar_height
            
            # Draw GT segments
            for start, end, class_name in gt_chunk:
                color = self.palette.get(class_name, '#CCCCCC')
                rect = Rectangle(
                    (start, y_gt), end - start, bar_height,
                    facecolor=color, edgecolor='white', linewidth=0.4
                )
                ax.add_patch(rect)
            
            # Draw Pred segments
            for start, end, class_name in pred_chunk:
                color = self.palette.get(class_name, '#CCCCCC')
                rect = Rectangle(
                    (start, y_pred), end - start, bar_height,
                    facecolor=color, edgecolor='white', linewidth=0.4
                )
                ax.add_patch(rect)
            
            # Row labels
            ax.text(-chunk_len * 0.015, y_gt + bar_height/2, 'GT',
                   ha='right', va='center', fontsize=8, fontweight='bold', color='#333333')
            ax.text(-chunk_len * 0.015, y_pred + bar_height/2, 'Pred',
                   ha='right', va='center', fontsize=8, fontweight='bold', color='#333333')
            
            # Time range label on right
            time_label = f"{int(chunk_start//60)}:{int(chunk_start%60):02d} - {int(chunk_end//60)}:{int(chunk_end%60):02d}"
            ax.text(1.01, 0.5, time_label, transform=ax.transAxes,
                   ha='left', va='center', fontsize=8, color='#666666')
            
            # Configure axes
            ax.set_xlim(0, chunk_len)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([])
            
            # Only show x-axis ticks on bottom panel
            if chunk_idx == n_chunks - 1:
                ax.set_xlabel('Time within segment (seconds)', fontsize=9)
                tick_interval = 30 if chunk_len > 60 else 10
                ax.set_xticks(np.arange(0, chunk_len + 1, tick_interval))
            else:
                ax.set_xticks([])
            
            # Minimal spines
            for spine in ['top', 'right', 'left']:
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_color('#DDDDDD')
            ax.spines['bottom'].set_linewidth(0.5)
            ax.grid(axis='x', color='#F5F5F5', linewidth=0.5, zorder=0)
        
        # Main title with metrics
        if self.binary_mode:
            subtitle = f"Acc: {metrics['accuracy']*100:.1f}% | Sens: {metrics['sensitivity']*100:.1f}% | Spec: {metrics['specificity']*100:.1f}%"
        else:
            subtitle = f"Accuracy: {metrics['accuracy']*100:.1f}%"
        
        fig.suptitle(
            f"Video: {video_name}\n{subtitle}",
            fontsize=10, fontweight='bold', y=1.0
        )
        
        # Legend at bottom
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
        
        return {
            'video_name': video_name,
            'gt_segments': gt_segments,
            'pred_segments': pred_segments,
            'duration_s': total_duration,
            'n_chunks': n_chunks,
            'metrics': metrics,
        }
    
    def generate_all_visualizations(
        self,
        clip_duration_s: float = 5.0,
        stride_s: float = 1.0,
        max_videos: int = 10,
        chunk_duration_s: float = 120.0,  # 2 minutes per row
        generate_individual: bool = True,
        generate_combined: bool = True,
    ):
        """
        Generate all qualitative visualizations.
        
        Args:
            clip_duration_s: Duration of each clip in seconds
            stride_s: Stride between clips in seconds  
            max_videos: Maximum number of videos to visualize
            chunk_duration_s: Duration of each time chunk (row) in visualization
            generate_individual: Whether to save individual video plots
            generate_combined: Whether to generate combined comparison plot
        """
        print("\n" + "="*60)
        print("🎨 GENERATING QUALITATIVE VISUALIZATIONS")
        print("="*60 + "\n")
        print(f"   Mode: Single plot per patient/video (full timeline)")
        
        # Collect predictions
        video_data = self.collect_video_predictions()
        
        # Sort videos by number of clips (show longer videos first)
        sorted_videos = sorted(
            video_data.keys(),
            key=lambda v: len(video_data[v]['gt_labels']),
            reverse=True
        )
        
        all_results = []
        
        for video_name in tqdm(sorted_videos[:max_videos], desc="Processing videos"):
            # Generate single full-video plot per patient
            result = self.generate_single_video_plot(
                video_name,
                video_data[video_name],
                clip_duration_s,
                stride_s,
                savepath=self.output_dir / f"patient_{video_name}.png" if generate_individual else None,
            )
            all_results.append(result)
        
        # Generate combined multi-video plot
        if generate_combined and len(all_results) > 1:
            plot_multi_video_comparison(
                video_results=all_results[:6],  # Top 6 videos
                class_names=self.class_names,
                palette=self.palette,
                title="Action Segmentation Comparison: Ground Truth vs Predictions",
                savepath=str(self.output_dir / "multi_video_comparison.png"),
            )
        
        # Save summary
        summary = {
            'n_videos': len(all_results),
            'avg_accuracy': np.mean([r['metrics']['accuracy'] for r in all_results]),
            'per_video': [
                {
                    'video_name': r['video_name'],
                    'duration_s': r['duration_s'],
                    'metrics': r['metrics'],
                }
                for r in all_results
            ]
        }
        
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Generated visualizations for {len(all_results)} videos")
        print(f"   Average accuracy: {summary['avg_accuracy']*100:.1f}%")
        print(f"   Saved to: {self.output_dir}")
        
        return all_results


# =============================================================================
# MAIN
# =============================================================================
def main():
    # import argparse
    # parser = argparse.ArgumentParser(description='Generate qualitative visualizations')
    # parser.add_argument('--config', type=str,
    #                    default='config/szr/config_skeleton_vsvig.yaml',
    #                    help='Path to config file')
    # parser.add_argument('--checkpoint', type=str, default=None,
    #                    help='Path to model checkpoint')
    # parser.add_argument('--binary', action='store_true',
    #                    help='Use binary mode (seizure vs normal)')
    # parser.add_argument('--max-videos', type=int, default=10,
    #                    help='Maximum number of videos to visualize')
    # parser.add_argument('--clip-duration', type=float, default=5.0,
    #                    help='Clip duration in seconds')
    # parser.add_argument('--stride', type=float, default=1.0,
    #                    help='Stride between clips in seconds')
    # args = parser.parse_args()
    
    # config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.config)
    config_path = '/mnt/ssd/Talha/reason/config/szr/config_skeleton_vsvig.yaml'
    checkpoint = '/mnt/ssd/Talha/reason/work_dir/vsvig3_f1_logic2/best_model.pth'
    clip_duration = 5.0
    stride = 0.0
    max_videos = 100
    chunk_duration = 240.0  # 2 minutes per row (adjust as needed: 60, 120, 180)
    
    evaluator = QualitativeEvaluator(
        config_path,
        checkpoint_path=checkpoint,
        binary_mode=False
    )
    
    evaluator.generate_all_visualizations(
        clip_duration_s=clip_duration,
        stride_s=stride,
        max_videos=max_videos,
        chunk_duration_s=chunk_duration,
    )


if __name__ == '__main__':
    main()
#%%
