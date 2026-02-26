"""
Statistics and Visualization Utilities for Seizure Labels
Generates comprehensive statistics, plots, and reports
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


def load_all_labels(labels_dir: str) -> Dict[str, np.ndarray]:
    """Load all label arrays from directory"""
    labels_dir = Path(labels_dir)
    labels = {}
    
    for npy_file in labels_dir.glob("*_labels.npy"):
        video_name = npy_file.stem.replace('_labels', '')
        labels[video_name] = np.load(npy_file)
    
    return labels


def load_label_map(labels_dir: str) -> Dict:
    """Load label mapping"""
    with open(Path(labels_dir) / 'label_map.json', 'r') as f:
        return json.load(f)


def compute_label_statistics(
    labels_dir: str,
    fps: int = 30
) -> Dict:
    """
    Compute comprehensive statistics from generated labels.
    
    Args:
        labels_dir: Directory containing label files
        fps: Frames per second
    
    Returns:
        Dictionary with statistics
    """
    labels = load_all_labels(labels_dir)
    label_map = load_label_map(labels_dir)
    id_to_label = label_map['id_to_label']
    
    stats = {
        'overall': {
            'total_videos': len(labels),
            'total_frames': 0,
            'total_duration_sec': 0,
            'total_duration_min': 0
        },
        'per_label': defaultdict(lambda: {
            'total_frames': 0,
            'total_duration_sec': 0,
            'occurrences': 0,  # Number of continuous segments
            'avg_segment_duration_sec': 0,
            'min_segment_duration_sec': float('inf'),
            'max_segment_duration_sec': 0,
            'videos_with_label': set()
        }),
        'per_video': {},
        'label_transitions': defaultdict(int)
    }
    
    for video_name, video_labels in labels.items():
        total_frames = len(video_labels)
        stats['overall']['total_frames'] += total_frames
        
        # Per-video statistics
        video_stats = {
            'total_frames': total_frames,
            'duration_sec': total_frames / fps,
            'label_distribution': {}
        }
        
        # Count labels in this video
        unique, counts = np.unique(video_labels, return_counts=True)
        for label_id, count in zip(unique, counts):
            label_name = id_to_label.get(str(label_id), f'unknown_{label_id}')
            video_stats['label_distribution'][label_name] = {
                'frames': int(count),
                'duration_sec': count / fps,
                'percentage': 100 * count / total_frames
            }
            
            # Update global per-label stats
            stats['per_label'][label_name]['total_frames'] += count
            stats['per_label'][label_name]['total_duration_sec'] += count / fps
            stats['per_label'][label_name]['videos_with_label'].add(video_name)
        
        # Analyze segments (continuous runs of same label)
        segment_starts = np.where(np.diff(video_labels, prepend=video_labels[0]-1) != 0)[0]
        segment_ends = np.append(segment_starts[1:], len(video_labels))
        
        for start, end in zip(segment_starts, segment_ends):
            label_id = video_labels[start]
            label_name = id_to_label.get(str(label_id), f'unknown_{label_id}')
            segment_duration = (end - start) / fps
            
            stats['per_label'][label_name]['occurrences'] += 1
            stats['per_label'][label_name]['min_segment_duration_sec'] = min(
                stats['per_label'][label_name]['min_segment_duration_sec'],
                segment_duration
            )
            stats['per_label'][label_name]['max_segment_duration_sec'] = max(
                stats['per_label'][label_name]['max_segment_duration_sec'],
                segment_duration
            )
        
        # Count transitions
        for i in range(len(video_labels) - 1):
            if video_labels[i] != video_labels[i + 1]:
                from_label = id_to_label.get(str(video_labels[i]), 'unknown')
                to_label = id_to_label.get(str(video_labels[i + 1]), 'unknown')
                stats['label_transitions'][f'{from_label} -> {to_label}'] += 1
        
        stats['per_video'][video_name] = video_stats
    
    # Finalize statistics
    stats['overall']['total_duration_sec'] = stats['overall']['total_frames'] / fps
    stats['overall']['total_duration_min'] = stats['overall']['total_duration_sec'] / 60
    
    # Calculate averages and convert sets to counts
    for label_name, label_stats in stats['per_label'].items():
        if label_stats['occurrences'] > 0:
            label_stats['avg_segment_duration_sec'] = (
                label_stats['total_duration_sec'] / label_stats['occurrences']
            )
        if label_stats['min_segment_duration_sec'] == float('inf'):
            label_stats['min_segment_duration_sec'] = 0
        label_stats['num_videos'] = len(label_stats['videos_with_label'])
        label_stats['videos_with_label'] = list(label_stats['videos_with_label'])
        
        # Calculate percentage of total
        if stats['overall']['total_frames'] > 0:
            label_stats['percentage'] = (
                100 * label_stats['total_frames'] / stats['overall']['total_frames']
            )
    
    # Convert defaultdicts to regular dicts
    stats['per_label'] = dict(stats['per_label'])
    stats['label_transitions'] = dict(stats['label_transitions'])
    
    return stats


def plot_label_distribution(
    labels_dir: str,
    output_path: str = None,
    fps: int = 30
):
    """
    Create bar plot of label distribution.
    """
    labels = load_all_labels(labels_dir)
    label_map = load_label_map(labels_dir)
    id_to_label = label_map['id_to_label']
    
    # Combine all labels
    all_labels = np.concatenate(list(labels.values()))
    
    # Count
    unique, counts = np.unique(all_labels, return_counts=True)
    
    # Prepare data
    label_names = [id_to_label.get(str(u), f'unknown_{u}') for u in unique]
    durations = counts / fps / 60  # Convert to minutes
    
    # Sort by count
    sorted_idx = np.argsort(counts)[::-1]
    label_names = [label_names[i] for i in sorted_idx]
    durations = durations[sorted_idx]
    counts_sorted = counts[sorted_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(label_names)), durations, color='steelblue', edgecolor='black')
    
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha='right')
    ax.set_ylabel('Duration (minutes)')
    ax.set_xlabel('Action Class')
    ax.set_title('Label Distribution Across All Videos')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts_sorted)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved distribution plot: {output_path}")
    else:
        plt.savefig(Path(labels_dir) / 'label_distribution.png', dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_per_video_composition(
    labels_dir: str,
    output_path: str = None,
    fps: int = 30
):
    """
    Create stacked bar plot showing label composition per video.
    """
    labels = load_all_labels(labels_dir)
    label_map = load_label_map(labels_dir)
    id_to_label = label_map['id_to_label']
    action_classes = label_map['action_classes']
    
    # Create matrix: videos x labels
    video_names = sorted(labels.keys())
    matrix = np.zeros((len(video_names), len(action_classes)))
    
    for i, video_name in enumerate(video_names):
        video_labels = labels[video_name]
        for j, label_name in enumerate(action_classes):
            label_id = int(label_map['label_to_id'][label_name])
            matrix[i, j] = np.sum(video_labels == label_id) / fps / 60  # minutes
    
    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use a colormap
    colors = plt.cm.tab20(np.linspace(0, 1, len(action_classes)))
    
    # Plot stacked bars
    bottom = np.zeros(len(video_names))
    for j, label_name in enumerate(action_classes):
        if matrix[:, j].sum() > 0:  # Only plot if label exists
            ax.bar(range(len(video_names)), matrix[:, j], bottom=bottom,
                   label=label_name, color=colors[j], edgecolor='white', linewidth=0.5)
            bottom += matrix[:, j]
    
    ax.set_xticks(range(len(video_names)))
    ax.set_xticklabels([v[:15] + '...' if len(v) > 15 else v for v in video_names],
                       rotation=90, fontsize=8)
    ax.set_ylabel('Duration (minutes)')
    ax.set_xlabel('Video')
    ax.set_title('Label Composition Per Video')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved composition plot: {output_path}")
    else:
        plt.savefig(Path(labels_dir) / 'per_video_composition.png', dpi=150, bbox_inches='tight')
    
    plt.close()


def plot_segment_duration_histogram(
    labels_dir: str,
    output_path: str = None,
    fps: int = 30
):
    """
    Plot histogram of segment durations per label.
    """
    labels = load_all_labels(labels_dir)
    label_map = load_label_map(labels_dir)
    id_to_label = label_map['id_to_label']
    
    # Collect segment durations per label
    segment_durations = defaultdict(list)
    
    for video_name, video_labels in labels.items():
        # Find segments
        changes = np.where(np.diff(video_labels) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [len(video_labels)]])
        
        for start, end in zip(starts, ends):
            label_id = video_labels[start]
            label_name = id_to_label.get(str(label_id), f'unknown_{label_id}')
            duration = (end - start) / fps
            segment_durations[label_name].append(duration)
    
    # Create subplots
    num_labels = len(segment_durations)
    cols = 4
    rows = (num_labels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
    axes = axes.flatten() if num_labels > 1 else [axes]
    
    for i, (label_name, durations) in enumerate(sorted(segment_durations.items())):
        ax = axes[i]
        ax.hist(durations, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'{label_name}\n(n={len(durations)})', fontsize=10)
        ax.set_xlabel('Duration (sec)')
        ax.set_ylabel('Count')
    
    # Hide unused subplots
    for i in range(len(segment_durations), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Segment Duration Distribution by Label', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved histogram: {output_path}")
    else:
        plt.savefig(Path(labels_dir) / 'segment_duration_histogram.png', dpi=150, bbox_inches='tight')
    
    plt.close()


def generate_statistics_report(
    labels_dir: str,
    output_path: str = None,
    fps: int = 30
) -> str:
    """
    Generate a comprehensive text report of labeling statistics.
    """
    stats = compute_label_statistics(labels_dir, fps)
    
    report = []
    report.append("=" * 70)
    report.append("SEIZURE VIDEO LABELING - STATISTICS REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 40)
    report.append(f"Total videos processed: {stats['overall']['total_videos']}")
    report.append(f"Total frames: {stats['overall']['total_frames']:,}")
    report.append(f"Total duration: {stats['overall']['total_duration_min']:.2f} minutes")
    report.append(f"               ({stats['overall']['total_duration_sec']/3600:.2f} hours)")
    report.append("")
    
    # Label distribution
    report.append("LABEL DISTRIBUTION")
    report.append("-" * 40)
    report.append(f"{'Label':<25} {'Frames':>10} {'Duration':>10} {'%':>8} {'Segments':>10}")
    report.append("-" * 70)
    
    # Sort by frame count
    sorted_labels = sorted(
        stats['per_label'].items(),
        key=lambda x: x[1]['total_frames'],
        reverse=True
    )
    
    for label_name, label_stats in sorted_labels:
        frames = label_stats['total_frames']
        duration = label_stats['total_duration_sec']
        pct = label_stats.get('percentage', 0)
        segments = label_stats['occurrences']
        
        duration_str = f"{duration:.1f}s" if duration < 60 else f"{duration/60:.1f}m"
        report.append(f"{label_name:<25} {frames:>10,} {duration_str:>10} {pct:>7.1f}% {segments:>10}")
    
    report.append("")
    
    # Segment duration statistics
    report.append("SEGMENT DURATION STATISTICS")
    report.append("-" * 40)
    report.append(f"{'Label':<25} {'Avg (sec)':>10} {'Min (sec)':>10} {'Max (sec)':>10}")
    report.append("-" * 55)
    
    for label_name, label_stats in sorted_labels:
        if label_stats['occurrences'] > 0:
            avg = label_stats['avg_segment_duration_sec']
            min_d = label_stats['min_segment_duration_sec']
            max_d = label_stats['max_segment_duration_sec']
            report.append(f"{label_name:<25} {avg:>10.2f} {min_d:>10.2f} {max_d:>10.2f}")
    
    report.append("")
    
    # Top label transitions
    report.append("TOP LABEL TRANSITIONS")
    report.append("-" * 40)
    sorted_transitions = sorted(
        stats['label_transitions'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:15]
    
    for transition, count in sorted_transitions:
        report.append(f"  {transition}: {count}")
    
    report.append("")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"Saved report: {output_path}")
    else:
        with open(Path(labels_dir) / 'statistics_report.txt', 'w') as f:
            f.write(report_text)
    
    return report_text


def generate_all_statistics(labels_dir: str, fps: int = 30):
    """
    Generate all statistics outputs: JSON, plots, and text report.
    """
    labels_dir = Path(labels_dir)
    
    print("Computing statistics...")
    stats = compute_label_statistics(labels_dir, fps)
    
    # Save JSON
    with open(labels_dir / 'detailed_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Saved: {labels_dir / 'detailed_statistics.json'}")
    
    # Generate plots
    print("Generating plots...")
    plot_label_distribution(str(labels_dir))
    plot_per_video_composition(str(labels_dir))
    plot_segment_duration_histogram(str(labels_dir))
    
    # Generate text report
    print("Generating report...")
    report = generate_statistics_report(str(labels_dir))
    print("\n" + report)
    
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate label statistics')
    parser.add_argument('labels_dir', help='Directory containing label files')
    parser.add_argument('--fps', type=int, default=30, help='Video FPS')
    
    args = parser.parse_args()
    
    generate_all_statistics(args.labels_dir, args.fps)
