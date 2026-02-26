"""
Main Pipeline for Seizure Video Labeling
Orchestrates video processing, VLM inference, and label generation
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from config import (
    FPS, WINDOW_SIZE_SEC, WINDOW_SIZE_FRAMES,
    VIDEO_DIR, OUTPUT_DIR, EXCEL_PATH,
    LABEL_MAP, LABEL_TO_ID, ACTION_CLASSES
)
from video_utils import (
    load_seizure_labels,
    get_video_info,
    extract_frame_from_window,
    generate_sliding_windows,
    create_frame_level_labels,
    VideoInfo
)
from vlm_classifier import get_classifier


class SeizureLabelingPipeline:
    """
    Complete pipeline for generating action labels from seizure monitoring videos.
    """
    
    def __init__(
        self,
        video_dir: str = VIDEO_DIR,
        output_dir: str = OUTPUT_DIR,
        excel_path: str = EXCEL_PATH,
        use_mock_vlm: bool = False,
        vlm_kwargs: dict = None
    ):
        """
        Initialize the labeling pipeline.
        
        Args:
            video_dir: Directory containing video files
            output_dir: Directory for output labels
            excel_path: Path to Excel file with seizure onset times
            use_mock_vlm: Use mock classifier for testing
            vlm_kwargs: Additional arguments for VLM classifier
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.excel_path = excel_path
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load seizure labels
        print(f"Loading seizure labels from: {excel_path}")
        self.labels_df = load_seizure_labels(excel_path)
        print(f"Loaded {len(self.labels_df)} seizure events")
        
        # Initialize VLM classifier
        vlm_kwargs = vlm_kwargs or {}
        self.classifier = get_classifier(use_mock=use_mock_vlm, **vlm_kwargs)
        
        # Statistics tracking
        self.stats = {
            'total_videos': 0,
            'total_frames': 0,
            'total_windows': 0,
            'label_counts': defaultdict(int),
            'label_durations_sec': defaultdict(float),
            'per_video_stats': [],
            'processing_errors': []
        }
    
    def discover_videos(self, pattern: str = "*.mp4") -> List[str]:
        """Discover video files in the video directory"""
        videos = list(self.video_dir.glob(pattern))
        videos.sort(key=lambda x: x.name)
        print(f"Found {len(videos)} video files")
        return [str(v) for v in videos]
    
    def process_video(
        self,
        video_path: str,
        save_individual: bool = True,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a single video and generate frame-level labels.
        
        Args:
            video_path: Path to video file
            save_individual: Save individual label file
            verbose: Print progress information
        
        Returns:
            Tuple of (labels_array, video_stats)
        """
        video_name = os.path.basename(video_path)
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing: {video_name}")
        
        # Get video info
        video_info = get_video_info(video_path, self.labels_df)
        
        if verbose:
            print(f"  Patient: {video_info.patient_id}")
            print(f"  Total frames: {video_info.total_frames}")
            print(f"  Duration: {video_info.duration_sec:.2f}s")
            print(f"  Seizure video: {video_info.is_seizure_video}")
            if video_info.clinical_onset_sec:
                print(f"  Clinical onset: {video_info.clinical_onset_sec:.2f}s")
                print(f"  Normal duration: {video_info.normal_duration_sec:.2f}s")
        
        # Calculate frame boundaries
        normal_end_frame = int(video_info.normal_duration_sec * FPS)
        seizure_start_frame = int(video_info.clinical_onset_sec * FPS) if video_info.clinical_onset_sec else None
        
        # Generate sliding windows for normal duration
        windows = generate_sliding_windows(
            video_info.total_frames,
            normal_end_frame,
            WINDOW_SIZE_FRAMES
        )
        
        if verbose:
            print(f"  Windows to process: {len(windows)}")
        
        # Process each window
        window_labels = []
        window_details = []
        
        for i, (start_frame, end_frame) in enumerate(tqdm(windows, desc="Classifying", disable=not verbose)):
            # Extract middle frame from window
            frame = extract_frame_from_window(
                video_path,
                start_frame,
                end_frame - start_frame,
                sample_position='middle'
            )
            
            if frame is None:
                if verbose:
                    print(f"  Warning: Could not extract frame at window {i}")
                label_name, label_id = 'unclear', LABEL_TO_ID['unclear']
                raw_output = "Frame extraction failed"
            else:
                # Classify using VLM
                label_name, label_id, raw_output = self.classifier.classify_frame(frame)
            
            window_labels.append((start_frame, end_frame, label_id))
            window_details.append({
                'window_idx': i,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'start_sec': start_frame / FPS,
                'end_sec': end_frame / FPS,
                'label': label_name,
                'label_id': label_id,
                'vlm_output': raw_output
            })
            
            # Update stats
            self.stats['label_counts'][label_name] += (end_frame - start_frame)
            self.stats['label_durations_sec'][label_name] += (end_frame - start_frame) / FPS
        
        # Create frame-level labels
        labels = create_frame_level_labels(
            window_labels,
            video_info.total_frames,
            seizure_start_frame,
            seizure_label_id=LABEL_TO_ID['seizure']
        )
        
        # Handle any unlabeled frames at the boundary
        # Fill gaps with nearest label
        labels = self._fill_label_gaps(labels)
        
        # Track seizure duration in stats
        if seizure_start_frame is not None:
            seizure_frames = video_info.total_frames - seizure_start_frame
            self.stats['label_counts']['seizure'] += seizure_frames
            self.stats['label_durations_sec']['seizure'] += seizure_frames / FPS
        
        # Video stats
        video_stats = {
            'video_name': video_name,
            'patient_id': video_info.patient_id,
            'total_frames': video_info.total_frames,
            'duration_sec': video_info.duration_sec,
            'normal_duration_sec': video_info.normal_duration_sec,
            'is_seizure_video': video_info.is_seizure_video,
            'clinical_onset_sec': video_info.clinical_onset_sec,
            'num_windows': len(windows),
            'window_details': window_details
        }
        
        # Save individual file
        if save_individual:
            output_path = self.output_dir / f"{Path(video_name).stem}_labels.npy"
            np.save(output_path, labels)
            
            # Save metadata
            metadata_path = self.output_dir / f"{Path(video_name).stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(video_stats, f, indent=2)
            
            if verbose:
                print(f"  Saved: {output_path}")
        
        # Update global stats
        self.stats['total_videos'] += 1
        self.stats['total_frames'] += video_info.total_frames
        self.stats['total_windows'] += len(windows)
        self.stats['per_video_stats'].append(video_stats)
        
        return labels, video_stats
    
    def _fill_label_gaps(self, labels: np.ndarray) -> np.ndarray:
        """Fill any -1 (unlabeled) gaps with nearest valid label"""
        # Find unlabeled positions
        unlabeled = labels == -1
        
        if not np.any(unlabeled):
            return labels
        
        # Forward fill
        labels_filled = labels.copy()
        last_valid = -1
        for i in range(len(labels_filled)):
            if labels_filled[i] != -1:
                last_valid = labels_filled[i]
            elif last_valid != -1:
                labels_filled[i] = last_valid
        
        # Backward fill for any remaining
        last_valid = -1
        for i in range(len(labels_filled) - 1, -1, -1):
            if labels_filled[i] != -1:
                last_valid = labels_filled[i]
            elif last_valid != -1:
                labels_filled[i] = last_valid
        
        # If still have -1, use 'unclear' label
        labels_filled[labels_filled == -1] = LABEL_TO_ID['unclear']
        
        return labels_filled
    
    def process_all_videos(
        self,
        video_paths: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Process all videos and generate labels.
        
        Args:
            video_paths: List of video paths (auto-discovers if None)
            verbose: Print progress
        
        Returns:
            Processing statistics
        """
        if video_paths is None:
            video_paths = self.discover_videos()
        
        all_labels = {}
        
        for video_path in video_paths:
            try:
                labels, stats = self.process_video(
                    video_path,
                    save_individual=True,
                    verbose=verbose
                )
                all_labels[os.path.basename(video_path)] = labels
            except Exception as e:
                error_msg = f"Error processing {video_path}: {str(e)}"
                print(f"\n  ERROR: {error_msg}")
                self.stats['processing_errors'].append(error_msg)
        
        return all_labels
    
    def save_combined_output(self) -> Dict[str, str]:
        """
        Save combined outputs: label map, all labels, and statistics.
        
        Returns:
            Dict of output file paths
        """
        output_files = {}
        
        # 1. Save label map
        label_map_path = self.output_dir / "label_map.json"
        with open(label_map_path, 'w') as f:
            json.dump({
                'id_to_label': LABEL_MAP,
                'label_to_id': LABEL_TO_ID,
                'action_classes': ACTION_CLASSES
            }, f, indent=2)
        output_files['label_map'] = str(label_map_path)
        print(f"\nSaved label map: {label_map_path}")
        
        # 2. Save statistics
        stats_summary = self._compute_statistics()
        stats_path = self.output_dir / "labeling_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats_summary, f, indent=2)
        output_files['statistics'] = str(stats_path)
        print(f"Saved statistics: {stats_path}")
        
        # 3. Save detailed per-video stats as CSV
        if self.stats['per_video_stats']:
            df_stats = pd.DataFrame([
                {
                    'video_name': s['video_name'],
                    'patient_id': s['patient_id'],
                    'total_frames': s['total_frames'],
                    'duration_sec': s['duration_sec'],
                    'normal_duration_sec': s['normal_duration_sec'],
                    'is_seizure_video': s['is_seizure_video'],
                    'clinical_onset_sec': s['clinical_onset_sec'],
                    'num_windows': s['num_windows']
                }
                for s in self.stats['per_video_stats']
            ])
            csv_path = self.output_dir / "per_video_stats.csv"
            df_stats.to_csv(csv_path, index=False)
            output_files['per_video_csv'] = str(csv_path)
            print(f"Saved per-video stats: {csv_path}")
        
        return output_files
    
    def _compute_statistics(self) -> Dict:
        """Compute comprehensive statistics"""
        stats = {
            'generated_at': datetime.now().isoformat(),
            'pipeline_settings': {
                'fps': FPS,
                'window_size_sec': WINDOW_SIZE_SEC,
                'window_size_frames': WINDOW_SIZE_FRAMES
            },
            'summary': {
                'total_videos_processed': self.stats['total_videos'],
                'total_frames': self.stats['total_frames'],
                'total_duration_sec': self.stats['total_frames'] / FPS,
                'total_duration_min': self.stats['total_frames'] / FPS / 60,
                'total_windows_classified': self.stats['total_windows']
            },
            'label_distribution': {
                'frame_counts': dict(self.stats['label_counts']),
                'duration_seconds': dict(self.stats['label_durations_sec']),
                'percentages': {}
            },
            'processing_errors': self.stats['processing_errors']
        }
        
        # Calculate percentages
        total_labeled_frames = sum(self.stats['label_counts'].values())
        if total_labeled_frames > 0:
            for label, count in self.stats['label_counts'].items():
                stats['label_distribution']['percentages'][label] = round(
                    100 * count / total_labeled_frames, 2
                )
        
        return stats
    
    def print_statistics(self):
        """Print a summary of labeling statistics"""
        print("\n" + "=" * 60)
        print("LABELING STATISTICS SUMMARY")
        print("=" * 60)
        
        print(f"\nVideos processed: {self.stats['total_videos']}")
        print(f"Total frames: {self.stats['total_frames']:,}")
        print(f"Total duration: {self.stats['total_frames'] / FPS / 60:.2f} minutes")
        print(f"Windows classified: {self.stats['total_windows']}")
        
        print("\nLabel Distribution:")
        print("-" * 40)
        
        total = sum(self.stats['label_counts'].values())
        for label_id, label_name in sorted(LABEL_MAP.items()):
            count = self.stats['label_counts'].get(label_name, 0)
            duration = self.stats['label_durations_sec'].get(label_name, 0)
            pct = 100 * count / total if total > 0 else 0
            print(f"  {label_id:2d}. {label_name:20s}: {count:8,} frames ({pct:5.1f}%) | {duration:7.1f}s")
        
        if self.stats['processing_errors']:
            print(f"\nErrors encountered: {len(self.stats['processing_errors'])}")
            for err in self.stats['processing_errors'][:5]:
                print(f"  - {err}")


def main():
    """Main entry point for the labeling pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Seizure Video Labeling Pipeline')
    parser.add_argument('--video-dir', default=VIDEO_DIR, help='Video directory')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Output directory')
    parser.add_argument('--excel-path', default=EXCEL_PATH, help='Excel file with seizure times')
    parser.add_argument('--mock', action='store_true', help='Use mock VLM for testing')
    parser.add_argument('--flash-attention', action='store_true', help='Use flash attention')
    parser.add_argument('--single-video', help='Process single video file')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    vlm_kwargs = {'use_flash_attention': args.flash_attention} if args.flash_attention else {}
    
    pipeline = SeizureLabelingPipeline(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        excel_path=args.excel_path,
        use_mock_vlm=args.mock,
        vlm_kwargs=vlm_kwargs
    )
    
    # Process videos
    if args.single_video:
        pipeline.process_video(args.single_video)
    else:
        pipeline.process_all_videos()
    
    # Save outputs and print stats
    pipeline.save_combined_output()
    pipeline.print_statistics()


if __name__ == "__main__":
    main()
