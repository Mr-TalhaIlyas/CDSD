#%%
"""
Simplified Pipeline for 5-Second Clip Labeling
Processes short video clips directly without windowing
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from config import (
    FPS,
    LABEL_MAP, LABEL_TO_ID, ACTION_CLASSES
)
from vlm_classifier import get_classifier

VIDEO_DIR_IEEE = '/data_hdd/talha/miccai_26/seizure_detection_pipeline/videos_raw/Normal/'
OUTPUT_DIR_IEEE = '/data_hdd/talha/miccai_26/seizure_detection_pipeline/vlm_labeller/ieee/'
class ShortClipLabelingPipeline:
    """
    Simplified pipeline for labeling 5-second video clips.
    Each clip gets a single label extracted from its middle frame.
    """
    
    def __init__(
        self,
        video_dir: str,
        output_dir: str = OUTPUT_DIR_IEEE,
        use_mock_vlm: bool = False,
        vlm_kwargs: dict = None,
        clip_duration_sec: float = 5.0
    ):
        """
        Initialize the labeling pipeline for short clips.
        
        Args:
            video_dir: Directory containing video clip files
            output_dir: Directory for output labels
            use_mock_vlm: Use mock classifier for testing
            vlm_kwargs: Additional arguments for VLM classifier
            clip_duration_sec: Expected duration of clips (default: 5.0s)
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.clip_duration_sec = clip_duration_sec
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize VLM classifier
        vlm_kwargs = vlm_kwargs or {}
        self.classifier = get_classifier(use_mock=use_mock_vlm, **vlm_kwargs)
        
        # Statistics tracking
        self.stats = {
            'total_clips': 0,
            'total_frames': 0,
            'label_counts': defaultdict(int),
            'label_durations_sec': defaultdict(float),
            'per_clip_stats': [],
            'processing_errors': []
        }
    
    def discover_videos(self, pattern: str = "*.mp4") -> List[str]:
        """Discover video clip files in the directory"""
        videos = list(self.video_dir.glob(pattern))
        videos.sort(key=lambda x: x.name)
        print(f"Found {len(videos)} video clips")
        return [str(v) for v in videos]
    
    def extract_middle_frame(self, video_path: str) -> Tuple[Optional[np.ndarray], int, float]:
        """
        Extract the middle frame from a video clip.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Tuple of (frame_rgb, total_frames, duration_sec)
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration_sec = total_frames / fps if fps > 0 else 0
        
        # Calculate middle frame index
        middle_frame_idx = total_frames // 2
        
        # Extract middle frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return None, total_frames, duration_sec
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb, total_frames, duration_sec
    
    def process_clip(
        self,
        video_path: str,
        save_individual: bool = True,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process a single 5-second clip and generate frame-level labels.
        
        Args:
            video_path: Path to video clip file
            save_individual: Save individual label file
            verbose: Print progress information
        
        Returns:
            Tuple of (labels_array, clip_stats)
        """
        video_name = os.path.basename(video_path)
        
        if verbose:
            print(f"\nProcessing: {video_name}")
        
        # Extract middle frame and get video info
        frame, total_frames, duration_sec = self.extract_middle_frame(video_path)
        
        if frame is None:
            error_msg = f"Could not extract frame from {video_name}"
            if verbose:
                print(f"  ERROR: {error_msg}")
            self.stats['processing_errors'].append(error_msg)
            
            # Create empty labels with 'unclear'
            labels = np.full(total_frames, LABEL_TO_ID['unclear'], dtype=np.int32)
            label_name = 'unclear'
            label_id = LABEL_TO_ID['unclear']
            raw_output = "Frame extraction failed"
        else:
            # Classify using VLM
            label_name, label_id, raw_output = self.classifier.classify_frame(frame)
            
            # Create frame-level labels (all frames get the same label)
            labels = np.full(total_frames, label_id, dtype=np.int32)
        
        if verbose:
            print(f"  Frames: {total_frames}, Duration: {duration_sec:.2f}s")
            print(f"  Label: {label_name} (ID: {label_id})")
        
        # Clip statistics
        clip_stats = {
            'video_name': video_name,
            'total_frames': total_frames,
            'duration_sec': duration_sec,
            'fps': total_frames / duration_sec if duration_sec > 0 else FPS,
            'label': label_name,
            'label_id': label_id,
            'vlm_output': raw_output,
            'middle_frame_idx': total_frames // 2
        }
        
        # Update statistics
        self.stats['label_counts'][label_name] += total_frames
        self.stats['label_durations_sec'][label_name] += duration_sec
        
        # Save individual file
        if save_individual:
            output_path = self.output_dir / f"{Path(video_name).stem}_labels.npy"
            np.save(output_path, labels)
            
            # Save metadata
            metadata_path = self.output_dir / f"{Path(video_name).stem}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(clip_stats, f, indent=2)
            
            if verbose:
                print(f"  Saved: {output_path}")
        
        # Update global stats
        self.stats['total_clips'] += 1
        self.stats['total_frames'] += total_frames
        self.stats['per_clip_stats'].append(clip_stats)
        
        return labels, clip_stats
    
    def process_all_clips(
        self,
        video_paths: Optional[List[str]] = None,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Process all video clips and generate labels.
        
        Args:
            video_paths: List of video paths (auto-discovers if None)
            verbose: Print progress
        
        Returns:
            Dictionary mapping video names to label arrays
        """
        if video_paths is None:
            video_paths = self.discover_videos()
        
        all_labels = {}
        
        print(f"\n{'='*60}")
        print(f"Processing {len(video_paths)} clips...")
        print(f"{'='*60}")
        
        for video_path in tqdm(video_paths, desc="Processing clips", disable=not verbose):
            try:
                labels, stats = self.process_clip(
                    video_path,
                    save_individual=True,
                    verbose=False  # Disable individual verbose in batch mode
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
        
        # 3. Save detailed per-clip stats as CSV
        if self.stats['per_clip_stats']:
            df_stats = pd.DataFrame(self.stats['per_clip_stats'])
            csv_path = self.output_dir / "per_clip_stats.csv"
            df_stats.to_csv(csv_path, index=False)
            output_files['per_clip_csv'] = str(csv_path)
            print(f"Saved per-clip stats: {csv_path}")
        
        return output_files
    
    def _compute_statistics(self) -> Dict:
        """Compute comprehensive statistics"""
        stats = {
            'generated_at': datetime.now().isoformat(),
            'pipeline_settings': {
                'fps': FPS,
                'clip_duration_sec': self.clip_duration_sec,
                'processing_mode': 'short_clips'
            },
            'summary': {
                'total_clips_processed': self.stats['total_clips'],
                'total_frames': self.stats['total_frames'],
                'total_duration_sec': self.stats['total_frames'] / FPS,
                'total_duration_min': self.stats['total_frames'] / FPS / 60,
                'avg_clip_duration_sec': (self.stats['total_frames'] / FPS / self.stats['total_clips']) 
                                          if self.stats['total_clips'] > 0 else 0
            },
            'label_distribution': {
                'frame_counts': dict(self.stats['label_counts']),
                'duration_seconds': dict(self.stats['label_durations_sec']),
                'clip_counts': {},
                'percentages': {}
            },
            'processing_errors': self.stats['processing_errors']
        }
        
        # Count clips per label
        label_clip_counts = defaultdict(int)
        for clip_stat in self.stats['per_clip_stats']:
            label_clip_counts[clip_stat['label']] += 1
        stats['label_distribution']['clip_counts'] = dict(label_clip_counts)
        
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
        print("LABELING STATISTICS SUMMARY (SHORT CLIPS)")
        print("=" * 60)
        
        print(f"\nClips processed: {self.stats['total_clips']}")
        print(f"Total frames: {self.stats['total_frames']:,}")
        print(f"Total duration: {self.stats['total_frames'] / FPS / 60:.2f} minutes")
        
        if self.stats['total_clips'] > 0:
            avg_duration = (self.stats['total_frames'] / FPS) / self.stats['total_clips']
            print(f"Avg clip duration: {avg_duration:.2f} seconds")
        
        print("\nLabel Distribution:")
        print("-" * 60)
        print(f"{'Label':<20} {'Clips':<8} {'Frames':<10} {'Duration':<12} {'%':<8}")
        print("-" * 60)
        
        # Count clips per label
        label_clip_counts = defaultdict(int)
        for clip_stat in self.stats['per_clip_stats']:
            label_clip_counts[clip_stat['label']] += 1
        
        total = sum(self.stats['label_counts'].values())
        for label_id, label_name in sorted(LABEL_MAP.items()):
            clip_count = label_clip_counts.get(label_name, 0)
            frame_count = self.stats['label_counts'].get(label_name, 0)
            duration = self.stats['label_durations_sec'].get(label_name, 0)
            pct = 100 * frame_count / total if total > 0 else 0
            print(f"{label_name:<20} {clip_count:<8} {frame_count:<10,} {duration:>7.1f}s    {pct:>5.1f}%")
        
        if self.stats['processing_errors']:
            print(f"\n⚠️  Errors encountered: {len(self.stats['processing_errors'])}")
            for err in self.stats['processing_errors'][:5]:
                print(f"  - {err}")
            if len(self.stats['processing_errors']) > 5:
                print(f"  ... and {len(self.stats['processing_errors']) - 5} more")


def main():
    """Main entry point for the short clip labeling pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Short Clip Labeling Pipeline (5sec clips)')
    parser.add_argument('--video-dir', default=VIDEO_DIR_IEEE, help='Directory with video clips')
    parser.add_argument('--output-dir', default=OUTPUT_DIR_IEEE, help='Output directory')
    parser.add_argument('--mock', action='store_true', help='Use mock VLM for testing')
    parser.add_argument('--flash-attention', default=False, action='store_true', help='Use flash attention')
    parser.add_argument('--single-clip', help='Process single video clip')
    parser.add_argument('--clip-duration', type=float, default=5.0, help='Expected clip duration in seconds')
    parser.add_argument('--pattern', default='*.mp4', help='Video file pattern (default: *.mp4)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    vlm_kwargs = {'use_flash_attention': args.flash_attention} if args.flash_attention else {}
    
    pipeline = ShortClipLabelingPipeline(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        use_mock_vlm=args.mock,
        vlm_kwargs=vlm_kwargs,
        clip_duration_sec=args.clip_duration
    )
    
    # Process clips
    if args.single_clip:
        print(f"\nProcessing single clip: {args.single_clip}")
        pipeline.process_clip(args.single_clip)
    else:
        # Discover and process all clips
        video_paths = pipeline.discover_videos(pattern=args.pattern)
        pipeline.process_all_clips(video_paths=video_paths, verbose=True)
    
    # Save outputs and print stats
    pipeline.save_combined_output()
    pipeline.print_statistics()
    
    print(f"\n✓ Processing complete! Output saved to: {pipeline.output_dir}")


if __name__ == "__main__":
    main()
#%%