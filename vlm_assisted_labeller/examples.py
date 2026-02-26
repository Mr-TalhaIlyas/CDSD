"""
Example Usage Script for Seizure Video Labeling Pipeline
Demonstrates how to use the pipeline for different scenarios
"""

import os
import numpy as np
import json
from pathlib import Path


def example_basic_usage():
    """
    Basic usage: Process all videos in a directory
    """
    from pipeline import SeizureLabelingPipeline
    
    # Initialize pipeline
    pipeline = SeizureLabelingPipeline(
        video_dir="./videos_raw/vids",
        output_dir="./labels_output",
        excel_path="./dataset_Label_-_Copy.xlsx",
        use_mock_vlm=False  # Set to True for testing without GPU
    )
    
    # Process all videos
    pipeline.process_all_videos()
    
    # Save combined outputs
    pipeline.save_combined_output()
    
    # Print statistics
    pipeline.print_statistics()


def example_single_video():
    """
    Process a single video file
    """
    from pipeline import SeizureLabelingPipeline
    
    pipeline = SeizureLabelingPipeline(
        video_dir="./videos_raw/vids",
        output_dir="./labels_output",
        excel_path="./dataset_Label_-_Copy.xlsx"
    )
    
    # Process single video
    labels, stats = pipeline.process_video(
        "./videos_raw/vids/pat01_000_Sz1PG.mp4",
        save_individual=True,
        verbose=True
    )
    
    print(f"Generated {len(labels)} frame labels")
    print(f"Unique labels: {np.unique(labels)}")


def example_custom_video_list():
    """
    Process a custom list of videos
    """
    from pipeline import SeizureLabelingPipeline
    
    # Your video file list
    video_files = [
        './videos_raw/vids/pat01_000_Sz1PG.mp4',
        './videos_raw/vids/pat01_001_Sz2PG.mp4',
        './videos_raw/vids/pat02_002_Sz1PG.mp4',
        './videos_raw/vids/pat02_003_Sz2PG.mp4',
        './videos_raw/vids/pat03_004_Sz1PG.mp4',
        './videos_raw/vids/pat03_005_Sz2PG.mp4',
        './videos_raw/vids/pat03_006_free.mp4',
        './videos_raw/vids/pat04_007_Sz1P.mp4',
        './videos_raw/vids/pat04_008_free.mp4',
        './videos_raw/vids/pat04_009_no-Sz2P.mp4',
        './videos_raw/vids/pat05_010_Sz1PG.mp4',
        './videos_raw/vids/pat05_011_Sz2PG.mp4',
    ]
    
    pipeline = SeizureLabelingPipeline(
        video_dir="./videos_raw/vids",
        output_dir="./labels_output",
        excel_path="./dataset_Label_-_Copy.xlsx"
    )
    
    # Process only specified videos
    pipeline.process_all_videos(video_paths=video_files)
    pipeline.save_combined_output()
    pipeline.print_statistics()


def example_load_labels():
    """
    Example: Load and use generated labels
    """
    # Load label map
    with open('./labels_output/label_map.json', 'r') as f:
        label_info = json.load(f)
    
    id_to_label = label_info['id_to_label']
    label_to_id = label_info['label_to_id']
    
    print("Label Map:")
    for idx, name in sorted(id_to_label.items(), key=lambda x: int(x[0])):
        print(f"  {idx}: {name}")
    
    # Load labels for a specific video
    labels = np.load('./labels_output/pat01_000_Sz1PG_labels.npy')
    
    print(f"\nLoaded labels shape: {labels.shape}")
    print(f"Labels dtype: {labels.dtype}")
    
    # Get frame counts per label
    unique, counts = np.unique(labels, return_counts=True)
    print("\nLabel distribution for this video:")
    for u, c in zip(unique, counts):
        print(f"  {id_to_label[str(u)]}: {c} frames ({c/30:.1f}s)")


def example_dataloader_integration():
    """
    Example: Integrate with PyTorch DataLoader
    """
    import torch
    from torch.utils.data import Dataset, DataLoader
    
    class SeizureVideoDataset(Dataset):
        """Example dataset class for training"""
        
        def __init__(self, video_paths, labels_dir, transform=None):
            self.video_paths = video_paths
            self.labels_dir = Path(labels_dir)
            self.transform = transform
            
            # Load label map
            with open(self.labels_dir / 'label_map.json', 'r') as f:
                self.label_info = json.load(f)
            
            self.num_classes = len(self.label_info['action_classes'])
            
            # Pre-load all labels
            self.labels = {}
            for vp in video_paths:
                video_name = Path(vp).stem
                label_path = self.labels_dir / f"{video_name}_labels.npy"
                if label_path.exists():
                    self.labels[vp] = np.load(label_path)
        
        def __len__(self):
            return sum(len(l) for l in self.labels.values())
        
        def get_labels_for_video(self, video_path):
            """Get all frame labels for a video"""
            return self.labels.get(video_path, None)
        
        def get_window_label(self, video_path, start_frame, window_size=150):
            """Get majority label for a window"""
            labels = self.labels.get(video_path)
            if labels is None:
                return None
            
            window_labels = labels[start_frame:start_frame + window_size]
            # Return majority label
            unique, counts = np.unique(window_labels, return_counts=True)
            return unique[np.argmax(counts)]
    
    # Usage
    dataset = SeizureVideoDataset(
        video_paths=['./videos_raw/vids/pat01_000_Sz1PG.mp4'],
        labels_dir='./labels_output'
    )
    
    print(f"Dataset has {len(dataset)} frames")
    print(f"Number of classes: {dataset.num_classes}")


def example_with_flash_attention():
    """
    Example: Use flash attention for faster inference
    """
    from pipeline import SeizureLabelingPipeline
    
    pipeline = SeizureLabelingPipeline(
        video_dir="./videos_raw/vids",
        output_dir="./labels_output",
        excel_path="./dataset_Label_-_Copy.xlsx",
        use_mock_vlm=False,
        vlm_kwargs={'use_flash_attention': True}
    )
    
    pipeline.process_all_videos()
    pipeline.save_combined_output()


def example_testing_without_gpu():
    """
    Example: Test pipeline without GPU using mock VLM
    """
    from pipeline import SeizureLabelingPipeline
    
    # Use mock classifier for testing
    pipeline = SeizureLabelingPipeline(
        video_dir="./videos_raw/vids",
        output_dir="./labels_output_test",
        excel_path="./dataset_Label_-_Copy.xlsx",
        use_mock_vlm=True  # Uses MockVLMClassifier
    )
    
    pipeline.process_all_videos()
    pipeline.save_combined_output()
    pipeline.print_statistics()


if __name__ == "__main__":
    # Run basic usage example
    print("Running basic usage example...")
    example_basic_usage()
