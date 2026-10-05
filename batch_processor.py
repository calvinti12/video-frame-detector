#!/usr/bin/env python3
"""
Batch processor for video frame cleanup.
Processes multiple videos in a directory or list with progress tracking.
"""

import argparse
import os
import sys
from pathlib import Path
from remove_letterbox import process_video


def process_batch(input_dir, output_dir, mode="blur", variance_threshold=18.0, brightness_cutoff=220.0, sample_every=10, blur_size=31, edge_sensitivity=0.3, spike_filter_window=5, video_extensions=None):
    """
    Process all videos in a directory.
    """
    if video_extensions is None:
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"]
    
    video_extensions = [ext.lower() for ext in video_extensions]
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    videos = []
    for ext in video_extensions:
        videos.extend(input_path.glob(f"*{ext}"))
    
    videos.sort()
    
    if not videos:
        print(f"No videos found in {input_dir}")
        return
    
    print(f"Found {len(videos)} videos to process")
    
    for idx, video_file in enumerate(videos, 1):
        print(f"\n[{idx}/{len(videos)}] Processing: {video_file.name}")
        
        output_file = output_path / f"cleaned_{video_file.stem}.mp4"
        
        try:
            process_video(
                str(video_file),
                str(output_file),
                mode=mode,
                variance_threshold=variance_threshold,
                brightness_cutoff=brightness_cutoff,
                sample_every=sample_every,
                blur_size=blur_size,
                edge_sensitivity=edge_sensitivity,
                spike_filter_window=spike_filter_window,
            )
            print(f"✓ Saved to: {output_file}")
        except Exception as e:
            print(f"✗ Error processing {video_file.name}: {e}")
            continue
    
    print(f"\nBatch processing complete. Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Batch process multiple videos for frame border cleanup.")
    parser.add_argument("--input-dir", required=True, help="Directory containing input videos")
    parser.add_argument("--output-dir", required=True, help="Directory to save cleaned videos")
    parser.add_argument("--mode", choices=["crop", "blur"], default="blur", help="Processing mode")
    parser.add_argument("--sample-every", type=int, default=10, help="Sample every Nth frame")
    parser.add_argument("--variance-threshold", type=float, default=18.0, help="Variance threshold for border detection")
    parser.add_argument("--brightness-cutoff", type=float, default=220.0, help="Brightness cutoff for dark borders")
    parser.add_argument("--blur-size", type=int, default=31, help="Blur kernel size")
    parser.add_argument("--edge-sensitivity", type=float, default=0.3, help="Edge detection sensitivity")
    parser.add_argument("--spike-filter-window", type=int, default=5, help="Spike filter window size")
    parser.add_argument("--extensions", nargs="+", default=[".mp4", ".avi", ".mov"], help="Video file extensions to process")
    
    args = parser.parse_args()
    
    try:
        process_batch(
            args.input_dir,
            args.output_dir,
            mode=args.mode,
            variance_threshold=args.variance_threshold,
            brightness_cutoff=args.brightness_cutoff,
            sample_every=args.sample_every,
            blur_size=args.blur_size,
            edge_sensitivity=args.edge_sensitivity,
            spike_filter_window=args.spike_filter_window,
            video_extensions=args.extensions,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
