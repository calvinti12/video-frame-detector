#!/usr/bin/env python3
"""
Dynamic overlay remover for videos with burnt-in titles/captions or frame overlays.
Detects top/bottom overlay bands by finding regions with consistently high edge/variance scores,
allows for gaps in the overlay (text breaks), and removes those regions with blur fill.
"""

import argparse
import os
import sys
from statistics import median

import cv2
import numpy as np


def row_score(frame):
    """Compute a per-row score that responds to high-contrast graphic overlays."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    edges = cv2.Canny(gray, 50, 150)

    scores = []
    for y in range(h):
        row = gray[y]
        row_var = float(row.var())
        edge_density = float(np.mean(edges[y] > 0))
        score = (row_var / 8.0) + (edge_density * 900.0)
        scores.append(score)
    return np.asarray(scores, dtype=np.float32)


def detect_overlay_rows(frame, gap_tolerance=100, min_band_size=20):
    """
    Detect top and bottom overlay rows by finding continuous (or near-continuous) high-score bands.
    Allows larger gaps in the band (text breaks, gaps between overlay elements) up to gap_tolerance pixels.
    """
    h, w = frame.shape[:2]
    scores = row_score(frame)

    if h < 100:
        return {"top": 0, "bottom": 0}

    # Use a lower threshold based on the overall distribution
    threshold = float(np.percentile(scores, 75))
    threshold = max(100.0, threshold * 0.5)

    # Find top band: rows near the top with high scores, allowing larger gaps
    top_band_start = None
    top_band_end = 0
    gap_count = 0

    for y in range(0, min(h // 2, int(h * 0.5))):
        if scores[y] >= threshold:
            if top_band_start is None:
                top_band_start = y
            top_band_end = y
            gap_count = 0
        else:
            if top_band_start is not None:
                gap_count += 1
                if gap_count > gap_tolerance:
                    break

    # Find bottom band: rows near the bottom with high scores, allowing larger gaps
    bottom_band_start = h
    bottom_band_end = h
    gap_count = 0

    for y in range(h - 1, max(h // 2, int(h * 0.5)) - 1, -1):
        if scores[y] >= threshold:
            if bottom_band_start == h:
                bottom_band_start = y
            bottom_band_end = y
            gap_count = 0
        else:
            if bottom_band_start != h:
                gap_count += 1
                if gap_count > gap_tolerance:
                    break

    top_cut = 0
    if top_band_start is not None and (top_band_end - top_band_start) >= min_band_size:
        # Include some margin to catch the full overlay
        top_cut = max(0, top_band_start - 5)

    bottom_cut = 0
    if bottom_band_start != h and (bottom_band_start - bottom_band_end) >= min_band_size:
        # Include some margin to catch the full overlay
        bottom_cut = max(0, h - bottom_band_start - 5)

    # Ignore tiny accidental detections
    if top_cut and top_cut < 15:
        top_cut = 0
    if bottom_cut and bottom_cut < 15:
        bottom_cut = 0

    return {"top": int(top_cut), "bottom": int(bottom_cut)}


def detect_bounds_across_video(video_path, sample_every=10, max_samples=100):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    sampled_top = []
    sampled_bottom = []
    frame_count = 0
    sample_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_count % sample_every != 0 and frame_count > 0:
            frame_count += 1
            continue

        bounds = detect_overlay_rows(frame, gap_tolerance=100, min_band_size=15)
        sampled_top.append(bounds["top"])
        sampled_bottom.append(bounds["bottom"])
        sample_count += 1
        if sample_count >= max_samples:
            break

        frame_count += 1

    cap.release()

    top = int(median(sampled_top)) if sampled_top else 0
    bottom = int(median(sampled_bottom)) if sampled_bottom else 0
    return {"top": top, "bottom": bottom}


def blur_fill_overlay(frame, top_cut, bottom_cut, blur_size=61):
    h, w = frame.shape[:2]
    if blur_size % 2 == 0:
        blur_size += 1

    # Blur the whole frame to create a background-like fill
    blurred = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
    result = blurred.copy()

    # Keep the original content in the center, blurred edges fill the removed overlay regions
    if top_cut > 0:
        result[0:top_cut, :] = frame[0:top_cut, :]
    if bottom_cut > 0:
        result[h - bottom_cut:h, :] = frame[h - bottom_cut:h, :]

    # Preserve the center content region
    center_start = top_cut
    center_end = h - bottom_cut if bottom_cut > 0 else h
    if center_end > center_start:
        result[center_start:center_end, :] = frame[center_start:center_end, :]

    return result


def process_video(input_path, output_path, blur_size=61, sample_every=10, max_samples=100):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    bounds = detect_bounds_across_video(input_path, sample_every=sample_every, max_samples=max_samples)
    print(f"Detected overlay bounds: top={bounds['top']} bottom={bounds['bottom']}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output_path}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out = blur_fill_overlay(frame, bounds["top"], bounds["bottom"], blur_size=blur_size)
        writer.write(out)

        frame_idx += 1
        if total_frames and frame_idx % max(1, total_frames // 10) == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    writer.release()
    print(f"Saved processed video to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Remove top/bottom overlays and blur the removed areas using the video itself as background.")
    parser.add_argument("--input", required=True, help="Path to source video")
    parser.add_argument("--output", required=True, help="Path to write cleaned video")
    parser.add_argument("--blur-size", type=int, default=61, help="Blur size used on the fill region")
    parser.add_argument("--sample-every", type=int, default=10, help="Sample every Nth frame for detection")
    parser.add_argument("--max-samples", type=int, default=100, help="Maximum detected samples across the video")
    args = parser.parse_args()

    try:
        process_video(args.input, args.output, args.blur_size, args.sample_every, args.max_samples)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
