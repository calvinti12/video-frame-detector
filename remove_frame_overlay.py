#!/usr/bin/env python3
"""
Specialized tool for removing top/bottom video overlays like news-title banners and caption bars,
then filling the removed space with a blurred version of the same video content.

This is tuned for layouts like:
- red/blue top banner: "BREAKING NEWS"
- white headline bar below it
- center actual video
- blurred background filled from the same video
"""

import argparse
import os
import sys
from statistics import median

import cv2
import numpy as np


def compute_row_metrics(frame):
    """Return row-wise metrics for detecting title/caption bands."""
    h, w = frame.shape[:2]
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    # Edge density and variance per row
    edges = cv2.Canny(gray, 50, 150)
    row_var = []
    row_edge = []
    row_mean = []

    for y in range(h):
        row = gray[y]
        row_var.append(float(row.var()))
        row_edge.append(float(np.sum(edges[y] > 0)) / float(w))
        row_mean.append(float(row.mean()))

    return row_var, row_edge, row_mean


def detect_overlay_rows(frame):
    """
    Detect top and bottom overlay rows based on strong high-contrast graphic blocks.
    """
    h, w = frame.shape[:2]
    row_var, row_edge, row_mean = compute_row_metrics(frame)

    # Score rows with high edge density and intense contrast, typical of text/logo graphics
    score = []
    for idx in range(h):
        s = row_edge[idx] * 1000.0 + (row_var[idx] / 30.0)
        score.append(s)

    # Top overlay: likely within first 35% of frame height
    top_rows = []
    for y in range(min(h // 2, int(h * 0.35))):
        if score[y] > 25.0 and row_edge[y] > 0.03:
            top_rows.append(y)

    # Bottom overlay: likely within last 25% of frame height
    bottom_rows = []
    for y in range(max(0, h - int(h * 0.25)), h):
        if score[y] > 20.0 and row_edge[y] > 0.03:
            bottom_rows.append(y)

    top_bound = 0
    if top_rows:
        # Start at first non-content row and extend until score drops
        candidate_top = min(top_rows)
        for y in range(candidate_top, min(h, candidate_top + 120)):
            if score[y] < 10.0:
                top_bound = y
                break
        else:
            top_bound = min(h // 3, max(top_rows) + 10)

    bottom_bound = 0
    if bottom_rows:
        candidate_bottom = max(bottom_rows)
        for y in range(candidate_bottom, max(0, candidate_bottom - 120), -1):
            if score[y] < 10.0:
                bottom_bound = h - y
                break
        else:
            bottom_bound = min(h // 4, h - min(bottom_rows))

    # Use a conservative estimate so we don't clip real content
    top_bound = max(0, min(top_bound, h // 3))
    bottom_bound = max(0, min(bottom_bound, h // 3))

    return {
        "top": int(top_bound),
        "bottom": int(bottom_bound),
    }


def detect_overlay_rows_across_video(video_path, sample_every=15, max_samples=80):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    samples_top = []
    samples_bottom = []
    frame_count = 0
    sample_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_count % sample_every != 0 and frame_count > 0:
            frame_count += 1
            continue

        bounds = detect_overlay_rows(frame)
        samples_top.append(bounds["top"])
        samples_bottom.append(bounds["bottom"])
        sample_count += 1

        if sample_count >= max_samples:
            break
        frame_count += 1

    cap.release()

    top_val = int(median(samples_top)) if samples_top else 0
    bottom_val = int(median(samples_bottom)) if samples_bottom else 0
    return {"top": top_val, "bottom": bottom_val}


def blur_fill_overlay(frame, top_cut, bottom_cut, blur_size=61):
    """Replace the removed top/bottom overlay region with a blurred version of the content."""
    h, w = frame.shape[:2]
    if blur_size % 2 == 0:
        blur_size += 1

    center_start = top_cut
    center_end = h - bottom_cut
    if center_end <= center_start:
        return cv2.GaussianBlur(frame, (blur_size, blur_size), 0)

    content = frame[center_start:center_end, :]
    if content.size == 0:
        return cv2.GaussianBlur(frame, (blur_size, blur_size), 0)

    # Resize content to full frame for blur fill, then insert original content back
    resized_content = cv2.resize(content, (w, h), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(resized_content, (blur_size, blur_size), 0)

    # Re-insert the original content in the center region
    result = blurred.copy()
    result[center_start:center_end, :] = frame[center_start:center_end, :]
    return result


def process_video(input_path, output_path, blur_size=61, sample_every=15, max_samples=80):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    bounds = detect_overlay_rows_across_video(input_path, sample_every=sample_every, max_samples=max_samples)
    print(f"Detected overlay bounds: top={bounds['top']} bottom={bounds['bottom']}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output: {output_path}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        out = blur_fill_overlay(frame, bounds["top"], bounds["bottom"], blur_size=blur_size)
        writer.write(out)

    cap.release()
    writer.release()
    print(f"Saved processed video to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Remove top/bottom video overlays and blur the remaining video as background.")
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to save cleaned video")
    parser.add_argument("--blur-size", type=int, default=61, help="Blur strength (odd integer; higher = stronger blur)")
    parser.add_argument("--sample-every", type=int, default=15, help="Sample every Nth frame for detection")
    parser.add_argument("--max-samples", type=int, default=80, help="Max frames to sample for detection")
    args = parser.parse_args()

    try:
        process_video(
            args.input,
            args.output,
            blur_size=args.blur_size,
            sample_every=args.sample_every,
            max_samples=args.max_samples,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
