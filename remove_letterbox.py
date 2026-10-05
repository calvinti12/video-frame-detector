#!/usr/bin/env python3
import argparse
import os
import sys
from statistics import median
from collections import deque

import cv2
import numpy as np


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def detect_border_with_edge_detection(frame, side, variance_threshold=18.0, brightness_cutoff=220.0, edge_sensitivity=0.3):
    """
    Enhanced border detection using:
    - Variance/brightness for dark borders
    - Edge detection (Canny) for colored/soft borders
    - Morphological operations for noise filtering
    """
    if frame.ndim != 3:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape[:2]

    # Apply bilateral filter to smooth while preserving edges
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Canny edge detection for colored borders
    edges = cv2.Canny(smooth, 50, 150)
    
    if side == "top":
        max_depth = height // 2
        for y in range(max_depth):
            row = gray[y]
            row_smooth = smooth[y]
            row_edges = edges[y]
            
            row_std = float(row.std())
            row_mean = float(row.mean())
            edge_density = float(np.sum(row_edges > 0)) / float(len(row_edges)) if len(row_edges) > 0 else 0
            
            # Uniform dark bar OR low edge content
            is_uniform = row_std <= variance_threshold and row_mean <= brightness_cutoff
            is_low_content = edge_density <= edge_sensitivity
            
            if is_uniform or is_low_content:
                continue
            else:
                return y
        return max_depth

    elif side == "bottom":
        max_depth = height // 2
        for y in range(height - 1, height - max_depth - 1, -1):
            if y < 0:
                break
            row = gray[y]
            row_smooth = smooth[y]
            row_edges = edges[y]
            
            row_std = float(row.std())
            row_mean = float(row.mean())
            edge_density = float(np.sum(row_edges > 0)) / float(len(row_edges)) if len(row_edges) > 0 else 0
            
            is_uniform = row_std <= variance_threshold and row_mean <= brightness_cutoff
            is_low_content = edge_density <= edge_sensitivity
            
            if is_uniform or is_low_content:
                continue
            else:
                return height - 1 - y
        return max_depth

    elif side == "left":
        max_depth = width // 2
        for x in range(max_depth):
            col = gray[:, x]
            col_smooth = smooth[:, x]
            col_edges = edges[:, x]
            
            col_std = float(col.std())
            col_mean = float(col.mean())
            edge_density = float(np.sum(col_edges > 0)) / float(len(col_edges)) if len(col_edges) > 0 else 0
            
            is_uniform = col_std <= variance_threshold and col_mean <= brightness_cutoff
            is_low_content = edge_density <= edge_sensitivity
            
            if is_uniform or is_low_content:
                continue
            else:
                return x
        return max_depth

    elif side == "right":
        max_depth = width // 2
        for x in range(width - 1, width - max_depth - 1, -1):
            if x < 0:
                break
            col = gray[:, x]
            col_smooth = smooth[:, x]
            col_edges = edges[:, x]
            
            col_std = float(col.std())
            col_mean = float(col.mean())
            edge_density = float(np.sum(col_edges > 0)) / float(len(col_edges)) if len(col_edges) > 0 else 0
            
            is_uniform = col_std <= variance_threshold and col_mean <= brightness_cutoff
            is_low_content = edge_density <= edge_sensitivity
            
            if is_uniform or is_low_content:
                continue
            else:
                return width - 1 - x
        return max_depth

    raise ValueError(f"Unsupported side: {side}")


class LetterboxDetector:
    def __init__(self, variance_threshold=18.0, brightness_cutoff=220.0, sample_every=10, max_samples=200, edge_sensitivity=0.3, spike_filter_window=5):
        self.variance_threshold = variance_threshold
        self.brightness_cutoff = brightness_cutoff
        self.sample_every = sample_every
        self.max_samples = max_samples
        self.edge_sensitivity = edge_sensitivity
        self.spike_filter_window = spike_filter_window

    def detect_global_bounds(self, video_path, progress_callback=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        sampled_top = []
        sampled_bottom = []
        sampled_left = []
        sampled_right = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_count % self.sample_every != 0 and frame_count > 0:
                frame_count += 1
                continue

            top = detect_border_with_edge_detection(
                frame,
                "top",
                variance_threshold=self.variance_threshold,
                brightness_cutoff=self.brightness_cutoff,
                edge_sensitivity=self.edge_sensitivity,
            )
            bottom = detect_border_with_edge_detection(
                frame,
                "bottom",
                variance_threshold=self.variance_threshold,
                brightness_cutoff=self.brightness_cutoff,
                edge_sensitivity=self.edge_sensitivity,
            )
            left = detect_border_with_edge_detection(
                frame,
                "left",
                variance_threshold=self.variance_threshold,
                brightness_cutoff=self.brightness_cutoff,
                edge_sensitivity=self.edge_sensitivity,
            )
            right = detect_border_with_edge_detection(
                frame,
                "right",
                variance_threshold=self.variance_threshold,
                brightness_cutoff=self.brightness_cutoff,
                edge_sensitivity=self.edge_sensitivity,
            )

            sampled_top.append(top)
            sampled_bottom.append(bottom)
            sampled_left.append(left)
            sampled_right.append(right)

            frame_count += 1
            if len(sampled_top) >= self.max_samples:
                break

            if progress_callback:
                progress = (frame_count / total_frames) * 100
                progress_callback(progress)

        cap.release()

        top = self._choose_value_with_spike_filter(sampled_top)
        bottom = self._choose_value_with_spike_filter(sampled_bottom)
        left = self._choose_value_with_spike_filter(sampled_left)
        right = self._choose_value_with_spike_filter(sampled_right)

        return {
            "top": int(top),
            "bottom": int(bottom),
            "left": int(left),
            "right": int(right),
        }

    def _choose_value_with_spike_filter(self, values):
        """Use median with spike filtering for short-burst anomalies."""
        if not values:
            return 0
        
        clean = [int(v) for v in values if int(v) > 0]
        if not clean:
            return 0
        
        # Apply running median to filter out brief spikes
        if len(clean) <= self.spike_filter_window:
            return median(clean)
        
        filtered = []
        for i in range(len(clean)):
            window_start = max(0, i - self.spike_filter_window // 2)
            window_end = min(len(clean), i + self.spike_filter_window // 2 + 1)
            window = clean[window_start:window_end]
            filtered.append(median(window))
        
        return int(median(filtered))

    def bounds_for_frame(self, frame, global_bounds):
        h, w = frame.shape[:2]
        top = max(global_bounds["top"], detect_border_with_edge_detection(frame, "top", self.variance_threshold, self.brightness_cutoff, self.edge_sensitivity))
        bottom = max(global_bounds["bottom"], detect_border_with_edge_detection(frame, "bottom", self.variance_threshold, self.brightness_cutoff, self.edge_sensitivity))
        left = max(global_bounds["left"], detect_border_with_edge_detection(frame, "left", self.variance_threshold, self.brightness_cutoff, self.edge_sensitivity))
        right = max(global_bounds["right"], detect_border_with_edge_detection(frame, "right", self.variance_threshold, self.brightness_cutoff, self.edge_sensitivity))

        top = min(top, h // 2)
        bottom = min(bottom, h // 2)
        left = min(left, w // 2)
        right = min(right, w // 2)

        return {
            "top": int(top),
            "bottom": int(bottom),
            "left": int(left),
            "right": int(right),
        }


def build_blurred_background(frame, bounds, blur_size=31):
    if blur_size < 1:
        blur_size = 1
    if blur_size % 2 == 0:
        blur_size += 1

    h, w = frame.shape[:2]
    top, bottom, left, right = bounds["top"], bounds["bottom"], bounds["left"], bounds["right"]

    result = cv2.GaussianBlur(frame, (blur_size, blur_size), 0)
    content_h = h - top - bottom
    content_w = w - left - right
    if content_h <= 0 or content_w <= 0:
        return frame.copy()

    source = frame[top : h - bottom, left : w - right]
    if source.shape[:2] != (content_h, content_w):
        source = cv2.resize(source, (content_w, content_h), interpolation=cv2.INTER_LINEAR)

    canvas_h = h
    canvas_w = w
    dst = result.copy()

    y0 = top
    x0 = left
    if y0 + content_h > canvas_h:
        y0 = max(0, canvas_h - content_h)
    if x0 + content_w > canvas_w:
        x0 = max(0, canvas_w - content_w)

    dst[y0 : y0 + content_h, x0 : x0 + content_w] = source
    return dst


def crop_frame(frame, bounds):
    h, w = frame.shape[:2]
    top = bounds["top"]
    bottom = bounds["bottom"]
    left = bounds["left"]
    right = bounds["right"]

    if h - top - bottom <= 0 or w - left - right <= 0:
        return frame.copy()

    return frame[top : h - bottom, left : w - right].copy()


def process_video(input_path, output_path, mode="blur", variance_threshold=18.0, brightness_cutoff=220.0, sample_every=10, blur_size=31, edge_sensitivity=0.3, spike_filter_window=5, progress_callback=None):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    detector = LetterboxDetector(
        variance_threshold=variance_threshold,
        brightness_cutoff=brightness_cutoff,
        sample_every=sample_every,
        edge_sensitivity=edge_sensitivity,
        spike_filter_window=spike_filter_window,
    )
    
    if progress_callback:
        progress_callback("Detecting borders...")
    
    global_bounds = detector.detect_global_bounds(input_path, progress_callback)
    print(f"Detected borders: top={global_bounds['top']}, bottom={global_bounds['bottom']}, left={global_bounds['left']}, right={global_bounds['right']}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer for output: {output_path}")

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        bounds = detector.bounds_for_frame(frame, global_bounds)
        if mode == "crop":
            processed = crop_frame(frame, bounds)
            processed_h, processed_w = processed.shape[:2]
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            y = (height - processed_h) // 2
            x = (width - processed_w) // 2
            canvas[y : y + processed_h, x : x + processed_w] = processed
            processed = canvas
        elif mode == "blur":
            processed = build_blurred_background(frame, bounds, blur_size=blur_size)
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'crop' or 'blur'.")

        writer.write(processed)
        frame_index += 1
        
        if progress_callback and total_frames > 0:
            progress = (frame_index / total_frames) * 100
            progress_callback(f"Processing: {progress:.1f}%")

    cap.release()
    writer.release()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Remove video frame borders and optionally blur the background behind the cropped content.")
    parser.add_argument("--input", required=True, help="Path to the source video")
    parser.add_argument("--output", required=True, help="Path to save the cleaned video")
    parser.add_argument("--mode", choices=["crop", "blur"], default="blur", help="Remove bars by cropping or by blurring the filled background")
    parser.add_argument("--sample-every", type=int, default=10, help="Sample every Nth frame during border detection")
    parser.add_argument("--variance-threshold", type=float, default=18.0, help="Lower values are more sensitive to uniform borders (colored/soft borders: 10-25)")
    parser.add_argument("--brightness-cutoff", type=float, default=220.0, help="Upper brightness limit for dark bars (colored bars: lower value like 150)")
    parser.add_argument("--blur-size", type=int, default=31, help="Gaussian blur kernel size used in blur-background mode")
    parser.add_argument("--edge-sensitivity", type=float, default=0.3, help="Edge detection sensitivity for colored borders (0.1-0.5)")
    parser.add_argument("--spike-filter-window", type=int, default=5, help="Window size for filtering brief border spikes")
    args = parser.parse_args()

    try:
        result = process_video(
            args.input,
            args.output,
            mode=args.mode,
            variance_threshold=args.variance_threshold,
            brightness_cutoff=args.brightness_cutoff,
            sample_every=args.sample_every,
            blur_size=args.blur_size,
            edge_sensitivity=args.edge_sensitivity,
            spike_filter_window=args.spike_filter_window,
        )
        print(f"Output saved to: {result}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
