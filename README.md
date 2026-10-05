#!/usr/bin/env python3
import argparse
import os
import sys
from statistics import median

import cv2
import numpy as np


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def border_width_for_side(frame, side, variance_threshold=18.0, brightness_cutoff=220.0):
    """Return the number of border pixels detected along a given edge."""
    if frame.ndim != 3:
        gray = frame
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape[:2]

    if side == "top":
        max_depth = height // 2
        width_pixels = width
        count = 0
        for y in range(max_depth):
            row = gray[y]
            row_std = float(row.std())
            row_mean = float(row.mean())
            if row_std <= variance_threshold and row_mean <= brightness_cutoff:
                count += 1
            else:
                break
        return count

    if side == "bottom":
        max_depth = height // 2
        count = 0
        for y in range(height - 1, height - max_depth - 1, -1):
            if y < 0:
                break
            row = gray[y]
            row_std = float(row.std())
            row_mean = float(row.mean())
            if row_std <= variance_threshold and row_mean <= brightness_cutoff:
                count += 1
            else:
                break
        return count

    if side == "left":
        max_depth = width // 2
        count = 0
        for x in range(max_depth):
            col = gray[:, x]
            col_std = float(col.std())
            col_mean = float(col.mean())
            if col_std <= variance_threshold and col_mean <= brightness_cutoff:
                count += 1
            else:
                break
        return count

    if side == "right":
        max_depth = width // 2
        count = 0
        for x in range(width - 1, width - max_depth - 1, -1):
            if x < 0:
                break
            col = gray[:, x]
            col_std = float(col.std())
            col_mean = float(col.mean())
            if col_std <= variance_threshold and col_mean <= brightness_cutoff:
                count += 1
            else:
                break
        return count

    raise ValueError(f"Unsupported side: {side}")


class LetterboxDetector:
    def __init__(self, variance_threshold=18.0, brightness_cutoff=220.0, sample_every=10, max_samples=200):
        self.variance_threshold = variance_threshold
        self.brightness_cutoff = brightness_cutoff
        self.sample_every = sample_every
        self.max_samples = max_samples

    def detect_global_bounds(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

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

            top = border_width_for_side(
                frame,
                "top",
                variance_threshold=self.variance_threshold,
                brightness_cutoff=self.brightness_cutoff,
            )
            bottom = border_width_for_side(
                frame,
                "bottom",
                variance_threshold=self.variance_threshold,
                brightness_cutoff=self.brightness_cutoff,
            )
            left = border_width_for_side(
                frame,
                "left",
                variance_threshold=self.variance_threshold,
                brightness_cutoff=self.brightness_cutoff,
            )
            right = border_width_for_side(
                frame,
                "right",
                variance_threshold=self.variance_threshold,
                brightness_cutoff=self.brightness_cutoff,
            )

            sampled_top.append(top)
            sampled_bottom.append(bottom)
            sampled_left.append(left)
            sampled_right.append(right)

            frame_count += 1
            if len(sampled_top) >= self.max_samples:
                break

        cap.release()

        top = self._choose_value(sampled_top)
        bottom = self._choose_value(sampled_bottom)
        left = self._choose_value(sampled_left)
        right = self._choose_value(sampled_right)

        return {
            "top": int(top),
            "bottom": int(bottom),
            "left": int(left),
            "right": int(right),
        }

    def _choose_value(self, values):
        if not values:
            return 0
        clean = [int(v) for v in values if int(v) > 0]
        if not clean:
            return 0
        return median(clean)

    def bounds_for_frame(self, frame, global_bounds):
        h, w = frame.shape[:2]
        top = max(global_bounds["top"], border_width_for_side(frame, "top", self.variance_threshold, self.brightness_cutoff))
        bottom = max(global_bounds["bottom"], border_width_for_side(frame, "bottom", self.variance_threshold, self.brightness_cutoff))
        left = max(global_bounds["left"], border_width_for_side(frame, "left", self.variance_threshold, self.brightness_cutoff))
        right = max(global_bounds["right"], border_width_for_side(frame, "right", self.variance_threshold, self.brightness_cutoff))

        # Do not crop beyond the valid frame size.
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


def process_video(input_path, output_path, mode="blur", variance_threshold=18.0, brightness_cutoff=220.0, sample_every=10, blur_size=31):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    detector = LetterboxDetector(
        variance_threshold=variance_threshold,
        brightness_cutoff=brightness_cutoff,
        sample_every=sample_every,
    )
    global_bounds = detector.detect_global_bounds(input_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open input video: {input_path}")

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

    cap.release()
    writer.release()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Remove video frame borders and optionally blur the background behind the cropped content.")
    parser.add_argument("--input", required=True, help="Path to the source video")
    parser.add_argument("--output", required=True, help="Path to save the cleaned video")
    parser.add_argument("--mode", choices=["crop", "blur"], default="blur", help="Remove bars by cropping or by blurring the filled background")
    parser.add_argument("--sample-every", type=int, default=10, help="Sample every Nth frame during border detection")
    parser.add_argument("--variance-threshold", type=float, default=18.0, help="Lower values are more sensitive to uniform borders")
    parser.add_argument("--brightness-cutoff", type=float, default=220.0, help="Upper brightness limit used when identifying dark bars")
    parser.add_argument("--blur-size", type=int, default=31, help="Gaussian blur kernel size used in blur-background mode")
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
        )
        print(f"Output saved to: {result}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


