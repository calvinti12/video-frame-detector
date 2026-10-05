#!/usr/bin/env python3
"""
Debug script to analyze overlay detection scores on a single frame.
"""

import sys
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

if len(sys.argv) < 2:
    print("Usage: python debug_detection.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Could not open: {video_path}")
    sys.exit(1)

# Grab first frame
ok, frame = cap.read()
cap.release()

if not ok:
    print("Could not read first frame")
    sys.exit(1)

h, w = frame.shape[:2]
scores = row_score(frame)

print(f"Frame size: {w}x{h}")
print(f"\nRow scores (sample every 10 rows):")
print("Row | Score")
print("----|-------")
for y in range(0, h, 10):
    print(f"{y:3d} | {scores[y]:7.1f}")

# Find zones
center_start = int(h * 0.35)
center_end = int(h * 0.65)
center_median = float(np.median(scores[center_start:center_end]))
print(f"\nCenter region ({center_start}-{center_end}):")
print(f"  Median score: {center_median:.1f}")
print(f"  Min score: {scores[center_start:center_end].min():.1f}")
print(f"  Max score: {scores[center_start:center_end].max():.1f}")

print(f"\nTop region (0-{int(h*0.45)}):")
print(f"  Median score: {np.median(scores[0:int(h*0.45)]):.1f}")
print(f"  Max score: {scores[0:int(h*0.45)].max():.1f}")

print(f"\nBottom region ({int(h*0.55)}-{h}):")
print(f"  Median score: {np.median(scores[int(h*0.55):h]):.1f}")
print(f"  Max score: {scores[int(h*0.55):h].max():.1f}")

threshold_top = max(30.0, center_median * 1.35)
threshold_bottom = max(30.0, center_median * 1.25)
print(f"\nThresholds:")
print(f"  Top threshold (1.35x center): {threshold_top:.1f}")
print(f"  Bottom threshold (1.25x center): {threshold_bottom:.1f}")
