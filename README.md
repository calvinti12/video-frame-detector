# Video Frame Border Cleaner

Professional tool for removing letterbox/frame borders from video while keeping the content stable and optionally blurring the background in the eliminated area.

## Features

✅ **Intelligent Border Detection**
- Detects black bars, gray bars, and colored frame borders
- Handles top/bottom/left/right borders independently
- Uses edge detection (Canny) for colored or soft borders
- Variance analysis for uniform dark borders

✅ **Spike Filtering**
- Automatically filters out brief border artifacts (1-2 frame spikes)
- Uses running median for smooth border detection across video
- Prevents jitter in detected border sizes

✅ **Multiple Processing Modes**
- **Blur mode**: Keep full frame size, blur the removed border areas
- **Crop mode**: Remove borders and center content

✅ **Flexible Interface**
- **CLI**: Command-line tool for scripting and automation
- **GUI**: User-friendly drag-and-drop interface with live parameter adjustment
- **Batch Processor**: Process multiple videos in a folder

✅ **Advanced Parameters**
- Variance threshold: Fine-tune border detection sensitivity
- Brightness cutoff: Target dark or lighter borders
- Edge sensitivity: Detect colored/soft borders
- Blur size: Control background blur strength
- Spike filter: Remove brief frame artifacts

## Installation

```bash
python 3.8+ required

# Clone or download this repository
cd video-frame-cleaner

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### GUI (Easiest)

```bash
python gui.py
```

Drag and drop your video, adjust parameters, and click "Process Video".

### CLI

```bash
# Basic usage (blur mode)
python remove_letterbox.py --input video.mp4 --output cleaned.mp4

# Crop mode
python remove_letterbox.py --input video.mp4 --output cleaned.mp4 --mode crop

# Colored/soft borders
python remove_letterbox.py --input video.mp4 --output cleaned.mp4 \
  --variance-threshold 12 \
  --brightness-cutoff 150 \
  --edge-sensitivity 0.4

# Brief frame spikes (flicker in borders)
python remove_letterbox.py --input video.mp4 --output cleaned.mp4 \
  --spike-filter-window 7
```

### Batch Processing

```bash
# Process all videos in a folder
python batch_processor.py --input-dir ./videos --output-dir ./cleaned

# With custom parameters
python batch_processor.py \
  --input-dir ./videos \
  --output-dir ./cleaned \
  --mode blur \
  --variance-threshold 15 \
  --brightness-cutoff 180
```

## Parameter Tuning Guide

### Black/Dark Borders (Default Settings)
```bash
python remove_letterbox.py --input video.mp4 --output cleaned.mp4
```

### Gray or Semi-Dark Borders
```bash
python remove_letterbox.py --input video.mp4 --output cleaned.mp4 \
  --variance-threshold 15 \
  --brightness-cutoff 200
```

### Colored Borders (Red, Green, Blue, etc.)
```bash
python remove_letterbox.py --input video.mp4 --output cleaned.mp4 \
  --variance-threshold 12 \
  --brightness-cutoff 150 \
  --edge-sensitivity 0.4
```

### Borders with Brief Spikes (1-3 frames of flicker)
```bash
python remove_letterbox.py --input video.mp4 --output cleaned.mp4 \
  --spike-filter-window 7
```

### Very Subtle or Soft Borders
```bash
python remove_letterbox.py --input video.mp4 --output cleaned.mp4 \
  --variance-threshold 8 \
  --edge-sensitivity 0.5
```

## Parameter Reference

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `--mode` | `crop`, `blur` | `blur` | Processing mode |
| `--variance-threshold` | 5-50 | 18.0 | Border uniformity threshold (lower = more sensitive) |
| `--brightness-cutoff` | 50-255 | 220.0 | Max brightness for dark borders (lower = darker required) |
| `--edge-sensitivity` | 0.1-0.8 | 0.3 | Colored border detection (higher = more sensitive) |
| `--blur-size` | 5-99 (odd) | 31 | Gaussian blur kernel for background (blur mode) |
| `--sample-every` | 1-100 | 10 | Detect borders every Nth frame |
| `--spike-filter-window` | 1-20 | 5 | Smooth brief border artifacts across frames |

## How It Works

1. **Sample Frames**: Scan every Nth frame to detect border patterns
2. **Dual Detection**: Uses both variance analysis and edge detection
3. **Median Filtering**: Takes median border size across samples
4. **Spike Filtering**: Removes anomalies with running median filter
5. **Per-Frame Processing**: Ensures consistency even with unstable borders
6. **Output**: Either crops or blurs based on selected mode

## Examples

### Example 1: YouTube Video with Black Letterbox
```bash
python remove_letterbox.py --input youtube_download.mp4 --output cleaned.mp4
```

### Example 2: Livestream with Gray Frame Overlay
```bash
python remove_letterbox.py --input stream.mp4 --output cleaned.mp4 \
  --brightness-cutoff 180 --variance-threshold 15
```

### Example 3: Multiple Videos with Different Border Styles
```bash
# Create input folder with videos
mkdir input_videos
cp video1.mp4 video2.mp4 video3.mp4 input_videos/

# Process all
python batch_processor.py --input-dir input_videos --output-dir output_videos
```

## Troubleshooting

**Issue**: Borders not detected
- **Solution**: Lower `--variance-threshold` (try 10-12)
- **Solution**: Lower `--brightness-cutoff` (try 150-180 for gray borders)

**Issue**: Content is being cropped
- **Solution**: Raise `--variance-threshold` (try 25-30)
- **Solution**: Raise `--brightness-cutoff` (try 240+)

**Issue**: Brief flickers in detected borders
- **Solution**: Increase `--spike-filter-window` (try 7-10)

**Issue**: Colored borders not detected
- **Solution**: Raise `--edge-sensitivity` (try 0.4-0.6)
- **Solution**: Lower `--brightness-cutoff` (try 100-150)

**Issue**: Output is too blurry (blur mode)
- **Solution**: Reduce `--blur-size` (try 15-21)

**Issue**: Output is not blurry enough (blur mode)
- **Solution**: Increase `--blur-size` (try 41-51)

## Performance Notes

- GUI: Single-threaded, suitable for videos up to 2 hours
- Batch processing: Processes one video at a time
- For videos >4GB: Use `--sample-every 20` or higher to speed up detection
- Output quality depends on input video codec; uses MP4V by default

## License

MIT License - See LICENSE file for details

## Support

For issues, feature requests, or sample frame analysis:
1. Test with GUI first to find optimal parameters
2. Note the detected border values from console output
3. Adjust parameters based on the troubleshooting guide above
