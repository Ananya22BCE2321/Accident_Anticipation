# Accident Anticipation Capstone Project

## Overview
This project focuses on accident anticipation by extracting and analyzing trajectories of vehicles and pedestrians from video data. Using YOLOv8 for object detection and ByteTrack for tracking, the system processes videos labeled as "positive" (accident-prone scenarios) and "negative" (safe scenarios) to generate trajectory data for machine learning model training.

The goal is to predict potential accidents based on object movements, speeds, and headings.

## Features
- **Object Detection and Tracking**: Detects and tracks cars, motorcycles, buses, trucks, and pedestrians using YOLOv8 and ByteTrack.
- **Trajectory Extraction**: Computes positions, speeds, and headings for each tracked object across video frames.
- **Data Labeling**: Automatically labels trajectories as "positive" or "negative" based on video source folders.
- **Visualization**: Saves annotated videos showing trajectories overlaid on the original footage.
- **Data Export**: Outputs trajectory data to CSV and per-video JSON files for analysis and training.

## Project Structure
```
.
├── b.ipynb                          # Main Jupyter notebook for processing videos
├── trajectories.csv                 # Consolidated trajectory data from all videos
├── yolov8n.pt                       # YOLOv8 nano model weights
├── data_processed/                  # Processed data (if any)
├── TU-DAT/
│   ├── Final Videos_processed/
│   │   ├── positive/                # Videos of accident-prone scenarios
│   │   └── negative/                # Videos of safe scenarios
│   └── ...                          # Other data folders
├── tracked_videos/                  # Output: Annotated videos with trajectories
├── trajectories/                    # Output: Per-video JSON trajectory files
└── README.md                        # This file
```

## Prerequisites
- Python 3.8+
- Jupyter Notebook or VS Code with Python extension
- GPU recommended for faster processing (optional)

## Installation
1. Clone or download the project repository.
2. Install required Python packages:
   ```
   pip install ultralytics supervision opencv-python pandas tqdm torch torchvision
   ```
   Or run the first cell in `b.ipynb` to install via `%pip`.

3. Ensure `yolov8n.pt` is in the project root (pre-downloaded).

## Usage
1. **Prepare Data**: Place your processed MP4 videos in `TU-DAT/Final Videos_processed/positive/` and `negative/` folders. Videos should be pre-processed to 12 FPS for consistency.

2. **Run the Notebook**:
   - Open `b.ipynb` in Jupyter or VS Code.
   - Execute the cells in order. The main function will:
     - Detect and track objects in all videos.
     - Extract trajectories with speeds and headings.
     - Save annotated videos to `tracked_videos/`.
     - Save per-video JSON to `trajectories/`.
     - Compile all data into `trajectories.csv`.

3. **Output Files**:
   - `trajectories.csv`: Columns include `video`, `label`, `frame`, `track_id`, `class`, `bbox`, `center_x`, `center_y`, `speed_mps`, `heading_deg`.
   - Tracked videos: Visualizations of detections and trajectories.
   - JSON files: Detailed frame-by-frame data per video.

## Configuration
- **FPS**: Set to 12 in the config (matches pre-processing).
- **Pixel to Meter**: Calibration factor (0.1) for speed calculations.
- **Confidence Threshold**: 0.3 for YOLO detections.
- **Object Classes**: Person (0), Car (2), Motorcycle (3), Bus (5), Truck (7).

## Training Your Model
Use `trajectories.csv` as input for your ML model:
- Features: Positions, speeds, headings, object classes.
- Labels: 'positive' or 'negative' for accident prediction.
- Consider sequence modeling (e.g., LSTMs) for temporal patterns.

## Troubleshooting
- **No Videos Found**: Check folder paths and ensure MP4 files are present.
- **Low Detections**: Adjust confidence threshold or check video quality.
- **Memory Issues**: Process fewer videos at a time by modifying the loop in `main()`.
- **Video Saving Fails**: Ensure OpenCV is installed and codec support is available.

## Contributing
This is a capstone project. For improvements, modify `b.ipynb` and test thoroughly.



## Acknowledgments
- YOLOv8 by Ultralytics
- Supervision library for tracking

- OpenCV for video processing
