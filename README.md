# Public Places Group Detection System

## Overview

The **Public Places Group Detection System** is an advanced AI-powered solution designed to detect and analyze groups of people in an public places. Using YOLO for object detection and ByteTrack for tracking, the system identifies individuals, couples, pairs, and larger groups in real-time from RTSP streams. It logs detailed group insights, including gender classification, group dynamics, and changes over time.

### Key Benefits
- Understand group dynamics in shared office spaces.
- Automate the collection of group statistics for behavior analysis.
- Scalable solution for large office settings with multiple cameras.
- Real-time insights into seating arrangements and proximity.

## Features

- **Group Detection**: Identifies groups based on proximity and seating arrangements.
- **Gender Classification**: Classifies individuals within groups as Male or Female using a YOLO-based gender detection model.
- **Dynamic Group Analysis**: Tracks group changes over time, identifying new, split, or unchanged groups.
- **Real-Time Logging**: Records group statistics, including group type, size, and status, into a CSV file.
- **Multi-Camera Support**: Processes multiple RTSP streams simultaneously.
- **Customizable Proximity Thresholds**: Fine-tune the detection criteria for different public area layouts.

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (for optimal performance).

### Software
- Python 3.8 or higher
- Required Libraries:
  - `opencv-python`
  - `numpy`
  - `torch`
  - `ultralytics`

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<username>/office-group-detection.git
cd office-group-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare YOLO Weights
Download the following model weights and place them in the project directory:
- `yolov8x.pt` (for object detection)
- `Gendermodel_Yolov8.pt` (for gender classification)

### 4. Configure Directories
Create a directory for storing CSV logs:
```bash
mkdir Allcsv
```

## Usage

### 1. Run the Group Detection System
Start processing an RTSP stream for group detection:
```bash
python groupdetectionwithcsv.py --rtsp_url <RTSP_STREAM_URL>
```

#### Command-Line Arguments
- `--headless`: Runs the system without a GUI.
- `--rtsp_url`: The URL of the RTSP stream to process.
- `--model_path`: Path to the YOLO model for object detection.
- `--gender_model_path`: Path to the YOLO model for gender classification.
- `--tracker_path`: Path to the ByteTrack configuration file.
- `--min_distance`: Minimum distance threshold for grouping.
- `--max_group_distance`: Maximum distance threshold for group detection.
- `--frame_skip`: Number of frames to skip for faster processing.
- `--detection_interval`: Interval (in seconds) between detections.

### Example
```bash
python groupdetectionwithcsv.py --rtsp_url rtsp://<camera_stream> \
    --model_path yolov8x.pt \
    --gender_model_path Gendermodel_Yolov8.pt \
    --tracker_path bytetrack.yaml \
    --min_distance 150 \
    --max_group_distance 300 \
    --frame_skip 3 \
    --detection_interval 0.1
```

### 2. Review Group Statistics
Group data is saved in the `Allcsv/group_counts.csv` file:

- **Columns**:
  - `Time`: Timestamp of the detection.
  - `Individuals`: Count of single-member groups.
  - `Couples`: Count of male-female pairs.
  - `Pairs`: Count of same-gender pairs.
  - `Groups`: Count of larger groups.
  - `Group ID`: Unique identifier for each group.
  - `Members`: List of members in the group.
  - `Status`: Group status (New, Unchanged, Split).

## Project Structure

```
├── groupdetectionwithcsv.py     # Main script for group detection
├── requirements.txt             # List of dependencies
├── yolov8x.pt                   # YOLO object detection model
├── Gendermodel_Yolov8.pt        # YOLO gender classification model
├── bytetrack.yaml               # ByteTrack configuration file
├── Allcsv/                      # Directory for CSV logs
    ├── group_counts.csv         # Group statistics log
```

## Screenshots

### Group Detection in Action
![Group Detection Demo](image.png)

### Group Statistics Log
- Example of `group_counts.csv`:
  ```csv
  Time,Individuals,Couples,Pairs,Groups,Group ID,Members,Status
  2025-01-10 10:15:30,1,2,0,3,G1,Id 1,Id 2,New
  ```

## Future Enhancements

- **Heatmap Visualization**: Generate heatmaps for group density and seating arrangements.
- **Behavioral Insights**: Provide analytics on group dynamics and interactions.
- **Integration with Access Control**: Link group data with office access control systems.
- **Edge Device Optimization**: Optimize the system for edge devices like NVIDIA Jetson Nano.

## License

This project was inspired by the research paper 'Identification and Tracking of Groups of People Using Object Detection and Object Tracking Techniques' by Tharuja Sandeepanie and Subha Fernando. The methodologies and grouping algorithm detailed in their work have significantly influenced the development of this system.

### Citation
If you use this project in your research, please consider citing their work:
```
@article{Sandeepanie2023,
  title={Identification and Tracking of Groups of People Using Object Detection and Object Tracking Techniques},
  author={Tharuja Sandeepanie and Subha Fernando},
  journal={International Journal on Advances in ICT for Emerging Regions},
  volume={16},
  number={1},
  year={2023},
  doi={10.4038/icter.v16i1.7259}
}
```

You can find the original work [here](http://doi.org/10.4038/icter.v16i1.7259).

This project is licensed under the [MIT License](LICENSE).
