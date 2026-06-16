# config.py
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # File paths
    model_path: str = "yolo11n.pt"
    input_video: str = "c:/Users/Abdo/Downloads/scoring 102.avi/scoring 102.avi"

    output_dir: Path = Path("./output")

    # Detection settings
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45

    # Tracking settings
    tracker_type: str = "bytetrack"  # or "botsort"
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.5

    # Pitch dimensions (meters)
    pitch_width: float = 68.0  # Pitch width (m)
    pitch_length: float = 105.0  # Pitch length (m)

    # Pitch reference points (for Homography)
    # Pixel coordinates in the video -> Real-world coordinates (meters)
    pitch_reference_points: dict = None

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "heatmaps").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "tracked_videos").mkdir(exist_ok=True)

        # Default reference points (update them for your specific video)
        self.pitch_reference_points = {
            "pixel_points": [
                [100, 200],  # Top-left corner
                [500, 200],  # Top-right corner
                [100, 400],  # Bottom-left corner
                [500, 400],  # Bottom-right corner
            ],
            "real_points": [
                [0, 0],  # (x, y) in meters
                [105, 0],
                [0, 68],
                [105, 68],
            ],
        }


config = Config()

