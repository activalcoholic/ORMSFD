# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SolutionConfig:
    """Manages configuration parameters for Ultralytics Vision AI solutions.

    The SolutionConfig class serves as a centralized configuration container for all the Ultralytics solution modules:
    https://docs.ultralytics.com/solutions/#solutions. It leverages Python `dataclass` for clear, type-safe, and
    maintainable parameter definitions.

    Attributes:
        source (str, optional): Path to the input source (video, RTSP, etc.). Only usable with Solutions CLI.
        model (str, optional): Path to the Ultralytics YOLO model to be used for inference.
        classes (list[int], optional): List of class indices to filter detections.
        show_conf (bool): Whether to show confidence scores on the visual output.
        show_labels (bool): Whether to display class labels on visual output.
        region (list[tuple[int, int]], optional): Polygonal region or line used by the solution.
        blur_ratio (float): Ratio used to blur objects in the video frames (0.0 to 1.0).
        vision_point (tuple[int, int]): Reference point for directional tracking or perspective drawing.
        crop_dir (str): Directory path to save cropped detection images.
        line_width (int): Width for visual display, e.g. bounding boxes and keypoints.
        records (int): Number of detection records to send email alerts.
        fps (float): Frame rate (Frames Per Second) for speed estimation calculation.
        max_hist (int): Maximum number of historical points or states stored per tracked object for speed estimation.
        meter_per_pixel (float): Scale for real-world measurement, used in speed or distance calculations.
        max_speed (int): Maximum speed limit (e.g., km/h or mph) used in visual alerts or constraints.
        show (bool): Whether to display the visual output on screen.
        iou (float): Intersection-over-Union threshold for detection filtering.
        conf (float): Confidence threshold for keeping predictions.
        device (str, optional): Device to run inference on (e.g., 'cpu', '0' for CUDA GPU).
        max_det (int): Maximum number of detections allowed per video frame.
        half (bool): Whether to use FP16 precision (requires a supported CUDA device).
        tracker (str): Path to tracking configuration YAML file (e.g., 'botsort.yaml').
        verbose (bool): Enable verbose logging output for debugging or diagnostics.
        data (str): Path to image directory used for similarity search.

    Methods:
        update: Update the configuration with user-defined keyword arguments and raise error on invalid keys.

    Examples:
        >>> from ultralytics.solutions.config import SolutionConfig
        >>> cfg = SolutionConfig(model="yolo26n.pt", region=[(0, 0), (100, 0), (100, 100), (0, 100)])
        >>> cfg.update(show=False, conf=0.3)
        >>> print(cfg.model)
    """

    source: str | None = None
    model: str | None = None
    classes: list[int] | None = None
    show_conf: bool = True
    show_labels: bool = True
    region: list[tuple[int, int]] | None = None
    blur_ratio: float = 0.5
    vision_point: tuple[int, int] = (20, 20)
    crop_dir: str = "cropped-detections"
    line_width: int = 2
    records: int = 5
    fps: float = 30.0
    max_hist: int = 5
    meter_per_pixel: float = 0.05
    max_speed: int = 120
    show: bool = False
    iou: float = 0.7
    conf: float = 0.25
    device: str | None = None
    max_det: int = 300
    half: bool = False
    tracker: str = "botsort.yaml"
    verbose: bool = True
    data: str = "images"

    def update(self, **kwargs: Any):
        """Update configuration parameters with new values provided as keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                url = "https://docs.ultralytics.com/solutions/#solutions-arguments"
                raise ValueError(f"{key} is not a valid solution argument, see {url}")

        return self
