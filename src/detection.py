import cv2
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from ultralytics import YOLO


class FootballDetector:
    """YOLO-based detector for players + ball.

    This wrapper is designed to match the expectations in `main.py`:
    - `detect_frame(frame)` returns:
        {
          'players': List[bbox],
          'ball': bbox or None
        }
      where bbox is (x1, y1, x2, y2, conf, cls)
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

    def _xyxy(self, boxes) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = boxes
        return int(x1), int(y1), int(x2), int(y2)

    def detect_frame(self, frame: np.ndarray) -> Dict[str, object]:
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            device=self.device,
        )[0]

        players: List[Tuple] = []
        ball: Optional[Tuple] = None

        if results.boxes is None or len(results.boxes) == 0:
            # Debug: no detections on this frame
            print("[detector] no boxes detected")
            return {"players": [], "ball": None}

        boxes = results.boxes
        # Debug only once per call (avoid flooding):
        try:
            cls_confs = []
            for bi in range(min(len(boxes), 5)):
                b = boxes[bi]
                cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
                conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
                cls_confs.append((cls_id, conf))
            print(f"[detector] boxes={len(boxes)} sample_cls_conf={cls_confs}")
        except Exception:
            pass

        for i in range(len(boxes)):
            b = boxes[i]
            cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)
            conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            bbox = (int(x1), int(y1), int(x2), int(y2), conf, cls_id)

            # Class mapping:
            # - ball_class_ids: classes treated as ball
            # - player_class_ids: classes treated as players
            # If player_class_ids is not set, we fallback to: cls_id not in ball_class_ids.
            from config import config
            ball_class_ids = config.ball_class_ids if config.ball_class_ids is not None else [0]
            player_class_ids = config.player_class_ids

            is_ball = cls_id in ball_class_ids
            if player_class_ids is not None:
                is_player = cls_id in player_class_ids
            else:
                is_player = not is_ball

            if is_ball:
                # pick the highest-confidence ball
                if ball is None or conf > ball[4]:
                    ball = bbox
            elif is_player:
                players.append(bbox)


        return {"players": players, "ball": ball}

