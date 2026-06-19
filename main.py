# changes
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import config
from src.detection import FootballDetector
from src.tracking import FootballTracker
from src.homography import HomographyTransformer
from src.metrics import FootballMetrics
class FootballAnalytics:

    def __init__(self, input_video: str | None = None):
        # detetctor
        self.detector = FootballDetector(
            model_path = config.model_path,
            conf_threshold = config.confidence_threshold
        )
        # tracking 
        self.tracker = FootballTracker()

        self.input_video = input_video or config.input_video

        # Homography 
        cap = cv2.VideoCapture(self.input_video)

        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError('cannot read video file!')
        self.homography = HomographyTransformer(
            frame_shape = first_frame.shape[:2],
            reference_points = config.pitch_reference_points
        )

        #metrics
        cap = cv2.VideoCapture(self.input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.metrics = FootballMetrics(self.homography, fps)

    
    def process_video(self):
        """video processing"""
        cap = cv2.VideoCapture(self.input_video)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Outputs
        output_path = config.output_dir  / 'tracked_videos' / 'output.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        all_player_pos = {}

        for frame_idx in tqdm(range(total_frames), desc='Processing'):
            ret, frame = cap.read()
            if not ret:
                break

            #Detections
            detections = self.detector.detect_frame(frame)
            #tracking players
            player_detections = detections['players']
            tracks = self.tracker.update(player_detections)

            # Save locations
            # Use the last center of the track, but require only 1 hit to keep short clips populated.
            for track in tracks:
                if track.hit_streak >= 1:
                    all_player_pos.setdefault(track.track_id, []).append(track.centers[-1])


            #draw results on frame
            frame = self.draw_results(frame, tracks, detections['ball'])
            out.write(frame)
        cap.release()
        out.release()

        # Calculate metrics
        # Filter out tracks with too few points (ignore single-point noise).
        cleaned_player_pos = {}
        for pid, pts in all_player_pos.items():
            if len(pts) >= 3:
                cleaned_player_pos[pid] = pts

        
        stabilized_player_pos = {}
        
        max_speed_mps = 60.0
        max_jump_m = max_speed_mps / (fps if fps else 30.0)

        for pid, pts in cleaned_player_pos.items():
            # loosen minimum points so more players appear in short clips
            if len(pts) < 2:
                continue

            kept = [pts[0]]
            prev = pts[0]

            # last accepted point in real coordinates (so we don't recompute too much)
            real_prev = self.homography.pixel_to_real([prev])[0]

            for cur in pts[1:]:
                real_cur = self.homography.pixel_to_real([cur])[0]

                # discard points that map outside the pitch area (helps when homography/ref points are imperfect)
                x_m, y_m = float(real_cur[0]), float(real_cur[1])
                if not (0.0 <= x_m <= 105.0 and 0.0 <= y_m <= 68.0):
                    continue

                jump_m = float(np.linalg.norm(real_cur - real_prev))

                if jump_m <= max_jump_m:
                    kept.append(cur)
                    prev = cur
                    real_prev = real_cur
                else:
                    # outlier: skip this point but keep the previous accepted one (prevents collapse)
                    continue

            # keep only meaningful tracks
            if len(kept) >= 2:
                stabilized_player_pos[pid] = kept


        # Compute & write report
        self._calculate_metrics(stabilized_player_pos)
        return output_path

    def draw_results(self, frame, tracks, ball):
        """draw boxes and nummbers on video"""
        for track in tracks:
            if track.hit_streak < 3:
                continue
            x1, y1, x2, y2 = track.bbox
            #draw green rectangle on player
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # player number
            cv2.putText(frame, f"P{track.track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if ball:
            # ball bbox may be (x1,y1,x2,y2,conf,cls) depending on detector implementation
            if len(ball) >= 4:
                x1, y1, x2, y2 = ball[:4]
            else:
                return frame
            # draw red rectangle on ball
            cv2.rectangle(frame, (x1, y1) ,(x2, y2), (0, 0, 255),2)

        return frame
    def _calculate_metrics(self, player_positions):
        """calulate metrics"""
        player_metrics = {}
        for player_id, positions in player_positions.items():
            # For short videos, allow metrics even with few samples.
            # Keep a tiny minimum so we don't divide by zero.
            if len(positions) >= 2:
                met = self.metrics.calculate_player_speed(positions)
                player_metrics[player_id] = met

        # Safety: if tracking produced no (or near-empty) tracks, still write an empty sheet.
        # This prevents downstream crashes in the UI.
        if not player_metrics:
            player_metrics = {}


        df = self.metrics.generate_report(player_metrics)
        report_path = config.output_dir / 'reports' / 'report.xlsx'
        # Ensure parent directory exists (fixes OSError on some environments)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(report_path) as writer:
            df.to_excel(writer, sheet_name='Players', index=False)
    
    
def main():
    system = FootballAnalytics()
    output_path = system.process_video()

if __name__ == '__main__':
    main()