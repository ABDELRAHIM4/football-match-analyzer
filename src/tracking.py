import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict
import cv2


class Track:
    def __init__(self, track_id: int, bbox: Tuple, center:Tuple):
        self.track_id = track_id
        self.bbox = bbox
        self.centers = [center]
        self.history = []
        self.age = 1
        self.hit_streak = 1

        # Kalman filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1,0,0,0],
                              [0,1,0,0]])
        self.kf.P *= 1000
        self.kf.R = 5
        self.kf.Q = 0.01
        self.kf.x[:2] = np.array(center).reshape(-1, 1)
    def predict(self):
        self.kf.predict()
        return self.kf.x[:2].flatten()
    

    def update(self, center: Tuple):
        self.kf.update(np.array(center).reshape(-1, 1))
        self.centers.append(center)
        self.age += 1
        self.hit_streak += 1
    def get_speed(self, fps: float) -> float:
        if len(self.centers) < 2:
            return 0.0
        distances = []
        for i in range(1, len(self.centers)):
            dist = np.linalg.norm(np.array(self.centers[i]) - np.array(self.centers[i - 1]))
            distances.append(dist)
        return np.mean(distances) * fps

class FootballTracker:
    def __init__(self, max_age: int = 30, min_hits = 3):
        self.tracks = []
        self.next_id = 0
        self.max_age = max_age
        self.min_hits = min_hits
    def update(self, detections: List[Tuple]) -> List[Track]:
        detection_centers = []
        for det in detections:
            x1, y1,x2,y2 = det[:4]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            detection_centers.append(center)
        track_predictions = []
        for track in self.tracks:
            pred = track.predict()
            track_predictions.append(pred)
        if track_predictions and detection_centers:
            cost_matrix = self._compute_cost_matrix(track_predictions, detection_centers)
            row_indices, column_indices = linear_sum_assignment(cost_matrix)
            matched_tracks = []
            matched_detections = []
            for r, c in zip(row_indices, column_indices):
                if cost_matrix[r,c] < 100:
                    self.tracks[r].update(detection_centers[c])
                    matched_tracks.append(r)
                    matched_detections.append(c)
            
            for i, c in enumerate(detection_centers):
                if i not in matched_detections:
                    bbox = detections[i][:4]
                    self._add_track(bbox, c)
            self.tracks = [track for i, track in enumerate(self.tracks) if i in matched_tracks or track.age < self.max_age]
        
        elif detection_centers:
            for det, center in zip(detections, detection_centers):
                self._add_track(det[:4], center)
        return self.tracks

    def _compute_cost_matrix(self, predictions: List, detections: List):
        cost_matrix = np.zeros((len(predictions), len(detections)))
        for i, pred in enumerate(predictions):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = np.linalg.norm(np.array(pred) - np.array(det))
        return cost_matrix

    def _add_track(self, bbox: Tuple, center: Tuple):
        track = Track(self.next_id, bbox, center)
        self.tracks.append(track)
        self.next_id += 1
    
    def get_positions(self) -> Dict[int, List]:
        """get positions of all players"""
        positions = {}
        for track in self.tracks:
            if track.hit_streak >= self.min_hits:
                positions[track.track_id] = track.centers
        return positions