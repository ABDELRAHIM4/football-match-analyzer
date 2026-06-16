# changes
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
import cv2
class FootballMetrics:

    def __init__(self, homography_transformer, fps: float):
        self.homography = homography_transformer
        self.fps = fps
    
    def calculate_player_speed(self, positions: List[Tuple]) -> Dict:
        """ Speed of player based on positions"""
        if len(positions) < 2:
            return {'max_speed': 0,'avg_speed': 0, 'instant_speeds': [], 'total_distance': 0}
        #Transform all positions to real 
        real_pos = []
        for pos in positions:
            real = self.homography.pixel_to_real([pos])[0]
            real_pos.append(real)
        # Calculate distance between positions
        distances = []
        for i in range(1, len(real_pos)):
            dist = np.linalg.norm(real_pos[i] - real_pos[i - 1])
            distances.append(dist)
        
        time_per_frame = 1.0 / self.fps
        speed = [dist / time_per_frame for dist in distances]


        return {
            'max_speed': max(speed) if speed else 0,'avg_speed': np.mean(speed) if speed else 0, 'instant_speeds': speed, 'total_distance': sum(distances)}
    def create_heatmap(self, positions:List[Tuple], grid_size: Tuple = (50, 34)) -> np.ndarray:


        heatmap = np.zeros(grid_size)
        for pos in positions:
            # pos expected in real-world meters (x: 0..105, y: 0..68)
            x = int(pos[0] / (105 / grid_size[0]))
            y = int(pos[1] / (68 / grid_size[1]))
            if 0 <= x < grid_size[0] and 0 <= y < grid_size[1]:
                heatmap[x, y] += 1

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap

    def generate_report(self, player_metrics: Dict) -> Tuple[pd.DataFrame, Dict]:
        """generate Excel report"""
        data = []
        for player_id, metrics in player_metrics.items():
            data.append({
                    'Player_id': player_id,
                    'Max Speed': round(metrics['max_speed'], 2),
                    'Avg Speed': round(metrics['avg_speed'], 2),
                    'Totale Distance': round(metrics['total_distance'], 2)
            })
        df = pd.DataFrame(data)
        return df
