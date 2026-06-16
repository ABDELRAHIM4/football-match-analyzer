import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict

class HomographyTransformer:


    def __init__(self, frame_shape: Tuple, reference_points: dict):
        self.frame_shape = frame_shape
        self.pixel_points = np.array(reference_points['pixel_points'], dtype=np.float32)
        self.real_points = np.array(reference_points['real_points'], dtype=np.float32)

        self.H,_ = cv2.findHomography(self.pixel_points, self.real_points)

    def pixel_to_real(self, pixel_points: List[Tuple]) -> np.ndarray:
        if not isinstance(pixel_points, np.ndarray):
            pixel_points = np.array(pixel_points, dtype= np.float32)
        if pixel_points.ndim == 1:
            pixel_points = pixel_points.reshape(1. -1)
        real_points = cv2.perspectiveTransform(pixel_points.reshape(-1, 1, 2), self.H)
        return real_points.reshape(-1, 2)

    def real_to_pixel(self, real_points: List[Tuple]) -> np.ndarray:
        real_points = np.array(real_points, dtype=np.float32)
        # Expected shape: (N, 2)
        if real_points.ndim == 1:
            real_points = real_points.reshape(1, -1)
        if real_points.ndim != 2 or real_points.shape[1] != 2:
            real_points = real_points.reshape(-1, 2)

        H_inv = np.linalg.inv(self.H)
        pixel_points = cv2.perspectiveTransform(real_points.reshape(-1, 1, 2), H_inv)
        return pixel_points.reshape(-1, 2)

    def draw_pitch_overlay(self, frame: np.ndarray) -> np.ndarray:

        pitch_lines_meter = [
            #Touchlines
            [(0, 0), (105, 0)], #Top touchline
            [(0, 68), (105, 68)], # Bottom touchline
            #Goal lines
            [(0, 0), (0, 68)], # left goal line
            [(105, 0), (105, 68)], # Right goal line
            # Halfwayline
            [(52.5, 0), (52.5, 68)], # Center line
            # Center cicle
            [(52.5, 34), (52.5, 34)], # Center spot
            # Penalty areas
            [(16.5, 0), (16.5, 16.5)],
            [(0, 16.5), (16.5, 16.5)],
            [(88.5, 0), (88.5, 16.5)],
            [(105, 16.5), (88.5, 16.5)],
        ]
        
        overlay = frame.copy()
        # Convert each linefrom meter to pixel and draw
        for line in pitch_lines_meter:
            start_pixel = self.real_to_pixel(line[0])[0]
            end_pixel = self.real_to_pixel(line[1])[0]
            cv2.line(
                overlay,
                tuple(start_pixel.astype(int)),
                tuple(end_pixel.astype(int)),
                (0, 255, 255),
                2,
            )  # Yellow lines

        
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        

    def calculate_real_distance(self, pix1: Tuple, pix2: Tuple) -> float:
        """calculate real world distances between 2 points"""
        real1 = self.pixel_to_real([pix1])[0]
        real2 = self.pixel_to_real([pix2])[0]

        distance = np.linalg.norm(real1 - real2)
        return distance