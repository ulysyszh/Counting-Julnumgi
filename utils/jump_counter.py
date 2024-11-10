import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from PIL import Image, ImageDraw, ImageFont


class JumpCounter:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO("yolov8n-pose.pt")

        # Initialize individual counters
        self.normal_jump_counter = 0
        self.knee_counter = 0
        self.jumping_jack_counter = 0
        self.criss_cross_counter = 0

        # Add total counter
        self.jump_counter = 0

        # Initialize states
        self.knee_position = None
        self.pelvic_position = None
        self.criss_cross_position = None

        # Initialize tracking variables
        self.left_hip_y = []
        self.prev_hip_y = None

        # Set thresholds
        self.JUMP_THRESHOLD = 0.003
        self.KNEE_ANGLE_THRESHOLD = 160
        self.JUMPING_JACK_ANGLE_THRESHOLD = 27

        # Initialize font
        try:
            self.font = ImageFont.truetype("assets/NanumGothic.ttf", 20)
        except:
            print("폰트 파일을 찾을 수 없습니다.")
            self.font = None

        # Add stabilization variables
        self.stabilization_frames = 30  # Wait for 30 frames before starting to count
        self.frame_count = 0
        self.is_stable = False

        # Add movement tracking
        self.movement_threshold = 0.01  # Adjust this value based on testing
        self.previous_positions = []
        self.min_movement_frames = 5  # Minimum frames of significant movement needed

    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points"""
        ba = point1 - point2
        bc = point3 - point2

        # Add error handling for zero vectors
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba == 0 or norm_bc == 0:
            return 0

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        # Clip the value to avoid domain errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def is_local_minimum(self, y_positions, index, window=3):
        """Check if point is local minimum"""
        if len(y_positions) < 2 * window + 1:
            return False

        center = y_positions[index]
        left = y_positions[max(0, index - window) : index]
        right = y_positions[index + 1 : min(len(y_positions), index + window + 1)]

        return all(center <= x for x in left) and all(center <= x for x in right)

    def detect_jumps(self, keypoints):
        """Detect different types of jumps based on keypoints"""
        # Extract relevant keypoints
        left_hip = keypoints[5][:2]
        right_hip = keypoints[6][:2]
        left_knee = keypoints[7][:2]
        right_knee = keypoints[8][:2]
        left_ankle = keypoints[9][:2]
        right_ankle = keypoints[10][:2]
        left_wrist = keypoints[3][:2]
        right_wrist = keypoints[4][:2]

        # Calculate average hip height
        hip_y = (left_hip[1] + right_hip[1]) / 2
        self.left_hip_y.append(hip_y)

        # Track previous total
        previous_total = self.jump_counter

        # Stabilization period
        if not self.is_stable:
            self.frame_count += 1
            if self.frame_count >= self.stabilization_frames:
                self.is_stable = True
            return

        # Track movement
        self.previous_positions.append(hip_y)
        if len(self.previous_positions) > self.min_movement_frames:
            self.previous_positions.pop(0)

        # Only proceed if there's significant movement
        if len(self.previous_positions) >= self.min_movement_frames:
            movement = max(self.previous_positions) - min(self.previous_positions)
            if movement < self.movement_threshold:
                return

        # Detect normal jumps
        if len(self.left_hip_y) > 5:
            if self.is_local_minimum(self.left_hip_y, -3):
                self.normal_jump_counter += 1
                self.jump_counter += 1

        # Calculate angles for knee raises
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

        # Detect knee raises
        if (
            left_knee_angle < self.KNEE_ANGLE_THRESHOLD
            or right_knee_angle < self.KNEE_ANGLE_THRESHOLD
        ):
            if self.knee_position is None:
                self.knee_position = "up"
        else:
            if self.knee_position == "up":
                self.knee_counter += 1
                self.jump_counter += 1
                self.knee_position = None

        # Calculate angles for jumping jacks
        left_arm_angle = self.calculate_angle(left_hip, left_wrist, right_wrist)

        # Detect jumping jacks
        if left_arm_angle > self.JUMPING_JACK_ANGLE_THRESHOLD:
            if self.pelvic_position is None:
                self.pelvic_position = "wide"
        else:
            if self.pelvic_position == "wide":
                self.jumping_jack_counter += 1
                self.jump_counter += 1
                self.pelvic_position = None

        # Detect criss cross
        wrist_distance = np.linalg.norm(left_wrist - right_wrist)
        if wrist_distance < 50:
            if self.criss_cross_position is None:
                self.criss_cross_position = "crossed"
        else:
            if self.criss_cross_position == "crossed":
                self.criss_cross_counter += 1
                self.jump_counter += 1
                self.criss_cross_position = None

    def draw_counters(self, frame):
        """Draw total jump counter on frame with Korean text"""
        # Convert frame to PIL Image for Korean text
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # Draw background rectangle
        x = frame.shape[1] - 150
        y = 0
        w = 150
        h = 40  # Reduced height for single counter
        cv2.rectangle(frame, (x, y), (x + w, y + h), (245, 117, 16), -1)

        # Draw only the total counter
        if self.font:
            text = f"줄넘기 개수: {self.jump_counter}"
            draw.text((x + 5, 15), text, font=self.font, fill=(255, 255, 255))

        # Convert back to OpenCV format
        frame[:] = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
