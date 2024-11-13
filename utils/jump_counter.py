import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
from PIL import Image, ImageDraw, ImageFont


class JumpCounter:
    def __init__(self):
        # YOLO 모델 초기화
        self.model = YOLO("yolov8n-pose.pt")

        # 개별 카운터 초기화
        self.normal_jump_counter = 0
        self.knee_counter = 0
        self.jumping_jack_counter = 0
        self.criss_cross_counter = 0

        # 총 카운터 추가
        self.jump_counter = 0

        # 상태 초기화
        self.knee_position = None
        self.pelvic_position = None
        self.criss_cross_position = None

        # 추적 변수 초기화
        self.left_hip_y = []
        self.prev_hip_y = None

        # thresholds 설정
        self.JUMP_THRESHOLD = 0.003
        self.KNEE_ANGLE_THRESHOLD = 160
        self.JUMPING_JACK_ANGLE_THRESHOLD = 27

        # 글꼴 초기화
        try:
            self.font = ImageFont.truetype("assets/NanumGothic.ttf", 20)
        except:
            print("폰트 파일을 찾을 수 없습니다.")
            self.font = None

        # stabilization 변수 추가
        self.stabilization_frames = 30  # Wait for 30 frames before starting to count
        self.frame_count = 0
        self.is_stable = False

        # 움직임 추적 추가
        self.movement_threshold = 0.01  # 테스트에 따라 값 조정하
        self.previous_positions = []
        self.min_movement_frames = 5  # 움직임 검출 최소 프레임

    def calculate_angle(self, point1, point2, point3):
        """세 점 사이의 각도 계산"""
        ba = point1 - point2
        bc = point3 - point2

        # zero벡터에 대 오류 처리
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        if norm_ba == 0 or norm_bc == 0:
            return 0

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        # 도메인 오류를 방지하기 위해 값을 잘라
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def is_local_minimum(self, y_positions, index, window=3):
        """국소 최소값인지 확인"""
        if len(y_positions) < 2 * window + 1:
            return False

        center = y_positions[index]
        left = y_positions[max(0, index - window) : index]
        right = y_positions[index + 1 : min(len(y_positions), index + window + 1)]

        return all(center <= x for x in left) and all(center <= x for x in right)

    def detect_jumps(self, keypoints):
        """키포인트 기반으로 다양한 유형의 점프 감지"""
        # 관련 키포인트 추출
        left_hip = keypoints[5][:2]
        right_hip = keypoints[6][:2]
        left_knee = keypoints[7][:2]
        right_knee = keypoints[8][:2]
        left_ankle = keypoints[9][:2]
        right_ankle = keypoints[10][:2]
        left_wrist = keypoints[3][:2]
        right_wrist = keypoints[4][:2]

        # average hip height 계산
        hip_y = (left_hip[1] + right_hip[1]) / 2
        self.left_hip_y.append(hip_y)

        # 이전 개수 총계
        previous_total = self.jump_counter

        # 안정화
        if not self.is_stable:
            self.frame_count += 1
            if self.frame_count >= self.stabilization_frames:
                self.is_stable = True
            return

        # 움직임 추적
        self.previous_positions.append(hip_y)
        if len(self.previous_positions) > self.min_movement_frames:
            self.previous_positions.pop(0)

        # 임계값이상의 움직임이 있는 경우에 진행
        if len(self.previous_positions) >= self.min_movement_frames:
            movement = max(self.previous_positions) - min(self.previous_positions)
            if movement < self.movement_threshold:
                return

        # 기본기 감지
        if len(self.left_hip_y) > 5:
            if self.is_local_minimum(self.left_hip_y, -3):
                self.normal_jump_counter += 1
                self.jump_counter += 1

        # 무릎 들어올리기 각도 계산
        left_knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

        # 무릎 들어올리기 감지
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

        # 옆흔들어뛰기 각도 계산
        left_arm_angle = self.calculate_angle(left_hip, left_wrist, right_wrist)

        # 옆흔들어뛰기 감지
        if left_arm_angle > self.JUMPING_JACK_ANGLE_THRESHOLD:
            if self.pelvic_position is None:
                self.pelvic_position = "wide"
        else:
            if self.pelvic_position == "wide":
                self.jumping_jack_counter += 1
                self.jump_counter += 1
                self.pelvic_position = None

        # 엇걸어풀어뛰기 감지
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
        """프레임에 총 점프 카운터 그리기"""
        # PIL 이미지로 변환
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)

        # 배경 사각형 그리기
        x = frame.shape[1] - 150
        y = 0
        w = 150
        h = 40  # 개수 인식 높이 조정
        cv2.rectangle(frame, (x, y), (x + w, y + h), (245, 117, 16), -1)

        # 총 개수만 나타기
        if self.font:
            text = f"줄넘기 개수: {self.jump_counter}"
            draw.text((x + 5, 15), text, font=self.font, fill=(255, 255, 255))

        # OpenCV format 변환
        frame[:] = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
