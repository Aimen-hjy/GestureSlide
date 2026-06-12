"""
手部检测模块 — 封装 MediaPipe Hands 进行手部关键点检测与手指状态判断
"""

import cv2
import numpy as np
import mediapipe as mp

import config


class HandDetector:
    """MediaPipe 手部检测器封装"""

    # MediaPipe 手部关键点索引
    # 参考: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    # 手指定义: (名称, TIP索引, PIP索引, MCP索引)
    FINGERS = [
        ("thumb",  THUMB_TIP,  THUMB_IP,   THUMB_MCP),
        ("index",  INDEX_TIP,  INDEX_PIP,  INDEX_MCP),
        ("middle", MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        ("ring",   RING_TIP,   RING_PIP,   RING_MCP),
        ("pinky",  PINKY_TIP,  PINKY_PIP,  PINKY_MCP),
    ]

    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=config.STATIC_IMAGE_MODE,
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def detect_hand(self, frame: np.ndarray) -> dict | None:
        """
        检测帧中的手部关键点

        Args:
            frame: BGR 图像帧

        Returns:
            手部关键点列表 (每个关键点为 {x, y, z} 归一化坐标)，
            未检测到手时返回 None
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if results.multi_hand_landmarks:
            # 只取第一只手
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
            return {"landmarks": landmarks, "handedness": handedness,
                    "raw": hand_landmarks}
        return None

    def get_finger_states(self, landmarks: list) -> list[bool]:
        """
        判断每根手指的伸展状态

        算法: 比较指尖到手腕距离 vs PIP关节到手腕距离
        指尖更远 → 伸展, 指尖更近 → 弯曲

        Args:
            landmarks: 21个手部关键点列表

        Returns:
            [拇指, 食指, 中指, 无名指, 小指] 的伸展状态
        """
        wrist = np.array([landmarks[self.WRIST]["x"],
                          landmarks[self.WRIST]["y"],
                          landmarks[self.WRIST]["z"]])

        finger_states = []
        for name, tip_idx, pip_idx, mcp_idx in self.FINGERS:
            tip = np.array([landmarks[tip_idx]["x"],
                            landmarks[tip_idx]["y"],
                            landmarks[tip_idx]["z"]])
            pip = np.array([landmarks[pip_idx]["x"],
                            landmarks[pip_idx]["y"],
                            landmarks[pip_idx]["z"]])

            tip_dist = np.linalg.norm(tip - wrist)
            pip_dist = np.linalg.norm(pip - wrist)

            if name == "thumb":
                # 拇指: 比较x坐标差异 (拇指横向运动)
                mcp = np.array([landmarks[self.THUMB_MCP]["x"],
                                landmarks[self.THUMB_MCP]["y"],
                                landmarks[self.THUMB_MCP]["z"]])
                ip = np.array([landmarks[self.THUMB_IP]["x"],
                               landmarks[self.THUMB_IP]["y"],
                               landmarks[self.THUMB_IP]["z"]])
                thumb_extended = abs(tip[0] - ip[0]) > 0.04
                finger_states.append(thumb_extended)
            else:
                # 其他手指: 比较距离
                finger_states.append(tip_dist > pip_dist * 0.92)

        return finger_states

    def get_fingertip_position(self, landmarks: list, finger_tip_idx: int) -> tuple:
        """获取指尖像素坐标"""
        h, w = config.FRAME_HEIGHT, config.FRAME_WIDTH
        lm = landmarks[finger_tip_idx]
        return (int(lm["x"] * w), int(lm["y"] * h))

    def get_hand_center(self, landmarks: list) -> tuple:
        """获取手掌中心像素坐标 (手腕与中指MCP的中点)"""
        h, w = config.FRAME_HEIGHT, config.FRAME_WIDTH
        wrist = landmarks[self.WRIST]
        mid_mcp = landmarks[self.MIDDLE_MCP]
        cx = (wrist["x"] + mid_mcp["x"]) / 2 * w
        cy = (wrist["y"] + mid_mcp["y"]) / 2 * h
        return (int(cx), int(cy))

    def get_thumb_index_distance(self, landmarks: list) -> float:
        """计算拇指尖与食指尖之间的归一化距离 (用于捏合检测)"""
        thumb_tip = np.array([landmarks[self.THUMB_TIP]["x"],
                              landmarks[self.THUMB_TIP]["y"]])
        index_tip = np.array([landmarks[self.INDEX_TIP]["x"],
                              landmarks[self.INDEX_TIP]["y"]])
        return float(np.linalg.norm(thumb_tip - index_tip))

    def draw_hand(self, frame: np.ndarray, raw_landmarks) -> np.ndarray:
        """在帧上绘制手部骨架和关键点"""
        self.mp_draw.draw_landmarks(
            frame,
            raw_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )
        return frame

    def close(self):
        """释放 MediaPipe 资源"""
        self.hands.close()
