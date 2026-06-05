"""
手势分类识别器 — 根据手部关键点和手指状态识别具体手势
"""

import time
import os
import logging
from enum import Enum, auto

import numpy as np

import config
from gesture_model import extract_features, GESTURE_TO_ID, ID_TO_GESTURE


class Gesture(Enum):
    """手势类型枚举"""
    NONE = auto()            # 无手势
    OPEN_PALM = auto()       # 五指张开
    FIST = auto()            # 握拳
    LEFT_POINT = auto()      # 食指左指
    RIGHT_POINT = auto()     # 食指右指
    DOWN_POINT = auto()      # 食指下指
    OK_SIGN = auto()         # OK手势 (拇指+食指圈, 其余伸展)
    POINT_INDEX = auto()     # 仅食指伸展 (朝上/向前, 进入鼠标模式)
    PINCH = auto()           # 拇指食指捏合
    PEACE_UP = auto()        # V手势朝上
    PEACE_DOWN = auto()      # V手势朝下
    THREE_FINGERS = auto()   # 食指+中指+无名指伸展
    THUMB_UP = auto()        # 仅拇指竖起
    TWO_FINGERS_CLOSE = auto()  # 食指中指并拢


class GestureClassifier:
    """手势识别分类器"""

    def __init__(self):
        # 手势防抖: 记录每种手势的上次触发时间
        self._last_trigger_time: dict[Gesture, float] = {}

        # 手势持续帧计数: 同一手势需持续N帧才确认
        self._gesture_frame_count: dict[Gesture, int] = {}
        self._last_raw_gesture: Gesture = Gesture.NONE

        # ML模型
        self._ml_model = None
        self._ml_scaler = None
        self._ml_enabled = False
        self._load_ml_model()

    def classify(self, hand_data: dict | None) -> Gesture:
        """
        根据手部数据识别当前手势

        Args:
            hand_data: HandDetector.detect_hand() 返回的数据, 包含:
                - landmarks: 21个关键点
                - handedness: "Left" / "Right"
                - finger_states: [thumb, index, middle, ring, pinky] 布尔列表

        Returns:
            识别出的 Gesture 枚举值
        """
        if hand_data is None:
            self._reset_swipe()
            return Gesture.NONE

        landmarks = hand_data["landmarks"]
        finger_states = hand_data.get("finger_states", [False] * 5)

        if len(finger_states) < 5:
            return Gesture.NONE

        thumb, index, middle, ring, pinky = finger_states

        # 1. 检测捏合 (拇指+食指靠近)
        pinch_distance = hand_data.get("thumb_index_dist", 1.0)
        if pinch_distance < config.PINCH_DISTANCE_THRESHOLD:
            # 拇指食捏合 + 其他手指弯曲 = PINCH (鼠标点击)
            if not middle and not ring and not pinky:
                return Gesture.PINCH
            # 拇指食捏合 + 其他手指伸展 = OK_SIGN
            if middle and ring and pinky:
                return Gesture.OK_SIGN

        # 2. 按手指组合模式识别静态手势 (ML或规则)
        if self._ml_enabled:
            gesture = self._classify_ml(hand_data)
        else:
            gesture = self._classify_static(thumb, index, middle, ring, pinky)

        return gesture

    def _classify_static(self, thumb: bool, index: bool,
                         middle: bool, ring: bool, pinky: bool) -> Gesture:
        """根据手指状态组合识别静态手势"""
        # 五指全握 → FIST
        if not any([thumb, index, middle, ring, pinky]):
            return Gesture.FIST

        # 五指全开 → OPEN_PALM
        if all([thumb, index, middle, ring, pinky]):
            return Gesture.OPEN_PALM

        # 仅食指伸展 → POINT_INDEX
        if index and not middle and not ring and not pinky:
            return Gesture.POINT_INDEX

        # 仅拇指竖起 → THUMB_UP
        if thumb and not index and not middle and not ring and not pinky:
            return Gesture.THUMB_UP

        # 仅小指伸展 → PINKY_ONLY
        if pinky and not index and not middle and not ring:
            return Gesture.PINKY_ONLY

        # 食指+中指伸展 (V手势) → PEACE
        if index and middle and not ring and not pinky:
            return Gesture.PEACE

        # 食指+中指+无名指伸展 → THREE_FINGERS
        if index and middle and ring and not pinky:
            return Gesture.THREE_FINGERS

        return Gesture.NONE

    def _load_ml_model(self):
        """加载ML模型, 若不存在则回退到规则分类"""
        import joblib

        model_path = config.ML_MODEL_PATH
        scaler_path = config.ML_SCALER_PATH

        if not os.path.exists(model_path):
            logging.info(f"ML model not found at {model_path}, "
                         f"using rule-based fallback")
            return

        try:
            self._ml_model = joblib.load(model_path)
            if os.path.exists(scaler_path):
                self._ml_scaler = joblib.load(scaler_path)
            self._ml_enabled = True
            logging.info(f"ML model loaded from {model_path}")
        except Exception as e:
            logging.warning(f"Failed to load ML model: {e}, "
                            f"using rule-based fallback")
            self._ml_model = None
            self._ml_enabled = False

    def _classify_ml(self, hand_data: dict) -> Gesture:
        """使用ML模型分类静态手势. 置信度不足时返回NONE"""
        # 提取69维特征
        features = extract_features(hand_data)
        X = features.reshape(1, -1)

        # 标准化 (仅坐标部分 [0:63])
        if self._ml_scaler is not None:
            X[:, :63] = self._ml_scaler.transform(X[:, :63])

        # 预测
        probas = self._ml_model.predict_proba(X)[0]
        max_idx = int(np.argmax(probas))
        max_prob = float(probas[max_idx])

        if max_prob < config.ML_CONFIDENCE_THRESHOLD:
            return Gesture.NONE

        # 将ML输出的字符串标签映射回 Gesture 枚举
        gesture_name = ID_TO_GESTURE.get(max_idx, "NONE")
        try:
            return Gesture[gesture_name]
        except KeyError:
            return Gesture.NONE

    def should_trigger(self, gesture: Gesture,
                       cooldown: float | None = None) -> bool:
        """
        检查手势是否可以触发 (防抖)

        Args:
            gesture: 待触发的手势
            cooldown: 自定义冷却时间，默认使用全局配置

        Returns:
            是否可以触发
        """
        if gesture == Gesture.NONE:
            return False

        if cooldown is None:
            cooldown = config.GESTURE_COOLDOWN

        now = time.time()
        last_time = self._last_trigger_time.get(gesture, 0)

        if now - last_time >= cooldown:
            self._last_trigger_time[gesture] = now
            return True

        return False

    def get_stable_gesture(self, raw_gesture: Gesture,
                           hold_frames: int = None) -> Gesture:
        """
        手势稳定器: 同一手势需持续N帧才返回确认

        Args:
            raw_gesture: 当前帧识别出的原始手势
            hold_frames: 需要持续的帧数

        Returns:
            稳定确认后的手势, 或 NONE
        """
        if hold_frames is None:
            hold_frames = config.GESTURE_HOLD_FRAMES

        if raw_gesture == self._last_raw_gesture:
            self._gesture_frame_count[raw_gesture] = (
                self._gesture_frame_count.get(raw_gesture, 0) + 1
            )
        else:
            self._gesture_frame_count.clear()
            self._gesture_frame_count[raw_gesture] = 1
            self._last_raw_gesture = raw_gesture

        count = self._gesture_frame_count.get(raw_gesture, 0)
        if count >= hold_frames:
            # 重置计数，避免持续触发
            self._gesture_frame_count[raw_gesture] = 0
            return raw_gesture

        return Gesture.NONE
