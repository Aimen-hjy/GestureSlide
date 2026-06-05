"""
PPT控制器 — 核心状态机, 协调手势识别与动作执行
"""

import time
from enum import Enum, auto

import numpy as np

import config
from hand_detector import HandDetector
from gesture_classifier import Gesture, GestureClassifier
from action_controller import ActionController


class ControllerMode(Enum):
    """控制器工作模式"""
    NORMAL = auto()    # 普通模式: 翻页/播放/音量
    MOUSE = auto()     # 鼠标模式: 光标移动/点击


class PPTController:
    """PPT手势控制核心状态机"""

    def __init__(self):
        self.hand_detector = HandDetector()
        self.gesture_classifier = GestureClassifier()
        self.action_controller = ActionController()

        # 当前模式
        self.mode = ControllerMode.NORMAL

        # 鼠标模式相关
        self._mouse_ref_x: float | None = None
        self._mouse_ref_y: float | None = None
        self._last_mouse_time = 0.0

        # 状态信息 (供UI渲染)
        self.current_gesture: Gesture = Gesture.NONE
        self.current_finger_states: list[bool] = [False] * 5
        self.status_message: str = "就绪"
        self.fps: float = 0.0
        self._fps_times: list[float] = []

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理一帧图像: 检测 → 识别 → 执行 → 渲染

        Args:
            frame: BGR 摄像头帧

        Returns:
            带有UI叠加层的输出帧
        """
        # FPS 计算
        self._calc_fps()

        # 1. 手部检测
        hand_data = self.hand_detector.detect_hand(frame)

        if hand_data is not None:
            # 获取手指状态
            landmarks = hand_data["landmarks"]
            finger_states = self.hand_detector.get_finger_states(landmarks)
            hand_data["finger_states"] = finger_states
            self.current_finger_states = finger_states

            # 计算拇指-食指距离 (捏合检测)
            thumb_index_dist = self.hand_detector.get_thumb_index_distance(landmarks)
            hand_data["thumb_index_dist"] = thumb_index_dist

            # 2. 手势分类
            raw_gesture = self.gesture_classifier.classify(hand_data)

            # 手势稳定 (需持续N帧)
            stable_gesture = self.gesture_classifier.get_stable_gesture(raw_gesture)
            self.current_gesture = raw_gesture  # 用于UI显示原始检测结果

            # 3. 根据模式和手势执行动作
            if stable_gesture != Gesture.NONE:
                self._handle_gesture(stable_gesture, hand_data)

            # 4. 绘制手部骨架
            frame = self.hand_detector.draw_hand(frame, hand_data["raw"])
        else:
            # 无手部检测
            self.current_gesture = Gesture.NONE
            self.current_finger_states = [False] * 5
            # 重置手势稳定器
            self.gesture_classifier.get_stable_gesture(Gesture.NONE)
            # 鼠标模式重置参考点
            self._mouse_ref_x = None
            self._mouse_ref_y = None

        # 5. 渲染UI叠加层
        frame = self._render_overlay(frame)
        return frame

    def _handle_gesture(self, gesture: Gesture, hand_data: dict):
        """
        根据当前模式和手势执行对应动作

        Args:
            gesture: 已确认的稳定手势
            hand_data: 手部检测数据
        """
        landmarks = hand_data["landmarks"]

        if self.mode == ControllerMode.NORMAL:
            self._handle_normal_mode(gesture, landmarks)
        elif self.mode == ControllerMode.MOUSE:
            self._handle_mouse_mode(gesture, landmarks)

    def _handle_normal_mode(self, gesture: Gesture, landmarks: list):
        """普通模式手势处理"""
        if gesture == Gesture.SWIPE_RIGHT:
            if self.gesture_classifier.should_trigger(
                    gesture, config.SWIPE_GESTURE_COOLDOWN):
                self.action_controller.next_slide()
                self.status_message = "下一页"

        elif gesture == Gesture.SWIPE_LEFT:
            if self.gesture_classifier.should_trigger(
                    gesture, config.SWIPE_GESTURE_COOLDOWN):
                self.action_controller.prev_slide()
                self.status_message = "上一页"

        elif gesture == Gesture.FIST:
            if self.gesture_classifier.should_trigger(gesture):
                self.action_controller.toggle_black_screen()
                self.status_message = "切换黑屏"

        elif gesture == Gesture.OK_SIGN:
            if self.gesture_classifier.should_trigger(gesture):
                self.action_controller.start_slideshow()
                self.status_message = "开始放映"

        elif gesture == Gesture.POINT_INDEX:
            if self.gesture_classifier.should_trigger(gesture):
                self._enter_mouse_mode(landmarks)
                self.status_message = "进入鼠标模式"

        elif gesture == Gesture.PEACE:
            if self.gesture_classifier.should_trigger(gesture):
                self.action_controller.volume_up()
                self.status_message = "音量+"

        elif gesture == Gesture.THREE_FINGERS:
            if self.gesture_classifier.should_trigger(gesture):
                self.action_controller.volume_down()
                self.status_message = "音量-"

        elif gesture == Gesture.THUMB_UP:
            if self.gesture_classifier.should_trigger(gesture):
                self.action_controller.start_slideshow()
                self.status_message = "开始放映"

        elif gesture == Gesture.PINKY_ONLY:
            if self.gesture_classifier.should_trigger(gesture):
                self.action_controller.exit_slideshow()
                self.status_message = "退出放映"

    def _handle_mouse_mode(self, gesture: Gesture, landmarks: list):
        """鼠标模式手势处理"""
        # 退出条件: 五指张开
        if gesture == Gesture.OPEN_PALM:
            if self.gesture_classifier.should_trigger(gesture):
                self._exit_mouse_mode()
                self.status_message = "退出鼠标模式, 就绪"
            return

        # 退出条件: 握拳
        if gesture == Gesture.FIST:
            if self.gesture_classifier.should_trigger(gesture):
                self._exit_mouse_mode()
                self.status_message = "退出鼠标模式, 就绪"
            return

        # 鼠标移动: 食指指向
        if gesture == Gesture.POINT_INDEX:
            self._handle_mouse_move(landmarks)

        # 鼠标点击: 捏合
        elif gesture == Gesture.PINCH:
            if self.gesture_classifier.should_trigger(
                    gesture, config.CLICK_COOLDOWN):
                self.action_controller.mouse_click()
                self.status_message = "鼠标点击"

    def _enter_mouse_mode(self, landmarks: list):
        """进入鼠标模式"""
        self.mode = ControllerMode.MOUSE
        # 记录食指指尖位置作为参考点
        tip_x, tip_y = self.hand_detector.get_fingertip_position(
            landmarks, HandDetector.INDEX_TIP)
        self._mouse_ref_x = tip_x
        self._mouse_ref_y = tip_y
        self.action_controller.reset_mouse_smoothing()
        print("[模式] 切换到鼠标模式")

    def _exit_mouse_mode(self):
        """退出鼠标模式"""
        self.mode = ControllerMode.NORMAL
        self._mouse_ref_x = None
        self._mouse_ref_y = None
        self.action_controller.reset_mouse_smoothing()
        print("[模式] 切换到普通模式")

    def _handle_mouse_move(self, landmarks: list):
        """
        处理鼠标移动: 计算食指指尖相对位移, 映射为鼠标移动
        """
        tip_x, tip_y = self.hand_detector.get_fingertip_position(
            landmarks, HandDetector.INDEX_TIP)

        if self._mouse_ref_x is None:
            self._mouse_ref_x = tip_x
            self._mouse_ref_y = tip_y
            return

        # 计算位移
        dx = tip_x - self._mouse_ref_x
        dy = tip_y - self._mouse_ref_y

        # 移动鼠标
        self.action_controller.move_mouse(dx, dy)

        # 更新参考点 (低通滤波平滑)
        self._mouse_ref_x += dx * 0.3
        self._mouse_ref_y += dy * 0.3

        now = time.time()
        if now - self._last_mouse_time > 0.5:
            self._last_mouse_time = now
            self.status_message = "鼠标模式: 移动食指控制光标"

    def _calc_fps(self):
        """计算FPS"""
        now = time.time()
        self._fps_times.append(now)
        # 保留最近30帧
        if len(self._fps_times) > 30:
            self._fps_times = self._fps_times[-30:]
        if len(self._fps_times) >= 2:
            duration = self._fps_times[-1] - self._fps_times[0]
            self.fps = (len(self._fps_times) - 1) / duration if duration > 0 else 0

    def _render_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        在帧上渲染UI信息叠加层

        包括:
        - 底部状态栏 (模式/手势/FPS)
        - 鼠标模式十字准星
        - 手指状态指示条
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # ── 底部状态栏 ──
        bar_y = h - config.STATUS_BAR_HEIGHT
        cv2.rectangle(overlay, (0, bar_y), (w, h),
                      config.COLOR_BG, -1)
        cv2.line(overlay, (0, bar_y), (w, bar_y),
                 config.COLOR_PRIMARY, 2)

        mode_text = ("鼠标模式" if self.mode == ControllerMode.MOUSE
                     else "普通模式")
        gesture_name = self.current_gesture.name if self.current_gesture else "NONE"

        # 第一行: 模式 + 手势
        line1 = f"Mode: {mode_text}  |  Gesture: {gesture_name}"
        cv2.putText(overlay, line1, (10, bar_y + 28),
                    config.FONT_FACE, config.FONT_SCALE,
                    config.COLOR_PRIMARY, config.FONT_THICKNESS)

        # 第二行: FPS + 状态
        line2 = f"FPS: {self.fps:.1f}  |  {self.status_message}"
        cv2.putText(overlay, line2, (10, bar_y + 58),
                    config.FONT_FACE, 0.55,
                    config.COLOR_TEXT, 1)

        # ── 手指状态指示 ──
        finger_names = ["T", "I", "M", "R", "P"]  # Thumb, Index, Middle, Ring, Pinky
        bar_x_start = w - 140
        for i, (name, is_extended) in enumerate(
                zip(finger_names, self.current_finger_states)):
            x = bar_x_start + i * 28
            color = config.COLOR_PRIMARY if is_extended else (80, 80, 80)
            cv2.circle(overlay, (x, bar_y - 15), 10, color, -1)
            cv2.putText(overlay, name, (x - 6, bar_y - 5),
                        config.FONT_FACE, 0.45,
                        config.COLOR_TEXT, 1)

        # ── 鼠标模式十字准星 ──
        if self.mode == ControllerMode.MOUSE and self._mouse_ref_x is not None:
            cx, cy = int(self._mouse_ref_x), int(self._mouse_ref_y)
            # 十字线
            cv2.line(overlay, (cx - 20, cy), (cx + 20, cy),
                     config.COLOR_SECONDARY, 2)
            cv2.line(overlay, (cx, cy - 20), (cx, cy + 20),
                     config.COLOR_SECONDARY, 2)
            # 外圈
            cv2.circle(overlay, (cx, cy), 25,
                       config.COLOR_SECONDARY, 2)
            cv2.circle(overlay, (cx, cy), 5,
                       config.COLOR_SECONDARY, -1)

        # ── 透明度混合 ──
        alpha = config.OVERLAY_ALPHA
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame

    def close(self):
        """释放所有资源"""
        self.hand_detector.close()
