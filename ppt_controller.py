"""
PPT Controller — Core state machine, coordinates gesture recognition & action execution
"""

import time
from enum import Enum, auto

import cv2
import numpy as np

import config
from hand_detector import HandDetector
from gesture_classifier import Gesture, GestureClassifier
from action_controller import ActionController


class ControllerMode(Enum):
    """Controller working mode"""
    NORMAL = auto()    # Normal mode: slide nav / slideshow / volume
    MOUSE = auto()     # Mouse mode: cursor move / click


class PPTController:
    """PPT gesture control core state machine"""

    def __init__(self):
        self.hand_detector = HandDetector()
        self.gesture_classifier = GestureClassifier()
        self.action_controller = ActionController()

        # Current mode
        self.mode = ControllerMode.NORMAL

        # Mouse mode state
        self._mouse_ref_x: float | None = None
        self._mouse_ref_y: float | None = None
        self._last_mouse_time = 0.0

        # State info (for UI rendering)
        self.current_gesture: Gesture = Gesture.NONE
        self.current_finger_states: list[bool] = [False] * 5
        self.status_message: str = "Ready"
        self.fps: float = 0.0
        self._fps_times: list[float] = []

        # Volume key hold state
        self._volume_active: str | None = None  # "up" / "down" / None
        self._volume_debounce: int = 0          # debounce for press
        self._volume_tick: int = 0              # frame counter for rapid press

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one frame: detect → classify → execute → render

        Args:
            frame: BGR camera frame

        Returns:
            Output frame with UI overlay
        """
        # FPS calculation
        self._calc_fps()

        # 1. Hand detection
        hand_data = self.hand_detector.detect_hand(frame)

        if hand_data is not None:
            # Get finger states
            landmarks = hand_data["landmarks"]
            finger_states = self.hand_detector.get_finger_states(landmarks)
            hand_data["finger_states"] = finger_states
            self.current_finger_states = finger_states

            # Compute thumb-index distance (pinch detection)
            thumb_index_dist = self.hand_detector.get_thumb_index_distance(landmarks)
            hand_data["thumb_index_dist"] = thumb_index_dist

            # 2. Gesture classification
            raw_gesture = self.gesture_classifier.classify(hand_data)
            self.current_gesture = raw_gesture  # Raw result for UI display

            # ── Volume rapid-press (raw gesture, avoids stabilizer gap) ──
            VOLUME_INTERVAL = 5  # press every N frames (~6/sec at 30fps)

            if raw_gesture in (Gesture.PEACE_UP, Gesture.PEACE_DOWN):
                target = "up" if raw_gesture == Gesture.PEACE_UP else "down"
                if self._volume_active == target:
                    self._volume_tick += 1
                    if self._volume_tick >= VOLUME_INTERVAL:
                        self.action_controller.volume_up_press() if target == "up" else self.action_controller.volume_down_press()
                        self._volume_tick = 0
                else:
                    self._volume_debounce += 1
                    if self._volume_debounce >= 3:
                        self._volume_active = target
                        self._volume_debounce = 0
                        self._volume_tick = 0
                        target_name = "Up" if target == "up" else "Down"
                        self.status_message = f"Volume {target_name}"
            else:
                self._volume_debounce = 0
                self._volume_active = None

            # Gesture stabilization (must persist N frames)
            stable_gesture = self.gesture_classifier.get_stable_gesture(raw_gesture)

            # 鼠标移动是连续动作，不应等待每 N 帧一次的稳定确认。
            if (self.mode == ControllerMode.MOUSE
                    and raw_gesture == Gesture.POINT_INDEX):
                self._handle_mouse_move(landmarks)

            # 3. Execute discrete actions based on stable gestures
            if stable_gesture != Gesture.NONE:
                self._handle_gesture(stable_gesture, hand_data)

            # 4. Draw hand skeleton
            frame = self.hand_detector.draw_hand(frame, hand_data["raw"])
        else:
            # No hand detected
            self.current_gesture = Gesture.NONE
            self.current_finger_states = [False] * 5
            # Reset gesture stabilizer
            self.gesture_classifier.get_stable_gesture(Gesture.NONE)
            # Reset mouse reference
            self._mouse_ref_x = None
            self._mouse_ref_y = None
            self.action_controller.reset_mouse_smoothing()
            # Release volume key
            self._volume_active = None

        # 5. Render UI overlay
        frame = self._render_overlay(frame)
        return frame

    def _handle_gesture(self, gesture: Gesture, hand_data: dict):
        """
        Execute action based on current mode and gesture

        Args:
            gesture: Confirmed stable gesture
            hand_data: Hand detection data
        """
        landmarks = hand_data["landmarks"]

        if self.mode == ControllerMode.NORMAL:
            self._handle_normal_mode(gesture, landmarks)
        elif self.mode == ControllerMode.MOUSE:
            self._handle_mouse_mode(gesture, landmarks)

    def _handle_normal_mode(self, gesture: Gesture, landmarks: list):
        """Normal mode gesture handling"""
        if gesture == Gesture.LEFT_POINT:
            if self.gesture_classifier.should_trigger(
                    gesture, config.SWIPE_GESTURE_COOLDOWN):
                self.action_controller.prev_slide()
                self.status_message = "Prev Slide"

        elif gesture == Gesture.RIGHT_POINT:
            if self.gesture_classifier.should_trigger(
                    gesture, config.SWIPE_GESTURE_COOLDOWN):
                self.action_controller.next_slide()
                self.status_message = "Next Slide"

        elif gesture == Gesture.OPEN_PALM:
            if self.gesture_classifier.should_trigger(gesture):
                self.action_controller.exit_slideshow()
                self.status_message = "Exit Slideshow"

        elif gesture == Gesture.POINT_INDEX:
            if self.gesture_classifier.should_trigger(gesture):
                self._enter_mouse_mode(landmarks)
                self.status_message = "Enter Mouse Mode"

        elif gesture == Gesture.OK_SIGN:
            if self.gesture_classifier.should_trigger(gesture):
                self.action_controller.start_slideshow()
                self.status_message = "Start Slideshow"

        elif gesture == Gesture.THUMB_UP:
            if self.gesture_classifier.should_trigger(gesture):
                self.action_controller.start_slideshow()
                self.status_message = "Start Slideshow"

    def _handle_mouse_mode(self, gesture: Gesture, landmarks: list):
        """Mouse mode gesture handling"""
        # Exit condition: open palm
        if gesture == Gesture.OPEN_PALM:
            if self.gesture_classifier.should_trigger(gesture):
                self._exit_mouse_mode()
                self.status_message = "Exit Mouse Mode, Ready"
            return

        # Mouse click: pinch
        if gesture == Gesture.PINCH:
            if self.gesture_classifier.should_trigger(
                    gesture, config.CLICK_COOLDOWN):
                self.action_controller.mouse_click()
                self.status_message = "Mouse Click"

    def _enter_mouse_mode(self, landmarks: list):
        """Enter mouse mode"""
        self.mode = ControllerMode.MOUSE
        # Record index fingertip position as reference
        tip_x, tip_y = self.hand_detector.get_fingertip_position(
            landmarks, HandDetector.INDEX_TIP)
        self._mouse_ref_x = tip_x
        self._mouse_ref_y = tip_y
        self.action_controller.reset_mouse_smoothing()
        print("[Mode] Switched to Mouse Mode")

    def _exit_mouse_mode(self):
        """Exit mouse mode"""
        self.mode = ControllerMode.NORMAL
        self._mouse_ref_x = None
        self._mouse_ref_y = None
        self.action_controller.reset_mouse_smoothing()
        print("[Mode] Switched to Normal Mode")

    def _handle_mouse_move(self, landmarks: list):
        """
        Handle mouse movement: compute relative displacement of index fingertip,
        map to screen mouse movement
        """
        tip_x, tip_y = self.hand_detector.get_fingertip_position(
            landmarks, HandDetector.INDEX_TIP)

        if self._mouse_ref_x is None:
            self._mouse_ref_x = tip_x
            self._mouse_ref_y = tip_y
            return

        # Compute displacement
        dx = tip_x - self._mouse_ref_x
        dy = tip_y - self._mouse_ref_y

        # Move mouse
        self.action_controller.move_mouse(dx, dy)

        # Update reference (low-pass filter smoothing)
        self._mouse_ref_x += dx * 0.3
        self._mouse_ref_y += dy * 0.3

        now = time.time()
        if now - self._last_mouse_time > 0.5:
            self._last_mouse_time = now
            self.status_message = "Mouse Mode: Move index finger"

    def _calc_fps(self):
        """Calculate FPS"""
        now = time.time()
        self._fps_times.append(now)
        # Keep last 30 frames
        if len(self._fps_times) > 30:
            self._fps_times = self._fps_times[-30:]
        if len(self._fps_times) >= 2:
            duration = self._fps_times[-1] - self._fps_times[0]
            self.fps = (len(self._fps_times) - 1) / duration if duration > 0 else 0

    def _render_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Render UI info overlay on frame

        Includes:
        - Bottom status bar (mode / gesture / FPS)
        - Mouse mode crosshair
        - Finger state indicators
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()

        # ── Bottom status bar ──
        bar_y = h - config.STATUS_BAR_HEIGHT
        cv2.rectangle(overlay, (0, bar_y), (w, h),
                      config.COLOR_BG, -1)
        cv2.line(overlay, (0, bar_y), (w, bar_y),
                 config.COLOR_PRIMARY, 2)

        mode_text = ("Mouse Mode" if self.mode == ControllerMode.MOUSE
                     else "Normal Mode")
        gesture_name = self.current_gesture.name if self.current_gesture else "NONE"

        # Line 1: Mode + Gesture
        line1 = f"Mode: {mode_text}  |  Gesture: {gesture_name}"
        cv2.putText(overlay, line1, (10, bar_y + 28),
                    config.FONT_FACE, config.FONT_SCALE,
                    config.COLOR_PRIMARY, config.FONT_THICKNESS)

        # Line 2: FPS + Status
        line2 = f"FPS: {self.fps:.1f}  |  {self.status_message}"
        cv2.putText(overlay, line2, (10, bar_y + 58),
                    config.FONT_FACE, 0.55,
                    config.COLOR_TEXT, 1)

        # ── Finger state indicators ──
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

        # ── Mouse mode crosshair ──
        if self.mode == ControllerMode.MOUSE and self._mouse_ref_x is not None:
            cx, cy = int(self._mouse_ref_x), int(self._mouse_ref_y)
            # Cross lines
            cv2.line(overlay, (cx - 20, cy), (cx + 20, cy),
                     config.COLOR_SECONDARY, 2)
            cv2.line(overlay, (cx, cy - 20), (cx, cy + 20),
                     config.COLOR_SECONDARY, 2)
            # Outer circle
            cv2.circle(overlay, (cx, cy), 25,
                       config.COLOR_SECONDARY, 2)
            cv2.circle(overlay, (cx, cy), 5,
                       config.COLOR_SECONDARY, -1)

        # ── Alpha blending ──
        alpha = config.OVERLAY_ALPHA
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame

    def close(self):
        """Release all resources"""
        self.hand_detector.close()
