"""
PPT Controller — Core state machine, coordinates gesture recognition & action execution.
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
    """Controller working mode."""
    NORMAL = auto()    # Normal mode: slide nav / slideshow / volume
    MOUSE = auto()     # Mouse mode: cursor move / click


class PPTController:
    """PPT gesture control core state machine."""

    def __init__(self):
        self.hand_detector = HandDetector()
        self.gesture_classifier = GestureClassifier()
        self.action_controller = ActionController()

        # Current mode
        self.mode = ControllerMode.NORMAL
        self.slideshow_started = False

        # Command safety gate is kept as an optional switch, but disabled by default.
        self.command_gate_enabled = bool(getattr(config, "COMMAND_GATE_ENABLED", False))
        self._command_active_until = 0.0

        # Variable-hold stabilizer. Low-risk gestures stay responsive; high-risk
        # gestures need more frames before execution.
        self._stable_candidate: Gesture = Gesture.NONE
        self._stable_candidate_frames = 0

        # Mouse mode state
        self._mouse_ref_x: float | None = None
        self._mouse_ref_y: float | None = None
        self._last_mouse_time = 0.0

        # State info (for UI/HUD rendering)
        self.current_gesture: Gesture = Gesture.NONE
        self.current_finger_states: list[bool] = [False] * 5
        self.current_confidence: float = 0.0
        self.current_backend: str = "rule"
        self.status_message: str = "Ready"
        self.fps: float = 0.0
        self._fps_times: list[float] = []

        # Volume key hold state
        self._volume_active: str | None = None  # "up" / "down" / None
        self._volume_debounce: int = 0          # debounce for press
        self._volume_tick: int = 0              # frame counter for rapid press

    # ==================== Command gate ====================

    def _is_unlock_gesture(self, gesture: Gesture) -> bool:
        return gesture in (Gesture.OK_SIGN, Gesture.THUMB_UP)

    def _is_command_gate_open(self) -> bool:
        if not self.command_gate_enabled:
            return True
        if self.mode == ControllerMode.MOUSE:
            return True
        return time.time() <= self._command_active_until

    def _gate_remaining(self) -> float:
        if not self.command_gate_enabled:
            return 0.0
        return max(0.0, self._command_active_until - time.time())

    def _activate_command_gate(self):
        if not self.command_gate_enabled:
            return
        timeout = float(getattr(config, "COMMAND_GATE_TIMEOUT", 5.0))
        self._command_active_until = time.time() + timeout

    def _extend_command_gate_after_action(self):
        if getattr(config, "COMMAND_GATE_EXTEND_ON_ACTION", True):
            self._activate_command_gate()

    def _should_execute_stable_gesture(self, gesture: Gesture) -> bool:
        if not self.command_gate_enabled:
            return True
        if self.mode == ControllerMode.MOUSE:
            return True
        if self._is_unlock_gesture(gesture):
            self._activate_command_gate()
            return True
        if self._is_command_gate_open():
            return True
        self.status_message = "Locked: show OK/Thumb Up before command"
        return False

    def gate_status_text(self) -> str:
        if not self.command_gate_enabled:
            return "Gate: OFF"
        if self.mode == ControllerMode.MOUSE:
            return "Gate: MOUSE"
        remaining = self._gate_remaining()
        if remaining > 0:
            return f"Gate: ACTIVE {remaining:.1f}s"
        return "Gate: LOCKED"

    # ==================== Practical recognition helpers ====================

    def _is_high_risk_gesture(self, gesture: Gesture) -> bool:
        return gesture in {
            Gesture.OPEN_PALM,      # exit slideshow
            Gesture.OK_SIGN,        # start slideshow
            Gesture.THUMB_UP,       # start slideshow
            Gesture.POINT_INDEX,    # enter mouse mode
            Gesture.PINCH,          # mouse click
        }

    def _required_hold_frames(self, gesture: Gesture) -> int:
        if self._is_high_risk_gesture(gesture):
            return int(getattr(config, "HIGH_RISK_GESTURE_HOLD_FRAMES", 10))
        return int(getattr(config, "GESTURE_HOLD_FRAMES", 5))

    def _stable_gesture_with_variable_hold(self, raw_gesture: Gesture) -> Gesture:
        """Confirm a raw gesture after a gesture-specific hold-frame count."""
        if raw_gesture == Gesture.NONE:
            self._stable_candidate = Gesture.NONE
            self._stable_candidate_frames = 0
            return Gesture.NONE

        if raw_gesture == self._stable_candidate:
            self._stable_candidate_frames += 1
        else:
            self._stable_candidate = raw_gesture
            self._stable_candidate_frames = 1

        if self._stable_candidate_frames >= self._required_hold_frames(raw_gesture):
            return raw_gesture
        return Gesture.NONE

    def _geometry_direction_override(self, raw_gesture: Gesture, hand_data: dict) -> Gesture:
        """Use index-finger geometry only to refine pointing candidates.

        The offline test showed the pure geometry rule recognizes most
        LEFT/RIGHT samples, but also turns local NONE-like boundary samples into
        LEFT_POINT. Therefore geometry must not create a page command from an
        unrelated model result. It only refines gestures already judged to be an
        index-pointing family by the classifier.
        """
        if not getattr(config, "GEOMETRY_DIRECTION_OVERRIDE", True):
            return raw_gesture

        pointing_candidates = {
            Gesture.LEFT_POINT,
            Gesture.RIGHT_POINT,
            Gesture.DOWN_POINT,
            Gesture.POINT_INDEX,
        }
        if raw_gesture not in pointing_candidates:
            return raw_gesture

        finger_states = hand_data.get("finger_states") or []
        if len(finger_states) < 5:
            return raw_gesture

        thumb, index, middle, ring, pinky = finger_states[:5]
        if not index or middle or ring or pinky:
            return raw_gesture

        landmarks = hand_data.get("landmarks") or []
        try:
            mcp = landmarks[HandDetector.INDEX_MCP]
            pip = landmarks[HandDetector.INDEX_PIP]
            tip = landmarks[HandDetector.INDEX_TIP]
        except Exception:
            return raw_gesture

        dx = float(tip["x"] - pip["x"])
        dy = float(tip["y"] - pip["y"])
        total_dx = float(tip["x"] - mcp["x"])
        total_dy = float(tip["y"] - mcp["y"])
        length = (total_dx ** 2 + total_dy ** 2) ** 0.5

        min_len = float(getattr(config, "GEOMETRY_DIRECTION_MIN_LENGTH", 0.08))
        ratio = float(getattr(config, "GEOMETRY_DIRECTION_RATIO", 1.30))
        if length < min_len:
            return raw_gesture

        if abs(dx) > abs(dy) * ratio:
            self.current_confidence = max(self.current_confidence, 0.95)
            self.current_backend = "ml+geom" if self.current_backend == "ml" else "geom"
            return Gesture.LEFT_POINT if dx < 0 else Gesture.RIGHT_POINT

        return raw_gesture

    # ==================== Main frame loop ====================

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one frame: detect → classify → execute → render."""
        self._calc_fps()
        hand_data = self.hand_detector.detect_hand(frame)

        if hand_data is not None:
            landmarks = hand_data["landmarks"]
            finger_states = self.hand_detector.get_finger_states(landmarks)
            hand_data["finger_states"] = finger_states
            self.current_finger_states = finger_states

            thumb_index_dist = self.hand_detector.get_thumb_index_distance(landmarks)
            hand_data["thumb_index_dist"] = thumb_index_dist

            raw_gesture = self.gesture_classifier.classify(hand_data)
            self.current_confidence = self.gesture_classifier.last_confidence
            self.current_backend = self.gesture_classifier.last_backend
            raw_gesture = self._geometry_direction_override(raw_gesture, hand_data)
            self.current_gesture = raw_gesture

            # Volume rapid-press remains a direct raw gesture path. It is not a
            # core PPT command and can be ignored during normal demo if unused.
            VOLUME_INTERVAL = 5  # press every N frames (~6/sec at 30fps)
            if (self._is_command_gate_open()
                    and raw_gesture in (Gesture.PEACE_UP, Gesture.PEACE_DOWN)):
                target = "up" if raw_gesture == Gesture.PEACE_UP else "down"
                if self._volume_active == target:
                    self._volume_tick += 1
                    if self._volume_tick >= VOLUME_INTERVAL:
                        if target == "up":
                            self.action_controller.volume_up_press()
                        else:
                            self.action_controller.volume_down_press()
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

            stable_gesture = self._stable_gesture_with_variable_hold(raw_gesture)

            # Mouse movement is continuous and should not wait for stable trigger.
            if (self.mode == ControllerMode.MOUSE
                    and raw_gesture == Gesture.POINT_INDEX):
                self._handle_mouse_move(landmarks)

            if stable_gesture != Gesture.NONE and self._should_execute_stable_gesture(stable_gesture):
                self._handle_gesture(stable_gesture, hand_data)

            frame = self.hand_detector.draw_hand(frame, hand_data["raw"])
        else:
            self.current_gesture = Gesture.NONE
            self.current_confidence = 0.0
            self.current_backend = "none"
            self.current_finger_states = [False] * 5
            self._stable_candidate = Gesture.NONE
            self._stable_candidate_frames = 0
            self._mouse_ref_x = None
            self._mouse_ref_y = None
            self.action_controller.reset_mouse_smoothing()
            self._volume_active = None

        return self._render_overlay(frame)

    def _handle_gesture(self, gesture: Gesture, hand_data: dict):
        """Execute action based on current mode and gesture."""
        landmarks = hand_data["landmarks"]
        if self.mode == ControllerMode.NORMAL:
            self._handle_normal_mode(gesture, landmarks)
        elif self.mode == ControllerMode.MOUSE:
            self._handle_mouse_mode(gesture, landmarks)

    def _handle_normal_mode(self, gesture: Gesture, landmarks: list):
        """Normal mode gesture handling."""
        if gesture == Gesture.LEFT_POINT:
            if self.gesture_classifier.should_trigger(gesture, config.SWIPE_GESTURE_COOLDOWN):
                self.action_controller.prev_slide()
                self.status_message = "Prev Slide"
                self._extend_command_gate_after_action()

        elif gesture == Gesture.RIGHT_POINT:
            if self.gesture_classifier.should_trigger(gesture, config.SWIPE_GESTURE_COOLDOWN):
                self.action_controller.next_slide()
                self.status_message = "Next Slide"
                self._extend_command_gate_after_action()

        elif gesture == Gesture.OPEN_PALM:
            if self.gesture_classifier.should_trigger(gesture):
                self.action_controller.exit_slideshow()
                self.slideshow_started = False
                self.status_message = "Exit Slideshow"
                self._command_active_until = 0.0

        elif gesture == Gesture.POINT_INDEX:
            if self.gesture_classifier.should_trigger(gesture):
                self._enter_mouse_mode(landmarks)
                self.status_message = "Enter Mouse Mode"
                self._extend_command_gate_after_action()

        elif gesture in (Gesture.OK_SIGN, Gesture.THUMB_UP):
            if self.gesture_classifier.should_trigger(gesture):
                self.action_controller.start_slideshow()
                self.slideshow_started = True
                self.status_message = "Start Slideshow"
                self._extend_command_gate_after_action()

    def _handle_mouse_mode(self, gesture: Gesture, landmarks: list):
        """Mouse mode gesture handling."""
        if gesture == Gesture.OPEN_PALM:
            if self.gesture_classifier.should_trigger(gesture):
                self._exit_mouse_mode()
                self.status_message = "Exit Mouse Mode, Ready"
                self._extend_command_gate_after_action()
            return

        if gesture == Gesture.PINCH:
            if self.gesture_classifier.should_trigger(gesture, config.CLICK_COOLDOWN):
                self.action_controller.mouse_click()
                self.status_message = "Mouse Click"

    def _enter_mouse_mode(self, landmarks: list):
        """Enter mouse mode."""
        self.mode = ControllerMode.MOUSE
        tip_x, tip_y = self.hand_detector.get_fingertip_position(
            landmarks, HandDetector.INDEX_TIP)
        self._mouse_ref_x = tip_x
        self._mouse_ref_y = tip_y
        self.action_controller.reset_mouse_smoothing()
        print("[Mode] Switched to Mouse Mode")

    def _exit_mouse_mode(self):
        """Exit mouse mode."""
        self.mode = ControllerMode.NORMAL
        self._mouse_ref_x = None
        self._mouse_ref_y = None
        self.action_controller.reset_mouse_smoothing()
        print("[Mode] Switched to Normal Mode")

    def _handle_mouse_move(self, landmarks: list):
        """Handle mouse movement by relative index fingertip displacement."""
        tip_x, tip_y = self.hand_detector.get_fingertip_position(
            landmarks, HandDetector.INDEX_TIP)

        if self._mouse_ref_x is None:
            self._mouse_ref_x = tip_x
            self._mouse_ref_y = tip_y
            return

        dx = tip_x - self._mouse_ref_x
        dy = tip_y - self._mouse_ref_y
        self.action_controller.move_mouse(dx, dy)

        self._mouse_ref_x += dx * 0.3
        self._mouse_ref_y += dy * 0.3

        now = time.time()
        if now - self._last_mouse_time > 0.5:
            self._last_mouse_time = now
            self.status_message = "Mouse Mode: Move index finger"

    def _calc_fps(self):
        """Calculate FPS."""
        now = time.time()
        self._fps_times.append(now)
        if len(self._fps_times) > 30:
            self._fps_times = self._fps_times[-30:]
        if len(self._fps_times) >= 2:
            duration = self._fps_times[-1] - self._fps_times[0]
            self.fps = (len(self._fps_times) - 1) / duration if duration > 0 else 0

    # ==================== Overlay rendering ====================

    def _action_hint(self) -> str:
        """Human-readable next action for the demo overlay/HUD."""
        if (self.command_gate_enabled
                and self.mode == ControllerMode.NORMAL
                and not self._is_command_gate_open()
                and self.current_gesture not in (Gesture.OK_SIGN, Gesture.THUMB_UP)):
            return "Locked: OK/Thumb Up activates commands"

        if self.mode == ControllerMode.MOUSE:
            mapping = {
                Gesture.POINT_INDEX: "Move cursor",
                Gesture.PINCH: "Click",
                Gesture.OPEN_PALM: "Exit mouse mode",
            }
            return mapping.get(self.current_gesture, "Mouse mode: index move, pinch click")

        mapping = {
            Gesture.LEFT_POINT: "Previous slide",
            Gesture.RIGHT_POINT: "Next slide",
            Gesture.OPEN_PALM: "Exit slideshow",
            Gesture.POINT_INDEX: "Enter mouse mode",
            Gesture.OK_SIGN: "Start slideshow",
            Gesture.THUMB_UP: "Start slideshow",
            Gesture.PEACE_UP: "Volume up",
            Gesture.PEACE_DOWN: "Volume down",
            Gesture.FIST: "Recognized, no action mapped",
            Gesture.DOWN_POINT: "Recognized, no action mapped",
            Gesture.THREE_FINGERS: "Recognized, no action mapped",
        }
        return mapping.get(self.current_gesture, "Hold a gesture steady to trigger")

    def _render_demo_panel(self, overlay: np.ndarray):
        if not getattr(config, "SHOW_DEMO_PANEL", True):
            return
        x, y, w, h = 10, 10, 390, 102
        cv2.rectangle(overlay, (x, y), (x + w, y + h), config.COLOR_BG, -1)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), config.COLOR_PRIMARY, 1)
        cv2.putText(overlay, "GestureSlide", (x + 12, y + 25),
                    config.FONT_FACE, 0.7, config.COLOR_PRIMARY, 2)
        cv2.putText(overlay, "MediaPipe Hands + Lightweight Classifier", (x + 12, y + 48),
                    config.FONT_FACE, 0.46, config.COLOR_TEXT, 1)
        cv2.putText(overlay, self.gate_status_text(), (x + 12, y + 70),
                    config.FONT_FACE, 0.48,
                    config.COLOR_PRIMARY if self._is_command_gate_open() else config.COLOR_WARNING, 1)
        cv2.putText(overlay, f"Action: {self._action_hint()}", (x + 12, y + 92),
                    config.FONT_FACE, 0.46, config.COLOR_SECONDARY, 1)

    def _render_confidence_bar(self, overlay: np.ndarray, x: int, y: int, width: int):
        conf = max(0.0, min(1.0, self.current_confidence))
        bar_h = 10
        cv2.rectangle(overlay, (x, y), (x + width, y + bar_h), config.COLOR_DIM, 1)
        fill_w = int(width * conf)
        color = config.COLOR_PRIMARY if conf >= config.ML_CONFIDENCE_THRESHOLD else config.COLOR_WARNING
        if fill_w > 0:
            cv2.rectangle(overlay, (x + 1, y + 1), (x + fill_w, y + bar_h - 1), color, -1)
        cv2.putText(overlay, f"conf {conf:.2f} | {self.current_backend}",
                    (x + width + 10, y + 10), config.FONT_FACE, 0.45,
                    config.COLOR_TEXT, 1)

    def _render_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Render UI info overlay on frame."""
        h, w = frame.shape[:2]
        overlay = frame.copy()

        self._render_demo_panel(overlay)

        # Bottom status bar
        bar_y = h - config.STATUS_BAR_HEIGHT
        cv2.rectangle(overlay, (0, bar_y), (w, h), config.COLOR_BG, -1)
        cv2.line(overlay, (0, bar_y), (w, bar_y), config.COLOR_PRIMARY, 2)

        mode_text = "Mouse Mode" if self.mode == ControllerMode.MOUSE else "Normal Mode"
        gesture_name = self.current_gesture.name if self.current_gesture else "NONE"

        line1 = f"Mode: {mode_text}  |  Gesture: {gesture_name}  |  {self.gate_status_text()}"
        cv2.putText(overlay, line1, (10, bar_y + 26),
                    config.FONT_FACE, config.FONT_SCALE,
                    config.COLOR_PRIMARY, config.FONT_THICKNESS)

        self._render_confidence_bar(overlay, 10, bar_y + 40, 170)

        line2 = f"FPS: {self.fps:.1f}  |  {self.status_message}"
        cv2.putText(overlay, line2, (10, bar_y + 72),
                    config.FONT_FACE, 0.55,
                    config.COLOR_TEXT, 1)

        line3 = f"Next action: {self._action_hint()}"
        cv2.putText(overlay, line3, (10, bar_y + 98),
                    config.FONT_FACE, 0.52,
                    config.COLOR_SECONDARY, 1)

        # Finger state indicators
        finger_names = ["T", "I", "M", "R", "P"]
        bar_x_start = w - 140
        for i, (name, is_extended) in enumerate(zip(finger_names, self.current_finger_states)):
            x = bar_x_start + i * 28
            color = config.COLOR_PRIMARY if is_extended else (80, 80, 80)
            cv2.circle(overlay, (x, bar_y - 15), 10, color, -1)
            cv2.putText(overlay, name, (x - 6, bar_y - 5),
                        config.FONT_FACE, 0.45,
                        config.COLOR_TEXT, 1)

        # Mouse mode crosshair
        if self.mode == ControllerMode.MOUSE and self._mouse_ref_x is not None:
            cx, cy = int(self._mouse_ref_x), int(self._mouse_ref_y)
            cv2.line(overlay, (cx - 20, cy), (cx + 20, cy),
                     config.COLOR_SECONDARY, 2)
            cv2.line(overlay, (cx, cy - 20), (cx, cy + 20),
                     config.COLOR_SECONDARY, 2)
            cv2.circle(overlay, (cx, cy), 25, config.COLOR_SECONDARY, 2)
            cv2.circle(overlay, (cx, cy), 5, config.COLOR_SECONDARY, -1)

        return cv2.addWeighted(overlay, config.OVERLAY_ALPHA, frame, 1 - config.OVERLAY_ALPHA, 0)

    def close(self):
        """Release all resources."""
        self.hand_detector.close()
