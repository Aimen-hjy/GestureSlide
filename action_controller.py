"""
动作执行器 — 通过 pyautogui 模拟键盘/鼠标操作
"""

import time
import pyautogui

import config

# pyautogui 安全设置
pyautogui.FAILSAFE = True  # 鼠标移到屏幕角落时触发异常
pyautogui.PAUSE = 0.01     # 每次操作后短暂暂停


class ActionController:
    """键盘/鼠标动作模拟器"""

    def __init__(self):
        self._screen_w, self._screen_h = pyautogui.size()
        # 鼠标平滑: 上次目标位置
        self._prev_target_x: float | None = None
        self._prev_target_y: float | None = None

    # ==================== PPT 翻页 ====================

    def next_slide(self):
        """下一页 (→ 键)"""
        pyautogui.press("right")
        print("[动作] 下一页")

    def prev_slide(self):
        """上一页 (← 键)"""
        pyautogui.press("left")
        print("[动作] 上一页")

    # ==================== 放映控制 ====================

    def start_slideshow(self):
        """开始放映 (F5)"""
        pyautogui.press("f5")
        print("[动作] 开始放映")

    def exit_slideshow(self):
        """退出放映 (Esc)"""
        pyautogui.press("escape")
        print("[动作] 退出放映")

    def toggle_black_screen(self):
        """切换黑屏 (B键)"""
        pyautogui.press("b")
        print("[动作] 切换黑屏")

    def toggle_white_screen(self):
        """切换白屏 (W键)"""
        pyautogui.press("w")
        print("[动作] 切换白屏")

    # ==================== 播放控制 ====================

    def pause_resume(self):
        """暂停/恢复 (Ctrl+P)"""
        pyautogui.hotkey("ctrl", "p")
        print("[动作] 暂停/恢复")

    # ==================== 音量控制 ====================

    def volume_up(self):
        """音量增大"""
        pyautogui.press("volumeup")
        print("[动作] 音量+")

    def volume_down(self):
        """音量减小"""
        pyautogui.press("volumedown")
        print("[动作] 音量-")

    def volume_mute(self):
        """静音切换"""
        pyautogui.press("volumemute")
        print("[动作] 静音切换")

    # ==================== 鼠标控制 ====================

    def move_mouse(self, dx: float, dy: float):
        """
        相对移动鼠标 (带平滑处理)

        Args:
            dx: x方向位移量 (像素)
            dy: y方向位移量 (像素)
        """
        # 死区过滤
        if abs(dx) < config.MOUSE_DEAD_ZONE and abs(dy) < config.MOUSE_DEAD_ZONE:
            return

        # 速度倍率
        dx *= config.MOUSE_SPEED
        dy *= config.MOUSE_SPEED

        # 平滑: 指数移动平均
        if self._prev_target_x is not None:
            dx = (config.MOUSE_SMOOTHING * dx +
                  (1 - config.MOUSE_SMOOTHING) * self._prev_target_x)
            dy = (config.MOUSE_SMOOTHING * dy +
                  (1 - config.MOUSE_SMOOTHING) * self._prev_target_y)

        self._prev_target_x = dx
        self._prev_target_y = dy

        # pyautogui 相对移动
        pyautogui.moveRel(int(dx), int(dy), duration=0.02)

    def reset_mouse_smoothing(self):
        """重置鼠标平滑状态 (模式切换时调用)"""
        self._prev_target_x = None
        self._prev_target_y = None

    def mouse_click(self):
        """鼠标左键点击"""
        pyautogui.click()
        print("[动作] 鼠标左键点击")

    def mouse_right_click(self):
        """鼠标右键点击"""
        pyautogui.rightClick()
        print("[动作] 鼠标右键点击")

    def mouse_double_click(self):
        """鼠标双击"""
        pyautogui.doubleClick()
        print("[动作] 鼠标双击")

    def mouse_scroll(self, amount: int):
        """
        鼠标滚轮滚动

        Args:
            amount: 滚动量, 正数向上, 负数向下
        """
        pyautogui.scroll(amount)
        print(f"[动作] 滚轮: {amount}")

    # ==================== 工具方法 ====================

    def get_screen_size(self) -> tuple:
        """获取屏幕尺寸"""
        return (self._screen_w, self._screen_h)
