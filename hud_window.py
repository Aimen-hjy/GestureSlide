"""Small always-on-top HUD for slideshow demos.

The HUD is intentionally independent from the OpenCV camera preview. It shows a
compact recognition/status panel that can remain visible while a PPT slideshow
is running.
"""

from __future__ import annotations

import tkinter as tk
from typing import Any


class GestureHUD:
    """Tiny topmost status window for live gesture feedback."""

    def __init__(self, position: str = "top-right", alpha: float = 0.88):
        self.root = tk.Tk()
        self.root.title("GestureSlide HUD")
        self.root.configure(bg="#111111")
        self.root.resizable(False, False)

        # Borderless, topmost, slightly transparent. Some Linux window managers
        # may still place exclusive fullscreen apps above all normal windows, but
        # this works well with most PPT/WPS/LibreOffice slideshow modes.
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", max(0.35, min(1.0, alpha)))
        except tk.TclError:
            pass
        try:
            self.root.attributes("-type", "splash")
        except tk.TclError:
            pass

        self._drag_start_x = 0
        self._drag_start_y = 0

        self.frame = tk.Frame(self.root, bg="#111111", bd=1, relief="solid")
        self.frame.pack(fill="both", expand=True)

        self.title_var = tk.StringVar(value="GestureSlide")
        self.gesture_var = tk.StringVar(value="Gesture: NONE")
        self.action_var = tk.StringVar(value="Action: waiting")
        self.detail_var = tk.StringVar(value="conf 0.00 | ready")

        self.title_label = tk.Label(
            self.frame, textvariable=self.title_var, bg="#111111", fg="#00ff88",
            font=("DejaVu Sans", 10, "bold"), anchor="w", padx=10, pady=2,
        )
        self.title_label.pack(fill="x")

        self.gesture_label = tk.Label(
            self.frame, textvariable=self.gesture_var, bg="#111111", fg="#ffffff",
            font=("DejaVu Sans", 14, "bold"), anchor="w", padx=10, pady=0,
        )
        self.gesture_label.pack(fill="x")

        self.action_label = tk.Label(
            self.frame, textvariable=self.action_var, bg="#111111", fg="#66e0ff",
            font=("DejaVu Sans", 10), anchor="w", padx=10, pady=0,
        )
        self.action_label.pack(fill="x")

        self.detail_label = tk.Label(
            self.frame, textvariable=self.detail_var, bg="#111111", fg="#bbbbbb",
            font=("DejaVu Sans", 9), anchor="w", padx=10, pady=2,
        )
        self.detail_label.pack(fill="x")

        for widget in (self.root, self.frame, self.title_label, self.gesture_label,
                       self.action_label, self.detail_label):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._drag)
            widget.bind("<Double-Button-1>", lambda _event: self.close())

        self._place(position)
        self.root.update_idletasks()
        self.root.update()

    def _place(self, position: str):
        width, height = 320, 96
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        margin = 24
        if position == "top-left":
            x, y = margin, margin
        elif position == "bottom-left":
            x, y = margin, max(margin, screen_h - height - margin)
        elif position == "bottom-right":
            x, y = max(margin, screen_w - width - margin), max(margin, screen_h - height - margin)
        else:
            x, y = max(margin, screen_w - width - margin), margin
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _start_drag(self, event):
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _drag(self, event):
        x = self.root.winfo_x() + event.x - self._drag_start_x
        y = self.root.winfo_y() + event.y - self._drag_start_y
        self.root.geometry(f"+{x}+{y}")

    def update(self, state: dict[str, Any]) -> bool:
        """Update the HUD. Returns False if the HUD has been closed."""
        try:
            gesture = str(state.get("gesture", "NONE"))
            action = str(state.get("action", "waiting"))
            status = str(state.get("status", "ready"))
            mode = str(state.get("mode", "Normal"))
            backend = str(state.get("backend", "rule"))
            confidence = float(state.get("confidence", 0.0))

            self.title_var.set(f"GestureSlide | {mode}")
            self.gesture_var.set(f"Gesture: {gesture}")
            self.action_var.set(f"Action: {action}")
            self.detail_var.set(f"conf {confidence:.2f} | {backend} | {status}")

            # Reassert topmost regularly because slideshow apps sometimes steal focus.
            self.root.attributes("-topmost", True)
            self.root.update_idletasks()
            self.root.update()
            return True
        except tk.TclError:
            return False

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass
