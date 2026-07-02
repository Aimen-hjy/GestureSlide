"""
基于MediaPipe的无接触手势PPT控制系统 - 主程序入口

使用方法:
  python main.py                         # 正常运行
  python main.py --collect               # 数据采集模式 (训练用)
  python main.py --ppt demo.pptx --start # 打开PPT并自动开始放映
  python main.py --headless --hud        # 正式演示：只显示悬浮识别HUD

按 'q' 键退出程序；headless 模式下使用 Ctrl+C 退出。
"""

import sys
import json
import time
import platform
import ctypes
import argparse
import os
import subprocess
from pathlib import Path

import cv2

import config
from hand_detector import HandDetector
from gesture_model import extract_features, GESTURE_NAMES
from ppt_controller import PPTController

# OpenCV 窗口名称
WINDOW_NAME = "Gesture PPT Controller - Press 'q' to quit"

# Windows API 常量
HWND_TOPMOST = -1
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001


def set_window_topmost():
    """尽量将 OpenCV 预览窗口置顶。"""
    try:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    if platform.system() != "Windows":
        return

    try:
        hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_NAME)
        if hwnd:
            SWP_SHOWWINDOW = 0x0040
            ctypes.windll.user32.SetWindowPos(
                hwnd,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW,
            )
    except Exception as exc:
        print(f"[警告] 无法设置窗口置顶: {exc}")


def setup_preview_window(preview: str):
    """配置调试预览窗口。"""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    if preview == "small":
        cv2.resizeWindow(WINDOW_NAME, 420, 315)
        cv2.moveWindow(WINDOW_NAME, 20, 60)
    else:
        cv2.resizeWindow(WINDOW_NAME, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    set_window_topmost()


def open_ppt_file(path: str):
    """用系统默认程序打开 PPT/PDF/演示文件。"""
    ppt_path = Path(path).expanduser().resolve()
    if not ppt_path.exists():
        raise FileNotFoundError(f"PPT file not found: {ppt_path}")

    system = platform.system()
    print(f"[系统] 打开演示文件: {ppt_path}")
    if system == "Windows":
        os.startfile(str(ppt_path))  # type: ignore[attr-defined]
    elif system == "Darwin":
        subprocess.Popen(["open", str(ppt_path)])
    else:
        subprocess.Popen(["xdg-open", str(ppt_path)])


def print_instructions():
    """打印操作说明。"""
    print("=" * 60)
    print("  基于MediaPipe的无接触手势PPT控制系统")
    print("=" * 60)
    print()
    print("  📷 请确保摄像头已连接")
    print("  🪟 可用 --hud 显示悬浮识别状态，不必显示完整摄像头画面")
    print()
    print("  普通模式:")
    print("    👈  食指左指       →  上一页")
    print("    👉  食指右指       →  下一页")
    print("    🖐  五指张开       →  退出放映 (Esc)")
    print("    👌  OK手势         →  开始放映 (F5)")
    print("    ☝️  食指朝上       →  进入鼠标模式")
    print("    ✌️  V手势朝上       →  音量增大")
    print("    ✌️  V手势朝下       →  音量减小")
    print("    👍  竖拇指         →  开始放映")
    print()
    print("  鼠标模式:")
    print("    ☝️  移动食指       →  移动光标")
    print("    🤏  拇指食指捏合   →  鼠标点击")
    print("    🖐  五指张开       →  退出鼠标模式")
    print()
    print("  调试: 按 'q' 退出；headless 模式下用 Ctrl+C 退出")
    print("-" * 60)


def run_collection_mode():
    """数据采集模式: 为每种手势录制训练样本。"""
    save_dir = Path("training_data")
    save_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[错误] 无法打开摄像头!")
        sys.exit(1)

    detector = HandDetector()
    samples = []
    current_label = 0
    recording = False
    frame_count = 0

    print()
    print("=" * 55)
    print("  数据采集模式 — 训练手势分类模型")
    print("=" * 55)
    print()
    print("  按键说明:")
    print("  0-9  切换手势标签 (a=10)")
    for i, name in enumerate(GESTURE_NAMES):
        key = str(i) if i <= 9 else "a"
        marker = " ←" if i == current_label else ""
        print(f"    {key} → {name}{marker}")
    print("  空格  开始/停止录制")
    print("  s     保存当前数据")
    print("  q     保存并退出")
    print()
    print("-" * 55)

    _topmost_set = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            label_name = GESTURE_NAMES[current_label]

            hand_data = detector.detect_hand(frame)
            if hand_data is not None:
                landmarks = hand_data["landmarks"]
                finger_states = detector.get_finger_states(landmarks)
                thumb_index_dist = detector.get_thumb_index_distance(landmarks)
                hand_data["finger_states"] = finger_states
                hand_data["thumb_index_dist"] = thumb_index_dist

                frame = detector.draw_hand(frame, hand_data["raw"])

                if recording and frame_count % 3 == 0:
                    features = extract_features(hand_data)
                    samples.append({
                        "features": features.tolist(),
                        "label": label_name,
                        "label_id": current_label,
                    })

                frame_count += 1

                finger_names = ["T", "I", "M", "R", "P"]
                for i, (name, is_ext) in enumerate(zip(finger_names, finger_states)):
                    x = w - 140 + i * 28
                    color = (0, 255, 0) if is_ext else (80, 80, 80)
                    cv2.circle(frame, (x, 40), 10, color, -1)
                    cv2.putText(frame, name, (x - 6, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
            key_label = str(current_label) if current_label <= 9 else "a"
            title = f"Label: {label_name} (key {key_label})"
            cv2.putText(overlay, title, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if recording:
                rec_text = f"● REC  |  Samples: {len(samples)}"
                rec_color = (0, 0, 255)
                cv2.circle(overlay, (w - 200, 25), 8, (0, 0, 255), -1)
            else:
                rec_text = "○ PAUSED (press SPACE to start)"
                rec_color = (150, 150, 150)
            cv2.putText(overlay, rec_text, (w - 350, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, rec_color, 1)

            cv2.rectangle(overlay, (0, h - 30), (w, h), (30, 30, 30), -1)
            hint = "0-9/a:label | SPACE:record | s:save | q:quit"
            cv2.putText(overlay, hint, (10, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            cv2.imshow(WINDOW_NAME, frame)
            if not _topmost_set:
                set_window_topmost()
                _topmost_set = True

            key = cv2.waitKey(1) & 0xFF

            if ord('0') <= key <= ord('9'):
                current_label = key - ord('0')
            elif key == ord('a') or key == ord('A'):
                current_label = 10
                print(f"[采集] 切换到标签: {GESTURE_NAMES[current_label]} ({current_label})")
            elif key == ord(' '):
                recording = not recording
                if recording:
                    print(f"[采集] ▶ 开始录制 {GESTURE_NAMES[current_label]}")
                else:
                    print(f"[采集] ⏸ 暂停录制 (已录 {len(samples)} 条)")
            elif key == ord('s'):
                if samples:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = save_dir / f"session_{ts}.json"
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(samples, f, ensure_ascii=False)
                    print(f"[采集] 已保存 {len(samples)} 条 → {path}")
                    samples.clear()
                else:
                    print("[采集] 无数据可保存")
            elif key == ord('q'):
                if samples:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = save_dir / f"session_{ts}.json"
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(samples, f, ensure_ascii=False)
                    print(f"[采集] 已保存 {len(samples)} 条 → {path}")
                print("\n[采集] 退出数据采集模式")
                break

    except KeyboardInterrupt:
        print("\n[采集] 检测到 Ctrl+C, 正在退出...")
        if samples:
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = save_dir / f"session_{ts}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False)
            print(f"[采集] 已保存 {len(samples)} 条 → {path}")
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[采集] 资源已释放")


def make_hud_state(controller: PPTController) -> dict:
    """Build a compact state dict for the floating HUD."""
    mode = "Mouse" if controller.mode.name == "MOUSE" else "Normal"
    gesture = controller.current_gesture.name if controller.current_gesture else "NONE"
    return {
        "mode": mode,
        "gesture": gesture,
        "action": controller._action_hint(),
        "status": controller.status_message,
        "confidence": controller.current_confidence,
        "backend": controller.current_backend,
    }


def main():
    parser = argparse.ArgumentParser(description="Gesture PPT Controller")
    parser.add_argument("--collect", action="store_true",
                        help="Run in data collection mode")
    parser.add_argument("--ppt", default=None,
                        help="Open this PPT/PDF/presentation file before starting the camera loop")
    parser.add_argument("--start", action="store_true",
                        help="After opening --ppt, automatically press F5 to start slideshow")
    parser.add_argument("--open-delay", type=float, default=4.0,
                        help="Seconds to wait after opening --ppt before pressing F5")
    parser.add_argument("--headless", action="store_true",
                        help="Run without the camera preview window; use Ctrl+C to exit")
    parser.add_argument("--preview", choices=("normal", "small"), default="normal",
                        help="Camera preview size when not using --headless")
    parser.add_argument("--hud", action="store_true",
                        help="Show a small always-on-top gesture status HUD")
    parser.add_argument("--hud-position", choices=("top-right", "top-left", "bottom-right", "bottom-left"),
                        default="top-right", help="Initial HUD position")
    args = parser.parse_args()

    if args.collect:
        run_collection_mode()
        return

    print_instructions()

    controller = PPTController()
    hud = None
    if args.hud:
        try:
            from hud_window import GestureHUD
            hud = GestureHUD(position=args.hud_position)
            print("[系统] 悬浮 HUD 已启动。双击 HUD 可关闭，拖拽可移动。")
        except Exception as exc:
            print(f"[警告] 无法启动 HUD: {exc}")
            print("[提示] Linux/conda 环境可能需要安装 tkinter，例如 conda install tk")

    if args.ppt:
        try:
            open_ppt_file(args.ppt)
            if args.start:
                print(f"[系统] 等待 {args.open_delay:.1f}s 后自动开始放映...")
                time.sleep(max(0.0, args.open_delay))
                controller.action_controller.start_slideshow()
                controller.slideshow_started = True
                print("[系统] 已发送 F5")
        except Exception as exc:
            print(f"[警告] 无法自动打开/开始演示: {exc}")

    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[错误] 无法打开摄像头! 请检查摄像头连接。")
        if hud is not None:
            hud.close()
        controller.close()
        sys.exit(1)

    if not args.headless:
        setup_preview_window(args.preview)

    print("[系统] 初始化完成，开始处理视频流...")
    if args.headless:
        print("[系统] Headless 模式：不显示摄像头窗口，使用 Ctrl+C 退出。")
    if args.hud:
        print("[系统] HUD 模式：放映时可查看当前手势/动作/置信度。")
    print()

    _topmost_set = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[警告] 无法读取视频帧")
                break

            if config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            output_frame = controller.process_frame(frame)

            if hud is not None:
                if not hud.update(make_hud_state(controller)):
                    hud = None

            if args.headless:
                time.sleep(0.001)
                continue

            cv2.imshow(WINDOW_NAME, output_frame)
            if not _topmost_set:
                set_window_topmost()
                _topmost_set = True

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[系统] 用户退出")
                break

    except KeyboardInterrupt:
        print("\n[系统] 检测到 Ctrl+C, 正在退出...")
    except Exception as e:
        print(f"\n[错误] 运行时异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if hud is not None:
            hud.close()
        controller.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[系统] 资源已释放, 程序结束。")


if __name__ == "__main__":
    main()
