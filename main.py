"""
基于MediaPipe的无接触手势PPT控制系统 - 主程序入口

使用方法:
  python main.py                # 正常运行
  python main.py --collect      # 数据采集模式 (训练用)

按 'q' 键退出程序
"""

import sys
import json
import time
import ctypes
import argparse
from pathlib import Path

import cv2

import config
from hand_detector import HandDetector
from gesture_classifier import GestureClassifier, Gesture
from gesture_model import extract_features, GESTURE_NAMES
from ppt_controller import PPTController, ControllerMode

# OpenCV 窗口名称
WINDOW_NAME = "Gesture PPT Controller - Press 'q' to quit"

# Windows API 常量
HWND_TOPMOST = -1
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001


def set_window_topmost():
    """将 OpenCV 窗口置顶"""
    hwnd = ctypes.windll.user32.FindWindowW(None, WINDOW_NAME)
    if hwnd:
        ctypes.windll.user32.SetWindowPos(
            hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)


def print_instructions():
    """打印操作说明"""
    print("=" * 55)
    print("  基于MediaPipe的无接触手势PPT控制系统")
    print("=" * 55)
    print()
    print("  📷 请确保摄像头已连接")
    print("  🖐  将手放在摄像头前，做出以下手势:")
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
    print("    🤏  拇指食捏合     →  鼠标点击")
    print("    🖐  五指张开       →  退出鼠标模式")
    print()
    print("  按 'q' 键退出程序")
    print()
    print("-" * 55)


def run_collection_mode():
    """数据采集模式: 为每种手势录制训练样本"""
    save_dir = Path("training_data")
    save_dir.mkdir(exist_ok=True)

    # 初始化
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[错误] 无法打开摄像头!")
        sys.exit(1)

    detector = HandDetector()
    samples = []
    current_label = 0       # 当前录制的标签ID
    recording = False       # 是否正在录制
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

            # 手部检测
            hand_data = detector.detect_hand(frame)
            if hand_data is not None:
                landmarks = hand_data["landmarks"]
                finger_states = detector.get_finger_states(landmarks)
                thumb_index_dist = detector.get_thumb_index_distance(landmarks)
                hand_data["finger_states"] = finger_states
                hand_data["thumb_index_dist"] = thumb_index_dist

                # 绘制手部骨架
                frame = detector.draw_hand(frame, hand_data["raw"])

                # 录制: 每3帧存一次 (减少重复)
                if recording and frame_count % 3 == 0:
                    features = extract_features(hand_data)
                    samples.append({
                        "features": features.tolist(),
                        "label": label_name,
                        "label_id": current_label,
                    })

                frame_count += 1

                # 显示手指状态圆点
                finger_names = ["T", "I", "M", "R", "P"]
                for i, (name, is_ext) in enumerate(
                        zip(finger_names, finger_states)):
                    x = w - 140 + i * 28
                    color = (0, 255, 0) if is_ext else (80, 80, 80)
                    cv2.circle(frame, (x, 40), 10, color, -1)
                    cv2.putText(frame, name, (x - 6, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1)

            # ── UI 叠加 ──
            overlay = frame.copy()

            # 顶部信息栏
            cv2.rectangle(overlay, (0, 0), (w, 70), (30, 30, 30), -1)
            key_label = str(current_label) if current_label <= 9 else "a"
            title = f"Label: {label_name} (key {key_label})"
            cv2.putText(overlay, title, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 录制状态
            if recording:
                rec_text = f"● REC  |  Samples: {len(samples)}"
                rec_color = (0, 0, 255)
                # 闪烁红点
                dot_x, dot_y = w - 200, 25
                cv2.circle(overlay, (dot_x, dot_y), 8, (0, 0, 255), -1)
            else:
                rec_text = "○ PAUSED (press SPACE to start)"
                rec_color = (150, 150, 150)
            cv2.putText(overlay, rec_text, (w - 350, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, rec_color, 1)

            # 底部按键提示
            cv2.rectangle(overlay, (0, h - 30), (w, h), (30, 30, 30), -1)
            hint = "0-9/a:label | SPACE:record | s:save | q:quit"
            cv2.putText(overlay, hint, (10, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            cv2.imshow(WINDOW_NAME, frame)
            if not _topmost_set:
                set_window_topmost()
                _topmost_set = True

            key = cv2.waitKey(1) & 0xFF

            # 数字键 0-9 / a: 切换标签
            if ord('0') <= key <= ord('9'):
                current_label = key - ord('0')
            elif key == ord('a') or key == ord('A'):
                current_label = 10
                print(f"[采集] 切换到标签: {GESTURE_NAMES[current_label]} ({current_label})")

            # 空格: 切换录制
            elif key == ord(' '):
                recording = not recording
                if recording:
                    print(f"[采集] ▶ 开始录制 {GESTURE_NAMES[current_label]}")
                else:
                    print(f"[采集] ⏸ 暂停录制 (已录 {len(samples)} 条)")

            # s: 保存
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

            # q: 退出
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


def main():
    parser = argparse.ArgumentParser(description="Gesture PPT Controller")
    parser.add_argument("--collect", action="store_true",
                        help="Run in data collection mode")
    args = parser.parse_args()

    if args.collect:
        run_collection_mode()
        return

    print_instructions()

    # 初始化摄像头
    cap = cv2.VideoCapture(config.CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("[错误] 无法打开摄像头! 请检查摄像头连接。")
        sys.exit(1)

    # 创建PPT控制器
    controller = PPTController()

    print("[系统] 初始化完成，开始处理视频流...")
    print()

    _topmost_set = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[警告] 无法读取视频帧")
                break

            # 水平镜像翻转 (自拍视角)
            if config.FLIP_HORIZONTAL:
                frame = cv2.flip(frame, 1)

            # 处理帧 (检测→识别→执行→渲染)
            output_frame = controller.process_frame(frame)

            # 显示输出
            cv2.imshow(WINDOW_NAME, output_frame)
            if not _topmost_set:
                set_window_topmost()
                _topmost_set = True

            # 按 'q' 键退出
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
        controller.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[系统] 资源已释放, 程序结束。")


if __name__ == "__main__":
    main()
