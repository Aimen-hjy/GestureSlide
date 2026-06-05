"""
基于MediaPipe的无接触手势PPT控制系统 - 主程序入口

功能:
  - 挥手翻页 (左滑上一页, 右滑下一页)
  - OK手势开始放映
  - 握拳切换黑屏
  - 食指指向进入鼠标模式
  - 捏合点击
  - V手势调高音量
  - 三指调低音量
  - 小指退出放映

使用方法:
  python main.py

按 'q' 键退出程序
"""

import sys
import cv2

import config
from ppt_controller import PPTController, ControllerMode


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
    print("    👉👈  挥手(左/右)  →  上一页/下一页")
    print("    ✊  握拳           →  切换黑屏/恢复")
    print("    👌  OK手势         →  开始放映 (F5)")
    print("    ☝️  食指指向       →  进入鼠标模式")
    print("    ✌️  V手势          →  音量增大")
    print("    🤟  三指           →  音量减小")
    print("    👍  竖拇指         →  开始放映")
    print("    🤙  小指           →  退出放映 (Esc)")
    print()
    print("  鼠标模式:")
    print("    ☝️  移动食指       →  移动光标")
    print("    🤏  拇指食捏合     →  鼠标点击")
    print("    🖐  五指张开       →  退出鼠标模式")
    print()
    print("  按 'q' 键退出程序")
    print()
    print("-" * 55)


def main():
    """主函数"""
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

    try:
        while True:
            # 读取帧
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
            cv2.imshow("Gesture PPT Controller - Press 'q' to quit",
                       output_frame)

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
        # 清理资源
        controller.close()
        cap.release()
        cv2.destroyAllWindows()
        print("[系统] 资源已释放, 程序结束。")


if __name__ == "__main__":
    main()
