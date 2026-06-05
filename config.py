"""
全局配置参数
"""

# ================== 摄像头设置 ==================
CAMERA_ID = 0               # 摄像头设备ID (0=默认摄像头)
FRAME_WIDTH = 640           # 帧宽度
FRAME_HEIGHT = 480          # 帧高度
FLIP_HORIZONTAL = True      # 水平镜像翻转 (自拍视角)

# ================== MediaPipe 设置 ==================
MIN_DETECTION_CONFIDENCE = 0.7   # 手部检测置信度阈值
MIN_TRACKING_CONFIDENCE = 0.7    # 手部追踪置信度阈值
MAX_NUM_HANDS = 1                # 最多检测手数 (单手控制)
STATIC_IMAGE_MODE = False        # 非静态图片模式

# ================== 手势识别阈值 ==================
SWIPE_THRESHOLD = 60        # 滑动最小像素距离
SWIPE_HISTORY_SIZE = 10     # 滑动历史帧数
SWIPE_COOLDOWN = 1.0        # 滑动冷却时间 (秒)
SWIPE_GESTURE_COOLDOWN = 1.5  # 翻页手势冷却时间 (秒)

PINCH_DISTANCE_THRESHOLD = 0.05  # 捏合距离阈值 (归一化坐标)
CLICK_COOLDOWN = 0.8        # 点击冷却时间 (秒)

GESTURE_COOLDOWN = 1.2      # 通用手势冷却时间 (秒)
GESTURE_HOLD_FRAMES = 5     # 手势需持续帧数才触发

# ================== 鼠标控制设置 ==================
MOUSE_SMOOTHING = 0.6       # 鼠标移动平滑系数 (0-1, 越小越平滑)
MOUSE_SPEED = 2.0           # 鼠标移动速度倍率
MOUSE_DEAD_ZONE = 15        # 鼠标移动死区 (像素)

# ================== UI显示设置 ==================
FONT_FACE = 0               # OpenCV 字体 (FONT_HERSHEY_SIMPLEX)
FONT_SCALE = 0.7            # 字体大小
FONT_THICKNESS = 2          # 字体粗细
OVERLAY_ALPHA = 0.6         # 叠加层透明度
STATUS_BAR_HEIGHT = 80      # 底部状态栏高度
COLOR_PRIMARY = (0, 255, 0)       # 主色调 (绿色)
COLOR_SECONDARY = (255, 255, 0)   # 次要色 (青色)
COLOR_WARNING = (0, 0, 255)       # 警告色 (红色)
COLOR_TEXT = (255, 255, 255)      # 文字颜色 (白色)
COLOR_BG = (30, 30, 30)           # 背景色 (深灰)
