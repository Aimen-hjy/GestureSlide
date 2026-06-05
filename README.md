# 基于MediaPipe的无接触手势PPT控制系统

通过摄像头实时识别手势，实现无接触控制PPT演示。支持翻页、放映、鼠标移动点击、音量调节等功能。

## 功能一览

### 普通模式
| 手势 | 动作 | 说明 |
|------|------|------|
| 👈 食指左指 | ← | 上一页 |
| 👉 食指右指 | → | 下一页 |
| 🖐 五指张开 | Esc | 退出PPT放映 |
| 👌 OK手势 | F5 | 开始PPT放映 |
| ☝️ 食指朝上 | 切换模式 | 进入鼠标控制模式 |
| ✌️ V手势朝上 | 音量+ | 调高系统音量 |
| ✌️ V手势朝下 | 音量- | 调低系统音量 |
| 👍 竖拇指 | F5 | 开始PPT放映 |

### 鼠标模式
| 手势 | 动作 | 说明 |
|------|------|------|
| ☝️ 移动食指 | 移动光标 | 食指控制鼠标指针 |
| 🤏 拇指食指捏合 | 左键点击 | 模拟鼠标点击 |
| 🖐 五指张开 | 退出 | 返回普通模式 |

## 系统架构

```
groupwork/
├── main.py                   # 主程序入口 + 可视化UI
├── config.py                 # 全局配置参数
├── hand_detector.py          # MediaPipe手部检测封装
├── gesture_classifier.py     # 手势分类识别器 (ML + 规则)
├── gesture_model.py          # ML模型定义 + 合成数据生成 + 特征提取
├── train_model.py            # 模型训练脚本
├── action_controller.py      # 键盘/鼠标动作执行器
├── ppt_controller.py         # PPT控制核心状态机
├── gesture_model.joblib      # 训练好的MLP模型 (自动生成)
├── gesture_scaler.joblib     # 特征标准化器 (自动生成)
├── requirements.txt          # Python依赖
└── README.md                 # 本文件
```

## 环境要求

- Python 3.8+
- Windows / macOS / Linux
- 摄像头 (内置或USB)

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成手势分类模型 (首次运行仅需一次)

```bash
python train_model.py
```

### 3. 运行程序

```bash
python main.py
```

### 4. 使用步骤

1. 确保摄像头正常工作
2. 打开你的PPT文件
3. 运行本程序
4. 使用手势控制PPT：
   - 做 **OK手势** 或 **竖拇指** 开始放映
   - **食指左指/右指** 翻上一页/下一页
   - **食指朝上** 进入鼠标模式，移动食指控制光标，捏合点击
   - **五指张开** 退出鼠标模式
   - 按 **q键** 退出程序

### 5. 配置调整

编辑 `config.py` 可调整：
- 摄像头分辨率和帧率
- ML模型置信度阈值
- 手势识别灵敏度
- 鼠标移动速度
- UI显示样式

## 数据采集（训练自定义模型）

默认使用合成数据训练模型。如需更高准确率，可采集真实手部数据重新训练。

### 1. 采集模式

```bash
python main.py --collect
```

采集界面操作：
| 按键 | 功能 |
|------|------|
| 0-9 / a | 选择手势标签：0=无手势 1=五指张开 2=握拳 3=左指 4=右指 5=下指 6=食指朝上 7=竖拇指 8=V朝上 9=V朝下 a=三指 |
| 空格 | 开始/停止录制 |
| s | 保存当前数据 |
| q | 保存数据并退出 |

每种手势建议在摄像头前变换角度、持续录制 10-30 秒，数据文件保存在 `training_data/` 目录下。

### 2. 用真实数据训练

```bash
# 仅真实数据
python train_model.py --real training_data/

# 真实 + 合成数据混合（数据量少时推荐）
python train_model.py --mix training_data/
```

训练完成后 `python main.py` 即使用新模型。可将 `training_data/` 分发给他人，对方执行相同命令即可复现。

## 技术原理

### 手部检测
使用 Google MediaPipe Hands 模型，实时检测21个手部关键点坐标。

### 手势识别 (ML + 规则混合)
- **ML 模型**：使用 MLP 神经网络（128→64，69维输入特征）直接分类 11 种静态手势（NONE / OPEN_PALM / FIST / LEFT_POINT / RIGHT_POINT / DOWN_POINT / POINT_INDEX / THUMB_UP / PEACE_UP / PEACE_DOWN / THREE_FINGERS），含方向变体。模型使用合成数据训练，准确率 >98%。
- **捏合检测**：计算拇指尖与食指尖的归一化距离，小于阈值判定为捏合（PINCH / OK_SIGN）。

### 防抖机制
- 手势需持续N帧才确认触发（避免瞬时误识别）
- 每种手势有独立冷却时间（避免重复触发）
- 鼠标移动采用指数平滑（避免抖动）

### 鼠标映射
食指指尖的相对位移按速度倍率映射为屏幕鼠标移动，带死区过滤和平滑处理。

## 依赖库

- **opencv-python** - 摄像头采集与图像处理
- **mediapipe** - Google手部关键点检测
- **pyautogui** - 键盘鼠标事件模拟
- **numpy** - 数值计算
- **scikit-learn** - MLP神经网络分类器

## 注意事项

1. 保持手部在摄像头视野内，光照充足时效果最佳
2. 手势动作要清晰、稳定，避免过快晃动
3. 鼠标模式下移动食指即可控制光标，不需要大幅度移动
4. 程序使用 `pyautogui.FAILSAFE`，将鼠标移到屏幕角落可紧急停止
5. 如遇摄像头无法打开，请检查 `config.py` 中的 `CAMERA_ID` 设置
6. 首次运行需先执行 `python train_model.py` 生成模型文件
7. 模型文件缺失时自动回退到规则分类，不影响基本使用
