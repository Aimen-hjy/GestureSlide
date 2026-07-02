# GestureSlide：基于 MediaPipe 的无接触手势 PPT 控制系统

GestureSlide 通过摄像头实时识别手势，实现对 PPT 放映的无接触控制。项目最终版本重点解决三个实际问题：

1. PPT 放映时仍能看到当前识别状态；
2. 左右翻页手势尽量稳定、直观；
3. 开始放映、退出放映、进入鼠标模式等高风险动作不容易误触发。

项目没有采用大型图片模型训练路线。运行时使用 MediaPipe Hands 提取 21 个手部关键点，再将关键点转换为 69 维结构化特征，由轻量分类器和少量几何规则完成手势判断。

```text
摄像头图像
   ↓
MediaPipe Hands：21 个手部关键点
   ↓
69 维结构化特征
   ↓
轻量分类器 + 食指方向几何修正
   ↓
连续帧稳定判断 + PPT 动作控制
```

## 最终保留的核心改进

### 1. 本地数据训练与多模型对比

最终演示模型使用本地采集数据训练，并保留特征级数据增强和类别平衡。训练流程会比较多种轻量模型，并按 `macro_f1` 保存表现最好的模型。

目前支持比较：

- MLP
- SVM
- RandomForest
- ExtraTrees
- HistGradientBoosting
- KNN

### 2. PPT 放映时的悬浮 HUD

普通 OpenCV 摄像头窗口在 PPT 全屏后容易被遮挡，因此项目增加了一个独立的简洁 HUD：

```bash
python main.py --headless --hud
```

HUD 只显示必要状态：

- 当前模式
- 当前识别手势
- 当前动作提示
- 置信度
- 识别后端

它不显示完整摄像头画面，适合正式演示时使用。

### 3. 左右翻页的几何方向修正

`LEFT_POINT` 和 `RIGHT_POINT` 是 PPT 控制中最核心的动作。最终版本不是完全依赖分类器判断方向，而是在模型判断为“指向类手势”后，再用食指关键点方向修正左/右。

这样既能提高左右方向稳定性，又避免单纯几何规则把 `NONE`、`THUMB_UP`、`THREE_FINGERS` 等非指向类误判成翻页命令。

### 4. 高风险动作更严格

普通翻页手势保持响应速度；高风险动作需要更长稳定帧后才会触发：

- `OPEN_PALM`：退出放映
- `OK_SIGN` / `THUMB_UP`：开始放映
- `POINT_INDEX`：进入鼠标模式
- `PINCH`：鼠标点击

相关配置在 `config.py` 中：

```python
GESTURE_HOLD_FRAMES = 5
HIGH_RISK_GESTURE_HOLD_FRAMES = 10
```

## 手势与动作

### 核心 PPT 控制

| 手势 | 动作 |
|------|------|
| 👌 OK 手势 | 开始放映 |
| 👍 竖拇指 | 开始放映 |
| 👈 食指左指 | 上一页 |
| 👉 食指右指 | 下一页 |
| 🖐 五指张开 | 退出放映 |

### 可选功能

| 手势 | 动作 |
|------|------|
| ☝️ 食指朝上 | 进入鼠标模式 |
| ☝️ 移动食指 | 移动鼠标 |
| 🤏 拇指食指捏合 | 鼠标点击 |
| ✌️ V 手势朝上 | 音量增大 |
| ✌️ V 手势朝下 | 音量减小 |

正式演示建议优先展示核心 PPT 控制功能；鼠标和音量控制属于可选功能。

## 项目结构

```text
GestureSlide/
├── main.py                         # 主程序入口、采集模式、HUD 启动参数
├── config.py                       # 摄像头、识别阈值、稳定帧等配置
├── hand_detector.py                # MediaPipe Hands 封装
├── gesture_model.py                # 69 维特征提取与基础模型工具
├── gesture_classifier.py           # ML/规则分类、置信度、冷却判断
├── ppt_controller.py               # PPT 控制状态机、几何方向修正、动作执行逻辑
├── action_controller.py            # 键盘/鼠标动作封装
├── hud_window.py                   # PPT 放映时的悬浮状态 HUD
├── train_model.py                  # 单模型训练入口
├── training_pipeline.py            # 数据加载、审计、增强、分组训练
├── tools/
│   ├── audit_training_data.py      # 训练数据审计
│   ├── compare_models.py           # 多模型比较并保存最佳模型
│   ├── evaluate_geometry_direction.py # 离线评估左右指几何方向规则
│   └── run_project_pipeline.sh     # 本地数据训练流水线
├── docs/
│   ├── CLASSROOM_DEMO.md           # 课堂展示说明
│   └── DATA_PIPELINE.md            # 本地数据采集与训练说明
├── gesture_model.joblib            # 训练后生成的模型
├── gesture_scaler.joblib           # 训练后生成的标准化器
└── requirements.txt
```

## 环境要求

- Python 3.10+
- 摄像头
- Windows / macOS / Linux

Linux 下如果使用 Wayland，`pyautogui` 的全局键鼠模拟可能受限。控制无效时建议切换到 Xorg 会话，或确保 PPT/WPS 放映窗口处于前台。

## 安装

```bash
# Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Windows
pip install -r requirements.txt
```

Debian/Ubuntu 如果提示 `externally-managed-environment`，不要使用系统级 `pip`，应使用虚拟环境。

如果 HUD 报 `tkinter` 相关错误，Ubuntu 可安装：

```bash
sudo apt install python3-tk
```

Conda 环境可安装：

```bash
conda install -y tk
```

## 训练最终演示模型

推荐使用本地数据训练：

```bash
bash tools/run_project_pipeline.sh
```

该流程会执行：

```text
语法检查 → 数据审计 → 特征增强 → 类别平衡 → 多模型比较 → 保存最佳模型
```

也可以手动运行模型比较：

```bash
python tools/compare_models.py \
  --data training_data/ \
  --split-strategy group \
  --augment \
  --augment-factor 2 \
  --balance-target 800 \
  --metric macro_f1
```

训练后会生成：

```text
gesture_model.joblib
gesture_scaler.joblib
```

## 运行

### 调试模式：显示摄像头预览和 HUD

```bash
python main.py --preview small --hud
```

适合查看手部骨架、识别结果和 HUD 是否一致。

### 正式演示模式：只显示 HUD

```bash
python main.py --headless --hud
```

手动打开 PPT 并进入放映后运行该命令，程序不会显示摄像头大窗口，只显示悬浮 HUD。使用 `Ctrl+C` 退出。

### 自动打开 PPT 并开始放映

```bash
python main.py --ppt "demo.pptx" --start --headless --hud
```

如果 WPS/PPT 打开较慢，可以延长等待时间：

```bash
python main.py --ppt "demo.pptx" --start --open-delay 6 --headless --hud
```

## 数据采集

```bash
python main.py --collect
```

采集界面按键：

| 按键 | 功能 |
|------|------|
| 0-9 / a | 选择手势标签 |
| 空格 | 开始/停止录制 |
| s | 保存当前数据 |
| q | 保存数据并退出 |

标签顺序：

```text
0 NONE
1 OPEN_PALM
2 FIST
3 LEFT_POINT
4 RIGHT_POINT
5 DOWN_POINT
6 POINT_INDEX
7 THUMB_UP
8 PEACE_UP
9 PEACE_DOWN
a THREE_FINGERS
```

采集建议：

- 每个手势分多次采集，保留多个独立 `session_*.json` 文件；
- 同一手势覆盖不同距离、角度、光照；
- `NONE` 应该录制“有手但不属于任何命令”的随机姿势，而不是空画面；
- 切换标签前先暂停录制，避免过渡帧被标错。

## 数据审计与几何规则测试

审计训练数据：

```bash
python tools/audit_training_data.py training_data/
```

离线测试左右指几何方向规则：

```bash
python tools/evaluate_geometry_direction.py training_data/
```

该工具用于检查当前阈值是否会把非指向类误判成 `LEFT_POINT` / `RIGHT_POINT`。最终代码中，几何规则只用于修正指向类手势方向，不会单独把 `NONE` 等类别改成翻页命令。

## 技术细节

### 69 维特征

每帧手部数据转换为 69 维特征：

- 63 维：21 个关键点相对手腕的 `(x, y, z)` 坐标；
- 5 维：拇指、食指、中指、无名指、小指是否伸展；
- 1 维：拇指尖与食指尖距离。

### 运行时识别逻辑

- 低置信度分类结果会被视为 `NONE`；
- OK 和捏合等动作有规则补充判断；
- 手势需要连续稳定若干帧才会触发；
- 每种动作有冷却时间，避免短时间重复触发；
- 左/右翻页在模型判断为指向类后，使用食指方向做二次修正；
- 高风险动作使用更长稳定帧，减少误开始、误退出、误进鼠标模式。

## 推荐展示流程

1. 启动 PPT/WPS 并进入放映；
2. 运行：

```bash
python main.py --headless --hud
```

3. 展示核心动作：开始放映、上一页、下一页、退出放映；
4. 说明 HUD 中的手势、置信度和识别后端；
5. 如需展示调试过程，再使用：

```bash
python main.py --preview small --hud
```
