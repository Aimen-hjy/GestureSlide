# 基于 MediaPipe 的无接触手势 PPT 控制系统

通过摄像头实时识别手势，实现无接触控制 PPT 演示。支持翻页、放映、鼠标移动点击、音量调节等功能。

本项目采用课堂展示友好的成熟架构：

```text
MediaPipe Hands 预训练视觉模型
        ↓
21 个 3D 手部关键点
        ↓
69 维结构化手势特征
        ↓
轻量分类器自动对比选择
        ↓
置信度阈值 + 连续帧稳定器 + 状态机动作控制
```

## 功能一览

### 普通模式
| 手势 | 动作 | 说明 |
|------|------|------|
| 👈 食指左指 | ← | 上一页 |
| 👉 食指右指 | → | 下一页 |
| 🖐 五指张开 | Esc | 退出 PPT 放映 |
| 👌 OK 手势 | F5 | 开始 PPT 放映 |
| ☝️ 食指朝上 | 切换模式 | 进入鼠标控制模式 |
| ✌️ V 手势朝上 | 音量+ | 调高系统音量 |
| ✌️ V 手势朝下 | 音量- | 调低系统音量 |
| 👍 竖拇指 | F5 | 开始 PPT 放映 |

### 鼠标模式
| 手势 | 动作 | 说明 |
|------|------|------|
| ☝️ 移动食指 | 移动光标 | 食指控制鼠标指针 |
| 🤏 拇指食指捏合 | 左键点击 | 模拟鼠标点击 |
| 🖐 五指张开 | 退出 | 返回普通模式 |

## 项目亮点

- **两阶段识别**：MediaPipe 负责视觉关键点，轻量模型负责手势语义分类。
- **模型自动选择**：支持 MLP、SVM、RandomForest、ExtraTrees、HistGradientBoosting、KNN 对比。
- **特征级数据增强**：模拟角度、距离和关键点抖动变化，不需要下载超大图片数据集。
- **分组评估**：按采集 session / source group 留出测试集，避免相邻帧泄漏导致准确率虚高。
- **可解释界面**：实时显示识别置信度、识别后端、动作预判和系统模式。
- **双状态机**：普通 PPT 控制模式 + 鼠标控制模式，适合课堂现场演示。

## 系统结构

```text
GestureSlide/
├── main.py                    # 主程序入口 + 采集模式
├── config.py                  # 全局配置参数
├── hand_detector.py           # MediaPipe 手部检测封装
├── gesture_model.py           # 特征提取、合成数据、ML 基础训练
├── training_pipeline.py       # 数据加载、审计、增强、分组训练
├── gesture_classifier.py      # ML/规则分类、置信度、防抖
├── ppt_controller.py          # PPT 控制核心状态机 + UI overlay
├── action_controller.py       # 键盘/鼠标动作执行器
├── train_model.py             # 单模型训练入口
├── tools/
│   ├── audit_training_data.py # 数据质量审计
│   ├── compare_models.py      # 多模型对比并保存最优模型
│   ├── import_image_folder.py # 通用图片文件夹数据导入
│   ├── download_rps_dataset.py# 轻量 RPS 数据集下载
│   └── run_project_pipeline.sh# 成熟训练流水线
├── docs/
│   ├── CLASSROOM_DEMO.md      # 课堂展示说明
│   └── DATA_PIPELINE.md       # 数据流程说明
├── gesture_model.joblib       # 自动生成，训练好的分类器
├── gesture_scaler.joblib      # 自动生成，特征标准化器
├── requirements.txt
└── README.md
```

## 环境要求

- Python 3.10+
- Windows / macOS / Linux
- 摄像头

## 快速开始

### 1. 安装依赖

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Windows
pip install -r requirements.txt
```

Debian/Ubuntu 若提示 `externally-managed-environment`，不要使用系统级 `pip`，应使用虚拟环境。

### 2. 训练成熟版模型

推荐先使用现有数据 + 特征增强 + 多模型对比：

```bash
bash tools/run_project_pipeline.sh
```

如果希望额外使用轻量外部数据集补充 `FIST / OPEN_PALM / PEACE_UP`：

```bash
USE_RPS=1 bash tools/run_project_pipeline.sh
```

该流程会自动执行：

```text
语法检查 → 数据审计 → 可选 RPS 导入 → 特征增强 → 多模型对比 → 保存最优模型
```

也可以手动运行模型对比：

```bash
python tools/compare_models.py \
  --data training_data/ \
  --split-strategy group \
  --augment \
  --augment-factor 2 \
  --balance-target 800 \
  --metric macro_f1
```

### 3. 运行程序

```bash
python main.py
```

## 使用步骤

1. 确保摄像头正常工作。
2. 打开 PPT 文件，并让 PPT / Impress 窗口位于前台。
3. 运行本程序。
4. 使用手势控制 PPT：
   - OK 手势或竖拇指：开始放映
   - 食指左指 / 右指：上一页 / 下一页
   - 食指朝上：进入鼠标模式
   - 食指移动：控制鼠标
   - 捏合：鼠标点击
   - 五指张开：退出鼠标模式或退出放映
   - 按 `q` 退出程序

## 数据采集与训练

### 采集模式

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

每种手势建议分多次、在不同距离、角度、光照下录制；每类至少保留 3 个独立采集文件。标签 `NONE` 应录制“有手但不属于任何命令”的随机姿势，而不是空画面。切换标签前先暂停录制，避免过渡帧被标错。

### 数据审计

```bash
python tools/audit_training_data.py training_data/
```

重点关注：

- 是否缺少 `NONE`
- 是否类别严重不平衡
- 每个类别是否至少有多个独立 group/session

### 单模型训练

```bash
python train_model.py --real training_data/ --augment --augment-factor 2 --balance-target 800
```

### 多模型比较并选择最佳模型

```bash
python tools/compare_models.py --data training_data/ --augment --balance-target 800
```

## 轻量外部数据

不建议直接下载 HaGRID 全量类别，因为单类压缩包可达几十 GB，而本项目不是图片 CNN 训练。更合理的路线是：

```text
本地真实数据为主 + 特征增强 + 少量轻量外部图片补充
```

可选 RPS 数据集导入：

```bash
python tools/download_rps_dataset.py --output-dir datasets/rps
python tools/import_image_folder.py \
  --dataset-root datasets/rps/rps \
  --map rock=FIST paper=OPEN_PALM scissors=PEACE_UP \
  --source-name rps_train \
  --output-dir training_data/imported
```

## 技术原理

### 手部检测

使用 MediaPipe Hands 模型实时检测 21 个手部关键点坐标。

### 特征构造

每帧手部数据转换为 69 维特征：

- 63 维：21 个关键点相对手腕的 `(x, y, z)` 坐标
- 5 维：拇指、食指、中指、无名指、小指是否伸展
- 1 维：拇指尖与食指尖距离

### 手势识别

后端分类器可以是 MLP、SVM、RandomForest、ExtraTrees、HistGradientBoosting 或 KNN。运行时通过 `predict_proba` 得到类别概率，置信度低于阈值时返回 `NONE`，减少误触发。

### 稳定性机制

- 手势需持续 N 帧才确认触发
- 每种动作有独立冷却时间
- 鼠标移动采用平滑和死区过滤
- 捏合和 OK 手势使用规则检测补充 ML 分类
- UI 显示置信度和下一步动作，便于调试和展示

## Linux / Wayland 注意事项

`pyautogui` 向当前获得焦点的窗口发送键盘事件。运行后应确保 PPT / Impress 放映窗口处于前台；预览窗口获得焦点时，翻页键可能发送到预览窗口。Wayland 会限制全局键鼠模拟，遇到控制无效时可改用 Xorg 会话，或实现针对 LibreOffice / PowerPoint 的专用控制后端。

## 更多说明

- 课堂展示与答辩说明：`docs/CLASSROOM_DEMO.md`
- 数据导入与训练说明：`docs/DATA_PIPELINE.md`
