# GestureSlide 课堂展示成熟版说明

本项目定位为大学课堂中的小型实时交互系统，而不是研究级大模型。优化目标是：保留原项目主框架，提升准确性、稳定性、可解释性和现场展示观感。

## 一句话定位

GestureSlide 采用“两阶段轻量识别架构”：

```text
MediaPipe Hands 预训练视觉模型 → 21 个 3D 手部关键点 → 69 维结构化特征 → 轻量分类器 → 状态机动作控制
```

因此项目不是“只有一个简单 MLP”，而是把复杂视觉感知交给 MediaPipe，后端再用轻量模型做手势语义分类。这种设计适合实时控制、低延迟和课堂现场演示。

## 成熟版的核心改进

### 1. 模型选择不再固定为 MLP

新增 `tools/compare_models.py`，可自动比较多种轻量分类器：

- MLP
- SVM
- RandomForest
- ExtraTrees
- HistGradientBoosting
- KNN

它会使用相同训练集、相同测试集、相同特征增强策略进行公平比较，并根据 `macro_f1` 默认选择最优模型，保存为：

```text
gesture_model.joblib
gesture_scaler.joblib
```

推荐命令：

```bash
python tools/compare_models.py \
  --data training_data/ \
  --split-strategy group \
  --augment \
  --augment-factor 2 \
  --balance-target 800 \
  --metric macro_f1
```

答辩时可以说：

> 我们不是直接假定 MLP 最优，而是对多种轻量模型进行了实验对比，最终选择兼顾准确率、鲁棒性和实时性的模型。

### 2. 本地特征增强训练

项目已经采集到数千条真实 MediaPipe 手部特征。与其下载数百 GB 外部原图，不如直接在 69 维特征上做轻量增强：

- 小角度旋转：模拟手势略微倾斜
- 随机缩放：模拟手离摄像头远近变化
- 坐标噪声：模拟 MediaPipe 关键点抖动
- 弱类过采样：缓解 `NONE` 等类别样本过少

增强只作用于训练集，不作用于测试集，因此不会把增强后的相似样本泄漏到测试集。

### 3. 质量更高但轻量的数据路线

不再建议直接下载 HaGRID 全量类别。HaGRID 质量高，但单类原图压缩包可达几十 GB。当前项目不是图片 CNN，而是 MediaPipe 特征 + 轻量分类器，因此全量图片成本过高。

成熟版保留两条数据扩展路线：

#### 路线 A：通用图片文件夹导入器

新增 `tools/import_image_folder.py`，可导入任何按类别放置的图片数据集：

```text
dataset_root/
  rock/
  paper/
  scissors/
```

示例映射：

```bash
python tools/import_image_folder.py \
  --dataset-root datasets/rps/rps \
  --map rock=FIST paper=OPEN_PALM scissors=PEACE_UP \
  --source-name rps_train \
  --output-dir training_data/imported
```

#### 路线 B：轻量 RPS 数据集

新增 `tools/download_rps_dataset.py`，可下载 Rock-Paper-Scissors 数据集。它只补充 3 个关键类：

```text
rock     → FIST
paper    → OPEN_PALM
scissors → PEACE_UP
```

这个数据集不能替代本地数据，但适合作为轻量高质量外部补充。

### 4. 置信度可视化

运行界面现在会显示：

- 当前识别手势
- 模型置信度条
- 当前识别后端：`ml` / `rule` / `rule-pinch`
- 下一步将触发的动作提示

这能让演示从“看起来像按键脚本”变成“可解释的人机交互系统”。老师可以直接看到模型何时有把握、何时因为置信度不足而不触发。

### 5. 动作预判展示面板

左上角新增展示面板：

```text
GestureSlide
MediaPipe Hands + MLP Classifier
Action: Next slide / Start slideshow / Move cursor ...
```

底部状态栏还会显示：

```text
Next action: Previous slide / Enter mouse mode / Volume up ...
```

观众不用看代码，也能理解系统当前处于什么状态、识别到了什么、准备执行什么动作。

## 一键成熟训练流程

### 不使用外部图片，仅用现有数据增强 + 模型对比

```bash
bash tools/run_project_pipeline.sh
```

### 使用轻量 RPS 外部数据补充

```bash
USE_RPS=1 bash tools/run_project_pipeline.sh
```

该脚本会执行：

```text
语法检查 → 数据审计 → 可选 RPS 导入 → 特征增强 → 多模型对比 → 保存最优模型
```

## 推荐演示流程

1. 打开 PPT，并让 PPT 窗口获得焦点。
2. 运行：

```bash
python main.py
```

3. 展示普通模式：
   - 竖拇指 / OK：开始放映
   - 左指 / 右指：翻页
   - V 朝上 / 朝下：音量调节
   - 五指张开：退出放映

4. 展示鼠标模式：
   - 食指朝上：进入鼠标模式
   - 食指移动：移动鼠标
   - 捏合：点击
   - 五指张开：退出鼠标模式

5. 重点讲解界面上的置信度条、动作预判和模型对比训练结果。

## 答辩时可以强调的技术特点

- 使用 MediaPipe Hands 实时提取 21 个手部关键点。
- 将关键点、手指状态、捏合距离组成 69 维结构化特征。
- 分类器不是固定死的，系统可自动比较 MLP、SVM、RandomForest、ExtraTrees、HGB 等轻量模型。
- 使用 group split 避免同一采集 session 的相邻帧同时进入训练和测试。
- 使用特征增强提升对角度、距离、关键点抖动的鲁棒性。
- 使用置信度阈值、连续帧稳定器和冷却时间减少误触发。
- 普通 PPT 控制和鼠标控制是两个状态机模式，便于扩展。
- 界面具备实时置信度和动作预判，可解释性强，适合课堂展示。
