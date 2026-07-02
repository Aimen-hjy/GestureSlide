# 本地训练数据与模型训练流程

最终演示模型使用本地摄像头采集数据训练。本文档只描述最终使用的本地数据路线。

## 1. 69 维手势特征

运行时每帧手部数据都会转换为 69 维结构化特征：

- 63 维：21 个关键点相对手腕的 `(x, y, z)` 坐标；
- 5 维：拇指、食指、中指、无名指、小指是否伸展；
- 1 维：拇指尖与食指尖距离。

训练数据中的每条样本都保存为：

```json
{
  "features": [...],
  "label": "RIGHT_POINT",
  "label_id": 4
}
```

## 2. 采集数据

```bash
python main.py --collect
```

采集界面按键：

| 按键 | 功能 |
|------|------|
| 0-9 / a | 选择手势标签 |
| 空格 | 开始/停止录制 |
| s | 保存当前数据 |
| q | 保存并退出 |

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

- 每个手势分多次采集，形成多个独立 `session_*.json` 文件；
- 同一手势覆盖不同距离、角度和光照；
- `NONE` 应采集“有手但不属于任何命令”的随机姿势，不是空画面；
- 切换标签前先暂停录制，避免过渡帧被标错。

## 3. 审计数据

```bash
python tools/audit_training_data.py training_data/
```

重点查看：

- 每类样本数量；
- 是否缺少 `NONE`；
- 是否类别严重不平衡；
- 每类是否来自多个独立 session/group。

旧数据没有 `source_group` 时，工具会把每个 `session_*.json` 文件当成一个 group。

## 4. 训练最终模型

推荐使用默认流水线：

```bash
bash tools/run_project_pipeline.sh
```

该流程会执行：

```text
语法检查 → 数据审计 → 特征增强 → 类别平衡 → 多模型比较 → 保存最佳模型
```

也可以手动运行：

```bash
python tools/compare_models.py \
  --data training_data/ \
  --split-strategy group \
  --augment \
  --augment-factor 2 \
  --balance-target 800 \
  --metric macro_f1
```

训练完成后生成：

```text
gesture_model.joblib
gesture_scaler.joblib
```

## 5. 为什么使用 group split

本项目数据通常按连续帧采集。如果随机切分，某个手势 session 的相邻帧可能同时出现在训练集和测试集，导致测试结果虚高。

`group` 划分会尽量按采集文件或 `source_group` 留出测试集，更接近真实泛化效果。

```bash
python train_model.py --real training_data/ --split-strategy group
```

## 6. 特征增强与类别平衡

最终训练保留特征级增强。

增强包括：

- 小角度旋转，模拟手势略微倾斜；
- 随机缩放，模拟手与摄像头距离变化；
- 坐标噪声，模拟 MediaPipe 关键点抖动；
- 对样本较少的类别做平衡增强。

增强只作用于训练集，不作用于测试集。

## 7. 左右指几何方向测试

项目提供离线测试工具，用于检查食指方向几何规则在已有数据上的表现：

```bash
python tools/evaluate_geometry_direction.py training_data/
```

这个工具不会训练模型，只会读取已有 69 维特征，统计几何规则对 `LEFT_POINT` / `RIGHT_POINT` 的识别情况，以及是否会把其他类别误判成翻页命令。

最终运行时采用保守策略：几何规则只修正模型已经判断为“指向类”的手势方向，不会单独把 `NONE` 等非指向类改成翻页命令。

## 8. 最终数据原则

最终演示模型建议保持：

```text
本地摄像头数据为主
+ 特征级增强
+ 类别平衡
+ group split 评估
+ 多模型比较
```
