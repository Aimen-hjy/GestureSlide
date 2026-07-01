# 训练数据与外部数据导入流程

本项目运行时模型使用 MediaPipe 21 个手部关键点提取 69 维特征：

- 63 维：21 个关键点相对手腕的 `(x, y, z)` 坐标
- 5 维：拇指、食指、中指、无名指、小指是否伸展
- 1 维：拇指尖与食指尖距离

因此外部图片数据不能直接用原始标签训练；必须用本项目同一套 MediaPipe + `extract_features()` 重新提取特征。

## 1. 审计现有采集数据

```bash
python tools/audit_training_data.py training_data/
```

重点查看：

- 是否缺少 `NONE`
- 每类样本是否过少或严重不平衡
- 每类是否至少来自 2 个以上 group/session

旧数据没有 `source_group` 时，工具会把每个 `session_*.json` 文件当成一个 group。

## 2. 使用更可靠的分组评估训练

推荐：

```bash
python train_model.py --real training_data/ --split-strategy group
```

这会把整个采集文件或外部数据的同一 `source_group` 留给测试集，避免同一段视频的相邻帧同时进入训练集和测试集。

如需对比旧结果，可运行：

```bash
python train_model.py --real training_data/ --split-strategy random
```

`random` 可能得到更高准确率，但通常会高估真实实机表现。

## 3. 导入 HaGRID / HaGRIDv2

HaGRIDv2 很大，不建议一次性下载完整数据。先只下载需要的类别，例如：

- `fist` → `FIST`
- `palm` / `stop` → `OPEN_PALM`
- `like` → `THUMB_UP`
- `peace` / `two_up` → `PEACE_UP`
- `peace_inverted` / `two_up_inverted` → `PEACE_DOWN`
- `three` / `three2` / `three3` → `THREE_FINGERS`
- `no_gesture` → `NONE`
- `point` / `one` → 通过食指关键点方向自动二次标注为 `LEFT_POINT` / `RIGHT_POINT` / `POINT_INDEX` / `DOWN_POINT`

按 HaGRID 官方下载脚本下载并解压后，目录应类似：

```text
hagrid_dataset/
  fist/
    00000000.jpg
  palm/
    00000000.jpg
hagrid_annotations/
  train/
    fist.json
    palm.json
  val/
  test/
```

导入示例：

```bash
python tools/import_hagrid.py \
  --dataset-root datasets/hagrid/hagrid_dataset \
  --annotations-root datasets/hagrid/hagrid_annotations \
  --splits train val \
  --targets fist palm stop like peace peace_inverted three no_gesture \
  --max-per-class 1000 \
  --output-dir training_data/imported
```

如果要导入 `point` / `one` 并自动按食指方向拆分：

```bash
python tools/import_hagrid.py \
  --dataset-root datasets/hagrid/hagrid_dataset \
  --annotations-root datasets/hagrid/hagrid_annotations \
  --splits train \
  --targets point one \
  --include-direction-classes \
  --max-per-class 1000 \
  --output-dir training_data/imported
```

导入脚本默认会做水平镜像，保持外部图片和实时摄像头自拍视角一致。若你运行时关闭了镜像，可加：

```bash
--no-flip-horizontal
```

## 4. 推荐数据比例

第一阶段建议：

- 40%：本机摄像头多 session 数据
- 40%：HaGRID 映射数据
- 20%：`NONE` 和困难负样本

`NONE` 应该是“画面中有手，但不属于任何命令”的随机姿势，而不是空画面。没有检测到手时程序本来就不会分类，空画面不能训练静态手势分类器。

## 5. 不要提交大型数据

`datasets/`、`external/`、模型缓存等目录应保留在本地，不要提交到 Git。仓库只保存转换脚本和小规模示例说明。
