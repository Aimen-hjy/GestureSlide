"""
手势识别ML模型 — 合成数据生成 + 训练 + 推理

使用 MLP 神经网络替代硬编码规则判断静态手势。
特征: 69维 (21个关键点相对手腕坐标×3 + 5手指状态 + 1拇指食指距离)
输出: 11类静态手势
"""

import os
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 手势枚举值映射 (与 gesture_classifier.py 的 Gesture 枚举对应)
# 仅包含静态手势 (可ML分类的)
GESTURE_NAMES = [
    "NONE",
    "OPEN_PALM",
    "FIST",
    "LEFT_POINT",
    "RIGHT_POINT",
    "DOWN_POINT",
    "POINT_INDEX",
    "THUMB_UP",
    "PEACE_UP",
    "PEACE_DOWN",
    "THREE_FINGERS",
]

GESTURE_TO_ID = {name: i for i, name in enumerate(GESTURE_NAMES)}
ID_TO_GESTURE = {i: name for i, name in enumerate(GESTURE_NAMES)}
N_CLASSES = len(GESTURE_NAMES)
FEATURE_DIM = 69
COORD_FEATURE_DIM = 63


def _print_evaluation(y_test: np.ndarray, y_pred: np.ndarray,
                      model: MLPClassifier) -> None:
    """按模型实际包含的类别输出分类报告与混淆矩阵。"""
    labels = np.asarray(model.classes_, dtype=np.int32)
    target_names = [
        ID_TO_GESTURE.get(int(label), str(label))
        for label in labels
    ]

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels))

# ──────────────────────────────────────────────────────────
# 手部骨架模板 (右手, 所有手指伸展的"理想"姿态)
# 坐标归一化到 [0,1], 手腕为原点, MediaPipe坐标系
# ──────────────────────────────────────────────────────────

class HandTemplate:
    """手部骨架模板 — 定义21个关键点在"五指张开"状态下的位置"""

    # 关节索引
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    # 每根手指的关节点索引链 [MCP, PIP, DIP, TIP]
    FINGER_CHAIN = {
        "thumb":  [THUMB_MCP,  THUMB_IP,  THUMB_TIP],
        "index":  [INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP],
        "middle": [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP],
        "ring":   [RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP],
        "pinky":  [PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP],
    }

    # 伸展姿态模板 (21点 × 3坐标, 相对手腕)
    # 这些值模拟一个正常大小、手指向上的手
    BASE_LANDMARKS = np.array([
        # 索引0: 手腕 (原点)
        0.000,  0.000,  0.000,
        # 索引1-4: 拇指 (CMC → MCP → IP → TIP)
        -0.040, -0.020, -0.060,
        -0.080, -0.070, -0.080,
        -0.100, -0.120, -0.090,
        -0.110, -0.170, -0.100,
        # 索引5-8: 食指 (MCP → PIP → DIP → TIP)
        -0.030, -0.090, -0.020,
        -0.025, -0.200, -0.015,
        -0.022, -0.300, -0.012,
        -0.020, -0.380, -0.010,
        # 索引9-12: 中指 (MCP → PIP → DIP → TIP)
         0.010, -0.100, -0.020,
         0.012, -0.220, -0.015,
         0.013, -0.330, -0.012,
         0.014, -0.410, -0.010,
        # 索引13-16: 无名指 (MCP → PIP → DIP → TIP)
         0.045, -0.085,  0.005,
         0.048, -0.195,  0.008,
         0.050, -0.290,  0.010,
         0.051, -0.365,  0.012,
        # 索引17-20: 小指 (MCP → PIP → DIP → TIP)
         0.080, -0.060,  0.030,
         0.085, -0.145,  0.035,
         0.088, -0.220,  0.038,
         0.090, -0.275,  0.040,
    ], dtype=np.float64)

    @classmethod
    def get_open_palm(cls) -> np.ndarray:
        """五指张开模板"""
        return cls.BASE_LANDMARKS.copy()

    @classmethod
    def get_bent_finger(cls, landmarks: np.ndarray,
                        finger_name: str) -> np.ndarray:
        """
        将指定手指弯曲 (缩短指尖到手腕的距离, 模拟卷曲)

        弯曲算法: 将 PIP/DIP/TIP 向 MCP 收缩, 使指尖离手腕更近
        """
        if finger_name == "thumb":
            # 拇指弯曲: 将IP和TIP向MCP收缩
            chain = cls.FINGER_CHAIN["thumb"]      # [MCP, IP, TIP]
            mcp_idx, ip_idx, tip_idx = chain
            # 收缩系数: MCP→IP 和 IP→TIP 都缩短
            for idx in chain[1:]:  # IP, TIP
                i3 = idx * 3
                # 朝 MCP 方向收缩 70%
                lm = landmarks[i3:i3+3].copy()
                mcp = landmarks[mcp_idx*3:(mcp_idx+1)*3]
                # 向MCP靠近
                new_lm = mcp + (lm - mcp) * 0.25
                landmarks[i3:i3+3] = new_lm
            # 拇指弯曲时调整x坐标使检测更明显
            landmarks[ip_idx*3] += 0.02     # IP x 偏移
            landmarks[tip_idx*3] += 0.04    # TIP x 靠近IP
        else:
            chain = cls.FINGER_CHAIN[finger_name]   # [MCP, PIP, DIP, TIP]
            mcp_idx = chain[0]
            mcp = landmarks[mcp_idx*3:(mcp_idx+1)*3]
            for idx in chain[2:]:  # DIP, TIP (保留PIP作为中间)
                i3 = idx * 3
                lm = landmarks[i3:i3+3].copy()
                # 弯曲: 向MCP靠拢 80%
                new_lm = mcp + (lm - mcp) * 0.30
                # 额外正向z偏移模拟手指向内弯曲
                new_lm[2] += 0.08
                landmarks[i3:i3+3] = new_lm
            # PIP也稍微收缩
            pip_idx = chain[1]
            pip = landmarks[pip_idx*3:(pip_idx+1)*3]
            landmarks[pip_idx*3:(pip_idx+1)*3] = mcp + (pip - mcp) * 0.65

        return landmarks

    @classmethod
    def compute_finger_states(cls, landmarks: np.ndarray) -> np.ndarray:
        """从关键点计算5根手指的伸展状态 (模拟 hand_detector.py 的算法)"""
        wrist = landmarks[0:3]
        finger_chains = [
            ("thumb",  cls.THUMB_MCP,  cls.THUMB_IP,  cls.THUMB_TIP),
            ("index",  cls.INDEX_MCP,  cls.INDEX_PIP,  cls.INDEX_TIP),
            ("middle", cls.MIDDLE_MCP, cls.MIDDLE_PIP, cls.MIDDLE_TIP),
            ("ring",   cls.RING_MCP,   cls.RING_PIP,   cls.RING_TIP),
            ("pinky",  cls.PINKY_MCP,  cls.PINKY_PIP,  cls.PINKY_TIP),
        ]
        states = []
        for name, mcp_idx, pip_idx, tip_idx in finger_chains:
            tip = landmarks[tip_idx*3:(tip_idx+1)*3]
            pip = landmarks[pip_idx*3:(pip_idx+1)*3]

            tip_dist = float(np.linalg.norm(tip - wrist))
            pip_dist = float(np.linalg.norm(pip - wrist))

            if name == "thumb":
                # 拇指: 比较 |TIP.x - IP.x| (模拟手部检测器逻辑)
                ip = landmarks[cls.THUMB_IP*3:(cls.THUMB_IP+1)*3]
                states.append(abs(tip[0] - ip[0]) > 0.04)
            else:
                states.append(tip_dist > pip_dist * 0.92)
        return np.array(states, dtype=np.float64)


def generate_synthetic_data(n_per_class: int = 1000,
                            noise_std: float = 0.015,
                            seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    为每种手势生成合成训练数据

    对每种手势:
    1. 从五指张开模板开始
    2. 弯曲不该伸展的手指
    3. 添加随机噪声/缩放/旋转
    4. 提取特征向量

    Args:
        n_per_class: 每类手势生成样本数
        noise_std: 高斯噪声标准差
        seed: 随机种子

    Returns:
        X: 特征矩阵 (n_samples × 69)
        y: 标签数组 (n_samples,)
    """
    rng = np.random.RandomState(seed)

    # 每种手势的手指伸展配置: [拇指, 食指, 中指, 无名指, 小指]
    GESTURE_FINGERS = {
        "OPEN_PALM":     [True,  True,  True,  True,  True],
        "FIST":          [False, False, False, False, False],
        "LEFT_POINT":    [False, True,  False, False, False],
        "RIGHT_POINT":   [False, True,  False, False, False],
        "DOWN_POINT":    [False, True,  False, False, False],
        "POINT_INDEX":   [False, True,  False, False, False],
        "THUMB_UP":      [True,  False, False, False, False],
        "PEACE_UP":      [False, True,  True,  False, False],
        "PEACE_DOWN":    [False, True,  True,  False, False],
        "THREE_FINGERS": [False, True,  True,  True,  False],
        "NONE":          None,
    }

    # 指尖方向偏移: (tip_y_shift) 正值=向下
    DIRECTION_SHIFT = {
        "LEFT_POINT":  (-0.18, 0.0),
        "RIGHT_POINT": (0.18, 0.0),
        "DOWN_POINT":  (0.0, 0.25),
        "POINT_INDEX": (0.0, 0.0),
        "PEACE_UP":    (0.0, 0.0),
        "PEACE_DOWN":  (0.0, 0.25),
    }

    X_list, y_list = [], []

    for gesture_name, finger_config in GESTURE_FINGERS.items():
        label_id = GESTURE_TO_ID[gesture_name]

        for i in range(n_per_class):
            # 1. 从张开模板开始
            landmarks = HandTemplate.get_open_palm()

            # 2. 根据手势弯曲对应手指
            if gesture_name == "NONE":
                # NONE: 随机选择一个非标准的手指组合
                random_fingers = rng.choice([True, False], size=5)
                # 避免碰巧生成已定义的手势 (概率很低, 但确保多样性)
                for name in ["index", "middle", "ring", "pinky", "thumb"]:
                    if rng.random() < 0.5:
                        # 部分弯曲
                        landmarks = HandTemplate.get_bent_finger(
                            landmarks, name)
            else:
                for finger_name, should_extend in zip(
                        ["thumb", "index", "middle", "ring", "pinky"],
                        finger_config):
                    if not should_extend:
                        landmarks = HandTemplate.get_bent_finger(
                            landmarks, finger_name)

            # 2.5 指尖方向偏移 (点指/V手势)
            if gesture_name in DIRECTION_SHIFT:
                dx, dy = DIRECTION_SHIFT[gesture_name]
                # 食指
                idx_tip = HandTemplate.INDEX_TIP * 3
                landmarks[idx_tip] += dx
                landmarks[idx_tip + 1] += dy
                # V手势需要同时偏移中指
                if gesture_name.startswith("PEACE"):
                    mid_tip = HandTemplate.MIDDLE_TIP * 3
                    landmarks[mid_tip] += dx
                    landmarks[mid_tip + 1] += dy

            # 3. 添加随机变化
            # a) 高斯噪声 (模拟检测抖动)
            noise = rng.normal(0, noise_std, size=63)
            landmarks += noise

            # b) 随机缩放 (±10%)
            scale = 1.0 + rng.normal(0, 0.05)
            landmarks *= scale

            # c) 全局旋转偏移 (绕y轴轻微旋转, 模拟手部倾斜)
            theta = rng.normal(0, 0.15)  # ~±8°
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            for j in range(21):
                x, y, z = landmarks[j*3], landmarks[j*3+1], landmarks[j*3+2]
                landmarks[j*3] = x * cos_t - z * sin_t
                landmarks[j*3+2] = x * sin_t + z * cos_t

            # d) 随机平移偏移 (模拟手在画面不同位置)
            offset_x = rng.normal(0, 0.08)
            offset_y = rng.normal(0, 0.08)
            for j in range(21):
                landmarks[j*3] += offset_x
                landmarks[j*3+1] += offset_y

            # 4. 提取特征
            features = extract_features_from_landmarks(landmarks, rng=rng)
            X_list.append(features)
            y_list.append(label_id)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def extract_features_from_landmarks(landmarks: np.ndarray,
                                    rng: np.random.RandomState | None = None
                                    ) -> np.ndarray:
    """
    从关键点坐标提取69维特征向量

    landmarks: 63维展平数组 (21点 × 3坐标)
    返回: 69维特征向量
      [0:63]   - 手腕相对坐标 (wx, wy, wz 归零)
      [63:68]  - 5根手指状态 (0.0/1.0)
      [68]     - 拇指-食指距离
    """
    # 手腕相对化 (手腕坐标归零)
    wrist = landmarks[0:3].copy()
    rel_landmarks = landmarks.copy()
    rel_landmarks[0:3] = 0.0     # 手腕本身
    for j in range(1, 21):
        rel_landmarks[j*3:j*3+3] -= wrist

    # 计算手指状态
    finger_states = HandTemplate.compute_finger_states(landmarks)

    # 计算拇指-食指距离 (2D)
    thumb_tip = landmarks[HandTemplate.THUMB_TIP*3:
                          HandTemplate.THUMB_TIP*3+2]   # 只取 x,y
    index_tip = landmarks[HandTemplate.INDEX_TIP*3:
                          HandTemplate.INDEX_TIP*3+2]
    pinch_dist = float(np.linalg.norm(thumb_tip - index_tip))

    # 拼成69维
    features = np.concatenate([
        rel_landmarks,           # 63
        finger_states,           # 5
        [pinch_dist],            # 1
    ])
    return features.astype(np.float64)


def extract_features(hand_data: dict) -> np.ndarray:
    """
    运行时从 hand_data 提取特征 (与 extract_features_from_landmarks 接口不同)

    hand_data: HandDetector.detect_hand() 返回的字典,
    含 landmarks (list of {x,y,z}), finger_states (list[bool]),
    thumb_index_dist (float)

    返回: 69维特征向量
    """
    landmarks_list = hand_data["landmarks"]
    # 展平为63维
    landmarks_flat = np.zeros(63, dtype=np.float64)
    for i, lm in enumerate(landmarks_list):
        landmarks_flat[i*3] = lm["x"]
        landmarks_flat[i*3+1] = lm["y"]
        landmarks_flat[i*3+2] = lm["z"]

    # 手腕相对化
    wrist = landmarks_flat[0:3].copy()
    rel_landmarks = landmarks_flat.copy()
    rel_landmarks[0:3] = 0.0
    for j in range(1, 21):
        rel_landmarks[j*3:j*3+3] -= wrist

    # 手指状态
    finger_states_raw = hand_data.get("finger_states", [False]*5)
    finger_states = np.array([1.0 if f else 0.0 for f in finger_states_raw],
                             dtype=np.float64)

    # 拇指-食指距离
    pinch_dist = float(hand_data.get("thumb_index_dist", 1.0))

    features = np.concatenate([rel_landmarks, finger_states, [pinch_dist]])
    return features.astype(np.float64)


def load_real_data(data_dir: str = "training_data") -> tuple[np.ndarray, np.ndarray]:
    """
    从采集的JSON文件中加载真实训练数据

    Args:
        data_dir: 存放 session_*.json 的目录

    Returns:
        X: 特征矩阵 (n_samples × 69)
        y: 标签数组 (n_samples,)
    """
    import glob
    data_path = Path(data_dir)
    json_files = sorted(glob.glob(str(data_path / "session_*.json")))

    if not json_files:
        raise FileNotFoundError(
            f"No session_*.json files found in {data_dir}. "
            f"Run 'python main.py --collect' first.")

    X_list, y_list = [], []
    class_counts = {name: 0 for name in GESTURE_NAMES}

    for path in json_files:
        with open(path, 'r', encoding='utf-8') as f:
            samples = json.load(f)

        if not isinstance(samples, list):
            raise ValueError(f"{path}: top-level JSON value must be a list")

        for sample_index, s in enumerate(samples):
            if not isinstance(s, dict):
                raise ValueError(
                    f"{path}: sample {sample_index} must be an object")

            try:
                features = np.asarray(s["features"], dtype=np.float64)
                label_id = int(s["label_id"])
                label_name = str(s["label"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: invalid sample {sample_index}: {exc}") from exc

            if features.shape != (FEATURE_DIM,):
                raise ValueError(
                    f"{path}: sample {sample_index} has feature shape "
                    f"{features.shape}, expected ({FEATURE_DIM},)")
            if not np.all(np.isfinite(features)):
                raise ValueError(
                    f"{path}: sample {sample_index} contains NaN or Inf")
            if label_id not in ID_TO_GESTURE:
                raise ValueError(
                    f"{path}: sample {sample_index} has unknown label_id "
                    f"{label_id}")
            if ID_TO_GESTURE[label_id] != label_name:
                raise ValueError(
                    f"{path}: sample {sample_index} label mismatch: "
                    f"id {label_id} means {ID_TO_GESTURE[label_id]}, "
                    f"but label is {label_name}")

            X_list.append(features)
            y_list.append(label_id)
            class_counts[label_name] += 1

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.int32)

    print(f"Loaded {len(X)} samples from {len(json_files)} file(s):")
    for name in GESTURE_NAMES:
        if class_counts[name] > 0:
            print(f"  {name}: {class_counts[name]}")

    return X, y


class GestureModel:
    """手势识别MLP模型 — 加载/推理/训练"""

    def __init__(self, model_path: str | None = None,
                 scaler_path: str | None = None):
        self.model: MLPClassifier | None = None
        self.scaler: StandardScaler | None = None
        self._loaded = False

        if model_path and os.path.exists(model_path):
            self.load(model_path, scaler_path)

    def load(self, model_path: str, scaler_path: str | None = None):
        """从磁盘加载模型和标准化器"""
        self.model = joblib.load(model_path)
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        self._loaded = True
        logging.info(f"ML model loaded from {model_path}")

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        """
        预测手势类别

        Args:
            features: 69维特征向量

        Returns:
            (gesture_name, confidence) e.g. ("PEACE", 0.92)
        """
        if self.model is None:
            return ("NONE", 0.0)

        features = np.asarray(features, dtype=np.float64)
        if features.shape != (FEATURE_DIM,):
            raise ValueError(
                f"Expected {FEATURE_DIM} features, got {features.shape}")

        X = features.reshape(1, -1).copy()
        if self.scaler is not None:
            X[:, :COORD_FEATURE_DIM] = self.scaler.transform(
                X[:, :COORD_FEATURE_DIM])

        probas = self.model.predict_proba(X)[0]
        max_idx = int(np.argmax(probas))
        max_prob = float(probas[max_idx])

        class_id = int(self.model.classes_[max_idx])
        gesture_name = ID_TO_GESTURE.get(class_id, "NONE")
        return (gesture_name, max_prob)

    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    @staticmethod
    def train_from_synthetic(n_per_class: int = 1500,
                             noise_std: float = 0.015,
                             test_size: float = 0.2,
                             model_path: str = "gesture_model.joblib",
                             scaler_path: str = "gesture_scaler.joblib",
                             seed: int = 42) -> dict:
        """
        使用合成数据训练模型并保存

        返回: 评估结果字典
        """
        print(f"Generating synthetic data ({n_per_class} samples × "
              f"{N_CLASSES} classes)...")
        X, y = generate_synthetic_data(
            n_per_class=n_per_class, noise_std=noise_std, seed=seed)

        # 分层划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed)

        # 标准化 (仅对坐标部分 [0:63])
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[:, :63] = scaler.fit_transform(X_train[:, :63])
        X_test_scaled[:, :63] = scaler.transform(X_test[:, :63])

        print(f"\nTraining MLP (hidden: 128,64)...")
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
            verbose=True,
        )

        model.fit(X_train_scaled, y_train)

        # 评估
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n{'='*55}")
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"{'='*55}")
        _print_evaluation(y_test, y_pred, model)

        # 保存
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

        return {
            "accuracy": acc,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "n_samples": len(X),
        }

    @staticmethod
    def train_from_real(data_dir: str = "training_data",
                        test_size: float = 0.2,
                        model_path: str = "gesture_model.joblib",
                        scaler_path: str = "gesture_scaler.joblib",
                        seed: int = 42) -> dict:
        """使用真实采集数据训练模型并保存"""
        X, y = load_real_data(data_dir)

        # 分层划分
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed)

        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[:, :63] = scaler.fit_transform(X_train[:, :63])
        X_test_scaled[:, :63] = scaler.transform(X_test[:, :63])

        print(f"\nTraining MLP (hidden: 128,64)...")
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
            verbose=True,
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n{'='*55}")
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"{'='*55}")
        _print_evaluation(y_test, y_pred, model)

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

        return {
            "accuracy": acc,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "n_samples": len(X),
        }

    @staticmethod
    def train_from_mixed(data_dir: str = "training_data",
                         synth_per_class: int = 500,
                         noise_std: float = 0.02,
                         test_size: float = 0.2,
                         model_path: str = "gesture_model.joblib",
                         scaler_path: str = "gesture_scaler.joblib",
                         seed: int = 42) -> dict:
        """混合真实数据 + 合成数据训练"""
        # 加载真实数据
        X_real, y_real = load_real_data(data_dir)
        print(f"Real data: {len(X_real)} samples")

        # 生成合成数据
        X_synth, y_synth = generate_synthetic_data(
            n_per_class=synth_per_class, noise_std=noise_std, seed=seed)
        print(f"Synthetic data: {len(X_synth)} samples")

        # 合并
        X = np.vstack([X_real, X_synth])
        y = np.hstack([y_real, y_synth])
        print(f"Total: {len(X)} samples")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed)

        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[:, :63] = scaler.fit_transform(X_train[:, :63])
        X_test_scaled[:, :63] = scaler.transform(X_test[:, :63])

        print(f"\nTraining MLP (hidden: 128,64)...")
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
            verbose=True,
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n{'='*55}")
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"{'='*55}")
        _print_evaluation(y_test, y_pred, model)

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

        return {
            "accuracy": acc,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "n_samples": len(X),
        }
