"""
CWRU轴承数据集加载与预处理模块
Case Western Reserve University Bearing Data Center

数据集下载地址: https://engineering.case.edu/bearingdatacenter
"""

import os
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torch


# CWRU数据集故障类型映射
FAULT_TYPES = {
    'N': 0,      # Normal 正常
    'B007': 1,   # Ball fault 0.007" 滚动体故障
    'B014': 2,   # Ball fault 0.014"
    'B021': 3,   # Ball fault 0.021"
    'IR007': 4,  # Inner race fault 0.007" 内圈故障
    'IR014': 5,
    'IR021': 6,
    'OR007': 7,  # Outer race fault 0.007" 外圈故障
    'OR014': 8,
    'OR021': 9,
}

# 负载条件 (马力 -> RPM近似)
LOAD_CONDITIONS = {
    0: 1797,  # 0 hp
    1: 1772,  # 1 hp
    2: 1750,  # 2 hp
    3: 1730,  # 3 hp
}


class CWRUDataset(Dataset):
    """CWRU轴承故障数据集"""

    def __init__(self, data_dir, window_size=512, stride=128, 
                 load_conditions=None, fault_types=None,
                 n_samples_per_class=None, transform=None,
                 return_condition=True):
        """
        Args:
            data_dir: 数据目录路径
            window_size: 滑动窗口长度
            stride: 滑动步长
            load_conditions: 使用的负载条件列表, None表示全部
            fault_types: 使用的故障类型列表, None表示全部
            n_samples_per_class: 每类采样数量(少样本设置), None表示全部
            transform: 数据增强变换
            return_condition: 是否返回条件向量
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.return_condition = return_condition
        
        # 默认使用全部负载和故障类型
        if load_conditions is None:
            self.load_conditions = [0, 1, 2, 3]
        else:
            self.load_conditions = load_conditions
            
        if fault_types is None:
            self.fault_types = list(FAULT_TYPES.keys())
        else:
            self.fault_types = fault_types
            
        self.n_samples_per_class = n_samples_per_class
        
        # 加载数据
        self.samples, self.labels, self.conditions = self._load_data()
        
    def _load_data(self):
        """加载所有数据并生成窗口样本"""
        all_samples = []
        all_labels = []
        all_conditions = []
        
        for load in self.load_conditions:
            for fault_type in self.fault_types:
                # 构建文件名
                filename = self._get_filename(fault_type, load)
                filepath = os.path.join(self.data_dir, filename)
                
                if not os.path.exists(filepath):
                    print(f"Warning: File not found: {filepath}")
                    continue
                    
                # 加载mat文件
                data = self._load_mat_file(filepath)
                
                if data is None:
                    continue
                
                # 滑动窗口切分
                windows = self._sliding_window(data)
                
                # 标签
                label = FAULT_TYPES[fault_type]
                labels = np.full(len(windows), label, dtype=np.int64)
                
                # 条件向量: [故障类型one-hot, 负载条件嵌入索引, 故障尺寸嵌入索引]
                fault_size = self._get_fault_size(fault_type)
                conditions = np.zeros((len(windows), 3), dtype=np.int64)
                conditions[:, 0] = label  # 故障类型索引
                conditions[:, 1] = load   # 负载条件索引
                conditions[:, 2] = fault_size  # 故障尺寸索引
                
                all_samples.append(windows)
                all_labels.append(labels)
                all_conditions.append(conditions)
        
        # 合并
        samples = np.concatenate(all_samples, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        conditions = np.concatenate(all_conditions, axis=0)
        
        # 少样本采样
        if self.n_samples_per_class is not None:
            samples, labels, conditions = self._few_shot_sample(
                samples, labels, conditions, self.n_samples_per_class
            )
        
        # 归一化
        samples = self._normalize(samples)
        
        # 增加通道维度 (N, L) -> (N, 1, L)
        samples = samples[:, np.newaxis, :].astype(np.float32)
        
        return samples, labels, conditions
    
    def _get_filename(self, fault_type, load):
        """根据故障类型和负载构建文件名（匹配下载的命名格式）"""
        if fault_type == 'N':
            return f"Normal_{load}.mat"
        else:
            return f"{fault_type}_{load}.mat"
    
    def _load_mat_file(self, filepath):
        """加载mat文件，返回DE端振动信号"""
        try:
            mat = loadmat(filepath)
            # 查找DE (Drive End) 数据
            for key in mat.keys():
                if 'DE' in key and 'time' in key:
                    return mat[key].flatten()
            # 如果找不到，尝试其他键
            for key in mat.keys():
                if 'DE' in key:
                    return mat[key].flatten()
            print(f"Warning: No DE data found in {filepath}")
            return None
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def _sliding_window(self, data):
        """滑动窗口切分"""
        n_samples = (len(data) - self.window_size) // self.stride + 1
        windows = np.zeros((n_samples, self.window_size), dtype=np.float32)
        for i in range(n_samples):
            start = i * self.stride
            windows[i] = data[start:start + self.window_size]
        return windows
    
    def _get_fault_size(self, fault_type):
        """获取故障尺寸索引"""
        if fault_type == 'N':
            return 0  # 正常
        elif '007' in fault_type:
            return 1  # 0.007英寸
        elif '014' in fault_type:
            return 2  # 0.014英寸
        elif '021' in fault_type:
            return 3  # 0.021英寸
        return 0
    
    def _few_shot_sample(self, samples, labels, conditions, n_per_class):
        """少样本采样：每类随机抽取n个样本"""
        unique_labels = np.unique(labels)
        sampled_indices = []
        
        for label in unique_labels:
            label_indices = np.where(labels == label)[0]
            if len(label_indices) > n_per_class:
                selected = np.random.choice(label_indices, n_per_class, replace=False)
            else:
                selected = label_indices
            sampled_indices.extend(selected)
        
        sampled_indices = np.array(sampled_indices)
        return samples[sampled_indices], labels[sampled_indices], conditions[sampled_indices]
    
    def _normalize(self, data):
        """Z-score归一化"""
        mean = np.mean(data)
        std = np.std(data) + 1e-8
        return (data - mean) / std
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = self.labels[idx]
        condition = self.conditions[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        sample = torch.from_numpy(sample).float()
        label = torch.tensor(label).long()
        condition = torch.from_numpy(condition).long()
        
        if self.return_condition:
            return sample, label, condition
        else:
            return sample, label


def get_cwru_dataloaders(data_dir, batch_size=64, window_size=512, stride=128,
                         n_train_per_class=None, source_load=None, target_load=None,
                         num_workers=0):
    """
    创建CWRU数据集的训练/验证/测试数据加载器
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        window_size: 窗口长度
        stride: 步长
        n_train_per_class: 训练集每类样本数 (少样本设置)
        source_load: 源域负载 (跨域实验用)
        target_load: 目标域负载 (跨域实验用)
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader, test_loader 或 (source_loaders, target_loaders)
    """
    if source_load is not None and target_load is not None:
        # 跨负载迁移实验
        source_train = CWRUDataset(
            data_dir, window_size, stride,
            load_conditions=[source_load],
            n_samples_per_class=n_train_per_class,
            return_condition=True
        )
        source_val = CWRUDataset(
            data_dir, window_size, stride,
            load_conditions=[source_load],
            n_samples_per_class=max(20, n_train_per_class // 5) if n_train_per_class else None,
            return_condition=True
        )
        target_test = CWRUDataset(
            data_dir, window_size, stride,
            load_conditions=[target_load],
            n_samples_per_class=None,  # 目标域用全部数据测试
            return_condition=True
        )
        
        source_train_loader = DataLoader(source_train, batch_size=batch_size, 
                                          shuffle=True, num_workers=num_workers)
        source_val_loader = DataLoader(source_val, batch_size=batch_size, 
                                        shuffle=False, num_workers=num_workers)
        target_test_loader = DataLoader(target_test, batch_size=batch_size, 
                                         shuffle=False, num_workers=num_workers)
        
        return (source_train_loader, source_val_loader, target_test_loader)
    else:
        # 标准实验
        full_dataset = CWRUDataset(
            data_dir, window_size, stride,
            n_samples_per_class=None,
            return_condition=True
        )
        
        # 按类别划分训练/验证/测试
        n_total = len(full_dataset)
        indices = np.random.permutation(n_total)
        
        train_ratio, val_ratio = 0.7, 0.1
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # 少样本：训练集限制每类样本数
        if n_train_per_class is not None:
            train_labels = full_dataset.labels[train_idx]
            sampled_train_idx = []
            for label in np.unique(train_labels):
                label_idx = train_idx[train_labels == label]
                if len(label_idx) > n_train_per_class:
                    selected = np.random.choice(label_idx, n_train_per_class, replace=False)
                else:
                    selected = label_idx
                sampled_train_idx.extend(selected)
            train_idx = np.array(sampled_train_idx)
        
        # 创建子集
        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
        test_dataset = torch.utils.data.Subset(full_dataset, test_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                   shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                  shuffle=False, num_workers=num_workers)
        
        return train_loader, val_loader, test_loader


# ==================== 合成数据生成器 (用于无数据时的测试) ====================

def generate_synthetic_cwru(n_samples_per_class=100, window_size=512, 
                            n_classes=10, n_loads=4, seed=42):
    """
    生成合成的CWRU风格数据，用于在没有真实数据时测试代码
    
    Args:
        n_samples_per_class: 每类样本数
        window_size: 窗口长度
        n_classes: 类别数
        n_loads: 负载条件数
        seed: 随机种子
    
    Returns:
        samples (N, 1, L), labels (N,), conditions (N, 3)
    """
    np.random.seed(seed)
    n_total = n_samples_per_class * n_classes
    
    samples = np.zeros((n_total, 1, window_size), dtype=np.float32)
    labels = np.zeros(n_total, dtype=np.int64)
    conditions = np.zeros((n_total, 3), dtype=np.int64)
    
    t = np.linspace(0, 1, window_size)
    
    idx = 0
    for cls in range(n_classes):
        for i in range(n_samples_per_class):
            # 基础信号 + 故障特征频率分量
            base_freq = 50 + cls * 10  # 不同类别不同基频
            signal = np.sin(2 * np.pi * base_freq * t)
            
            # 添加故障特征
            if cls > 0:  # 非正常
                fault_freq = 100 + cls * 20
                signal += 0.3 * np.sin(2 * np.pi * fault_freq * t)
                # 冲击成分
                impulse = np.exp(-((t - 0.3 * (i % 3))**2) / 0.001)
                signal += 0.5 * impulse
            
            # 负载相关的振幅变化
            load = i % n_loads
            amplitude = 1.0 + 0.1 * load
            signal *= amplitude
            
            # 添加噪声
            noise = np.random.randn(window_size) * 0.1
            signal += noise
            
            samples[idx, 0] = signal
            labels[idx] = cls
            conditions[idx, 0] = cls  # 故障类型
            conditions[idx, 1] = load  # 负载条件
            conditions[idx, 2] = (cls - 1) % 3 if cls > 0 else 0  # 故障尺寸
            
            idx += 1
    
    return samples, labels, conditions


if __name__ == "__main__":
    # 测试合成数据生成
    samples, labels, conditions = generate_synthetic_cwru(
        n_samples_per_class=50, window_size=512
    )
    print(f"合成CWRU数据: samples={samples.shape}, labels={labels.shape}")
    print(f"  类别分布: {np.bincount(labels)}")
    print(f"  条件范围: fault={conditions[:,0].min()}-{conditions[:,0].max()}, "
          f"load={conditions[:,1].min()}-{conditions[:,1].max()}, "
          f"size={conditions[:,2].min()}-{conditions[:,2].max()}")
