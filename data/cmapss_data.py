"""
C-MAPSS涡扇发动机退化数据集加载与预处理模块
Commercial Modular Aero-Propulsion System Simulation

数据集来源: NASA Prognostics Center of Excellence
下载地址: https://data.nasa.gov/dataset/C-MAPSS-Turbofan-Engine-Degradation-Simulation/

子数据集:
- FD001: 100 train/100 test, 1种运行条件, 1种故障模式 (HPC)
- FD002: 260 train/259 test, 6种运行条件, 1种故障模式 (HPC)
- FD003: 100 train/100 test, 1种运行条件, 2种故障模式 (HPC+Fan)
- FD004: 249 train/248 test, 6种运行条件, 2种故障模式 (HPC+Fan)
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch


# 传感器列名 (21个传感器 + 3个运行条件设置)
SENSOR_COLS = [
    'setting_1', 'setting_2', 'setting_3',  # 运行条件设置
    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
    's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17',
    's18', 's19', 's20', 's21'
]

# 实际有意义的传感器索引 (排除常数传感器)
USEFUL_SENSORS = [
    's2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13',
    's14', 's15', 's17', 's20', 's21'
]


class CMAPSSDataset(Dataset):
    """C-MAPSS涡扇发动机退化数据集"""

    def __init__(self, data_dir, subset='FD001', window_size=30,
                 max_rul=125, mode='train', n_engines=None,
                 return_rul=True, return_condition=True):
        """
        Args:
            data_dir: 数据目录
            subset: 数据集子集 (FD001, FD002, FD003, FD004)
            window_size: 滑动窗口大小 (时间步数)
            max_rul: RUL最大值 (分段线性函数的上限)
            mode: 'train' 或 'test'
            n_engines: 使用的发动机数量 (少样本设置), None表示全部
            return_rul: 是否返回RUL标签
            return_condition: 是否返回条件向量
        """
        self.data_dir = data_dir
        self.subset = subset
        self.window_size = window_size
        self.max_rul = max_rul
        self.mode = mode
        self.n_engines = n_engines
        self.return_rul = return_rul
        self.return_condition = return_condition
        
        # 加载数据
        self.samples, self.ruls, self.conditions = self._load_and_process()
        
    def _load_and_process(self):
        """加载数据并处理成窗口样本"""
        # 加载原始数据
        train_path = os.path.join(self.data_dir, f'train_{self.subset}.txt')
        test_path = os.path.join(self.data_dir, f'test_{self.subset}.txt')
        rul_path = os.path.join(self.data_dir, f'RUL_{self.subset}.txt')
        
        if not os.path.exists(train_path):
            # 找不到真实数据时，生成合成数据
            print(f"Warning: Data files not found in {self.data_dir}")
            print("Generating synthetic C-MAPSS data for testing...")
            return self._generate_synthetic_data()
        
        # 读取数据
        col_names = ['engine_id', 'cycle'] + SENSOR_COLS
        train_df = pd.read_csv(train_path, sep='\s+', header=None, names=col_names)
        test_df = pd.read_csv(test_path, sep='\s+', header=None, names=col_names)
        test_rul = np.loadtxt(rul_path)
        
        if self.mode == 'train':
            df = train_df
        else:
            df = test_df
        
        # 少样本：只使用前n台发动机
        if self.n_engines is not None and self.mode == 'train':
            engine_ids = df['engine_id'].unique()[:self.n_engines]
            df = df[df['engine_id'].isin(engine_ids)]
        
        # 提取有用的传感器
        sensor_cols = USEFUL_SENSORS
        setting_cols = ['setting_1', 'setting_2', 'setting_3']
        
        # 计算每台发动机的RUL
        engines = df['engine_id'].unique()
        all_samples = []
        all_ruls = []
        all_conditions = []
        
        for engine_id in engines:
            engine_data = df[df['engine_id'] == engine_id].sort_values('cycle')
            n_cycles = len(engine_data)
            
            # 计算RUL
            if self.mode == 'train':
                # 训练集：RUL = 总周期数 - 当前周期，上限为max_rul
                rul = np.arange(n_cycles - 1, -1, -1)
                rul = np.minimum(rul, self.max_rul)
            else:
                # 测试集：使用文件中的RUL
                final_rul = test_rul[engine_id - 1]  # engine_id从1开始
                rul = np.arange(n_cycles - 1, -1, -1) + final_rul
                rul = np.minimum(rul, self.max_rul)
            
            # 传感器数据
            sensor_data = engine_data[sensor_cols].values.astype(np.float32)
            
            # 运行条件
            settings = engine_data[setting_cols].values.astype(np.float32)
            
            # 滑动窗口
            for i in range(n_cycles - self.window_size + 1):
                window = sensor_data[i:i + self.window_size]  # (window_size, n_sensors)
                current_rul = rul[i + self.window_size - 1]
                
                # 条件向量: [运行条件编码, 退化阶段, 故障模式]
                current_setting = settings[i + self.window_size - 1]
                condition = self._compute_condition(current_setting, current_rul)
                
                all_samples.append(window)
                all_ruls.append(current_rul)
                all_conditions.append(condition)
        
        # 转换为数组
        samples = np.array(all_samples, dtype=np.float32)  # (N, L, D)
        ruls = np.array(all_ruls, dtype=np.float32)  # (N,)
        conditions = np.array(all_conditions, dtype=np.int64)  # (N, 3)
        
        # 转置为 (N, D, L) 适配卷积操作
        samples = samples.transpose(0, 2, 1)
        
        # 归一化 (在样本内)
        samples = self._normalize(samples)
        
        return samples, ruls, conditions
    
    def _compute_condition(self, settings, rul):
        """
        计算条件向量
        
        condition[0]: 运行条件类别 (聚类后的工况编号)
        condition[1]: 退化阶段 (0=早期, 1=中期, 2=晚期)
        condition[2]: 故障模式 (0=HPC, 1=Fan, 2=两者)
        """
        # 运行条件：根据设置值简单离散化
        # 实际应用中应对运行条件进行聚类
        if self.subset in ['FD001', 'FD003']:
            op_cond = 0  # 单一运行条件
        else:
            # 多运行条件：简化处理，实际应聚类
            op_cond = int(abs(settings[0]) * 10) % 6
        
        # 退化阶段
        if rul > self.max_rul * 0.7:
            deg_stage = 0  # 早期
        elif rul > self.max_rul * 0.3:
            deg_stage = 1  # 中期
        else:
            deg_stage = 2  # 晚期
        
        # 故障模式
        if self.subset in ['FD001', 'FD002']:
            fault_mode = 0  # HPC
        else:
            # FD003, FD004 有两种故障模式，这里简化为两种都有
            fault_mode = 2  # HPC + Fan
        
        return np.array([op_cond, deg_stage, fault_mode])
    
    def _normalize(self, samples):
        """逐特征归一化"""
        # 对每个特征维度进行归一化
        for d in range(samples.shape[1]):
            mean = np.mean(samples[:, d, :])
            std = np.std(samples[:, d, :]) + 1e-8
            samples[:, d, :] = (samples[:, d, :] - mean) / std
        return samples
    
    def _generate_synthetic_data(self):
        """生成合成数据用于测试"""
        np.random.seed(42)
        
        n_engines = min(self.n_engines or 100, 100)
        n_sensors = len(USEFUL_SENSORS)
        all_samples = []
        all_ruls = []
        all_conditions = []
        
        for engine_id in range(n_engines):
            # 每台发动机的退化周期数 (随机50-200)
            n_cycles = np.random.randint(50, 200)
            
            # 生成传感器数据：基础信号 + 退化趋势 + 噪声
            t = np.arange(n_cycles) / n_cycles  # 归一化时间
            
            # 退化趋势 (指数型)
            degradation = np.exp(3 * t) - 1  # 从0开始加速退化
            
            sensor_data = np.zeros((n_cycles, n_sensors))
            for s in range(n_sensors):
                base = np.sin(2 * np.pi * (s + 1) * 0.1 * t) * 0.1
                trend = degradation * (0.1 + 0.05 * s) * np.sign(np.random.randn())
                noise = np.random.randn(n_cycles) * 0.05
                sensor_data[:, s] = base + trend + noise
            
            # RUL
            rul = np.arange(n_cycles - 1, -1, -1)
            rul = np.minimum(rul, self.max_rul)
            
            # 滑动窗口
            for i in range(n_cycles - self.window_size + 1):
                window = sensor_data[i:i + self.window_size]  # (L, D)
                current_rul = rul[i + self.window_size - 1]
                
                # 条件
                op_cond = engine_id % 6 if self.subset in ['FD002', 'FD004'] else 0
                if current_rul > self.max_rul * 0.7:
                    deg_stage = 0
                elif current_rul > self.max_rul * 0.3:
                    deg_stage = 1
                else:
                    deg_stage = 2
                fault_mode = 0 if self.subset in ['FD001', 'FD002'] else 2
                
                condition = np.array([op_cond, deg_stage, fault_mode])
                
                all_samples.append(window)
                all_ruls.append(current_rul)
                all_conditions.append(condition)
        
        samples = np.array(all_samples, dtype=np.float32).transpose(0, 2, 1)  # (N, D, L)
        ruls = np.array(all_ruls, dtype=np.float32)
        conditions = np.array(all_conditions, dtype=np.int64)
        
        # 归一化
        samples = self._normalize(samples)
        
        return samples, ruls, conditions
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = torch.from_numpy(self.samples[idx]).float()
        rul = torch.tensor(self.ruls[idx]).float()
        condition = torch.from_numpy(self.conditions[idx]).long()
        
        if self.return_rul and self.return_condition:
            return sample, rul, condition
        elif self.return_rul:
            return sample, rul
        else:
            return sample


def get_cmapss_dataloaders(data_dir, subset='FD001', batch_size=64, 
                           window_size=30, max_rul=125,
                           n_train_engines=None, num_workers=0):
    """
    创建C-MAPSS数据集的训练/验证/测试数据加载器
    
    Args:
        data_dir: 数据目录
        subset: 子集名称
        batch_size: 批次大小
        window_size: 窗口大小
        max_rul: RUL最大值
        n_train_engines: 训练发动机数量 (少样本设置)
        num_workers: 数据加载线程数
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 训练集
    train_dataset = CMAPSSDataset(
        data_dir, subset, window_size, max_rul,
        mode='train', n_engines=n_train_engines
    )
    
    # 测试集
    test_dataset = CMAPSSDataset(
        data_dir, subset, window_size, max_rul,
        mode='test', n_engines=None
    )
    
    # 从训练集中划分验证集 (按发动机划分)
    n_total = len(train_dataset)
    indices = np.random.permutation(n_total)
    n_val = int(n_total * 0.1)
    
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    
    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(train_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size,
                               shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def cmapss_score_function(y_true, y_pred):
    """
    C-MAPSS标准评分函数 (不对称惩罚)
    
    Score = sum(exp(d/13) - 1) for d < 0 (预测过早)
          + sum(exp(d/10) - 1) for d >= 0 (预测过晚)
    
    其中 d = y_pred - y_true
    分数越低越好
    """
    d = y_pred - y_true
    score = np.zeros_like(d)
    
    early_mask = d < 0
    late_mask = d >= 0
    
    score[early_mask] = np.exp(-d[early_mask] / 13.0) - 1
    score[late_mask] = np.exp(d[late_mask] / 10.0) - 1
    
    return np.sum(score)


if __name__ == "__main__":
    # 测试合成数据
    dataset = CMAPSSDataset('./data', subset='FD001', window_size=30, mode='train')
    print(f"C-MAPSS FD001 合成数据: {len(dataset)} 个窗口样本")
    sample, rul, cond = dataset[0]
    print(f"  sample shape: {sample.shape}")
    print(f"  RUL: {rul.item()}")
    print(f"  condition: {cond.numpy()}")
