# -*- coding: utf-8 -*-
"""
PU轴承数据集加载器
"""
import os
import numpy as np
import scipy.io as sio


CLASS_NAMES = ['Normal', 'Inner_Race', 'Outer_Race']


def bearing_to_class(bearing_code):
    if bearing_code.startswith('K0'):
        return 0  # Normal
    elif bearing_code.startswith('KA'):
        return 1  # Inner Race Fault
    elif bearing_code.startswith('KI'):
        return 2  # Outer Race Fault
    else:
        raise ValueError(f"Unknown bearing: {bearing_code}")


def load_vibration_from_mat(mat_path):
    """从PU .mat文件加载振动信号（vibration_1通道）"""
    mat = sio.loadmat(mat_path)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    if not keys:
        return None
    
    data = mat[keys[0]]
    Y = data['Y'][0, 0]  # shape: (1, 7)
    
    # 找vibration_1通道（通常是索引6）
    vib_signal = None
    for i in range(Y.shape[1]):
        name = str(Y['Name'][0, i][0]) if Y['Name'][0, i].size > 0 else ''
        if 'vibration' in name.lower():
            vib_signal = Y['Data'][0, i].flatten()
            break
    
    if vib_signal is None:
        # fallback: 用最长的信号
        max_len = 0
        for i in range(Y.shape[1]):
            d = Y['Data'][0, i].flatten()
            if len(d) > max_len:
                max_len = len(d)
                vib_signal = d
    
    return vib_signal.astype(np.float32)


def extract_windows(signal, window_size, stride):
    n = (len(signal) - window_size) // stride + 1
    if n <= 0:
        return np.zeros((0, window_size), dtype=np.float32)
    idx = np.arange(n)[:, None] * stride + np.arange(window_size)
    return signal[idx].astype(np.float32)


class PUDataset:
    def __init__(self, data_dir, bearing_codes, condition='N15_M07_F10',
                 window_size=512, stride=128, max_files_per_bearing=10):
        self.data_dir = data_dir
        self.condition = condition
        self.window_size = window_size
        self.stride = stride
        
        all_windows = []
        all_labels = []
        
        for bearing in bearing_codes:
            bearing_dir = os.path.join(data_dir, bearing)
            if not os.path.isdir(bearing_dir):
                continue
            
            label = bearing_to_class(bearing)
            file_count = 0
            
            mat_files = sorted([
                f for f in os.listdir(bearing_dir)
                if f.endswith('.mat') and f.startswith(condition)
            ])
            
            for fname in mat_files[:max_files_per_bearing]:
                mat_path = os.path.join(bearing_dir, fname)
                signal = load_vibration_from_mat(mat_path)
                
                if signal is None or len(signal) < window_size:
                    continue
                
                windows = extract_windows(signal, window_size, stride)
                if len(windows) > 0:
                    all_windows.append(windows)
                    all_labels.extend([label] * len(windows))
                    file_count += 1
        
        if all_windows:
            self.samples = np.vstack(all_windows)[:, np.newaxis, :]
        else:
            self.samples = np.zeros((0, 1, window_size), dtype=np.float32)
        self.labels = np.array(all_labels, dtype=np.int64)
    
    def __len__(self):
        return len(self.labels)
    
    def summary(self):
        classes, counts = np.unique(self.labels, return_counts=True)
        print(f"PU Dataset ({self.condition}):")
        print(f"  Total: {len(self.labels)} samples, shape={self.samples.shape}")
        for c, cnt in zip(classes, counts):
            print(f"  {CLASS_NAMES[c]}: {cnt}")


if __name__ == '__main__':
    data_dir = 'data/pu_data'
    bearings = ['K001', 'K002', 'K003', 'KA04', 'KA15', 'KI04', 'KI14']
    
    for cond in ['N15_M07_F10', 'N09_M07_F10']:
        ds = PUDataset(data_dir, bearings, condition=cond,
                       window_size=512, stride=256, max_files_per_bearing=5)
        ds.summary()
        print()
