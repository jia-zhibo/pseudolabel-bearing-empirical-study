# -*- coding: utf-8 -*-
"""测试真实CWRU数据加载与跨负载划分"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cwru_data import get_cwru_dataloaders, CWRUDataset
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cwru_real')

# 1. 加载全部数据（跨负载0/1/2为训练，3为测试）
print('=== 跨负载划分测试 ===')
source_load, target_load = 0, 3
for sl, tl in [(0, 3), (1, 3), (2, 3)]:
    source_train, source_val, target_test = get_cwru_dataloaders(
        DATA_DIR, batch_size=64, window_size=1024, stride=512,
        n_train_per_class=None, source_load=sl, target_load=tl, num_workers=0
    )
    print(f'  train(0hp)={len(source_train.dataset)} val={len(source_val.dataset)} test({tl}hp)={len(target_test.dataset)}')

# 2. 检查类别分布
print('\n=== 类别分布 (窗口大小1024, 步长512) ===')
full_ds = CWRUDataset(DATA_DIR, window_size=1024, stride=512)
print(f'Total windows: {len(full_ds)}')
print(f'Unique labels: {sorted(set(full_ds.labels.tolist()))}')
from collections import Counter
print(f'Label counts: {dict(sorted(Counter(full_ds.labels.tolist()).items()))}')

# 3. 少样本采样测试
print('\n=== 少样本 (n=20 per class) ===')
train_loader, val_loader, test_loader = get_cwru_dataloaders(
    DATA_DIR, batch_size=64, window_size=1024, stride=512,
    n_train_per_class=20, num_workers=0
)
print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')

# 4. 检查信号质量
batch = next(iter(test_loader))
sample, label, cond = batch[0][0], batch[1][0], batch[2][0]
print(f'\nSample shape: {sample.shape}, label={label.item()}, condition={cond.tolist()}')
print(f'Signal range: [{sample.min():.3f}, {sample.max():.3f}], std={sample.std():.3f}')
