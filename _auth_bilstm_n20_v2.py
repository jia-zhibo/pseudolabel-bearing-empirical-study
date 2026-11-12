# -*- coding: utf-8 -*-
"""
对比多种域适应方法在n=50上的稳定性
1. Baseline
2. MMD (多核MMD)
3. CORAL (简单特征对齐)
4. 伪标签自训练
5. DANN (不同lambda)
"""
import os
import sys
import json
import time
import os as _os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ[_v] = "10"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
from cwru_data import CWRUDataset
from models.detector.domain_adapt import compute_mmd
from utils.train_utils import set_seed
import torch as _torch
_torch.set_num_threads(10)


# ============================================================
# 存活心跳
# ------------------------------------------------------------
# 本脚本在每个 epoch 刷新一次 results/_n20_alive.heartbeat 的 mtime；
# 外部 run_n20.bat 依据该文件判断「是否仍有实例在推进」，从而在实例意外终止后
# 约 10 分钟内自动接管续跑（此前只能靠日志更新判断，pseudo 单轮最长 40 分钟，
# 接管延迟高达 50 分钟）。
# 只调用 os.utime，不参与任何随机数生成或数据流转，不影响可复现性。
# ============================================================
_ALIVE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'results', '_n20_alive.heartbeat')


def _beat():
    try:
        os.utime(_ALIVE_FILE, None)
    except OSError:
        try:
            open(_ALIVE_FILE, 'a').close()
        except OSError:
            pass


# ============================================================
# 数据
# ============================================================

def load_data_full(data_dir, load_condition, window_size=512, stride=128):
    dataset = CWRUDataset(
        data_dir, window_size=window_size, stride=stride,
        load_conditions=[load_condition],
        n_samples_per_class=None, return_condition=True
    )
    samples = torch.from_numpy(dataset.samples).float()
    labels = torch.from_numpy(dataset.labels).long()
    return samples, labels


def few_shot_split(source_x, source_y, n_train, n_val, seed=42):
    np.random.seed(seed)
    classes = torch.unique(source_y)
    train_idx, val_idx = [], []
    for c in classes:
        c_idx = (source_y == c.item()).nonzero().squeeze().numpy()
        np.random.shuffle(c_idx)
        train_idx.extend(c_idx[:n_train].tolist())
        val_idx.extend(c_idx[n_train:n_train + n_val].tolist())
    return (source_x[torch.tensor(train_idx, dtype=torch.long)],
            source_y[torch.tensor(train_idx, dtype=torch.long)],
            source_x[torch.tensor(val_idx, dtype=torch.long)],
            source_y[torch.tensor(val_idx, dtype=torch.long)])


# ============================================================
# 基础分类器
# ============================================================

class BaseClassifier(nn.Module):
    """CNN + LSTM + Attention 分类器"""
    def __init__(self, n_classes=10, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.lstm = nn.LSTM(128, hidden_dim, 2, batch_first=True,
                           bidirectional=True, dropout=dropout)
        self.attn = nn.MultiheadAttention(hidden_dim * 2, 4, dropout=dropout, batch_first=True)
        feat_dim = hidden_dim * 2
        self.cls = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(feat_dim // 2, n_classes),
        )
    
    def forward(self, x, return_feat=False):
        x = self.conv(x).transpose(1, 2)
        x, _ = self.lstm(x)
        attn_out, _ = self.attn(x, x, x)
        feat = attn_out.mean(dim=1)
        out = self.cls(feat)
        if return_feat:
            return out, feat
        return out


def train_eval(model, train_loader, val_loader, test_loader,
               device, epochs=80, lr=2e-3, wd=1e-4):
    """通用训练评估函数"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    best_state = None
    patience = 20
    no_improve = 0
    
    for epoch in range(epochs):
        _beat()
        model.train()
        for batch in train_loader:
            x = batch[0].to(device)
            y = batch[1].to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                _, predicted = output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    model.load_state_dict(best_state)
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            probs = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    from sklearn.metrics import f1_score, roc_auc_score
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    return {'accuracy': accuracy, 'f1_macro': f1_macro, 'auc': auc, 'best_val_acc': best_val_acc}


# ============================================================
# 方法1: MMD域适应
# ============================================================

def train_mmd(train_x, train_y, test_x, val_x, val_y, test_y,
              device, n_classes, seed, epochs=80, lambda_mmd=0.5):
    set_seed(seed)
    model = BaseClassifier(n_classes=n_classes).to(device)
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    target_loader = DataLoader(TensorDataset(test_x, torch.zeros(len(test_x))),
                               batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=128)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=256)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        _beat()
        model.train()
        p = epoch / epochs
        lam = lambda_mmd * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
        target_iter = iter(target_loader)
        
        for src_x, src_y in train_loader:
            src_x, src_y = src_x.to(device), src_y.to(device)
            try:
                tgt_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_x, _ = next(target_iter)
            tgt_x = tgt_x.to(device)
            
            src_logits, src_feat = model(src_x, return_feat=True)
            cls_loss = criterion(src_logits, src_y)
            
            _, tgt_feat = model(tgt_x, return_feat=True)
            mmd_loss = compute_mmd(src_feat, tgt_feat)
            
            loss = cls_loss + lam * mmd_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                _, predicted = output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 20:
                break
    
    model.load_state_dict(best_state)
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            probs = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    from sklearn.metrics import f1_score, roc_auc_score
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    return {'accuracy': accuracy, 'f1_macro': f1_macro, 'auc': auc, 'best_val_acc': best_val_acc}


# ============================================================
# 方法2: CORAL (简单特征对齐)
# ============================================================

def coral_loss(source_feat, target_feat):
    """CORAL loss: align covariance matrices"""
    n_s = source_feat.size(0)
    n_t = target_feat.size(0)
    
    # Source covariance
    src_mean = source_feat.mean(dim=0, keepdim=True)
    src_centered = source_feat - src_mean
    src_cov = (src_centered.t() @ src_centered) / (n_s - 1)
    
    # Target covariance
    tgt_mean = target_feat.mean(dim=0, keepdim=True)
    tgt_centered = target_feat - tgt_mean
    tgt_cov = (tgt_centered.t() @ tgt_centered) / (n_t - 1)
    
    # Frobenius norm of difference
    loss = torch.norm(src_cov - tgt_cov, p='fro') ** 2
    # Normalize by feature dimension squared
    loss = loss / (4 * source_feat.size(1) ** 2)
    return loss


def train_coral(train_x, train_y, test_x, val_x, val_y, test_y,
                device, n_classes, seed, epochs=80, lambda_coral=0.5):
    set_seed(seed)
    model = BaseClassifier(n_classes=n_classes).to(device)
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    target_loader = DataLoader(TensorDataset(test_x, torch.zeros(len(test_x))),
                               batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=128)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=256)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        _beat()
        model.train()
        p = epoch / epochs
        lam = lambda_coral * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
        target_iter = iter(target_loader)
        
        for src_x, src_y in train_loader:
            src_x, src_y = src_x.to(device), src_y.to(device)
            try:
                tgt_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_x, _ = next(target_iter)
            tgt_x = tgt_x.to(device)
            
            src_logits, src_feat = model(src_x, return_feat=True)
            cls_loss = criterion(src_logits, src_y)
            
            _, tgt_feat = model(tgt_x, return_feat=True)
            coral_l = coral_loss(src_feat, tgt_feat)
            
            loss = cls_loss + lam * coral_l
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
        
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                _, predicted = output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 20:
                break
    
    model.load_state_dict(best_state)
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            probs = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    from sklearn.metrics import f1_score, roc_auc_score
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    return {'accuracy': accuracy, 'f1_macro': f1_macro, 'auc': auc, 'best_val_acc': best_val_acc}


# ============================================================
# 方法3: 伪标签自训练
# ============================================================

def train_pseudo_label(train_x, train_y, test_x, val_x, val_y, test_y,
                       device, n_classes, seed, epochs=80,
                       confidence_threshold=0.9, n_rounds=3):
    """
    伪标签自训练：
    1. 在源域数据上训练初始模型
    2. 用模型预测目标域伪标签
    3. 选择高置信度伪标签加入训练集
    4. 重新训练
    5. 重复n_rounds轮
    """
    set_seed(seed)
    
    current_train_x = train_x.clone()
    current_train_y = train_y.clone()
    
    for round_idx in range(n_rounds):
        # Train on current data
        model = BaseClassifier(n_classes=n_classes).to(device)
        train_loader = DataLoader(
            TensorDataset(current_train_x, current_train_y),
            batch_size=64, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=128)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0
        best_state = None
        no_improve = 0
        
        for epoch in range(epochs):
            _beat()
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    output = model(x)
                    _, predicted = output.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()
            
            val_acc = correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 20:
                    break
        
        model.load_state_dict(best_state)
        model.eval()
        
        # Generate pseudo-labels on target data
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=256)
        all_probs = []
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                output = model(x)
                probs = F.softmax(output, dim=1)
                all_probs.append(probs.cpu())
        
        all_probs = torch.cat(all_probs, dim=0)
        conf, pred = all_probs.max(dim=1)
        
        # Select high-confidence pseudo-labels
        mask = conf > confidence_threshold
        n_pseudo = mask.sum().item()
        
        if n_pseudo == 0 or round_idx == n_rounds - 1:
            # No more pseudo-labels or last round - use this model as final
            break
        
        # Add pseudo-labeled data to training set
        pseudo_x = test_x[mask]
        pseudo_y = pred[mask]
        
        current_train_x = torch.cat([current_train_x, pseudo_x])
        current_train_y = torch.cat([current_train_y, pseudo_y])
    
    # Final evaluation
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=256)
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            probs = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    from sklearn.metrics import f1_score, roc_auc_score
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    accuracy = (all_preds == all_labels).mean()
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except:
        auc = 0.0
    
    return {
        'accuracy': accuracy, 'f1_macro': f1_macro, 'auc': auc,
        'best_val_acc': best_val_acc,
        'n_pseudo_final': len(current_train_x) - len(train_x),
        'n_rounds': round_idx + 1,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    device = torch.device('cpu')
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cwru_real')
    source_load = 0
    target_load = 3
    n_train = 50
    n_val = 10
    n_repeats = 20
    
    print("=" * 70)
    print("AUTHORITATIVE RE-RUN  threads=10  torch=" + _torch.__version__ +
          "  proto=no-instrumentation")
    print("Domain Adaptation Method Comparison (n=50, 20 repeats)")
    print("start: " + time.strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)
    
    source_x, source_y = load_data_full(data_dir, source_load)
    target_x, target_y = load_data_full(data_dir, target_load)
    n_classes = len(torch.unique(source_y))
    
    print("Source: %d samples, Target: %d samples, %d classes" % (
        len(source_x), len(target_x), n_classes))
    
    # 修正原始脚本的种子缺陷：
    # train_eval()（baseline 专用）从未调用 set_seed，而 train_mmd / train_coral /
    # train_pseudo_label 都调用了。因此 baseline 的模型初始化与数据打乱依赖进程残留的
    # torch 随机状态，每次执行都不同（实测同种子两次得 77.08% 与 64.49%，
    # 而同批的 MMD/CORAL/伪标签三次完全一致）。这正是已发表 Table 1 无法复现的原因。
    # 这里在 baseline 训练前显式播种，使其与其余三个方法同样可复现。
    def _train_baseline(tr_x, tr_y, te_x, vx, vy, te_y, d, nc, s):
        set_seed(s)
        return train_eval(
            BaseClassifier(nc).to(d),
            DataLoader(TensorDataset(tr_x, tr_y), batch_size=64, shuffle=True),
            DataLoader(TensorDataset(vx, vy), batch_size=128),
            DataLoader(TensorDataset(te_x, te_y), batch_size=256),
            d, epochs=80
        )

    methods = {
        'baseline': _train_baseline,
        'mmd_0.5': lambda tr_x, tr_y, te_x, vx, vy, te_y, d, nc, s:
            train_mmd(tr_x, tr_y, te_x, vx, vy, te_y, d, nc, s, epochs=80, lambda_mmd=0.5),
        'coral_0.5': lambda tr_x, tr_y, te_x, vx, vy, te_y, d, nc, s:
            train_coral(tr_x, tr_y, te_x, vx, vy, te_y, d, nc, s, epochs=80, lambda_coral=0.5),
        'pseudo_0.9': lambda tr_x, tr_y, te_x, vx, vy, te_y, d, nc, s:
            train_pseudo_label(tr_x, tr_y, te_x, vx, vy, te_y, d, nc, s, 
                              epochs=80, confidence_threshold=0.9),
    }
    
    all_results = {name: [] for name in methods}

    OUT = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                        'results', 'authoritative_bilstm_n50_n20.json')

    def _save(state, note=''):
        """方法级增量保存：state[name] = {rep_index: result}。

        粒度从「每个 repeat 结束」细化到「每个方法结束」，因为 pseudo 分支单轮
        可达数十分钟，一旦中途被终止，整轮进度都会丢失。
        """
        out = {}
        for nm in methods:
            rs = [state[nm][i] for i in sorted(state[nm])]
            if not rs:
                continue
            accs = np.array([r['accuracy'] for r in rs])
            f1s = np.array([r['f1_macro'] for r in rs])
            out[nm] = {'mean_acc': float(accs.mean()), 'std_acc': float(accs.std()),
                       'mean_f1': float(f1s.mean()), 'std_f1': float(f1s.std()),
                       'accuracies': accs.tolist(), 'f1s': f1s.tolist(),
                       'reps_done': [i for i in sorted(state[nm])]}
        ndone = min([len(state[nm]) for nm in methods] + [0])
        out['_meta'] = {'threads': 10, 'torch': _torch.__version__,
                        'proto': 'no per-epoch instrumentation (matches original)',
                        'n_train': 50, 'n_val': 10, 'n_repeats_total': 20,
                        'n_repeats_done': ndone, 'epochs': 80,
                        'seeds': [r * 1000 for r in range(20)],
                        'window_size': 512, 'stride': 128,
                        'progress': {nm: len(state[nm]) for nm in methods}}
        # 原子写盘：先写临时文件再替换，避免进程被强杀时留下半截 JSON
        _tmp = OUT + '.tmp'
        with open(_tmp, 'w') as f:
            json.dump(out, f, indent=2)
        _os.replace(_tmp, OUT)
        print('  [saved: %s | progress %s | @%s]' % (
            note, {nm: len(state[nm]) for nm in methods},
            time.strftime('%H:%M:%S')), flush=True)

    # ---------------- 断点续跑：恢复已完成结果 ----------------
    state = {nm: {} for nm in methods}
    if _os.path.exists(OUT):
        try:
            with open(OUT) as f:
                prev = json.load(f)
        except Exception as _e:
            print('  [WARN] 检查点无法解析(%s)，本次从零开始' % _e, flush=True)
            prev = {}
        for nm in methods:
            if nm in prev and 'accuracies' in prev[nm] and 'f1s' in prev[nm]:
                # 优先用显式记录的 rep 下标，避免各方法进度不齐时错位
                idxs = prev[nm].get('reps_done') or list(
                    range(len(prev[nm]['accuracies'])))
                for i, (a, f1) in zip(idxs, zip(prev[nm]['accuracies'],
                                                prev[nm]['f1s'])):
                    state[nm][i] = {'accuracy': a, 'f1_macro': f1}
        if any(state[nm] for nm in methods):
            print('  [RESUMED from %s: %s]' % (
                _os.path.basename(OUT), {nm: len(state[nm]) for nm in methods}),
                flush=True)

    T_START = time.time()

    for rep in range(n_repeats):
        seed = rep * 1000
        print("\n--- Repeat %d/%d (seed=%d) ---" % (rep + 1, n_repeats, seed), flush=True)
        if rep > 0:
            _el = (time.time() - T_START) / 60.0          # minutes so far
            _per = _el / rep                              # minutes per repeat
            print("  [pace: %.1f min/rep | ETA: %.1f h remaining @%s]" % (
                _per, _per * (n_repeats - rep) / 60.0,
                time.strftime('%H:%M:%S')), flush=True)

        tr_x, tr_y, val_x, val_y = few_shot_split(
            source_x, source_y, n_train, n_val, seed=seed)

        for name, func in methods.items():
            if rep in state[name]:
                print("  %-15s: %.2f%% [cached]" % (
                    name, state[name][rep]['accuracy'] * 100), flush=True)
                continue
            t0 = time.time()
            r = func(tr_x, tr_y, target_x, val_x, val_y, target_y,
                     device, n_classes, seed)
            elapsed = (time.time() - t0) / 60
            state[name][rep] = r
            print("  %-15s: %.2f%% [%.1fmin] @%s" % (
                name, r['accuracy'] * 100, elapsed,
                time.strftime('%H:%M:%S')), flush=True)
            _save(state, '%s rep%d' % (name, rep + 1))

    # 收尾：还原成列表形式，供下方 summary / 落盘复用
    all_results = {nm: [state[nm][i] for i in sorted(state[nm])] for nm in methods}

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    baseline_accs = np.array([r['accuracy'] for r in all_results['baseline']])
    
    print('%-15s %-18s %-12s %-10s %-10s' % (
        'Method', 'Accuracy', 'Δ(pp)', 'p-value', 'sig'))
    print('-' * 65)
    
    results_summary = {}
    for name in methods:
        accs = np.array([r['accuracy'] for r in all_results[name]])
        f1s = np.array([r['f1_macro'] for r in all_results[name]])
        
        if name == 'baseline':
            print('%-15s %5.2f%% ± %5.2f%%  %+5.2f   %-10s %s' % (
                name, accs.mean() * 100, accs.std() * 100, 0.0, '-', '-'))
        else:
            diffs = accs - baseline_accs
            t_stat, p_val = stats.ttest_rel(accs, baseline_accs)
            sig = '*' if p_val < 0.05 else 'ns'
            print('%-15s %5.2f%% ± %5.2f%%  %+5.2f   %.4f    %s' % (
                name, accs.mean() * 100, accs.std() * 100,
                diffs.mean() * 100, p_val, sig))
        
        results_summary[name] = {
            'mean_acc': float(accs.mean()),
            'std_acc': float(accs.std()),
            'mean_f1': float(f1s.mean()),
            'std_f1': float(f1s.std()),
            'accuracies': accs.tolist(),
        }
    
    # Save
    os.makedirs('results', exist_ok=True)
    results_summary['_meta'] = {
        'threads': 10, 'torch': _torch.__version__,
        'proto': 'no per-epoch instrumentation (matches original)',
        'n_train': 50, 'n_val': 10, 'n_repeats': 20, 'n_repeats_done': 20,
        'epochs': 80,
        'seeds': [r * 1000 for r in range(20)],
        'window_size': 512, 'stride': 128,
    }
    _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                       'results', 'authoritative_bilstm_n50_n20.json')
    with open(_p, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print('\nSaved to: ' + _p)


if __name__ == '__main__':
    main()
