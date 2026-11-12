# -*- coding: utf-8 -*-
"""
MSSP返修综合实验 - 一次性解决所有P1级实验问题
1. 同架构（FastCNN）CWRU+PU 跨数据集对比（10次重复）- 解决P1-1 + P1-3
2. DANN 现代基线方法 - 解决P1-4
3. 消融实验3次重复 - 解决P1-2
4. 特征提取 + 保存用于t-SNE可视化 - 解决P1-6
5. 混淆矩阵数据保存 - 解决P1-6
"""
import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
from pu_data import PUDataset
from cwru_data import CWRUDataset


# ============================================================
# Models
# ============================================================
class FastCNN(nn.Module):
    """Lightweight CNN for fast experiments"""
    def __init__(self, n_classes=3, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=2, padding=7),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )
    
    def forward(self, x, return_feat=False):
        feat = self.features(x).squeeze(-1)
        out = self.classifier(feat)
        if return_feat:
            return out, feat
        return out


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lam, None


class DANN(nn.Module):
    """Domain Adversarial Neural Network"""
    def __init__(self, n_classes=3, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=2, padding=7),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )
        self.domain_cls = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )
    
    def forward(self, x, lam=0.0):
        feat = self.features(x).squeeze(-1)
        out_cls = self.classifier(feat)
        # Gradient reversal for domain classifier
        feat_rev = GradientReversal.apply(feat, lam)
        out_dom = self.domain_cls(feat_rev)
        return out_cls, out_dom, feat


# ============================================================
# Utility functions
# ============================================================
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


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, test_loader, device, return_details=False):
    model.eval()
    all_preds, all_labels, all_feats = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if hasattr(model, 'domain_cls'):
                out, _, feat = model(x, lam=0.0)
            else:
                out, feat = model(x, return_feat=True)
            _, predicted = out.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_feats.append(feat.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_feats = np.concatenate(all_feats, axis=0)
    
    from sklearn.metrics import f1_score
    result = {
        'accuracy': float((all_preds == all_labels).mean()),
        'f1_macro': float(f1_score(all_labels, all_preds, average='macro')),
    }
    if return_details:
        result['predictions'] = all_preds.tolist()
        result['labels'] = all_labels.tolist()
        result['features'] = all_feats.tolist()
        result['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()
    return result


# ============================================================
# Training functions
# ============================================================
def train_baseline(train_x, train_y, val_x, val_y, n_classes, device,
                   seed=42, epochs=40, batch_size=64):
    set_seed(seed)
    model = FastCNN(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=256)
    
    best_val_acc, best_state, no_imp = 0, None, 0
    train_losses, val_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        ep_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        train_losses.append(ep_loss / len(train_loader))
        
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, p = model(x).max(1)
                total += y.size(0)
                correct += p.eq(y).sum().item()
        va = correct / total
        val_accs.append(va)
        
        if va > best_val_acc:
            best_val_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 12:
                break
    
    model.load_state_dict(best_state)
    return model, {'train_losses': train_losses, 'val_accs': val_accs}


def train_mmd(train_x, train_y, val_x, val_y, test_x, n_classes, device,
              seed=42, epochs=40, batch_size=64, lam=0.5):
    """MMD domain adaptation (source labeled + target unlabeled)"""
    set_seed(seed)
    model = FastCNN(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    cls_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Create mixed batches: half source, half target
    src_ds = TensorDataset(train_x, train_y)
    tgt_ds = TensorDataset(test_x, torch.zeros(len(test_x), dtype=torch.long))
    src_loader = DataLoader(src_ds, batch_size=batch_size//2, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size//2, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=256)
    
    def guassian_kernel(src, tgt, kernels=[1.0, 5.0, 10.0]):
        n_src, n_tgt = src.shape[0], tgt.shape[0]
        n_total = n_src + n_tgt
        total = torch.cat([src, tgt], dim=0)
        total0 = total.unsqueeze(0).expand(n_total, n_total, total.shape[1])
        total1 = total.unsqueeze(1).expand(n_total, n_total, total.shape[1])
        L2 = ((total0 - total1) ** 2).sum(2)
        
        kernel_val = 0
        for k in kernels:
            kernel_val += torch.exp(-L2 / k)
        
        XX = kernel_val[:n_src, :n_src].mean()
        YY = kernel_val[n_src:, n_src:].mean()
        XY = kernel_val[:n_src, n_src:].mean()
        return XX + YY - 2 * XY
    
    best_val_acc, best_state, no_imp = 0, None, 0
    
    for epoch in range(epochs):
        model.train()
        # Iterate through both loaders
        tgt_iter = iter(tgt_loader)
        for sx, sy in src_loader:
            try:
                tx, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tx, _ = next(tgt_iter)
            
            sx, sy = sx.to(device), sy.to(device)
            tx = tx.to(device)
            
            optimizer.zero_grad()
            s_out, s_feat = model(sx, return_feat=True)
            t_out, t_feat = model(tx, return_feat=True)
            
            cls_loss = cls_crit(s_out, sy)
            mmd_loss = guassian_kernel(s_feat, t_feat)
            
            loss = cls_loss + lam * mmd_loss
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, p = model(x).max(1)
                total += y.size(0)
                correct += p.eq(y).sum().item()
        va = correct / total
        
        if va > best_val_acc:
            best_val_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 12:
                break
    
    model.load_state_dict(best_state)
    return model, {}


def train_dann(train_x, train_y, val_x, val_y, test_x, n_classes, device,
               seed=42, epochs=40, batch_size=64, lam=0.5):
    """DANN adversarial domain adaptation"""
    set_seed(seed)
    model = DANN(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    cls_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    dom_crit = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    src_ds = TensorDataset(train_x, train_y)
    tgt_ds = TensorDataset(test_x, torch.zeros(len(test_x), dtype=torch.long))
    src_loader = DataLoader(src_ds, batch_size=batch_size//2, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size//2, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=256)
    
    best_val_acc, best_state, no_imp = 0, None, 0
    
    for epoch in range(epochs):
        model.train()
        p = epoch / epochs
        lambda_val = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0  # Gradual increase
        
        tgt_iter = iter(tgt_loader)
        for sx, sy in src_loader:
            try:
                tx, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tx, _ = next(tgt_iter)
            
            sx, sy = sx.to(device), sy.to(device)
            tx = tx.to(device)
            
            # Domain labels: 0=source, 1=target
            dom_src = torch.zeros(len(sx), dtype=torch.long).to(device)
            dom_tgt = torch.ones(len(tx), dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            s_cls, s_dom, _ = model(sx, lam=lambda_val * lam)
            t_cls, t_dom, _ = model(tx, lam=lambda_val * lam)
            
            cls_loss = cls_crit(s_cls, sy)
            dom_loss = dom_crit(s_dom, dom_src) + dom_crit(t_dom, dom_tgt)
            
            loss = cls_loss + dom_loss
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out, _, _ = model(x, lam=0.0)
                _, p = out.max(1)
                total += y.size(0)
                correct += p.eq(y).sum().item()
        va = correct / total
        
        if va > best_val_acc:
            best_val_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 12:
                break
    
    model.load_state_dict(best_state)
    return model, {}


def train_coral(train_x, train_y, val_x, val_y, test_x, n_classes, device,
                seed=42, epochs=40, batch_size=64, lam=0.5):
    """CORAL domain adaptation"""
    set_seed(seed)
    model = FastCNN(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    cls_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    src_ds = TensorDataset(train_x, train_y)
    tgt_ds = TensorDataset(test_x, torch.zeros(len(test_x), dtype=torch.long))
    src_loader = DataLoader(src_ds, batch_size=batch_size//2, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size//2, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=256)
    
    best_val_acc, best_state, no_imp = 0, None, 0
    
    for epoch in range(epochs):
        model.train()
        tgt_iter = iter(tgt_loader)
        for sx, sy in src_loader:
            try:
                tx, _ = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tx, _ = next(tgt_iter)
            
            sx, sy = sx.to(device), sy.to(device)
            tx = tx.to(device)
            
            optimizer.zero_grad()
            s_out, s_feat = model(sx, return_feat=True)
            t_out, t_feat = model(tx, return_feat=True)
            
            cls_loss = cls_crit(s_out, sy)
            
            # CORAL loss: align covariance
            s_cov = torch.cov(s_feat.T)
            t_cov = torch.cov(t_feat.T)
            coral_loss = torch.norm(s_cov - t_cov, p='fro') ** 2 / (4 * s_feat.shape[1] ** 2)
            
            loss = cls_loss + lam * coral_loss
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, p = model(x).max(1)
                total += y.size(0)
                correct += p.eq(y).sum().item()
        va = correct / total
        
        if va > best_val_acc:
            best_val_acc = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 12:
                break
    
    model.load_state_dict(best_state)
    return model, {}


def train_pseudo(train_x, train_y, test_x, val_x, val_y, test_y,
                 n_classes, device, seed=42, conf_thresh=0.9,
                 n_rounds=2, pseudo_weight=1.0, epochs=40):
    """Pseudo-label self-training"""
    set_seed(seed)
    batch_size = 64
    
    cur_x, cur_y = train_x.clone(), train_y.clone()
    cur_w = torch.ones(len(train_y), dtype=torch.float32)
    
    best_model = None
    best_val = 0
    
    for rnd in range(n_rounds):
        model = FastCNN(n_classes=n_classes).to(device)
        ds = TensorDataset(cur_x, cur_y, cur_w)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=256)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(reduction='none')
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_acc, best_st, no_imp = 0, None, 0
        
        for epoch in range(epochs):
            model.train()
            for x, y, w in loader:
                x, y, w = x.to(device), y.to(device), w.to(device)
                optimizer.zero_grad()
                loss = (crit(model(x), y) * w).mean()
                loss.backward()
                optimizer.step()
            sched.step()
            
            model.eval()
            c = t = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    _, p = model(x).max(1)
                    t += y.size(0)
                    c += p.eq(y).sum().item()
            va = c / t
            if va > best_acc:
                best_acc = va
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= 12:
                    break
        
        model.load_state_dict(best_st)
        if best_acc > best_val:
            best_val = best_acc
            best_model = {k: v.clone() for k, v in model.state_dict().items()}
        
        if rnd == n_rounds - 1:
            break
        
        # Generate pseudo labels
        model.eval()
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=512)
        probs_list = []
        with torch.no_grad():
            for x, _ in test_loader:
                probs_list.append(F.softmax(model(x.to(device)), dim=1).cpu())
        
        all_probs = torch.cat(probs_list, dim=0)
        conf, pred = all_probs.max(dim=1)
        mask = conf > conf_thresh
        n_pseudo = mask.sum().item()
        
        if n_pseudo == 0:
            break
        
        cur_x = torch.cat([cur_x, test_x[mask]])
        cur_y = torch.cat([cur_y, pred[mask]])
        cur_w = torch.cat([cur_w, torch.full((n_pseudo,), pseudo_weight)])
    
    # Load best model
    final_model = FastCNN(n_classes=n_classes).to(device)
    final_model.load_state_dict(best_model)
    return final_model, {}


# ============================================================
# Main experiments
# ============================================================
def run_full_comparison(src_x, src_y, tgt_x, tgt_y, n_classes, device,
                        dataset_name, n_repeats=10):
    """Run full comparison: Baseline, CORAL, MMD, DANN, Pseudo-Label"""
    print(f"\n{'='*70}")
    print(f"  Full comparison on {dataset_name} (n_repeats={n_repeats}, FastCNN)")
    print(f"{'='*70}")
    
    methods = ['baseline', 'coral', 'mmd', 'dann', 'pseudo']
    results = {m: {'accuracies': [], 'f1s': []} for m in methods}
    
    for rep in range(n_repeats):
        seed = rep * 1000 + 42
        print(f"\n  Rep {rep+1}/{n_repeats} (seed={seed})...")
        t0 = time.time()
        
        tr_x, tr_y, vx, vy = few_shot_split(src_x, src_y, 50, 10, seed=seed)
        
        # Baseline
        m, _ = train_baseline(tr_x, tr_y, vx, vy, n_classes, device, seed=seed)
        r = evaluate(m, DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512), device)
        results['baseline']['accuracies'].append(r['accuracy'])
        results['baseline']['f1s'].append(r['f1_macro'])
        print(f"    Baseline: {r['accuracy']*100:.2f}%", end='')
        
        # CORAL
        m, _ = train_coral(tr_x, tr_y, vx, vy, tgt_x, n_classes, device, seed=seed)
        r = evaluate(m, DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512), device)
        results['coral']['accuracies'].append(r['accuracy'])
        results['coral']['f1s'].append(r['f1_macro'])
        print(f", CORAL: {r['accuracy']*100:.2f}%", end='')
        
        # MMD
        m, _ = train_mmd(tr_x, tr_y, vx, vy, tgt_x, n_classes, device, seed=seed)
        r = evaluate(m, DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512), device)
        results['mmd']['accuracies'].append(r['accuracy'])
        results['mmd']['f1s'].append(r['f1_macro'])
        print(f", MMD: {r['accuracy']*100:.2f}%", end='')
        
        # DANN
        m, _ = train_dann(tr_x, tr_y, vx, vy, tgt_x, n_classes, device, seed=seed)
        r = evaluate(m, DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512), device)
        results['dann']['accuracies'].append(r['accuracy'])
        results['dann']['f1s'].append(r['f1_macro'])
        print(f", DANN: {r['accuracy']*100:.2f}%", end='')
        
        # Pseudo-label
        m, _ = train_pseudo(tr_x, tr_y, tgt_x, vx, vy, tgt_y, n_classes, device, seed=seed)
        r = evaluate(m, DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512), device)
        results['pseudo']['accuracies'].append(r['accuracy'])
        results['pseudo']['f1s'].append(r['f1_macro'])
        print(f", Pseudo: {r['accuracy']*100:.2f}%  [{time.time()-t0:.1f}s]")
    
    # Summary
    print(f"\n  Summary ({dataset_name}, n={n_repeats}):")
    for m in methods:
        accs = results[m]['accuracies']
        print(f"    {m:10s}: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    
    # Compute mean and std
    for m in methods:
        results[m]['mean_acc'] = float(np.mean(results[m]['accuracies']))
        results[m]['std_acc'] = float(np.std(results[m]['accuracies']))
        results[m]['mean_f1'] = float(np.mean(results[m]['f1s']))
        results[m]['std_f1'] = float(np.std(results[m]['f1s']))
    
    return results


def run_ablation(src_x, src_y, tgt_x, tgt_y, n_classes, device, n_repeats=3):
    """Ablation study with multiple repeats"""
    print(f"\n{'='*70}")
    print(f"  Ablation study (n_repeats={n_repeats}, FastCNN)")
    print(f"{'='*70}")
    
    abl_results = {}
    
    # Baseline
    print("\n  Baseline...")
    base_accs = []
    for rep in range(n_repeats):
        seed = rep * 1000 + 42
        tr_x, tr_y, vx, vy = few_shot_split(src_x, src_y, 50, 10, seed=seed)
        m, _ = train_baseline(tr_x, tr_y, vx, vy, n_classes, device, seed=seed)
        r = evaluate(m, DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512), device)
        base_accs.append(r['accuracy'])
    abl_results['baseline'] = {
        'mean': float(np.mean(base_accs)),
        'std': float(np.std(base_accs)),
        'accuracies': base_accs,
    }
    print(f"    {np.mean(base_accs)*100:.2f}% ± {np.std(base_accs)*100:.2f}%")
    
    # Confidence threshold
    print("\n  Confidence threshold...")
    for ct in [0.8, 0.9, 0.95]:
        accs = []
        for rep in range(n_repeats):
            seed = rep * 1000 + 42
            tr_x, tr_y, vx, vy = few_shot_split(src_x, src_y, 50, 10, seed=seed)
            m, _ = train_pseudo(tr_x, tr_y, tgt_x, vx, vy, tgt_y, n_classes, device,
                                seed=seed, conf_thresh=ct, n_rounds=2)
            r = evaluate(m, DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512), device)
            accs.append(r['accuracy'])
        key = f'conf_{ct:.2f}'
        abl_results[key] = {
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
            'accuracies': accs,
        }
        print(f"    τ={ct}: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    
    # Number of rounds
    print("\n  Number of rounds...")
    for nr in [1, 2, 3]:
        accs = []
        for rep in range(n_repeats):
            seed = rep * 1000 + 42
            tr_x, tr_y, vx, vy = few_shot_split(src_x, src_y, 50, 10, seed=seed)
            m, _ = train_pseudo(tr_x, tr_y, tgt_x, vx, vy, tgt_y, n_classes, device,
                                seed=seed, conf_thresh=0.9, n_rounds=nr)
            r = evaluate(m, DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512), device)
            accs.append(r['accuracy'])
        key = f'rounds_{nr}'
        abl_results[key] = {
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
            'accuracies': accs,
        }
        print(f"    R={nr}: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    
    # Pseudo weight
    print("\n  Pseudo weight...")
    for pw in [0.5, 0.7, 1.0]:
        accs = []
        for rep in range(n_repeats):
            seed = rep * 1000 + 42
            tr_x, tr_y, vx, vy = few_shot_split(src_x, src_y, 50, 10, seed=seed)
            m, _ = train_pseudo(tr_x, tr_y, tgt_x, vx, vy, tgt_y, n_classes, device,
                                seed=seed, conf_thresh=0.9, n_rounds=2, pseudo_weight=pw)
            r = evaluate(m, DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512), device)
            accs.append(r['accuracy'])
        key = f'weight_{pw:.1f}'
        abl_results[key] = {
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
            'accuracies': accs,
        }
        print(f"    α={pw}: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    
    return abl_results


def run_visualization(src_x, src_y, tgt_x, tgt_y, n_classes, device, dataset_name):
    """Generate features for t-SNE visualization and confusion matrices"""
    print(f"\n{'='*70}")
    print(f"  Visualization data generation ({dataset_name})")
    print(f"{'='*70}")
    
    seed = 42
    tr_x, tr_y, vx, vy = few_shot_split(src_x, src_y, 50, 10, seed=seed)
    
    vis_data = {}
    
    # Baseline features
    print("  Training baseline...")
    m, _ = train_baseline(tr_x, tr_y, vx, vy, n_classes, device, seed=seed)
    r = evaluate(m, DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512), device, return_details=True)
    vis_data['baseline'] = {
        'features': r['features'],
        'labels': r['labels'],
        'predictions': r['predictions'],
        'confusion_matrix': r['confusion_matrix'],
        'accuracy': r['accuracy'],
    }
    print(f"    Acc: {r['accuracy']*100:.2f}%")
    
    # Pseudo-label features
    print("  Training pseudo-label...")
    m, _ = train_pseudo(tr_x, tr_y, tgt_x, vx, vy, tgt_y, n_classes, device, seed=seed)
    r = evaluate(m, DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512), device, return_details=True)
    vis_data['pseudo'] = {
        'features': r['features'],
        'labels': r['labels'],
        'predictions': r['predictions'],
        'confusion_matrix': r['confusion_matrix'],
        'accuracy': r['accuracy'],
    }
    print(f"    Acc: {r['accuracy']*100:.2f}%")
    
    # DANN features
    print("  Training DANN...")
    m, _ = train_dann(tr_x, tr_y, vx, vy, tgt_x, n_classes, device, seed=seed)
    r = evaluate(m, DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512), device, return_details=True)
    vis_data['dann'] = {
        'features': r['features'],
        'labels': r['labels'],
        'predictions': r['predictions'],
        'confusion_matrix': r['confusion_matrix'],
        'accuracy': r['accuracy'],
    }
    print(f"    Acc: {r['accuracy']*100:.2f}%")
    
    # t-SNE embedding (subsample for speed)
    print("  Computing t-SNE...")
    n_samples = min(500, len(tgt_y))
    idx = np.random.choice(len(tgt_y), n_samples, replace=False)
    
    for method in ['baseline', 'pseudo', 'dann']:
        feats = np.array(vis_data[method]['features'])[idx]
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples//5))
        emb = tsne.fit_transform(feats)
        vis_data[method]['tsne_2d'] = emb.tolist()
        vis_data[method]['tsne_labels'] = np.array(vis_data[method]['labels'])[idx].tolist()
    
    return vis_data


def main():
    device = torch.device('cpu')
    data_dir_cwru = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'cwru_real')
    data_dir_pu = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'pu_data')
    
    t_start = time.time()
    os.makedirs('results', exist_ok=True)
    
    # Load CWRU
    print("Loading CWRU data...")
    t0 = time.time()
    cwru_src = CWRUDataset(data_dir_cwru, window_size=512, stride=1024,
                           load_conditions=[0], return_condition=False)
    cwru_tgt = CWRUDataset(data_dir_cwru, window_size=512, stride=1024,
                           load_conditions=[3], return_condition=False)
    print(f"  Done in {time.time()-t0:.1f}s (src={len(cwru_src)}, tgt={len(cwru_tgt)})")
    
    cwru_src_x = torch.from_numpy(cwru_src.samples).float()
    cwru_src_y = torch.from_numpy(cwru_src.labels).long()
    cwru_tgt_x = torch.from_numpy(cwru_tgt.samples).float()
    cwru_tgt_y = torch.from_numpy(cwru_tgt.labels).long()
    n_cls_cwru = len(torch.unique(cwru_src_y))
    print(f"  Classes: {n_cls_cwru}")
    
    # Load PU
    print("\nLoading PU data...")
    t0 = time.time()
    bearings = ['K001', 'K002', 'K003', 'KA04', 'KA15', 'KI04', 'KI14']
    pu_src = PUDataset(data_dir_pu, bearings, condition='N15_M07_F10',
                       window_size=512, stride=1024, max_files_per_bearing=5)
    pu_tgt = PUDataset(data_dir_pu, bearings, condition='N09_M07_F10',
                       window_size=512, stride=1024, max_files_per_bearing=5)
    print(f"  Done in {time.time()-t0:.1f}s")
    pu_src.summary()
    pu_tgt.summary()
    
    pu_src_x = torch.from_numpy(pu_src.samples).float()
    pu_src_y = torch.from_numpy(pu_src.labels).long()
    pu_tgt_x = torch.from_numpy(pu_tgt.samples).float()
    pu_tgt_y = torch.from_numpy(pu_tgt.labels).long()
    n_cls_pu = 3
    
    # ============================================================
    # Experiment 1: CWRU full comparison (10 repeats)
    # ============================================================
    cwru_results = run_full_comparison(
        cwru_src_x, cwru_src_y, cwru_tgt_x, cwru_tgt_y,
        n_cls_cwru, device, 'CWRU', n_repeats=10
    )
    with open('results/revision_cwru_10repeats.json', 'w') as f:
        json.dump(cwru_results, f, indent=2)
    
    # ============================================================
    # Experiment 2: PU full comparison (10 repeats, same architecture)
    # ============================================================
    pu_results = run_full_comparison(
        pu_src_x, pu_src_y, pu_tgt_x, pu_tgt_y,
        n_cls_pu, device, 'PU', n_repeats=10
    )
    with open('results/revision_pu_10repeats.json', 'w') as f:
        json.dump(pu_results, f, indent=2)
    
    # ============================================================
    # Experiment 3: Ablation (3 repeats)
    # ============================================================
    abl_results = run_ablation(
        cwru_src_x, cwru_src_y, cwru_tgt_x, cwru_tgt_y,
        n_cls_cwru, device, n_repeats=3
    )
    with open('results/revision_ablation_3repeats.json', 'w') as f:
        json.dump(abl_results, f, indent=2)
    
    # ============================================================
    # Experiment 4: Visualization data
    # ============================================================
    cwru_vis = run_visualization(
        cwru_src_x, cwru_src_y, cwru_tgt_x, cwru_tgt_y,
        n_cls_cwru, device, 'CWRU'
    )
    with open('results/revision_cwru_visualization.json', 'w') as f:
        json.dump(cwru_vis, f, indent=2)
    
    pu_vis = run_visualization(
        pu_src_x, pu_src_y, pu_tgt_x, pu_tgt_y,
        n_cls_pu, device, 'PU'
    )
    with open('results/revision_pu_visualization.json', 'w') as f:
        json.dump(pu_vis, f, indent=2)
    
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"ALL EXPERIMENTS COMPLETE! Total time: {elapsed/60:.1f} min")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
