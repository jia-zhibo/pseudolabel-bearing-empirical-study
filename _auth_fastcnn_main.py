# -*- coding: utf-8 -*-
"""
全套同口径插桩重跑（CWRU + PU，FastCNN，n=10）。

为什么必须"同口径"：
  实测发现，CPU 上只要在训练循环内额外做一次前向推理（记录逐 epoch 目标域精度所需的操作），
  后续训练的数值轨迹就会改变（conv 的内存布局/算子选择被推理过程改写），即使固定线程数与
  随机种子：
        训练中不推理 -> 0.789916（= 已发表值）
        每 epoch 推理 -> 0.766645
  因此若只对伪标签分支插桩而其他方法不插桩，表格内的比较会被该副作用污染。
  这里对 baseline / CORAL / MMD / DANN / Pseudo-Label / 损失对齐版 Pseudo-Label
  六个条目施加完全相同的插桩量，保证比较公平。

输出：results/revision_full_instrumented.json
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

torch.set_num_threads(10)

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, 'results')
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'data'))

from cwru_data import CWRUDataset
from pu_data import PUDataset
from _run_revision_experiments import few_shot_split, set_seed, FastCNN, DANN, evaluate

EPOCHS = 40
N_REPEATS = 10
SEEDS = [r * 1000 + 42 for r in range(N_REPEATS)]


def guassian_kernel(src, tgt, kernels=(1.0, 5.0, 10.0)):
    n_src, n_tgt = src.shape[0], tgt.shape[0]
    n_total = n_src + n_tgt
    total = torch.cat([src, tgt], dim=0)
    total0 = total.unsqueeze(0).expand(n_total, n_total, total.shape[1])
    total1 = total.unsqueeze(1).expand(n_total, n_total, total.shape[1])
    L2 = ((total0 - total1) ** 2).sum(2)
    kernel_val = 0
    for k in kernels:
        kernel_val = kernel_val + torch.exp(-L2 / k)
    XX = kernel_val[:n_src, :n_src].mean()
    YY = kernel_val[n_src:, n_src:].mean()
    XY = kernel_val[:n_src, n_src:].mean()
    YX = kernel_val[n_src:, :n_src].mean()
    return XX + YY - XY - YX


def _record(model, val_loader, tgt_loader, device, dann=False):
    """统一的逐 epoch 记录：源域验证精度 + 目标域精度。"""
    model.eval()
    with torch.no_grad():
        c = t = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x, lam=0.0) if dann else model(x)
            out = out[0] if dann else out
            _, p = out.max(1)
            t += y.size(0); c += p.eq(y).sum().item()
        va = c / t
        c = t = 0
        for x, y in tgt_loader:
            x, y = x.to(device), y.to(device)
            out = model(x, lam=0.0) if dann else model(x)
            out = out[0] if dann else out
            _, p = out.max(1)
            t += y.size(0); c += p.eq(y).sum().item()
        ta = c / t
    return va, ta


# ---------------------------------------------------------------- baseline
def tr_baseline(tr_x, tr_y, vx, vy, tx, ty, n_classes, device, seed=42, epochs=EPOCHS):
    from _run_curves_full import train_baseline_log
    m, h = train_baseline_log(tr_x, tr_y, vx, vy, tx, ty, n_classes, device,
                              seed=seed, epochs=epochs)
    return m, [h]


# ---------------------------------------------------------------- pseudo
def tr_pseudo(tr_x, tr_y, vx, vy, tx, ty, n_classes, device, seed=42, epochs=EPOCHS,
              ls=0.0):
    from _run_curves_full import train_pseudo_log
    m, h, ps = train_pseudo_log(tr_x, tr_y, tx, vx, vy, ty, n_classes, device,
                                seed=seed, conf_thresh=0.9, n_rounds=2,
                                pseudo_weight=1.0, epochs=epochs, label_smoothing=ls)
    return m, h, ps


# ---------------------------------------------------------------- MMD
def tr_mmd(tr_x, tr_y, vx, vy, tx, ty, n_classes, device, seed=42, epochs=EPOCHS, lam=0.5):
    set_seed(seed)
    model = FastCNN(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    cls_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    src_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=32, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(TensorDataset(tx, torch.zeros(len(tx), dtype=torch.long)),
                            batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(vx, vy), batch_size=256)
    tgt_eval = DataLoader(TensorDataset(tx, ty), batch_size=512)

    best_val, best_st, no_imp = 0, None, 0
    hist = {'train_loss': [], 'val_acc': [], 'target_acc': []}
    for epoch in range(epochs):
        model.train()
        running, nb = 0.0, 0
        t_iter = iter(tgt_loader)
        for sx, sy in src_loader:
            try:
                tx_, _ = next(t_iter)
            except StopIteration:
                t_iter = iter(tgt_loader)
                tx_, _ = next(t_iter)
            sx, sy, tx_ = sx.to(device), sy.to(device), tx_.to(device)
            optimizer.zero_grad()
            s_out, s_feat = model(sx, return_feat=True)
            t_out, t_feat = model(tx_, return_feat=True)
            loss = cls_crit(s_out, sy) + lam * guassian_kernel(s_feat, t_feat)
            loss.backward(); optimizer.step()
            running += loss.item(); nb += 1
        scheduler.step()
        hist['train_loss'].append(running / max(nb, 1))
        va, ta = _record(model, val_loader, tgt_eval, device)
        hist['val_acc'].append(va); hist['target_acc'].append(ta)
        if va > best_val:
            best_val, no_imp = va, 0
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= 12:
                break
    model.load_state_dict(best_st)
    return model, [{'round': 0, 'n_labeled': int(tr_x.shape[0]), **hist}]


# ---------------------------------------------------------------- CORAL
def tr_coral(tr_x, tr_y, vx, vy, tx, ty, n_classes, device, seed=42, epochs=EPOCHS, lam=0.5):
    set_seed(seed)
    model = FastCNN(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    cls_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    src_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=32, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(TensorDataset(tx, torch.zeros(len(tx), dtype=torch.long)),
                            batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(vx, vy), batch_size=256)
    tgt_eval = DataLoader(TensorDataset(tx, ty), batch_size=512)

    best_val, best_st, no_imp = 0, None, 0
    hist = {'train_loss': [], 'val_acc': [], 'target_acc': []}
    for epoch in range(epochs):
        model.train()
        running, nb = 0.0, 0
        t_iter = iter(tgt_loader)
        for sx, sy in src_loader:
            try:
                tx_, _ = next(t_iter)
            except StopIteration:
                t_iter = iter(tgt_loader)
                tx_, _ = next(t_iter)
            sx, sy, tx_ = sx.to(device), sy.to(device), tx_.to(device)
            optimizer.zero_grad()
            s_out, s_feat = model(sx, return_feat=True)
            t_out, t_feat = model(tx_, return_feat=True)
            cls_loss = cls_crit(s_out, sy)
            s_cov = torch.cov(s_feat.T)
            t_cov = torch.cov(t_feat.T)
            coral_loss = torch.norm(s_cov - t_cov, p='fro') ** 2 / (4 * s_feat.shape[1] ** 2)
            loss = cls_loss + lam * coral_loss
            loss.backward(); optimizer.step()
            running += loss.item(); nb += 1
        scheduler.step()
        hist['train_loss'].append(running / max(nb, 1))
        va, ta = _record(model, val_loader, tgt_eval, device)
        hist['val_acc'].append(va); hist['target_acc'].append(ta)
        if va > best_val:
            best_val, no_imp = va, 0
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= 12:
                break
    model.load_state_dict(best_st)
    return model, [{'round': 0, 'n_labeled': int(tr_x.shape[0]), **hist}]


# ---------------------------------------------------------------- DANN
def tr_dann(tr_x, tr_y, vx, vy, tx, ty, n_classes, device, seed=42, epochs=EPOCHS, lam=0.5):
    set_seed(seed)
    model = DANN(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    cls_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    dom_crit = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    src_loader = DataLoader(TensorDataset(tr_x, tr_y), batch_size=32, shuffle=True, drop_last=True)
    tgt_loader = DataLoader(TensorDataset(tx, torch.zeros(len(tx), dtype=torch.long)),
                            batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(vx, vy), batch_size=256)
    tgt_eval = DataLoader(TensorDataset(tx, ty), batch_size=512)

    best_val, best_st, no_imp = 0, None, 0
    hist = {'train_loss': [], 'val_acc': [], 'target_acc': []}
    for epoch in range(epochs):
        model.train()
        running, nb = 0.0, 0
        lam_p = epoch / epochs
        lambda_val = 2.0 / (1.0 + np.exp(-10 * lam_p)) - 1.0
        t_iter = iter(tgt_loader)
        for sx, sy in src_loader:
            try:
                tx_, _ = next(t_iter)
            except StopIteration:
                t_iter = iter(tgt_loader)
                tx_, _ = next(t_iter)
            sx, sy, tx_ = sx.to(device), sy.to(device), tx_.to(device)
            dom_src = torch.zeros(len(sx), dtype=torch.long).to(device)
            dom_tgt = torch.ones(len(tx_), dtype=torch.long).to(device)
            optimizer.zero_grad()
            s_cls, s_dom, _ = model(sx, lam=lambda_val * lam)
            t_cls, t_dom, _ = model(tx_, lam=lambda_val * lam)
            loss = cls_crit(s_cls, sy) + dom_crit(s_dom, dom_src) + dom_crit(t_dom, dom_tgt)
            loss.backward(); optimizer.step()
            running += loss.item(); nb += 1
        scheduler.step()
        hist['train_loss'].append(running / max(nb, 1))
        va, ta = _record(model, val_loader, tgt_eval, device, dann=True)
        hist['val_acc'].append(va); hist['target_acc'].append(ta)
        if va > best_val:
            best_val, no_imp = va, 0
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= 12:
                break
    model.load_state_dict(best_st)
    return model, [{'round': 0, 'n_labeled': int(tr_x.shape[0]), **hist}]


METHODS = [
    ('baseline', lambda *a, **k: tr_baseline(*a, **k)),
    ('coral', lambda *a, **k: tr_coral(*a, **k)),
    ('mmd', lambda *a, **k: tr_mmd(*a, **k)),
    ('dann', lambda *a, **k: tr_dann(*a, **k)),
    ('pseudo_plain', lambda *a, **k: tr_pseudo(*a, ls=0.0, **k)),
    ('pseudo_matched', lambda *a, **k: tr_pseudo(*a, ls=0.1, **k)),
]


def load_data():
    cdir = os.path.join(BASE, 'data', 'cwru_real')
    pdir = os.path.join(BASE, 'data', 'pu_data')
    s = CWRUDataset(cdir, window_size=512, stride=1024, load_conditions=[0], return_condition=False)
    t = CWRUDataset(cdir, window_size=512, stride=1024, load_conditions=[3], return_condition=False)
    cwru = (torch.from_numpy(s.samples).float(), torch.from_numpy(s.labels).long(),
            torch.from_numpy(t.samples).float(), torch.from_numpy(t.labels).long())
    bearings = ['K001', 'K002', 'K003', 'KA04', 'KA15', 'KI04', 'KI14']
    ps = PUDataset(pdir, bearings, condition='N15_M07_F10', window_size=512, stride=1024,
                   max_files_per_bearing=5)
    pt = PUDataset(pdir, bearings, condition='N09_M07_F10', window_size=512, stride=1024,
                   max_files_per_bearing=5)
    pu = (torch.from_numpy(ps.samples).float(), torch.from_numpy(ps.labels).long(),
          torch.from_numpy(pt.samples).float(), torch.from_numpy(pt.labels).long())
    return cwru, pu


def main():
    device = torch.device('cpu')
    t0 = time.time()
    cwru, pu = load_data()
    out = {'meta': {'n_repeats': N_REPEATS, 'seeds': SEEDS, 'epochs_max': EPOCHS,
                    'torch': torch.__version__, 'threads': torch.get_num_threads(),
                    'instrumentation': 'identical per-epoch val+target evaluation '
                                       'applied to every method',
                    'purpose': 'authoritative re-run for bit-exact verification'}}
    for name, (sx, sy, tx, ty) in (('cwru', cwru), ('pu', pu)):
        n_classes = len(torch.unique(sy))
        tgt_eval = DataLoader(TensorDataset(tx, ty), batch_size=512)
        print('\n' + '=' * 72)
        print('%s  (classes=%d, src=%d, tgt=%d)' % (name.upper(), n_classes, len(sx), len(tx)))
        print('=' * 72, flush=True)
        out[name] = {}
        for mname, fn in METHODS:
            accs, hists = [], []
            for seed in SEEDS:
                tr_x, tr_y, vx, vy = few_shot_split(sx, sy, 50, 10, seed=seed)
                res = fn(tr_x, tr_y, vx, vy, tx, ty, n_classes, device, seed=seed)
                m, h = res[0], res[1]
                accs.append(float(evaluate(m, tgt_eval, device)['accuracy']))
                hists.append(h)
            out[name][mname] = {'accuracies': accs,
                                'mean': float(np.mean(accs)),
                                'std': float(np.std(accs, ddof=1)),
                                'history': hists}
            print('  %-16s mean=%.2f%%  sd=%.2f  %s'
                  % (mname, np.mean(accs) * 100, np.std(accs, ddof=1) * 100,
                     [round(a * 100, 2) for a in accs]), flush=True)
    path = os.path.join(RESULTS_DIR, 'authoritative_fastcnn_main.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), indent=2)
    print('\nSaved: %s' % path)
    print('Total %.1f min' % ((time.time() - t0) / 60))


if __name__ == '__main__':
    main()
