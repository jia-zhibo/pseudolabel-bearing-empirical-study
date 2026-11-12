# -*- coding: utf-8 -*-
"""
整体重跑消融实验 + 采集真实逐 epoch 训练曲线（供 §6.4 使用）。

为什么必须整体重跑（而不是只补曲线）：
  * 基准(train_baseline)路径可在本环境逐位复现：0.7899/0.7847/0.5915 与已发表完全一致；
  * 但伪标签路径在 round≥2 无法复现已发表数值——已发表的伪标签行由旧 torch 版本产出。
    已验证：直接调用原始未改动的 train_pseudo 得到 [0.7809, 0.7440, 0.5107]，
    与本次记录版一致，而与已发表的 [0.8197, 0.7440, 0.5158] 不同。
    机理：置信度阈值 τ=0.9 会放大浮点级差异——边界样本的取舍改变伪标签集合，
    进而改变后续轮次的训练数据，属于该方法的固有敏感性。
  * 因此若只补曲线，曲线端点会与 Table 3 不一致。故整套消融在上述两种方式下重跑，
    表格与曲线来自同一次执行，端点严格自洽。

两种配置（这是一个重要的方法学发现）：
  plain    = 论文原始配置：伪标签分支用普通交叉熵，而其余 4 个方法(含 baseline)
             都用 label_smoothing=0.1；
  aligned  = 对齐配置：伪标签分支同样使用 label_smoothing=0.1。
两者对比用于§6.4/§7.4：loss 正则化 → 置信度校准 → 伪标签精度 → 自训练成败。
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

# CPU 训练结果与线程数相关（已实测：同一 seed，threads=10 与 12 给出不同数值），
# 因此显式固定线程数，保证结果可复现。threads=10 在本机可逐位复现已发表
# revision_*_10repeats.json 中的 baseline 结果。
torch.set_num_threads(10)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'data'))

from cwru_data import CWRUDataset
from _run_revision_experiments import few_shot_split, set_seed, FastCNN, evaluate

RESULTS_DIR = os.path.join(BASE, 'results')
N_REPEATS = 3
EPOCHS = 40
SEEDS = [r * 1000 + 42 for r in range(N_REPEATS)]


# ------------------------------------------------------------
# 带逐 epoch 记录的自训练（train_pseudo 的等价改写）
# ------------------------------------------------------------
def train_pseudo_log(train_x, train_y, test_x, val_x, val_y, test_y,
                     n_classes, device, seed=42, conf_thresh=0.9,
                     n_rounds=2, pseudo_weight=1.0, epochs=40, label_smoothing=0.0):
    set_seed(seed)
    cur_x, cur_y = train_x.clone(), train_y.clone()
    cur_w = torch.ones(len(train_y), dtype=torch.float32)

    best_model, best_val = None, 0
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=256)
    tgt_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=512)

    history, pseudo_stats = [], []

    for rnd in range(n_rounds):
        model = FastCNN(n_classes=n_classes).to(device)
        loader = DataLoader(TensorDataset(cur_x, cur_y, cur_w), batch_size=64, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(reduction='none', label_smoothing=label_smoothing)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc, best_st, no_imp = 0, None, 0
        ep_loss, ep_val, ep_tgt = [], [], []

        for _ in range(epochs):
            model.train()
            running, nb = 0.0, 0
            for x, y, w in loader:
                x, y, w = x.to(device), y.to(device), w.to(device)
                optimizer.zero_grad()
                loss = (crit(model(x), y) * w).mean()
                loss.backward()
                optimizer.step()
                running += loss.item()
                nb += 1
            sched.step()
            ep_loss.append(running / max(nb, 1))

            model.eval()
            with torch.no_grad():          # 记录不消耗 RNG
                c = t = 0
                for x, y in val_loader:
                    _, p = model(x.to(device)).max(1)
                    t += y.size(0); c += p.eq(y).sum().item()
                ep_val.append(c / t)
                c = t = 0
                for x, y in tgt_loader:
                    _, p = model(x.to(device)).max(1)
                    t += y.size(0); c += p.eq(y).sum().item()
                ep_tgt.append(c / t)

            if ep_val[-1] > best_acc:
                best_acc, no_imp = ep_val[-1], 0
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_imp += 1
                if no_imp >= 12:
                    break

        model.load_state_dict(best_st)
        if best_acc > best_val:
            best_val = best_acc
            best_model = {k: v.clone() for k, v in model.state_dict().items()}

        history.append({'round': rnd, 'n_labeled': int(cur_x.shape[0]),
                        'train_loss': [float(v) for v in ep_loss],
                        'val_acc': [float(v) for v in ep_val],
                        'target_acc': [float(v) for v in ep_tgt]})

        if rnd == n_rounds - 1:
            break

        model.eval()
        with torch.no_grad():
            probs = torch.cat([torch.softmax(model(x.to(device)), dim=1).cpu()
                               for x, _ in DataLoader(TensorDataset(test_x, test_y), batch_size=512)])
        conf, pred = probs.max(dim=1)
        mask = conf > conf_thresh
        n_sel = int(mask.sum())
        prec = float(pred[mask].eq(test_y[mask]).sum().item() / n_sel) if n_sel else 0.0
        pseudo_stats.append({'round': rnd, 'n_selected': n_sel,
                             'selection_rate': float(n_sel / test_x.shape[0]),
                             'pseudo_precision': prec,
                             'target_acc_at_selection': float(ep_tgt[-1])})
        if n_sel == 0:
            break
        cur_x = torch.cat([cur_x, test_x[mask]])
        cur_y = torch.cat([cur_y, pred[mask]])
        cur_w = torch.cat([cur_w, torch.full((n_sel,), pseudo_weight)])

    final = FastCNN(n_classes=n_classes).to(device)
    final.load_state_dict(best_model)
    return final, history, pseudo_stats


def train_baseline_log(train_x, train_y, val_x, val_y, test_x, test_y,
                       n_classes, device, seed=42, epochs=40):
    set_seed(seed)
    model = FastCNN(n_classes=n_classes).to(device)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=256)
    tgt_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=512)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)   # 与原始 train_baseline 一致
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_st, no_imp = 0, None, 0
    ep_loss, ep_val, ep_tgt = [], [], []
    for _ in range(epochs):
        model.train()
        running, nb = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad()
            loss = crit(model(x), y)
            loss.backward(); optimizer.step()
            running += loss.item(); nb += 1
        sched.step()
        ep_loss.append(running / max(nb, 1))
        model.eval()
        with torch.no_grad():
            c = t = 0
            for x, y in val_loader:
                _, p = model(x.to(device)).max(1)
                t += y.size(0); c += p.eq(y).sum().item()
            ep_val.append(c / t)
            c = t = 0
            for x, y in tgt_loader:
                _, p = model(x.to(device)).max(1)
                t += y.size(0); c += p.eq(y).sum().item()
            ep_tgt.append(c / t)
        if ep_val[-1] > best_acc:
            best_acc, no_imp = ep_val[-1], 0
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= 12:
                break
    model.load_state_dict(best_st)
    return model, {'round': 0, 'n_labeled': int(train_x.shape[0]),
                   'train_loss': [float(v) for v in ep_loss],
                   'val_acc': [float(v) for v in ep_val],
                   'target_acc': [float(v) for v in ep_tgt]}


# ------------------------------------------------------------
#  ablation 配置（与原 run_ablation 完全一致）
# ------------------------------------------------------------
CONFIGS = {
    'conf_0.80': dict(conf_thresh=0.8, n_rounds=2),
    'conf_0.90': dict(conf_thresh=0.9, n_rounds=2),
    'conf_0.95': dict(conf_thresh=0.95, n_rounds=2),
    'rounds_1': dict(conf_thresh=0.9, n_rounds=1),
    'rounds_2': dict(conf_thresh=0.9, n_rounds=2),
    'rounds_3': dict(conf_thresh=0.9, n_rounds=3),
    'weight_0.5': dict(conf_thresh=0.9, n_rounds=2, pseudo_weight=0.5),
    'weight_0.7': dict(conf_thresh=0.9, n_rounds=2, pseudo_weight=0.7),
    'weight_1.0': dict(conf_thresh=0.9, n_rounds=2, pseudo_weight=1.0),
}


def main():
    device = torch.device('cpu')
    dd = os.path.join(BASE, 'data', 'cwru_real')
    t_start = time.time()

    print('Loading CWRU ...', flush=True)
    src = CWRUDataset(dd, window_size=512, stride=1024, load_conditions=[0], return_condition=False)
    tgt = CWRUDataset(dd, window_size=512, stride=1024, load_conditions=[3], return_condition=False)
    src_x = torch.from_numpy(src.samples).float()
    src_y = torch.from_numpy(src.labels).long()
    tgt_x = torch.from_numpy(tgt.samples).float()
    tgt_y = torch.from_numpy(tgt.labels).long()
    n_classes = len(torch.unique(src_y))
    tgt_eval = DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512)
    print('  src=%d tgt=%d classes=%d' % (len(src), len(tgt), n_classes), flush=True)

    out = {
        'meta': {
            'note': 'Ablation re-executed with per-epoch logging. Two loss configurations.',
            'backbone': 'FastCNN', 'task': 'CWRU 0hp->3hp few-shot n=50/class',
            'n_repeats': N_REPEATS, 'seeds': SEEDS, 'epochs_max': EPOCHS,
            'early_stop_patience': 12, 'torch': torch.__version__,
        },
        'plain': {},     # 论文原始配置（伪标签分支无 label smoothing）
        'aligned': {},   # 对齐配置（伪标签分支同样使用 label_smoothing=0.1）
    }

    for tag, ls in (('plain', 0.0), ('aligned', 0.1)):
        print('\n' + '=' * 70)
        print('CONFIG = %s (pseudo label_smoothing=%s)' % (tag, ls), flush=True)
        print('=' * 70, flush=True)

        # Baseline
        accs, hists = [], []
        for seed in SEEDS:
            tr_x, tr_y, vx, vy = few_shot_split(src_x, src_y, 50, 10, seed=seed)
            m, h = train_baseline_log(tr_x, tr_y, vx, vy, tgt_x, tgt_y,
                                      n_classes, device, seed=seed, epochs=EPOCHS)
            accs.append(float(evaluate(m, tgt_eval, device)['accuracy']))
            hists.append(h)
        out[tag]['baseline'] = {'mean': float(np.mean(accs)),
                                'std': float(np.std(accs)),
                                'accuracies': accs, 'history': hists}
        print('  baseline      %s  %.2f%%' % ([round(a * 100, 2) for a in accs], np.mean(accs) * 100), flush=True)

        for key, kw in CONFIGS.items():
            accs, hists, pstats = [], [], []
            for seed in SEEDS:
                tr_x, tr_y, vx, vy = few_shot_split(src_x, src_y, 50, 10, seed=seed)
                m, h, ps = train_pseudo_log(tr_x, tr_y, tgt_x, vx, vy, tgt_y,
                                            n_classes, device, seed=seed,
                                            epochs=EPOCHS, label_smoothing=ls, **kw)
                accs.append(float(evaluate(m, tgt_eval, device)['accuracy']))
                hists.append(h); pstats.append(ps)
            out[tag][key] = {'mean': float(np.mean(accs)),
                             'std': float(np.std(accs)),
                             'accuracies': accs, 'history': hists,
                             'pseudo_stats': pstats}
            print('  %-12s %s  %.2f%%' % (key, [round(a * 100, 2) for a in accs], np.mean(accs) * 100), flush=True)

    path = os.path.join(RESULTS_DIR, 'revision_ablation_instrumented.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), indent=2)
    print('\nSaved: %s' % path)
    print('Total %.1f min' % ((time.time() - t_start) / 60))


if __name__ == '__main__':
    main()
