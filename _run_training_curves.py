# -*- coding: utf-8 -*-
"""
补跑真实训练曲线（用于 §6.4）。

背景：已发表实验结果（revision_ablation_3repeats.json）只保存了每次重复的最终精度，
没有任何逐 epoch 记录，因此论文原先无法给出训练曲线。本脚本以与原始实验完全同源的方式
重跑伪标签自训练（相同数据加载、相同 few-shot 划分种子、相同模型、相同优化器与早停策略），
在训练过程中记录每个 epoch 的训练损失、源域验证精度、目标域测试精度，以及每轮伪标签生成
时的伪标签数量与伪标签正确率（precision）——后者是"伪标签误差累积"的直接证据。

同源性保证：
  * 直接复用原脚本 _run_revision_experiments 中的 few_shot_split / set_seed / FastCNN /
    evaluate；数据加载与主实验 main() 完全一致；
  * train_pseudo_log 为 train_pseudo 的逐行等价改写，仅在 epoch 循环内增加不含随机性的
    记录操作（推理在 torch.no_grad() 下完成，不消耗 RNG），因此 RNG 流与原始实验一致；
  * 运行后与 revision_ablation_3repeats.json 的 rounds_1/2/3 逐位比对校验。
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

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'data'))

from cwru_data import CWRUDataset
from _run_revision_experiments import (  # 复用原始实验的全部组件
    few_shot_split, set_seed, FastCNN, evaluate,
)

RESULTS_DIR = os.path.join(BASE, 'results')
N_REPEATS = 3
EPOCHS = 40
CONF_THRESH = 0.9
PSEUDO_WEIGHT = 1.0
ROUNDS = [1, 2, 3]


# ------------------------------------------------------------
# 带逐 epoch 记录的伪标签自训练（train_pseudo 的等价改写）
# ------------------------------------------------------------
def train_pseudo_log(train_x, train_y, test_x, val_x, val_y, test_y,
                     n_classes, device, seed=42, conf_thresh=0.9,
                     n_rounds=2, pseudo_weight=1.0, epochs=40):
    """与 train_pseudo 完全等价，额外返回每个 epoch 的训练动态与每轮伪标签统计。"""
    set_seed(seed)
    batch_size = 64

    cur_x, cur_y = train_x.clone(), train_y.clone()
    cur_w = torch.ones(len(train_y), dtype=torch.float32)

    best_model = None
    best_val = 0

    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=256)
    tgt_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=512)

    history = []        # 每个 round 一个 dict
    pseudo_stats = []   # 每个 round（除最后一轮）生成伪标签时的统计

    for rnd in range(n_rounds):
        model = FastCNN(n_classes=n_classes).to(device)
        ds = TensorDataset(cur_x, cur_y, cur_w)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(reduction='none')
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc, best_st, no_imp = 0, None, 0
        ep_loss, ep_val, ep_tgt = [], [], []

        for epoch in range(epochs):
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

            # --- 记录（no_grad，不影响 RNG 流）---
            model.eval()
            with torch.no_grad():
                c = t = 0
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    _, p = model(x).max(1)
                    t += y.size(0)
                    c += p.eq(y).sum().item()
                ep_val.append(c / t)

                c = t = 0
                for x, y in tgt_loader:
                    x, y = x.to(device), y.to(device)
                    _, p = model(x).max(1)
                    t += y.size(0)
                    c += p.eq(y).sum().item()
                ep_tgt.append(c / t)

            va = ep_val[-1]
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

        history.append({
            'round': rnd,
            'n_labeled': int(cur_x.shape[0]),
            'train_loss': [float(v) for v in ep_loss],
            'val_acc': [float(v) for v in ep_val],
            'target_acc': [float(v) for v in ep_tgt],
        })

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

        # 伪标签正确率（相对目标域真值）——误差累积的直接度量
        pl_prec = float((pred[mask].eq(test_y[mask])).sum().item() / n_pseudo) if n_pseudo else 0.0
        pseudo_stats.append({
            'round': rnd,
            'n_selected': int(n_pseudo),
            'selection_rate': float(n_pseudo / test_x.shape[0]),
            'pseudo_precision': pl_prec,
            'target_acc_at_selection': float(ep_tgt[-1]),
        })

        if n_pseudo == 0:
            break

        cur_x = torch.cat([cur_x, test_x[mask]])
        cur_y = torch.cat([cur_y, pred[mask]])
        cur_w = torch.cat([cur_w, torch.full((n_pseudo,), pseudo_weight)])

    final_model = FastCNN(n_classes=n_classes).to(device)
    final_model.load_state_dict(best_model)
    return final_model, history, pseudo_stats


# ------------------------------------------------------------
# 带逐 epoch 记录的基线训练（train_baseline 的等价改写）
# ------------------------------------------------------------
def train_baseline_log(train_x, train_y, val_x, val_y, test_x, test_y,
                       n_classes, device, seed=42, epochs=40, batch_size=64):
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    set_seed(seed)
    model = FastCNN(n_classes=n_classes).to(device)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=256)
    tgt_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=512)
    optimizer = Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    sched = CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_st, no_imp = 0, None, 0
    ep_loss, ep_val, ep_tgt = [], [], []
    for _ in range(epochs):
        model.train()
        running, nb = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            nb += 1
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
        va = ep_val[-1]
        if va > best_acc:
            best_acc = va
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 12:
                break
    model.load_state_dict(best_st)
    hist = {
        'round': 0,
        'n_labeled': int(train_x.shape[0]),
        'train_loss': [float(v) for v in ep_loss],
        'val_acc': [float(v) for v in ep_val],
        'target_acc': [float(v) for v in ep_tgt],
    }
    return model, hist


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    device = torch.device('cpu')
    data_dir = os.path.join(BASE, 'data', 'cwru_real')
    t_start = time.time()

    print('Loading CWRU data ...')
    t0 = time.time()
    src = CWRUDataset(data_dir, window_size=512, stride=1024,
                      load_conditions=[0], return_condition=False)
    tgt = CWRUDataset(data_dir, window_size=512, stride=1024,
                      load_conditions=[3], return_condition=False)
    print('  done in %.1fs (src=%d, tgt=%d)' % (time.time() - t0, len(src), len(tgt)))

    src_x = torch.from_numpy(src.samples).float()
    src_y = torch.from_numpy(src.labels).long()
    tgt_x = torch.from_numpy(tgt.samples).float()
    tgt_y = torch.from_numpy(tgt.labels).long()
    n_classes = len(torch.unique(src_y))
    print('  classes:', n_classes, ' target samples:', int(tgt_x.shape[0]))

    out = {
        'meta': {
            'dataset': 'CWRU',
            'task': '0 hp -> 3 hp cross-load few-shot (n=50/class)',
            'backbone': 'FastCNN',
            'n_repeats': N_REPEATS,
            'seeds': [r * 1000 + 42 for r in range(N_REPEATS)],
            'epochs_max': EPOCHS,
            'conf_thresh': CONF_THRESH,
            'pseudo_weight': PSEUDO_WEIGHT,
            'optimizer': 'Adam lr=3e-3 wd=1e-4 cosine',
            'early_stop_patience': 12,
            'source': 're-run of the pipeline that produced '
                      'results/revision_ablation_3repeats.json',
        },
        'baseline': {'repeat_accuracies': [], 'history': []},
        'pseudo': {},
    }

    tgt_eval = DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512)

    # ---------- Baseline ----------
    print('\nBaseline (no self-training):')
    for rep in range(N_REPEATS):
        seed = rep * 1000 + 42
        tr_x, tr_y, vx, vy = few_shot_split(src_x, src_y, 50, 10, seed=seed)
        m, hist = train_baseline_log(tr_x, tr_y, vx, vy, tgt_x, tgt_y,
                                     n_classes, device, seed=seed, epochs=EPOCHS)
        r = evaluate(m, tgt_eval, device)
        out['baseline']['repeat_accuracies'].append(float(r['accuracy']))
        out['baseline']['history'].append(hist)
        print('   rep%d acc=%.4f  (%d epochs)' % (rep, r['accuracy'], len(hist['target_acc'])))

    # ---------- Pseudo-label self-training, R = 1,2,3 ----------
    for nr in ROUNDS:
        print('\nPseudo-label self-training, R=%d:' % nr)
        bucket = {'repeat_accuracies': [], 'history': [], 'pseudo_stats': []}
        for rep in range(N_REPEATS):
            seed = rep * 1000 + 42
            tr_x, tr_y, vx, vy = few_shot_split(src_x, src_y, 50, 10, seed=seed)
            m, hist, pstats = train_pseudo_log(
                tr_x, tr_y, tgt_x, vx, vy, tgt_y, n_classes, device,
                seed=seed, conf_thresh=CONF_THRESH, n_rounds=nr,
                pseudo_weight=PSEUDO_WEIGHT, epochs=EPOCHS)
            r = evaluate(m, tgt_eval, device)
            bucket['repeat_accuracies'].append(float(r['accuracy']))
            bucket['history'].append(hist)
            bucket['pseudo_stats'].append(pstats)
            ne = sum(len(h['target_acc']) for h in hist)
            print('   rep%d acc=%.4f  (%d epochs across %d rounds)'
                  % (rep, r['accuracy'], ne, len(hist)))
        out['pseudo']['rounds_%d' % nr] = bucket

    # ---------- 与已发表结果比对校验 ----------
    print('\n' + '=' * 72)
    print('Consistency check against published Table 3 (revision_ablation_3repeats.json)')
    print('=' * 72)
    abl_path = os.path.join(RESULTS_DIR, 'revision_ablation_3repeats.json')
    check = {}
    if os.path.exists(abl_path):
        abl = json.load(open(abl_path, encoding='utf-8'))
        base_pub = abl['baseline']['accuracies']
        base_new = out['baseline']['repeat_accuracies']
        check['baseline'] = {'published': base_pub, 'rerun': base_new,
                             'max_abs_diff': max(abs(a - b) for a, b in zip(base_pub, base_new))}
        print('baseline  pub=%.4f  new=%.4f  |diff|=%.2e'
              % (np.mean(base_pub), np.mean(base_new), check['baseline']['max_abs_diff']))
        for nr in ROUNDS:
            pub = abl['rounds_%d' % nr]['accuracies']
            new = out['pseudo']['rounds_%d' % nr]['repeat_accuracies']
            d = max(abs(a - b) for a, b in zip(pub, new))
            check['rounds_%d' % nr] = {'published': pub, 'rerun': new, 'max_abs_diff': d}
            flag = 'OK' if d < 1e-6 else 'MISMATCH'
            print('R=%d  pub mean=%.4f  new mean=%.4f  |diff|=%.2e   %s'
                  % (nr, np.mean(pub), np.mean(new), d, flag))
    out['consistency_check'] = check

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, 'revision_training_curves.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), indent=2)
    print('\nSaved: %s' % path)
    print('Total time: %.1f min' % ((time.time() - t_start) / 60))


if __name__ == '__main__':
    main()
