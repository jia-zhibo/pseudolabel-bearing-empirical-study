# -*- coding: utf-8 -*-
"""
补跑：loss 对齐后的伪标签自训练（n=10 重复，CWRU + PU）。

动机：论文原配置中，5 个方法里只有伪标签分支没有 label_smoothing
（其余 4 个含 baseline 均为 label_smoothing=0.1）。为保证比较公平，这里在**完全相同的
数据与超参**下，仅给伪标签分支补上 label_smoothing=0.1，重跑 n=10，
用于判断"伪标签在弱骨干下必然退化"这一边界结论是否依赖于该配置差异。

本脚本只重跑 pseudo 一行；baseline/CORAL/MMD/DANN 的已发表数值沿用
results/revision_cwru_10repeats.json 与 revision_pu_10repeats.json（其配置本身已一致）。
"""
import os
import sys
import json
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# CPU 训练结果与线程数相关（已实测：同一 seed，threads=10 与 12 给出不同数值），
# 因此显式固定线程数，保证结果可复现。threads=10 在本机可逐位复现已发表
# revision_*_10repeats.json 中的 baseline 结果。
torch.set_num_threads(10)

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'data'))

from cwru_data import CWRUDataset
from pu_data import PUDataset
from _run_revision_experiments import few_shot_split, evaluate
from _run_curves_full import train_pseudo_log

RESULTS_DIR = os.path.join(BASE, 'results')
N_REPEATS = 10
EPOCHS = 40


def load_datasets():
    device = torch.device('cpu')
    cwru_dir = os.path.join(BASE, 'data', 'cwru_real')
    pu_dir = os.path.join(BASE, 'data', 'pu_data')
    src = CWRUDataset(cwru_dir, window_size=512, stride=1024, load_conditions=[0], return_condition=False)
    tgt = CWRUDataset(cwru_dir, window_size=512, stride=1024, load_conditions=[3], return_condition=False)
    cwru = (torch.from_numpy(src.samples).float(), torch.from_numpy(src.labels).long(),
            torch.from_numpy(tgt.samples).float(), torch.from_numpy(tgt.labels).long())
    bearings = ['K001', 'K002', 'K003', 'KA04', 'KA15', 'KI04', 'KI14']
    ps = PUDataset(pu_dir, bearings, condition='N15_M07_F10', window_size=512, stride=1024,
                   max_files_per_bearing=5)
    pt = PUDataset(pu_dir, bearings, condition='N09_M07_F10', window_size=512, stride=1024,
                   max_files_per_bearing=5)
    pu = (torch.from_numpy(ps.samples).float(), torch.from_numpy(ps.labels).long(),
          torch.from_numpy(pt.samples).float(), torch.from_numpy(pt.labels).long())
    return device, cwru, pu


def run_pseudo_10(src_x, src_y, tgt_x, tgt_y, n_classes, device, name, n_target_shot=50):
    print('\n=== %s: matched-loss pseudo-label self-training (n=%d) ===' % (name, N_REPEATS), flush=True)
    tgt_eval = DataLoader(TensorDataset(tgt_x, tgt_y), batch_size=512)
    accs, all_ps = [], []
    for rep in range(N_REPEATS):
        seed = rep * 1000 + 42
        tr_x, tr_y, vx, vy = few_shot_split(src_x, src_y, n_target_shot, 10, seed=seed)
        m, hist, ps = train_pseudo_log(tr_x, tr_y, tgt_x, vx, vy, tgt_y, n_classes, device,
                                       seed=seed, conf_thresh=0.9, n_rounds=2,
                                       pseudo_weight=1.0, epochs=EPOCHS, label_smoothing=0.1)
        a = float(evaluate(m, tgt_eval, device)['accuracy'])
        accs.append(a)
        all_ps.append(ps)
        print('  rep%2d seed=%5d acc=%.4f  n_sel=%s prec=%s'
              % (rep, seed, a,
                 [p['n_selected'] for p in ps],
                 [round(p['pseudo_precision'], 3) for p in ps]), flush=True)
    return {'accuracies': accs,
            'mean_acc': float(np.mean(accs)),
            'std_acc': float(np.std(accs))}


def main():
    device, cwru, pu = load_datasets()
    src_x, src_y, tgt_x, tgt_y = cwru
    n_cls_cwru = len(torch.unique(src_y))
    print('CWRU classes=%d src=%d tgt=%d' % (n_cls_cwru, len(src_x), len(tgt_x)), flush=True)
    ps_x, ps_y, pt_x, pt_y = pu
    print('PU classes=%d src=%d tgt=%d' % (len(torch.unique(ps_y)), len(ps_x), len(pt_x)), flush=True)

    res = {}
    res['cwru'] = run_pseudo_10(src_x, src_y, tgt_x, tgt_y, n_cls_cwru, device, 'CWRU')
    res['pu'] = run_pseudo_10(ps_x, ps_y, pt_x, pt_y, 3, device, 'PU')

    # 与已发表（非对齐）结果对比
    print('\n' + '=' * 72)
    print('对比：原配置(已发表) vs 对齐配置')
    print('=' * 72)
    for ds, fn, key in (('CWRU', 'revision_cwru_10repeats.json', 'cwru'),
                        ('PU', 'revision_pu_10repeats.json', 'pu')):
        old = json.load(open(os.path.join(RESULTS_DIR, fn), encoding='utf-8'))
        ob = old['baseline']['accuracies']
        op = old['pseudo']['accuracies']
        nb = ob
        np_ = res[key]['accuracies']
        res[key + '_comparison'] = {
            'baseline_old_mean': float(np.mean(ob)), 'baseline_old_std_ddof1': float(np.std(ob, ddof=1)),
            'pseudo_old_mean': float(np.mean(op)), 'pseudo_old_std_ddof1': float(np.std(op, ddof=1)),
            'pseudo_matched_mean': float(np.mean(np_)), 'pseudo_matched_std_ddof1': float(np.std(np_, ddof=1)),
            'delta_matched_minus_baseline_pp': float((np.mean(np_) - np.mean(ob)) * 100),
            'delta_old_minus_baseline_pp': float((np.mean(op) - np.mean(ob)) * 100),
        }
        c = res[key + '_comparison']
        print('%s: baseline=%.2f%%  pseudo(原)=%.2f%% (Δ%+.2fpp)  pseudo(对齐)=%.2f%% (Δ%+.2fpp)'
              % (ds, c['baseline_old_mean'] * 100, c['pseudo_old_mean'] * 100,
                 c['delta_old_minus_baseline_pp'], c['pseudo_matched_mean'] * 100,
                 c['delta_matched_minus_baseline_pp']))

    res['meta'] = {'n_repeats': N_REPEATS, 'seeds': [r * 1000 + 42 for r in range(N_REPEATS)],
                   'conf_thresh': 0.9, 'n_rounds': 2, 'pseudo_weight': 1.0,
                   'label_smoothing': 0.1, 'epochs_max': EPOCHS, 'torch': torch.__version__}
    path = os.path.join(RESULTS_DIR, 'revision_pseudo_matched_loss.json')
    json.dump(res, open(path, 'w', encoding='utf-8'), indent=2)
    print('\nSaved: %s' % path)


if __name__ == '__main__':
    main()
