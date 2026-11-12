# -*- coding: utf-8 -*-
"""从权威重跑结果重算全部表格统计量（ddof=1，统一单侧检验）。"""
import json, os
import numpy as np
from scipy import stats

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def ld(n):
    with open(os.path.join(R, n)) as f:
        return json.load(f)


bilstm = ld('authoritative_bilstm_n50.json')
full = ld('authoritative_fastcnn_main.json')
abl = ld('authoritative_fastcnn_ablation.json')

KEYMAP = [('baseline', 'baseline'), ('coral', 'coral'), ('mmd', 'mmd'),
          ('dann', 'dann'), ('pseudo', 'pseudo_plain'), ('pseudo_lm', 'pseudo_matched')]


def reshape(ds):
    out = {}
    for nk, ok in KEYMAP:
        a = np.array(full[ds][ok]['accuracies'])
        out[nk] = {'mean_acc': float(a.mean()), 'std_acc': float(a.std(ddof=1)),
                   'accuracies': a.tolist()}
    return out


cwru, pu = reshape('cwru'), reshape('pu')


def pt(a, b):
    a, b = np.array(a), np.array(b)
    t1, p1 = stats.ttest_rel(a, b, alternative='greater')
    t2, p2 = stats.ttest_rel(a, b, alternative='two-sided')
    d = a - b
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    return p1, p2, dz


def row(name, m, base):
    p1, p2, dz = pt(m['accuracies'], base['accuracies'])
    delta = (m['mean_acc'] - base['mean_acc']) * 100
    print('  %-12s %6.2f ±%5.2f   %+7.2fpp   p1=%.4f p2=%.4f  d=%+.3f  %s'
          % (name, m['mean_acc'] * 100, m['std_acc'] * 100, delta, p1, p2, dz,
             '*' if (p1 < 0.05 and dz > 0) else ''))
    return delta, p1, p2, dz


print('=' * 88)
print('TABLE 1  BiLSTM-Attention, CWRU 0hp->3hp, n=50, 5 repeats  (authoritative, threads=10)')
print('=' * 88)
b = bilstm['baseline']
bn = {'mean_acc': float(np.mean(b['accuracies'])),
      'std_acc': float(np.std(b['accuracies'], ddof=1)),
      'accuracies': b['accuracies']}
row('baseline', bn, bn)
T1 = {}
for k in ('mmd_0.5', 'coral_0.5', 'pseudo_0.9'):
    m = bilstm[k]
    mn = {'mean_acc': float(np.mean(m['accuracies'])),
          'std_acc': float(np.std(m['accuracies'], ddof=1)),
          'accuracies': m['accuracies']}
    T1[k] = row(k, mn, bn)

print()
print('  f1-macro: baseline %.2f  mmd %.2f  coral %.2f  pseudo %.2f'
      % (b['mean_f1'] * 100, bilstm['mmd_0.5']['mean_f1'] * 100,
         bilstm['coral_0.5']['mean_f1'] * 100, bilstm['pseudo_0.9']['mean_f1'] * 100))

for tag, d in (('TABLE 2a  FastCNN CWRU', cwru), ('TABLE 2b  FastCNN PU', pu)):
    print()
    print('=' * 88)
    print('%s  (authoritative, threads=10, n=10)' % tag)
    print('=' * 88)
    row('baseline', d['baseline'], d['baseline'])
    for k in ('coral', 'mmd', 'dann', 'pseudo', 'pseudo_lm'):
        row(k, d[k], d['baseline'])

print()
print('=' * 88)
print('TABLE 3/4  FastCNN CWRU ablation  (authoritative, threads=10, n=3)')
print('=' * 88)
for cfg, label in (('plain', 'as-run (ls=0.0)'), ('aligned', 'loss-matched (ls=0.1)')):
    print(' [%s]' % label)
    for k in sorted(abl[cfg].keys()):
        v = abl[cfg][k]
        a = np.array(v['accuracies'] if 'accuracies' in v else [])
        if len(a) == 0:
            continue
        print('   %-14s %6.2f ±%5.2f   %s' % (k, a.mean() * 100, a.std(ddof=1) * 100,
                                              [round(x * 100, 2) for x in a.tolist()]))

# ---- 旧值对比 ----
print()
print('=' * 88)
print('OLD vs NEW  (Table 1)')
print('=' * 88)
old = ld('cwru_n50_da_comparison.json')
for k in ('baseline', 'mmd_0.5', 'coral_0.5', 'pseudo_0.9'):
    o = np.array(old[k]['accuracies'])
    n = np.array(bilstm[k]['accuracies'])
    print('  %-11s old %6.2f  new %6.2f   bit-exact=%s   max|dev|=%.2fpp'
          % (k, o.mean() * 100, n.mean() * 100, np.array_equal(o, n),
             np.abs(o - n).max() * 100))
