# -*- coding: utf-8 -*-
"""v4 最终独立审计：从原始 JSON 完全独立地重算所有报告值，与 docx 表格逐格比对。

审计口径：
  - 样本标准差 ddof=1（与 Cohen's d 同口径）
  - 统一单侧配对 t 检验 H1: mu_method > mu_baseline
  - 星号规则 * <=> (p < 0.05 且 d > 0)
"""
import json
import os
import numpy as np
from scipy import stats
from docx import Document

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DOCX = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'MSSP_Submission_PseudoLabel_BearingFaultDiagnosis_v7.docx')


def load(n):
    with open(os.path.join(BASE, n), encoding='utf-8') as f:
        return json.load(f)


def sd1(v):
    return float(np.std(v, ddof=1))


def paired(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    t, p = stats.ttest_rel(a, b, alternative='greater')
    d = a - b
    s = d.std(ddof=1)
    return t, p, (d.mean() / s if s > 0 else 0.0)


def star(p, d):
    return '*' if (p < 0.05 and d > 0) else ''


n50 = load('authoritative_bilstm_n50_n20.json')
n5_old = load('authoritative_bilstm_n50.json')
full = load('authoritative_fastcnn_main.json')
abl = load('authoritative_fastcnn_ablation.json')

KEYMAP = [('baseline', 'baseline'), ('coral', 'coral'), ('mmd', 'mmd'),
          ('dann', 'dann'), ('pseudo', 'pseudo_plain'), ('pseudo_lm', 'pseudo_matched')]


def rs(ds):
    out = {}
    for k, s in KEYMAP:
        out[k] = {'accuracies': list(full[ds][s]['accuracies'])}
    return out


cwru, pu = rs('cwru'), rs('pu')

doc = Document(DOCX)
T = {i: t for i, t in enumerate(doc.tables)}
cells = lambda ti: [[c.text.strip() for c in r.cells] for r in T[ti].rows]

fails, checks = [], [0]


def cmp(tag, got, want):
    checks[0] += 1
    if got != want:
        fails.append((tag, got, want))
    print('%-4s %-62s got=%-22s want=%-22s'
          % ('OK' if got == want else 'ERR', tag[:62], got[:22], want[:22]))


print('=' * 108)
print('Table 1  (CWRU / BiLSTM / n=20)')
print('=' * 108)
t1 = cells(0)
spec1 = [('baseline', 'Baseline', None),
         ('coral_0.5', 'CORAL (λ=0.5)', 'CORAL'),
         ('mmd_0.5', 'MMD (λ=0.5)', 'MMD'),
         ('pseudo_0.9', 'Pseudo-Label (τ=0.9)', 'Pseudo')]
bacc = n50['baseline']['accuracies']
bmean = float(np.mean(bacc)) * 100
for key, label, name in spec1:
    a = n50[key]['accuracies']
    mean = float(np.mean(a)) * 100
    s = sd1(a) * 100
    row = next(r for r in t1[1:] if r[0] == label)
    cmp('T1 %s accuracy' % (name or 'Baseline'), row[1], '%.2f ± %.2f' % (mean, s))
    cmp('T1 %s delta' % (name or 'Baseline'), row[3],
        '0.00' if name is None else ('+%.2f' if mean >= bmean else '%.2f') % (mean - bmean))
    cmp('T1 %s std col' % (name or 'Baseline'), row[4], '%.2f' % s)
    if name:
        t, p, d = paired(a, bacc)
        cmp('T1 %s p' % name, row[5], '%.4f%s' % (p, star(p, d)))

print()
print('=' * 108)
print('Table 2  (FastCNN, CWRU n=10 / PU n=10, 含损失对齐列)')
print('=' * 108)
t2 = cells(1)
meths = [('baseline', 'Baseline'), ('coral', 'CORAL'), ('mmd', 'MMD'),
         ('dann', 'DANN'), ('pseudo', 'Pseudo-Label'), ('pseudo_lm', 'Pseudo-LM')]
for ds, src in (('CWRU', cwru), ('PU', pu)):
    b = src['baseline']['accuracies']
    bm = float(np.mean(b)) * 100
    acc_row = next(r for r in t2[1:] if r[0] == ds and r[1] == 'Acc (%)')
    d_row = next(r for r in t2[1:] if r[0] == ds and r[1] == 'Δ (pp)')
    for j, (k, name) in enumerate(meths):
        a = src[k]['accuracies']
        m = float(np.mean(a)) * 100
        s = sd1(a) * 100
        p = d = None
        if k != 'baseline':
            _t, p, d = paired(a, b)
        want = '%.2f±%.2f%s' % (m, s, star(p, d) if p is not None else '')
        cmp('T2 %s %s acc' % (ds, name), acc_row[2 + j], want)
        cmp('T2 %s %s delta' % (ds, name), d_row[2 + j],
            '0.00' if k == 'baseline' else ('+%.2f' if m >= bm else '%.2f') % (m - bm))

print()
print('=' * 108)
print('Table 3 / Table 4  (Ablation, FastCNN, CWRU, n=3, as-run / loss-matched)')
print('=' * 108)
t3, t4 = cells(2), cells(3)
for ti, cfg, tname in ((t3, 'plain', 'T3 as-run'), (t4, 'aligned', 'T4 loss-matched')):
    b = abl[cfg]['baseline']['accuracies']
    bm = float(np.mean(b)) * 100
    seen = set()
    for r in ti[1:]:
        setting = r[1]
        if setting == 'Baseline':
            key = 'baseline'
        elif setting.startswith('τ'):
            key = 'conf_' + setting.split('=')[1].strip()
        elif setting.startswith('R'):
            key = 'rounds_' + setting.split('=')[1].strip()
        elif setting.startswith('α'):
            key = 'weight_' + setting.split('=')[1].strip()
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        a = abl[cfg][key]['accuracies']
        m = float(np.mean(a)) * 100
        s = sd1(a) * 100
        cmp('%s %s acc' % (tname, setting), r[2], '%.2f' % m)
        cmp('%s %s std' % (tname, setting), r[3], '%.2f' % s)
        # 基线行的 Δ 按稿例写作 0.00（无正号）
        cmp('%s %s delta' % (tname, setting), r[4],
            '0.00' if key == 'baseline' else ('+%.2f' if m >= bm else '%.2f') % (m - bm))

print()
print('=' * 108)
print('Table A.4  (all paired tests, 含两行 loss-matched)')
print('=' * 108)
t7 = cells(7)
expect = [
    ('CWRU', 'BiLSTM', n50, 'coral_0.5', 'CORAL vs Baseline'),
    ('CWRU', 'BiLSTM', n50, 'mmd_0.5', 'MMD vs Baseline'),
    ('CWRU', 'BiLSTM', n50, 'pseudo_0.9', 'Pseudo vs Baseline'),
    ('CWRU', 'FastCNN', cwru, 'coral', 'CORAL vs Baseline'),
    ('CWRU', 'FastCNN', cwru, 'mmd', 'MMD vs Baseline'),
    ('CWRU', 'FastCNN', cwru, 'dann', 'DANN vs Baseline'),
    ('CWRU', 'FastCNN', cwru, 'pseudo', 'Pseudo vs Baseline'),
    ('CWRU', 'FastCNN', cwru, 'pseudo_lm', 'Pseudo (loss-matched) vs Baseline'),
    ('PU', 'FastCNN', pu, 'coral', 'CORAL vs Baseline'),
    ('PU', 'FastCNN', pu, 'mmd', 'MMD vs Baseline'),
    ('PU', 'FastCNN', pu, 'dann', 'DANN vs Baseline'),
    ('PU', 'FastCNN', pu, 'pseudo', 'Pseudo vs Baseline'),
    ('PU', 'FastCNN', pu, 'pseudo_lm', 'Pseudo (loss-matched) vs Baseline'),
]
print('  A.4 行数 = %d（应为 %d）' % (len(t7) - 1, len(expect)))
if len(t7) - 1 != len(expect):
    fails.append(('A.4 row count', len(t7) - 1, len(expect)))
for ds, bb, src, key, label in expect:
    t, p, d = paired(src[key]['accuracies'], src['baseline']['accuracies'])
    row = next((r for r in t7[1:] if r[0] == ds and r[1] == bb and r[2] == label), None)
    if row is None:
        fails.append(('A.4 missing row', label, ds))
        print('ERR  缺少行 %s / %s / %s' % (ds, bb, label))
        continue
    cmp('A.4 %s/%s/%s t' % (ds, bb, key), row[3], '%.3f' % t)
    cmp('A.4 %s/%s/%s p' % (ds, bb, key), row[4], '%.4f%s' % (p, star(p, d)))
    cmp('A.4 %s/%s/%s d' % (ds, bb, key), row[5], '%.3f' % d)

print()
print('=' * 108)
print('星号规则全局自检')
print('=' * 108)
starred = []
for ds, bb, src, key, label in expect:
    t, p, d = paired(src[key]['accuracies'], src['baseline']['accuracies'])
    s = star(p, d)
    if s:
        starred.append('%s/%s/%s' % (ds, bb, label))
    print('  %-6s %-8s %-40s p=%.4f d=%+.3f -> %s' % (ds, bb, label, p, d, s or '-'))
print('  标星行数 =', len(starred), starred)

print()
print('=' * 108)
print('文档结构自检')
print('=' * 108)
heads = [p.text.strip() for p in doc.paragraphs if p.text.strip() and
         (p.text.strip()[0].isdigit() and '.' in p.text[:6])]
dups = [h for h in set(heads) if heads.count(h) > 1]
print('  一级/二级标题数 =', len(heads))
print('  重复标题 =', dups if dups else '无')
if dups:
    fails.append(('duplicate headings', dups, None))
figcaps = [p.text.strip() for p in doc.paragraphs if p.text.strip().startswith('Fig.')]
print('  图注 =', [c[:12] for c in figcaps])
tabcaps = [p.text.strip().split(chr(10))[0] for p in doc.paragraphs
           if p.text.strip().startswith('Table')]
print('  表注 =', tabcaps)

print()
print('=' * 108)
print('v7 专项：A.1 行数、n=20 前 5 轮与 n=5 权威结果逐位一致、正文 CI 与独立复算一致')
print('=' * 108)

t5 = cells(4)          # Table A.1
_maps = [('baseline', 1), ('coral_0.5', 2), ('mmd_0.5', 3), ('pseudo_0.9', 4)]
_rows5 = t5[1:]
checks[0] += 1
if len(_rows5) != len(n50['baseline']['accuracies']):
    fails.append(('A.1 row count', len(_rows5), len(n50['baseline']['accuracies'])))
print('%-4s %-62s got=%-22s want=%-22s' % (
    'OK' if len(_rows5) == len(n50['baseline']['accuracies']) else 'ERR',
    'A.1 row count', len(_rows5), len(n50['baseline']['accuracies'])))

for key, col in _maps:
    for i, r in enumerate(_rows5):
        want = '%.2f' % (n50[key]['accuracies'][i] * 100)
        cmp('A.1 run%d %s' % (i + 1, key), r[col], want)
        cmp('A.1 run%d %s label' % (i + 1, key), r[0], str(i + 1))

for key, col in _maps:
    _a = n50[key]['accuracies'][:5]
    _b = n5_old[key]['accuracies'][:5]
    same = len(_a) == len(_b) and all(x == y for x, y in zip(_a, _b))
    checks[0] += 1
    if not same:
        fails.append(('bit-exact first5 %s' % key, _a, _b))
    print('%-4s %-62s %s' % ('OK' if same else 'ERR',
                             'first 5 seeds bit-exact vs n=5 authoritative: ' + key,
                             'identical' if same else 'DIFFERS'))

# 正文里的 95% CI 必须与独立复算一致
_diff = (np.array(n50['pseudo_0.9']['accuracies'], float) -
         np.array(n50['baseline']['accuracies'], float))
_nn = len(_diff)
_se = _diff.std(ddof=1) / np.sqrt(_nn)
_tc = stats.t.ppf(0.975, _nn - 1)
_lo = (_diff.mean() - _tc * _se) * 100
_hi = (_diff.mean() + _tc * _se) * 100
_doc_text = chr(10).join(p.text for p in doc.paragraphs)
_ci_txt = '%+.2f to %+.2f pp' % (_lo, _hi)
checks[0] += 1
_ok = _ci_txt in _doc_text
if not _ok:
    fails.append(('CI sentence missing', _ci_txt, 'in body text'))
print('%-4s %-62s got=%-22s' % ('OK' if _ok else 'ERR',
                                'body contains independently recomputed 95% CI',
                                _ci_txt if _ok else 'NOT FOUND: ' + _ci_txt))


print()
print('=' * 108)
print('结论：检查 %d 项，失败 %d 项' % (checks[0], len(fails)))
if fails:
    for f in fails:
        print('  ERR', f)
else:
    print('  全部数值与独立复算完全一致 ✔')
