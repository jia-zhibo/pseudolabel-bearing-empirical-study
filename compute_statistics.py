# -*- coding: utf-8 -*-
"""
Recompute every manuscript table from the raw JSON files.

Plain-Python implementation (no scipy / numpy required) so the numbers can be
verified in any environment. Convention, identical to the manuscript:

  * one-sided paired t-test, H1: mu(method) > mu(baseline)
  * '*'  <=>  (p < 0.05 AND Cohen's d > 0)
  * all SDs use ddof = 1
  * accuracies are stored as fractions in the JSON, printed here as percent

Usage:  python compute_statistics.py
"""
import os
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')


# ---------- Student-t survival function via the regularised incomplete beta ----------
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_sf_greater(t, df):
    """P(T > t) for a t variate with df degrees of freedom."""
    p = 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))
    return p if t > 0 else 1.0 - p


def paired(a, b):
    """Return (one-sided p, delta in pp, Cohen's d_z) for H1: mean(a-b) > 0."""
    n = len(a)
    assert n == len(b) and n > 1
    d = [x - y for x, y in zip(a, b)]
    md = sum(d) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1))
    if sd == 0.0:
        return (0.0 if md > 0 else 1.0), md * 100.0, 0.0
    t = md / (sd / math.sqrt(n))
    p = _t_sf_greater(t, n - 1)
    return p, md * 100.0, md / sd


def mean_sd(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0
    return m * 100.0, s * 100.0


def load(name):
    with open(os.path.join(R, name), 'r', encoding='utf-8') as f:
        return json.load(f)


def banner(t):
    print('\n' + '=' * 92)
    print(t)
    print('=' * 92)


def show_table(d, dataset_keys, order, title):
    banner(title)
    base = [float(x) for x in d['baseline']['accuracies']]
    bm, bs = mean_sd(base)
    print('  %-30s %6.2f +/- %5.2f  (n=%d)' % ('Baseline', bm, bs, len(base)))
    for label, key in order:
        v = [float(x) for x in d[key]['accuracies']]
        m, s = mean_sd(v)
        p, delta, dz = paired(v, base)
        print('  %-30s %6.2f +/- %5.2f  delta=%+6.2f pp  p=%.4f  d=%+.3f  %s'
              % (label, m, s, delta, p, dz, '*' if (p < 0.05 and dz > 0) else ''))


def main():
    # ---------------- Table 1 : BiLSTM n=20 (AUTHORITATIVE) ----------------
    t1 = load('authoritative_bilstm_n50_n20.json')
    meta = t1.get('_meta', {})
    banner('TABLE 1  BiLSTM-Attention, CWRU 0hp->3hp, n=50/class, %d repeats '
           '(threads=%s, epochs=%s, window=%s, stride=%s)'
           % (meta.get('n_repeats'), meta.get('threads'), meta.get('epochs'),
              meta.get('window_size'), meta.get('stride')))
    b = [float(x) for x in t1['baseline']['accuracies']]
    bm, bs = mean_sd(b)
    print('  %-24s %6.2f +/- %5.2f   delta= 0.00 pp' % ('Baseline', bm, bs))
    for label, key in (('CORAL (lambda=0.5)', 'coral_0.5'),
                       ('MMD (lambda=0.5)', 'mmd_0.5'),
                       ('Pseudo-Label (tau=0.9)', 'pseudo_0.9')):
        v = [float(x) for x in t1[key]['accuracies']]
        m, s = mean_sd(v)
        p, delta, dz = paired(v, b)
        print('  %-24s %6.2f +/- %5.2f   delta=%+6.2f pp  p=%.4f  d=%+.3f  %s'
              % (label, m, s, delta, p, dz, '*' if (p < 0.05 and dz > 0) else ''))

    # 95% CI of the pseudo-label paired difference (t critical, df=19)
    pv = [float(x) for x in t1['pseudo_0.9']['accuracies']]
    diff = [x - y for x, y in zip(pv, b)]
    n = len(diff)
    md = sum(diff) / n
    sd = math.sqrt(sum((x - md) ** 2 for x in diff) / (n - 1))
    # t_{0.975,19} = 2.0930
    tcrit = 2.093024054408314
    half = tcrit * sd / math.sqrt(n)
    print('  pseudo 95%% CI of paired difference: %+.2f to %+.2f pp'
          % (md * 100 - half * 100, md * 100 + half * 100))

    # ---------------- Section 7 negative control (n=5, unseeded) ----------------
    old = load('cwru_n50_da_comparison.json')
    banner('SECTION 7 NEGATIVE CONTROL  n=5, unseeded baseline (NOT used in any table)')
    ob = [float(x) for x in old['baseline']['accuracies']]
    for label, key in (('Pseudo-Label', 'pseudo_0.9'), ('MMD', 'mmd_0.5'),
                       ('CORAL', 'coral_0.5')):
        v = [float(x) for x in old[key]['accuracies']]
        p, delta, dz = paired(v, ob)
        print('  %-16s delta=%+6.2f pp  p=%.4f  d=%+.3f' % (label, delta, p, dz))

    # ---------------- Table 2 : FastCNN n=10 ----------------
    t2 = load('authoritative_fastcnn_main.json')
    order = [('CORAL', 'coral'), ('MMD', 'mmd'), ('DANN', 'dann'),
             ('Pseudo-Label', 'pseudo_plain'),
             ('Pseudo-Label loss-matched', 'pseudo_matched')]
    for ds, label in (('cwru', 'CWRU'), ('pu', 'PU')):
        show_table(t2[ds], None, order,
                   'TABLE 2  FastCNN %s, n=10 repeats (threads=%s, torch=%s)'
                   % (label, t2['meta'].get('threads'), t2['meta'].get('torch')))

    # ---------------- Tables 3 & 4 : ablation n=3 ----------------
    ab = load('authoritative_fastcnn_ablation.json')
    for cfg, tno, note in (('plain', 3, 'as-run objective'),
                           ('aligned', 4, 'loss-matched objective')):
        banner('TABLE %d  Ablation, %s, n=3 repeats' % (tno, note))
        keys = sorted(ab[cfg].keys())
        for k in keys:
            v = [float(x) for x in ab[cfg][k]['accuracies']]
            m, s = mean_sd(v)
            print('  %-16s %6.2f +/- %5.2f' % (k, m, s))

    banner('DONE - all values recomputed from raw per-repetition accuracies.')


if __name__ == '__main__':
    main()
