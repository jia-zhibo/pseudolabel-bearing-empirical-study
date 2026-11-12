# -*- coding: utf-8 -*-
"""验证下载的CWRU .mat文件完整性"""
import os
import numpy as np
from scipy.io import loadmat

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cwru_real')

files = sorted(os.listdir(DATA_DIR))
print(f'Total files: {len(files)}')

ok, bad = 0, []
for fname in files:
    path = os.path.join(DATA_DIR, fname)
    try:
        mat = loadmat(path)
        de_key = [k for k in mat.keys() if 'DE' in k and 'time' in k]
        rpm_key = [k for k in mat.keys() if 'RPM' in k]
        if de_key:
            sig = mat[de_key[0]].flatten()
            rpm = mat[rpm_key[0]].flatten()[0] if rpm_key else 'N/A'
            print(f'OK  {fname:20s} samples={len(sig):7d} rpm={rpm} de_key={de_key[0]}')
            ok += 1
        else:
            print(f'BAD {fname:20s} no DE time key, keys={list(mat.keys())[:5]}')
            bad.append(fname)
    except Exception as e:
        print(f'ERR {fname:20s} {e}')
        bad.append(fname)

print(f'\n{ok}/{len(files)} files valid')
if bad:
    print('Bad files:', bad)
