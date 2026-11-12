# -*- coding: utf-8 -*-
"""
下载CWRU轴承数据集（12kHz驱动端）
官方文件编号映射，4个负载 × 10类
"""
import os
import urllib.request
import time

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cwru_real')
os.makedirs(OUT_DIR, exist_ok=True)

# CWRU官方文件编号（12kHz驱动端）
# 正常基线
NORMAL = {0: '97', 1: '98', 2: '99', 3: '100'}
# 驱动端故障 12kHz（DE time）
FAULT_12K = {
    'IR007': {0: '105', 1: '106', 2: '107', 3: '108'},
    'B007':  {0: '118', 1: '119', 2: '120', 3: '121'},
    'OR007': {0: '130', 1: '131', 2: '132', 3: '133'},
    'IR014': {0: '169', 1: '170', 2: '171', 3: '172'},
    'B014':  {0: '185', 1: '186', 2: '187', 3: '188'},
    'OR014': {0: '197', 1: '198', 2: '199', 3: '200'},
    'IR021': {0: '209', 1: '210', 2: '211', 3: '212'},
    'B021':  {0: '222', 1: '223', 2: '224', 3: '225'},
    'OR021': {0: '234', 1: '235', 2: '236', 3: '237'},
}

BASE_URLS = [
    'https://engineering.case.edu/sites/default/files/{}.mat',
    'https://raw.githubusercontent.com/XiongMeijing/CWRU-1/master/{}.mat',
]

def download_file(file_id, out_path):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return True, 'exists'
    for base in BASE_URLS:
        url = base.format(file_id)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) < 1000:
                continue
            with open(out_path, 'wb') as f:
                f.write(data)
            return True, url
        except Exception as e:
            continue
    return False, 'all sources failed'

def main():
    tasks = []
    # 正常
    for load, fid in NORMAL.items():
        tasks.append((f'Normal_{load}.mat', fid, load, 'N'))
    # 故障
    for fault, loads in FAULT_12K.items():
        for load, fid in loads.items():
            # 统一命名: 故障类型_负载.mat
            tasks.append((f'{fault}_{load}.mat', fid, load, fault))

    ok, fail = 0, 0
    for name, fid, load, ftype in tasks:
        out = os.path.join(OUT_DIR, name)
        success, src = download_file(fid, out)
        status = 'OK' if success else 'FAIL'
        size = os.path.getsize(out) if success and os.path.exists(out) else 0
        print(f'[{status}] {name:20s} id={fid:4s} load={load} size={size/1024:.0f}KB src={src[:60]}')
        if success:
            ok += 1
        else:
            fail += 1
        time.sleep(0.3)
    print(f'\nTotal: {ok} OK, {fail} FAIL')

if __name__ == '__main__':
    main()
