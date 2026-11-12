# -*- coding: utf-8 -*-
"""
PU轴承数据集下载（修正版：.rar格式）
"""
import os
import ssl
import urllib.request
import time

ssl._create_default_https_context = ssl._create_unverified_context

PU_BASE_URL = "https://groups.uni-paderborn.de/kat/BearingDataCenter/"

# 最小集：健康(K001,K002,K003) + 内圈故障(KA04,KA15) + 外圈故障(KI04,KI14)
MINIMAL_BEARINGS = ['K001', 'K002', 'K003', 'KA04', 'KA15', 'KI04', 'KI14']

# 完整健康集
HEALTHY = ['K001', 'K002', 'K003', 'K004', 'K005', 'K006']
# 人工损伤-内圈
INNER_FAULT = ['KA04', 'KA15', 'KA16', 'KA22', 'KA30']
# 人工损伤-外圈
OUTER_FAULT = ['KI04', 'KI14', 'KI16', 'KI17', 'KI18']


def download_file(url, filepath, max_retries=3):
    for attempt in range(max_retries):
        try:
            name = os.path.basename(filepath)
            print(f"  [{attempt+1}/{max_retries}] Downloading {name}...")
            urllib.request.urlretrieve(url, filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"    Done: {size_mb:.1f} MB")
            return True
        except Exception as e:
            print(f"    Failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(3)
    return False


def main():
    import sys
    subset = sys.argv[1] if len(sys.argv) > 1 else 'minimal'
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'data', 'pu_data')
    os.makedirs(output_dir, exist_ok=True)
    
    if subset == 'minimal':
        bearings = MINIMAL_BEARINGS
    elif subset == 'healthy':
        bearings = HEALTHY
    elif subset == 'full':
        bearings = HEALTHY + INNER_FAULT + OUTER_FAULT
    else:
        bearings = [subset]  # single file
    
    total_mb = len(bearings) * 160
    print(f"PU Bearing Dataset Download ({subset})")
    print(f"Files: {len(bearings)} x ~160MB = ~{total_mb}MB")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    success = 0
    failed = []
    
    for i, bearing in enumerate(bearings):
        filename = f"{bearing}.rar"
        url = PU_BASE_URL + filename
        filepath = os.path.join(output_dir, filename)
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 50_000_000:
            print(f"[{i+1}/{len(bearings)}] {filename}: already exists")
            success += 1
            continue
        
        print(f"[{i+1}/{len(bearings)}]", end=' ')
        if download_file(url, filepath):
            success += 1
        else:
            failed.append(bearing)
    
    print("\n" + "=" * 60)
    print(f"Result: {success}/{len(bearings)} successful")
    if failed:
        print(f"Failed: {failed}")
    print(f"Saved to: {output_dir}")


if __name__ == '__main__':
    main()
