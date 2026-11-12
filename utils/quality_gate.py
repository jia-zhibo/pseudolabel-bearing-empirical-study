"""
质量门控与评估指标工具

包含:
1. 三层质量门控 (统计一致性、专家评审(自动化)、下游效用)
2. 故障诊断评估指标
3. RUL预测评估指标
"""

import numpy as np
from scipy import stats
from scipy.signal import correlate
from scipy.fft import fft
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
# 一、三层质量门控
# ============================================================

class QualityGate:
    """
    三层质量门控
    
    Layer 1: 统计一致性检验 (KS检验 + 自相关误差 + FFT谱误差)
    Layer 2: 自动化专家评审 (基于特征的分类器判别)
    Layer 3: 下游效用验证 (验证集性能增益 + 比例回滚)
    """
    
    def __init__(self, 
                 # Layer 1 参数
                 ks_alpha=0.05,
                 acf_threshold=0.2,
                 fft_threshold=0.2,
                 # Layer 2 参数  
                 expert_threshold=0.3,
                 # Layer 3 参数
                 min_gain=0.0,
                 max_synthetic_ratio=5.0,
                 rollback_rate=0.5,
                 ):
        self.ks_alpha = ks_alpha
        self.acf_threshold = acf_threshold
        self.fft_threshold = fft_threshold
        self.expert_threshold = expert_threshold
        self.min_gain = min_gain
        self.max_synthetic_ratio = max_synthetic_ratio
        self.rollback_rate = rollback_rate
        
        # 状态记录
        self.stats_history = []
        self.gate_stats = {'layer1_pass': 0, 'layer2_pass': 0, 'layer3_pass': 0,
                           'layer1_fail': 0, 'layer2_fail': 0, 'layer3_fail': 0}
    
    # ==================== Layer 1: 统计一致性检验 ====================
    
    def layer1_statistical_check(self, real_samples, synthetic_samples):
        """
        第一层：统计一致性检验
        
        检查:
        1. KS检验 (Kolmogorov-Smirnov): 边缘分布一致性
        2. 自相关函数 (ACF) 相对误差
        3. FFT幅值谱相对误差
        
        Args:
            real_samples: (N, C, L) 真实样本
            synthetic_samples: (M, C, L) 合成样本
        
        Returns:
            pass_mask: (M,) 布尔数组，表示每个合成样本是否通过
            metrics: 检验指标字典
        """
        M = synthetic_samples.shape[0]
        pass_mask = np.ones(M, dtype=bool)
        metrics = {}
        
        # 展平用于单变量检验
        real_flat = real_samples.reshape(-1, real_samples.shape[-1])  # (N*C, L)
        syn_flat = synthetic_samples.reshape(-1, synthetic_samples.shape[-1])  # (M*C, L)
        
        # 1. KS检验 - 对每个合成样本与真实样本总体比较
        ks_pass = np.ones(M, dtype=bool)
        ks_stats_list = []
        
        for i in range(M):
            sample_i = synthetic_samples[i].flatten()  # 单样本所有通道
            real_all = real_samples.reshape(-1)  # 所有真实数据
            
            ks_stat, p_value = stats.ks_2samp(sample_i, real_all)
            ks_stats_list.append(ks_stat)
            
            if p_value < self.ks_alpha:
                ks_pass[i] = False
        
        pass_mask &= ks_pass
        metrics['ks_pass_rate'] = ks_pass.mean()
        metrics['ks_stat_mean'] = np.mean(ks_stats_list)
        
        # 2. 自相关函数误差
        acf_pass = np.ones(M, dtype=bool)
        acf_errors = []
        
        # 计算真实样本的平均自相关
        real_acf = self._compute_mean_acf(real_flat)
        
        for i in range(M):
            sample_acf = self._compute_mean_acf(syn_flat[i*synthetic_samples.shape[1]:
                                                            (i+1)*synthetic_samples.shape[1]])
            rel_error = np.mean(np.abs(sample_acf - real_acf)) / (np.mean(np.abs(real_acf)) + 1e-8)
            acf_errors.append(rel_error)
            
            if rel_error > self.acf_threshold:
                acf_pass[i] = False
        
        pass_mask &= acf_pass
        metrics['acf_pass_rate'] = acf_pass.mean()
        metrics['acf_error_mean'] = np.mean(acf_errors)
        
        # 3. FFT谱误差
        fft_pass = np.ones(M, dtype=bool)
        fft_errors = []
        
        real_fft = self._compute_mean_fft_magnitude(real_flat)
        
        for i in range(M):
            sample_fft = self._compute_mean_fft_magnitude(syn_flat[i*synthetic_samples.shape[1]:
                                                                      (i+1)*synthetic_samples.shape[1]])
            rel_error = np.mean(np.abs(sample_fft - real_fft)) / (np.mean(np.abs(real_fft)) + 1e-8)
            fft_errors.append(rel_error)
            
            if rel_error > self.fft_threshold:
                fft_pass[i] = False
        
        pass_mask &= fft_pass
        metrics['fft_pass_rate'] = fft_pass.mean()
        metrics['fft_error_mean'] = np.mean(fft_errors)
        metrics['layer1_overall_pass_rate'] = pass_mask.mean()
        
        self.gate_stats['layer1_pass'] += pass_mask.sum()
        self.gate_stats['layer1_fail'] += (~pass_mask).sum()
        
        return pass_mask, metrics
    
    def _compute_mean_acf(self, signals, max_lag=50):
        """计算一组信号的平均自相关函数"""
        acfs = []
        for sig in signals:
            sig = sig - np.mean(sig)
            acf = correlate(sig, sig, mode='full')
            acf = acf[len(acf)//2:len(acf)//2 + max_lag]
            acf /= acf[0]  # 归一化
            acfs.append(acf)
        return np.mean(acfs, axis=0)
    
    def _compute_mean_fft_magnitude(self, signals):
        """计算一组信号的平均FFT幅值谱"""
        ffts = []
        for sig in signals:
            spectrum = np.abs(fft(sig))
            ffts.append(spectrum[:len(spectrum)//2])
        return np.mean(ffts, axis=0)
    
    # ==================== Layer 2: 自动化专家评审 ====================
    
    def layer2_expert_review(self, real_samples, synthetic_samples):
        """
        第二层：盲法专家评审（自动化替代）
        
        使用特征相似度来模拟专家判别。
        真实工业场景中由领域专家进行盲法评审。
        
        Args:
            real_samples: (N, C, L)
            synthetic_samples: (M, C, L)
        
        Returns:
            pass_mask: (M,)
            metrics: 指标字典
        """
        M = synthetic_samples.shape[0]
        pass_mask = np.ones(M, dtype=bool)
        
        # 提取统计特征
        real_features = self._extract_features(real_samples)
        syn_features = self._extract_features(synthetic_samples)
        
        # 计算每个合成样本与真实特征分布的距离
        real_mean = np.mean(real_features, axis=0)
        real_std = np.std(real_features, axis=0) + 1e-8
        
        # 标准化后的马氏距离近似
        distances = np.zeros(M)
        for i in range(M):
            normalized_dist = np.abs(syn_features[i] - real_mean) / real_std
            distances[i] = np.mean(normalized_dist)
        
        # 阈值判断
        pass_mask = distances < self.expert_threshold * np.mean(real_std) * len(real_mean)
        # 简化：用距离排序，保留前70%
        threshold = np.percentile(distances, 70)
        pass_mask = distances <= threshold
        
        metrics = {
            'expert_dist_mean': np.mean(distances),
            'expert_dist_std': np.std(distances),
            'layer2_pass_rate': pass_mask.mean(),
        }
        
        self.gate_stats['layer2_pass'] += pass_mask.sum()
        self.gate_stats['layer2_fail'] += (~pass_mask).sum()
        
        return pass_mask, metrics
    
    def _extract_features(self, samples):
        """
        提取时序统计特征
        
        包括: 均值、方差、峰度、偏度、RMS、波峰因子、脉冲因子
        """
        N = samples.shape[0]
        samples_2d = samples.reshape(N, -1)  # (N, C*L)
        
        features = []
        
        for i in range(N):
            sig = samples_2d[i]
            feat = []
            
            # 基本统计量
            feat.append(np.mean(sig))
            feat.append(np.std(sig))
            feat.append(stats.kurtosis(sig))
            feat.append(stats.skew(sig))
            
            # 时域指标
            rms = np.sqrt(np.mean(sig**2))
            feat.append(rms)
            feat.append(np.max(np.abs(sig)) / (rms + 1e-8))  # 波峰因子
            
            # 简单频域特征
            spectrum = np.abs(fft(sig))
            feat.append(np.mean(spectrum))
            feat.append(np.std(spectrum))
            
            features.append(feat)
        
        return np.array(features)
    
    # ==================== Layer 3: 下游效用验证 ====================
    
    def layer3_downstream_utility(self, base_performance, augmented_performance,
                                   current_ratio, synthetic_count, real_count):
        """
        第三层：下游效用验证 + 比例回滚
        
        Args:
            base_performance: 基线性能 (如F1或RMSE)
            augmented_performance: 增强后性能
            current_ratio: 当前合成/真实比例
            synthetic_count: 合成样本数
            real_count: 真实样本数
        
        Returns:
            passed: 是否通过
            new_ratio: 调整后的比例
            metrics: 指标字典
        """
        # 计算增益 (对于分类指标，越大越好；对于RMSE，越小越好)
        # 这里假设performance是"越大越好"的指标 (如F1, Accuracy)
        gain = augmented_performance - base_performance
        
        # 检查是否超过比例上限
        ratio_exceeded = current_ratio > self.max_synthetic_ratio
        
        passed = gain > self.min_gain and not ratio_exceeded
        
        # 计算新的比例
        if gain < self.min_gain:
            # 性能下降，回滚比例
            new_ratio = max(0.5, current_ratio * self.rollback_rate)
            rollback_reason = 'negative_gain'
        elif ratio_exceeded:
            new_ratio = self.max_synthetic_ratio
            rollback_reason = 'ratio_cap'
        else:
            new_ratio = current_ratio
            rollback_reason = 'none'
        
        metrics = {
            'base_performance': base_performance,
            'augmented_performance': augmented_performance,
            'gain': gain,
            'current_ratio': current_ratio,
            'new_ratio': new_ratio,
            'rollback_reason': rollback_reason,
            'passed': passed,
        }
        
        if passed:
            self.gate_stats['layer3_pass'] += 1
        else:
            self.gate_stats['layer3_fail'] += 1
        
        return passed, new_ratio, metrics
    
    # ==================== 完整门控流程 ====================
    
    def gate_samples(self, real_samples, synthetic_samples, conditions=None,
                     base_performance=None, downstream_fn=None):
        """
        完整的三层门控流程
        
        Args:
            real_samples: (N, C, L) 真实样本
            synthetic_samples: (M, C, L) 合成样本
            conditions: (M, 3) 条件向量 (可选)
            base_performance: 基线性能 (Layer 3需要)
            downstream_fn: 下游评估函数 (Layer 3需要)
        
        Returns:
            admitted_samples: 通过门控的样本
            admitted_conditions: 通过门控的条件
            gate_metrics: 门控指标
        """
        M = synthetic_samples.shape[0]
        
        # Layer 1: 统计一致性
        l1_pass, l1_metrics = self.layer1_statistical_check(real_samples, synthetic_samples)
        l1_samples = synthetic_samples[l1_pass]
        l1_conditions = conditions[l1_pass] if conditions is not None else None
        
        # Layer 2: 专家评审
        if l1_pass.sum() > 0:
            l2_pass, l2_metrics = self.layer2_expert_review(real_samples, l1_samples)
            l2_samples = l1_samples[l2_pass]
            l2_conditions = l1_conditions[l2_pass] if l1_conditions is not None else None
        else:
            l2_samples = synthetic_samples[:0]
            l2_conditions = conditions[:0] if conditions is not None else None
            l2_metrics = {'layer2_pass_rate': 0.0}
        
        # Layer 3: 下游效用 (需要外部评估函数)
        if downstream_fn is not None and base_performance is not None and len(l2_samples) > 0:
            # 评估增强后的性能
            augmented_performance = downstream_fn(l2_samples, l2_conditions)
            current_ratio = len(l2_samples) / len(real_samples)
            
            l3_pass, new_ratio, l3_metrics = self.layer3_downstream_utility(
                base_performance, augmented_performance, current_ratio,
                len(l2_samples), len(real_samples)
            )
            
            # 如果通过，返回所有l2样本；如果不通过，按比例减少
            if l3_pass:
                final_samples = l2_samples
                final_conditions = l2_conditions
            else:
                # 按比例减少
                n_keep = int(len(l2_samples) * (new_ratio / current_ratio))
                indices = np.random.choice(len(l2_samples), n_keep, replace=False)
                final_samples = l2_samples[indices]
                final_conditions = l2_conditions[indices] if l2_conditions is not None else None
        else:
            final_samples = l2_samples
            final_conditions = l2_conditions
            l3_metrics = {'note': 'Layer 3 skipped (no downstream function provided)'}
        
        gate_metrics = {
            'layer1': l1_metrics,
            'layer2': l2_metrics,
            'layer3': l3_metrics,
            'total_input': M,
            'total_admitted': len(final_samples),
            'admission_rate': len(final_samples) / M if M > 0 else 0,
        }
        
        return final_samples, final_conditions, gate_metrics
    
    def reset_stats(self):
        """重置统计"""
        self.gate_stats = {'layer1_pass': 0, 'layer2_pass': 0, 'layer3_pass': 0,
                           'layer1_fail': 0, 'layer2_fail': 0, 'layer3_fail': 0}


# ============================================================
# 二、故障诊断评估指标
# ============================================================

def classification_metrics(y_true, y_pred, average='macro'):
    """
    计算分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        average: 平均方式
    
    Returns:
        metrics字典
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    return metrics


def per_class_metrics(y_true, y_pred, n_classes=None):
    """每类别的指标"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1
    
    results = {}
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[c] = {'precision': precision, 'recall': recall, 'f1': f1,
                       'support': np.sum(y_true == c)}
    
    return results


# ============================================================
# 三、RUL预测评估指标
# ============================================================

def rul_metrics(y_true, y_pred):
    """
    计算RUL预测评估指标
    
    Args:
        y_true: 真实RUL值 (n_samples,)
        y_pred: 预测RUL值 (n_samples,)
    
    Returns:
        metrics字典
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mae = np.mean(np.abs(y_true - y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    # 对称MAPE
    mape = np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)) * 100
    
    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'mape': mape,
    }
    
    return metrics


def cmapss_score(y_true, y_pred):
    """
    C-MAPSS标准评分函数
    
    Score = sum(exp(d/13) - 1) for d < 0 (预测过早)
          + sum(exp(d/10) - 1) for d >= 0 (预测过晚)
    
    分数越低越好
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    d = y_pred - y_true
    score = np.zeros_like(d)
    
    early_mask = d < 0
    late_mask = d >= 0
    
    score[early_mask] = np.exp(-d[early_mask] / 13.0) - 1
    score[late_mask] = np.exp(d[late_mask] / 10.0) - 1
    
    return np.sum(score)


def quantile_coverage(y_true, y_pred_quantiles, quantiles):
    """
    计算分位数区间的经验覆盖率
    
    Args:
        y_true: 真实值 (n,)
        y_pred_quantiles: 预测分位数 (n, n_quantiles)
        quantiles: 分位数列表
    
    Returns:
        coverage字典
    """
    y_true = np.array(y_true).flatten()
    y_pred_quantiles = np.array(y_pred_quantiles)
    
    coverages = {}
    for i, q in enumerate(quantiles):
        coverages[f'quantile_{q}'] = np.mean(y_true <= y_pred_quantiles[:, i])
    
    # 90%区间覆盖率 (如果有0.05和0.95分位数，或0.1和0.9)
    if 0.1 in quantiles and 0.9 in quantiles:
        i_low = quantiles.index(0.1)
        i_high = quantiles.index(0.9)
        in_interval = (y_true >= y_pred_quantiles[:, i_low]) & (y_true <= y_pred_quantiles[:, i_high])
        coverages['interval_80pct'] = np.mean(in_interval)
    
    if 0.05 in quantiles and 0.95 in quantiles:
        i_low = quantiles.index(0.05)
        i_high = quantiles.index(0.95)
        in_interval = (y_true >= y_pred_quantiles[:, i_low]) & (y_true <= y_pred_quantiles[:, i_high])
        coverages['interval_90pct'] = np.mean(in_interval)
    
    return coverages


# ============================================================
# 四、生成质量评估指标
# ============================================================

def generation_quality_metrics(real_samples, fake_samples):
    """
    生成质量评估指标 (简化版FID-like)
    
    基于特征空间的分布距离
    """
    from sklearn.decomposition import PCA
    
    real_flat = real_samples.reshape(len(real_samples), -1)
    fake_flat = fake_samples.reshape(len(fake_samples), -1)
    
    # PCA降维
    n_components = min(50, real_flat.shape[1], len(real_flat) - 1, len(fake_flat) - 1)
    pca = PCA(n_components=n_components)
    
    all_data = np.vstack([real_flat, fake_flat])
    all_pca = pca.fit_transform(all_data)
    
    real_pca = all_pca[:len(real_samples)]
    fake_pca = all_pca[len(real_samples):]
    
    # 计算均值和协方差
    mu_real = np.mean(real_pca, axis=0)
    mu_fake = np.mean(fake_pca, axis=0)
    cov_real = np.cov(real_pca, rowvar=False)
    cov_fake = np.cov(fake_pca, rowvar=False)
    
    # Fréchet距离 (简化，不使用矩阵平方根)
    diff = mu_real - mu_fake
    mean_diff = np.sum(diff ** 2)
    cov_diff = np.sum((cov_real - cov_fake) ** 2)
    
    fid_approx = mean_diff + cov_diff
    
    metrics = {
        'fid_approx': fid_approx,
        'mean_diff': mean_diff,
        'cov_diff': cov_diff,
    }
    
    return metrics


if __name__ == "__main__":
    # 测试质量门控
    print("Testing QualityGate...")
    
    # 生成测试数据
    np.random.seed(42)
    real = np.random.randn(100, 1, 128)
    synthetic_good = np.random.randn(50, 1, 128) * 0.95 + 0.05  # 接近真实
    synthetic_bad = np.random.randn(50, 1, 128) * 2.0 + 1.0  # 分布偏移
    
    gate = QualityGate(ks_alpha=0.05, acf_threshold=0.5, fft_threshold=0.5)
    
    # 测试好的合成样本
    pass_mask, metrics = gate.layer1_statistical_check(real, synthetic_good)
    print(f"\nGood synthetic - Layer1 pass rate: {metrics['layer1_overall_pass_rate']:.2%}")
    print(f"  KS pass: {metrics['ks_pass_rate']:.2%}, ACF pass: {metrics['acf_pass_rate']:.2%}, "
          f"FFT pass: {metrics['fft_pass_rate']:.2%}")
    
    # 测试差的合成样本
    pass_mask, metrics = gate.layer1_statistical_check(real, synthetic_bad)
    print(f"Bad synthetic - Layer1 pass rate: {metrics['layer1_overall_pass_rate']:.2%}")
    
    # 测试Layer 2
    pass_mask, metrics = gate.layer2_expert_review(real, synthetic_good)
    print(f"\nGood synthetic - Layer2 pass rate: {metrics['layer2_pass_rate']:.2%}")
    
    # 测试分类指标
    print("\nTesting classification metrics...")
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 2])
    metrics = classification_metrics(y_true, y_pred)
    print(f"  Acc: {metrics['accuracy']:.2f}, F1: {metrics['f1']:.2f}")
    
    # 测试RUL指标
    print("\nTesting RUL metrics...")
    y_true = np.array([10, 20, 30, 40, 50])
    y_pred = np.array([12, 18, 33, 38, 55])
    metrics = rul_metrics(y_true, y_pred)
    score = cmapss_score(y_true, y_pred)
    print(f"  RMSE: {metrics['rmse']:.2f}, Score: {score:.2f}")
    
    print("\nAll utility functions OK!")
