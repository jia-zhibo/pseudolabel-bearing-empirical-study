# -*- coding: utf-8 -*-
"""
域适应故障分类模型
- DANN: Domain Adversarial Neural Network (对抗域适应)
- MMD: Maximum Mean Discrepancy (分布对齐)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalLayer(Function):
    """梯度反转层（DANN核心）"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


class DomainDiscriminator(nn.Module):
    """域判别器：判断样本来自源域还是目标域"""
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2),
        )
    
    def forward(self, x):
        return self.net(x)


class DANNClassifier(nn.Module):
    """
    DANN: 特征提取器 + 分类器 + 域判别器
    训练时：
    - 特征提取器试图提取域不变特征（骗过域判别器）
    - 分类器试图正确分类故障类型
    - 域判别器试图区分源域/目标域
    """
    def __init__(self, in_channels=1, n_classes=10, hidden_dim=64,
                 lstm_layers=2, n_heads=4, dropout=0.3):
        super().__init__()
        
        # CNN特征提取 (same as FaultClassifier front end)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=hidden_dim,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        
        feat_dim = hidden_dim * 2
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, n_classes),
        )
        
        # Domain discriminator
        self.domain_discriminator = DomainDiscriminator(feat_dim, feat_dim // 2)
        
        self.dropout = dropout
    
    def extract_features(self, x):
        """提取特征（CNN + LSTM + Attention）"""
        # x: (B, 1, L)
        x = self.conv_layers(x)  # (B, 128, L')
        x = x.transpose(1, 2)   # (B, L', 128)
        
        x, _ = self.lstm(x)     # (B, L', hidden*2)
        
        # Self-attention pooling
        attn_out, _ = self.attention(x, x, x)  # (B, L', hidden*2)
        feat = attn_out.mean(dim=1)  # (B, hidden*2)
        
        return feat
    
    def forward(self, x, alpha=1.0, return_domain=False):
        """
        Args:
            x: input signal (B, 1, L)
            alpha: GRL强度（训练前期低，后期高）
            return_domain: 是否返回域判别结果
        """
        feat = self.extract_features(x)
        
        # Classification
        cls_logits = self.classifier(feat)
        
        if return_domain:
            # Domain discrimination (with gradient reversal)
            reversed_feat = grad_reverse(feat, alpha)
            domain_logits = self.domain_discriminator(reversed_feat)
            return cls_logits, domain_logits, feat
        
        return cls_logits


def compute_mmd(source_feat, target_feat, kernel_mul=2.0, kernel_num=5):
    """
    计算多核MMD距离（最大均值差异）
    衡量源域和目标域特征分布的差异
    """
    n_samples = min(source_feat.shape[0], target_feat.shape[0])
    source_feat = source_feat[:n_samples]
    target_feat = target_feat[:n_samples]
    
    # 计算高斯核的带宽
    total = torch.cat([source_feat, target_feat], dim=0)
    total_norm = (total ** 2).sum(dim=1, keepdim=True)
    distances = total_norm + total_norm.t() - 2 * total @ total.t()
    
    # 多个核带宽
    bandwidths = []
    median_dist = torch.median(distances[distances > 0])
    for i in range(kernel_num):
        bandwidths.append(median_dist * (kernel_mul ** (i - kernel_num // 2)))
    
    # 多核
    kernels = 0
    for bw in bandwidths:
        kernels += torch.exp(-distances / (2 * bw))
    
    # MMD = k_ss + k_tt - 2*k_st
    n = n_samples
    k_ss = kernels[:n, :n].mean()
    k_tt = kernels[n:, n:].mean()
    k_st = kernels[:n, n:].mean()
    
    mmd = k_ss + k_tt - 2 * k_st
    return mmd


class MMDClassifier(nn.Module):
    """
    MMD域适应分类器
    在分类损失基础上加入MMD域对齐损失
    """
    def __init__(self, in_channels=1, n_classes=10, hidden_dim=64,
                 lstm_layers=2, n_heads=4, dropout=0.3):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=hidden_dim,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        
        feat_dim = hidden_dim * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, n_classes),
        )
        
        self.dropout = dropout
    
    def extract_features(self, x):
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        attn_out, _ = self.attention(x, x, x)
        feat = attn_out.mean(dim=1)
        return feat
    
    def forward(self, x):
        feat = self.extract_features(x)
        cls_logits = self.classifier(feat)
        return cls_logits, feat
