"""
下游检测与预测模型

包含:
1. CNN-BiLSTM-Attention 故障分类器 (用于CWRU等诊断任务)
2. CNN-BiLSTM-Attention RUL预测器 (用于C-MAPSS等预测任务)

对应论文中的协同设计检测器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ==================== 注意力模块 ====================

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    
    def __init__(self, d_k):
        super().__init__()
        self.scale = math.sqrt(d_k)
    
    def forward(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力"""
    
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
            attn_weights: (B, H, L, L)
        """
        B, L, D = x.shape
        
        # 线性投影并分头
        q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B, H, L, d)
        k = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        
        # 注意力计算
        output, attn_weights = self.attention(q, k, v, mask)
        
        # 拼接并输出投影
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        output = self.w_o(output)
        
        return output, attn_weights


# ==================== 多尺度卷积模块 ====================

class MultiScaleConv1d(nn.Module):
    """
    多尺度一维卷积
    并行使用不同大小的卷积核，捕捉不同尺度的特征
    """
    
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 7)):
        super().__init__()
        
        n_convs = len(kernel_sizes)
        # 分配通道数，确保总和等于out_channels
        base_ch = out_channels // n_convs
        remainder = out_channels % n_convs
        
        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            ch = base_ch + (1 if i < remainder else 0)
            padding = k // 2
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, ch, k, padding=padding),
                nn.BatchNorm1d(ch),
                nn.ReLU(),
            ))
    
    def forward(self, x):
        """
        Args:
            x: (B, C_in, L)
        Returns:
            (B, C_out, L)
        """
        outputs = []
        for conv in self.convs:
            outputs.append(conv(x))
        return torch.cat(outputs, dim=1)


# ==================== CNN-BiLSTM-Attention 骨干网络 ====================

class CNNBiLSTMAttnBackbone(nn.Module):
    """
    CNN-BiLSTM-Attention 骨干网络
    
    结构:
    1. 多尺度1D卷积 (捕捉局部瞬态特征)
    2. 双向LSTM (建模长程时序依赖)
    3. 多头注意力 (重点关注关键时段)
    
    输出: 池化后的全局特征向量
    """
    
    def __init__(self, in_channels=1, hidden_dim=128, lstm_layers=2,
                 n_heads=4, kernel_sizes=(3, 5, 7), dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 多尺度卷积
        self.conv = MultiScaleConv1d(in_channels, hidden_dim, kernel_sizes)
        
        # 池化降维
        self.pool = nn.MaxPool1d(2)
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )
        
        # 注意力层
        self.attention = MultiHeadAttention(hidden_dim * 2, n_heads)
        self.attn_norm = nn.LayerNorm(hidden_dim * 2)
        
        # 输出投影
        self.output_dim = hidden_dim * 2
        
    def forward(self, x):
        """
        Args:
            x: (B, C_in, L) 输入时序
        
        Returns:
            pooled_feature: (B, hidden_dim*2) 全局池化特征
            attn_weights: (B, H, L', L') 注意力权重
            lstm_out: (B, L', hidden_dim*2) LSTM输出序列
        """
        B, C, L = x.shape
        
        # 卷积特征提取
        conv_out = self.conv(x)  # (B, hidden_dim, L)
        conv_out = self.pool(conv_out)  # (B, hidden_dim, L/2)
        
        # 转换为序列格式
        seq = conv_out.transpose(1, 2)  # (B, L/2, hidden_dim)
        
        # BiLSTM
        lstm_out, _ = self.lstm(seq)  # (B, L/2, hidden_dim*2)
        
        # 注意力
        attn_out, attn_weights = self.attention(lstm_out)
        attn_out = self.attn_norm(attn_out + lstm_out)  # 残差连接
        
        # 全局平均池化
        pooled = torch.mean(attn_out, dim=1)  # (B, hidden_dim*2)
        
        return pooled, attn_weights, attn_out


# ==================== 故障分类器 ====================

class FaultClassifier(nn.Module):
    """
    故障分类器
    
    基于CNN-BiLSTM-Attention骨干，添加分类头
    """
    
    def __init__(self, in_channels=1, n_classes=10, hidden_dim=128,
                 lstm_layers=2, n_heads=4, dropout=0.2):
        super().__init__()
        
        self.backbone = CNNBiLSTMAttnBackbone(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
    
    def forward(self, x, return_attn=False):
        """
        Args:
            x: (B, C, L)
            return_attn: 是否返回注意力权重
        
        Returns:
            logits: (B, n_classes)
            attn_weights (可选): 注意力权重
        """
        pooled, attn_weights, _ = self.backbone(x)
        logits = self.classifier(pooled)
        
        if return_attn:
            return logits, attn_weights
        return logits


# ==================== RUL预测器 ====================

class RULPredictor(nn.Module):
    """
    剩余使用寿命 (RUL) 预测器
    
    基于CNN-BiLSTM-Attention骨干，添加回归头
    支持分位数回归输出
    """
    
    def __init__(self, in_channels=14, hidden_dim=128, lstm_layers=2,
                 n_heads=4, dropout=0.2, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        
        self.quantiles = quantiles
        
        self.backbone = CNNBiLSTMAttnBackbone(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            lstm_layers=lstm_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        
        # 健康指标 (HI) 回归头
        self.hi_head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        
        # RUL分位数回归头
        self.rul_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.backbone.output_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in quantiles
        ])
    
    def forward(self, x, return_attn=False):
        """
        Args:
            x: (B, C, L)
            return_attn: 是否返回注意力权重
        
        Returns:
            rul_pred: (B, n_quantiles) RUL分位数预测
            hi_pred: (B, 1) 健康指标预测
            attn_weights (可选)
        """
        pooled, attn_weights, _ = self.backbone(x)
        
        hi_pred = self.hi_head(pooled)  # (B, 1)
        
        rul_preds = []
        for head in self.rul_heads:
            rul_preds.append(head(pooled))
        rul_pred = torch.cat(rul_preds, dim=-1)  # (B, n_quantiles)
        
        if return_attn:
            return rul_pred, hi_pred, attn_weights
        return rul_pred, hi_pred


def quantile_loss(pred, target, quantiles):
    """
    分位数回归损失 (Pinball loss)
    
    Args:
        pred: (B, n_quantiles) 预测值
        target: (B,) 真实值
        quantiles: 分位数列表
    
    Returns:
        loss: 标量损失
    """
    target = target.unsqueeze(-1)  # (B, 1)
    errors = target - pred  # (B, n_q)
    
    q = torch.tensor(quantiles, device=pred.device).unsqueeze(0)  # (1, n_q)
    
    loss = torch.max(q * errors, (q - 1) * errors)
    return torch.mean(loss)


# ==================== GAN增强基线生成器 ====================

class GANGenerator(nn.Module):
    """
    条件GAN生成器 (基线对比用)
    基于LSTM的条件时序生成器
    """
    
    def __init__(self, latent_dim=100, n_classes=10, out_channels=1,
                 seq_len=512, hidden_dim=128):
        super().__init__()
        
        self.seq_len = seq_len
        self.out_channels = out_channels
        
        # 标签嵌入
        self.label_embed = nn.Embedding(n_classes, latent_dim)
        
        # LSTM生成器
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
        )
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_channels),
            nn.Tanh(),
        )
    
    def forward(self, z, labels):
        """
        Args:
            z: (B, latent_dim) 噪声
            labels: (B,) 类别标签
        Returns:
            (B, out_channels, seq_len)
        """
        # 条件注入
        label_emb = self.label_embed(labels)  # (B, latent_dim)
        z = z + label_emb
        
        # 扩展为序列
        seq = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, L, latent_dim)
        
        # LSTM生成
        lstm_out, _ = self.lstm(seq)  # (B, L, hidden_dim)
        
        # 输出
        out = self.output(lstm_out)  # (B, L, C)
        out = out.transpose(1, 2)  # (B, C, L)
        
        return out


class GANDiscriminator(nn.Module):
    """
    GAN判别器 (基线对比用)
    """
    
    def __init__(self, in_channels=1, seq_len=512, n_classes=10, hidden_dim=128):
        super().__init__()
        
        self.label_embed = nn.Embedding(n_classes, seq_len * in_channels)
        self.seq_len = seq_len
        self.in_channels = in_channels
        
        # 卷积判别器
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels * 2, hidden_dim, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )
        
        # 计算卷积输出长度
        conv_out_len = seq_len // 8
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 4 * conv_out_len, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x, labels):
        """
        Args:
            x: (B, C, L)
            labels: (B,)
        Returns:
            (B, 1) 真假概率
        """
        B = x.shape[0]
        # 条件拼接
        label_emb = self.label_embed(labels).view(B, self.in_channels, self.seq_len)
        x = torch.cat([x, label_emb], dim=1)  # (B, 2C, L)
        
        conv_out = self.conv(x)
        conv_out = conv_out.view(B, -1)
        out = self.fc(conv_out)
        return out


if __name__ == "__main__":
    # 测试故障分类器
    print("Testing FaultClassifier...")
    classifier = FaultClassifier(in_channels=1, n_classes=10, hidden_dim=64, lstm_layers=2)
    x = torch.randn(4, 1, 512)
    logits, attn = classifier(x, return_attn=True)
    print(f"  Input: {x.shape}")
    print(f"  Output logits: {logits.shape}")
    print(f"  Attention weights: {attn.shape}")
    
    # 测试RUL预测器
    print("\nTesting RULPredictor...")
    predictor = RULPredictor(in_channels=14, hidden_dim=64, lstm_layers=2)
    x = torch.randn(4, 14, 30)
    rul_pred, hi_pred = predictor(x)
    print(f"  Input: {x.shape}")
    print(f"  RUL pred: {rul_pred.shape} (quantiles: {predictor.quantiles})")
    print(f"  HI pred: {hi_pred.shape}")
    
    # 测试分位数损失
    target = torch.randn(4)
    loss = quantile_loss(rul_pred, target, predictor.quantiles)
    print(f"  Quantile loss: {loss.item():.4f}")
    
    # 测试GAN
    print("\nTesting GAN...")
    gen = GANGenerator(latent_dim=64, n_classes=10, out_channels=1, seq_len=128, hidden_dim=64)
    disc = GANDiscriminator(in_channels=1, seq_len=128, n_classes=10, hidden_dim=64)
    z = torch.randn(4, 64)
    labels = torch.randint(0, 10, (4,))
    fake = gen(z, labels)
    print(f"  Generated: {fake.shape}")
    d_out = disc(fake, labels)
    print(f"  Discriminator output: {d_out.shape}")
    
    print("\nAll detector models OK!")
