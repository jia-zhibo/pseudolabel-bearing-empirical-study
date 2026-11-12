"""
训练工具函数

包含:
1. 扩散模型训练器
2. 分类器训练器
3. RUL预测器训练器
4. 通用训练循环工具
"""

import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class DiffusionTrainer:
    """扩散模型训练器"""
    
    def __init__(self, model, train_loader, val_loader=None,
                 lr=1e-4, weight_decay=1e-5, device='cuda',
                 save_dir='./checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100)
        
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch in pbar:
            # 处理不同格式的数据
            if len(batch) == 3:
                x, labels, conditions = batch
                x = x.to(self.device)
                conditions = conditions.to(self.device)
                
                # 前向传播
                loss = self.model(x, conditions)
            elif len(batch) == 2:
                x, labels = batch
                x = x.to(self.device)
                labels = labels.to(self.device)
                
                # 对于基线模型，可能使用标签作为条件
                if hasattr(self.model, 'denoiser') and hasattr(self.model.denoiser, 'cond_embed'):
                    loss = self.model(x, labels)
                else:
                    # 无条件或使用标签
                    loss = self.model(x, labels) if hasattr(self.model, 'n_classes') else self.model(x)
            else:
                x = batch[0].to(self.device)
                loss = self.model(x)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / n_batches
        self.history['train_loss'].append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        """验证"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for batch in tqdm(self.val_loader, desc=f'Epoch {epoch} [Val]'):
            if len(batch) == 3:
                x, labels, conditions = batch
                x = x.to(self.device)
                conditions = conditions.to(self.device)
                loss = self.model(x, conditions)
            elif len(batch) == 2:
                x, labels = batch
                x = x.to(self.device)
                labels = labels.to(self.device)
                loss = self.model(x, labels)
            else:
                x = batch[0].to(self.device)
                loss = self.model(x)
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        self.history['val_loss'].append(avg_loss)
        
        # 保存最佳模型
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.save_checkpoint('best_model.pt')
        
        return avg_loss
    
    def train(self, n_epochs, save_freq=10):
        """完整训练"""
        print(f"Training diffusion model for {n_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            self.scheduler.step()
            
            if val_loss is not None:
                print(f"Epoch {epoch}/{n_epochs} - Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            else:
                print(f"Epoch {epoch}/{n_epochs} - Train Loss: {train_loss:.4f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if epoch % save_freq == 0:
                self.save_checkpoint(f'epoch_{epoch}.pt')
        
        print(f"Training complete. Best val loss: {self.best_loss:.4f}")
        return self.history
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
        }, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")


class ClassifierTrainer:
    """故障分类器训练器"""
    
    def __init__(self, model, train_loader, val_loader=None, test_loader=None,
                 lr=1e-3, weight_decay=1e-4, device='cuda',
                 save_dir='./checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50)
        
        self.best_f1 = 0
        self.history = {'train_loss': [], 'train_acc': [],
                        'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch in pbar:
            if len(batch) >= 2:
                x, labels = batch[0], batch[1]
            else:
                x = batch[0]
                labels = batch[1] if len(batch) > 1 else None
            
            x = x.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(x)
            loss = self.criterion(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.2%}'})
        
        avg_loss = total_loss / n_batches
        accuracy = correct / total
        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(accuracy)
        return avg_loss, accuracy
    
    @torch.no_grad()
    def evaluate(self, loader, name='Val'):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        n_batches = 0
        
        for batch in tqdm(loader, desc=f'[{name}]'):
            x, labels = batch[0], batch[1]
            x = x.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.model(x)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return avg_loss, accuracy, f1, all_preds, all_labels
    
    def train(self, n_epochs, save_freq=10, early_stop_patience=20):
        print(f"Training classifier for {n_epochs} epochs...")
        
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(1, n_epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            
            if self.val_loader is not None:
                val_loss, val_acc, val_f1, _, _ = self.evaluate(self.val_loader, 'Val')
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
                self.history['val_f1'].append(val_f1)
                
                print(f"Epoch {epoch}/{n_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | "
                      f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}, F1: {val_f1:.4f}")
                
                if val_f1 > best_f1:
                    best_f1 = val_f1
                    self.save_checkpoint('best_classifier.pt')
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                print(f"Epoch {epoch}/{n_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%}")
            
            self.scheduler.step()
            
            if epoch % save_freq == 0:
                self.save_checkpoint(f'classifier_epoch_{epoch}.pt')
            
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # 测试集评估
        if self.test_loader is not None:
            print("\nEvaluating on test set...")
            self.load_checkpoint('best_classifier.pt')
            test_loss, test_acc, test_f1, test_preds, test_labels = self.evaluate(self.test_loader, 'Test')
            print(f"Test Results - Loss: {test_loss:.4f}, Acc: {test_acc:.2%}, F1: {test_f1:.4f}")
            self.history['test_metrics'] = {
                'loss': test_loss, 'accuracy': test_acc, 'f1': test_f1
            }
        
        return self.history
    
    def save_checkpoint(self, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'history': self.history,
        }, path)
    
    def load_checkpoint(self, filename):
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_f1 = checkpoint['best_f1']
        self.history = checkpoint['history']


class RULTrainer:
    """RUL预测器训练器"""
    
    def __init__(self, model, train_loader, val_loader=None, test_loader=None,
                 lr=1e-3, weight_decay=1e-4, device='cuda',
                 quantiles=[0.1, 0.5, 0.9], save_dir='./checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.quantiles = quantiles
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50)
        
        self.best_rmse = float('inf')
        self.history = {'train_loss': [], 'val_rmse': [], 'val_score': []}
    
    def quantile_loss(self, pred, target):
        """分位数损失"""
        target = target.unsqueeze(-1)
        errors = target - pred
        q = torch.tensor(self.quantiles, device=pred.device).unsqueeze(0)
        loss = torch.max(q * errors, (q - 1) * errors)
        return torch.mean(loss)
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [Train]')
        for batch in pbar:
            x, rul = batch[0], batch[1]
            x = x.to(self.device)
            rul = rul.to(self.device)
            
            rul_pred, hi_pred = self.model(x)
            loss = self.quantile_loss(rul_pred, rul)
            
            # 添加HI辅助损失
            hi_target = 1.0 - (rul / 125.0).clamp(0, 1)  # 简单的HI构造
            hi_loss = nn.MSELoss()(hi_pred.squeeze(-1), hi_target)
            loss = loss + 0.1 * hi_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / n_batches
        self.history['train_loss'].append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, loader, name='Val'):
        self.model.eval()
        all_preds = []
        all_targets = []
        
        for batch in tqdm(loader, desc=f'[{name}]'):
            x, rul = batch[0], batch[1]
            x = x.to(self.device)
            
            rul_pred, _ = self.model(x)
            # 取中位数作为点预测
            median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
            point_pred = rul_pred[:, median_idx]
            
            all_preds.extend(point_pred.cpu().numpy())
            all_targets.extend(rul.numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        from data.cmapss_data import cmapss_score
        rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
        mae = np.mean(np.abs(all_preds - all_targets))
        score = cmapss_score(all_targets, all_preds)
        
        return rmse, mae, score, all_preds, all_targets
    
    def train(self, n_epochs, save_freq=10, early_stop_patience=20):
        print(f"Training RUL predictor for {n_epochs} epochs...")
        
        best_rmse = float('inf')
        patience_counter = 0
        
        for epoch in range(1, n_epochs + 1):
            train_loss = self.train_epoch(epoch)
            
            if self.val_loader is not None:
                val_rmse, val_mae, val_score, _, _ = self.evaluate(self.val_loader, 'Val')
                self.history['val_rmse'].append(val_rmse)
                self.history['val_score'].append(val_score)
                
                print(f"Epoch {epoch}/{n_epochs} - "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val RMSE: {val_rmse:.2f}, MAE: {val_mae:.2f}, Score: {val_score:.1f}")
                
                if val_rmse < best_rmse:
                    best_rmse = val_rmse
                    self.save_checkpoint('best_rul_predictor.pt')
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                print(f"Epoch {epoch}/{n_epochs} - Train Loss: {train_loss:.4f}")
            
            self.scheduler.step()
            
            if epoch % save_freq == 0:
                self.save_checkpoint(f'rul_epoch_{epoch}.pt')
            
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # 测试集评估
        if self.test_loader is not None:
            print("\nEvaluating on test set...")
            self.load_checkpoint('best_rul_predictor.pt')
            test_rmse, test_mae, test_score, test_preds, test_labels = self.evaluate(self.test_loader, 'Test')
            print(f"Test Results - RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, Score: {test_score:.1f}")
            self.history['test_metrics'] = {
                'rmse': test_rmse, 'mae': test_mae, 'score': test_score
            }
        
        return self.history
    
    def save_checkpoint(self, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_rmse': self.best_rmse,
            'history': self.history,
        }, path)
    
    def load_checkpoint(self, filename):
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_rmse = checkpoint['best_rmse']
        self.history = checkpoint['history']


def set_seed(seed=42):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
