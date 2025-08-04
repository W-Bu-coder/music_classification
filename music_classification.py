import pandas as pd
import numpy as np
import torch
import os
import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from tqdm import tqdm
import time
import random
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from sklearn.metrics import accuracy_score, classification_report
warnings.filterwarnings('ignore')

def load_all_features():
    """加载所有特征文件（内存优化版）"""
    print("📂 加载所有特征文件...")

    feature_files = glob.glob(os.path.join(FEATURE_PATH, 'features_*.npz'))
    if not feature_files:
        print("❌ 没有找到特征文件！")
        return None, None, None
    
    # 初始化空列表（预分配内存更高效）
    all_features = []
    all_labels = []
    all_track_ids = []
    
    for file in feature_files:
        print(f"   加载: {file}")
        with np.load(file) as data:  # 使用上下文管理器确保文件及时关闭
            # 关键修改1：立即转换为float32减少内存
            all_features.append(data['features'].astype(np.float32))
            all_labels.append(data['labels'])
            all_track_ids.append(data['track_ids'])
    
    # 关键修改2：分步合并+及时释放内存
    features = np.concatenate(all_features, axis=0)
    del all_features  # 立即删除临时变量
    labels = np.concatenate(all_labels, axis=0)
    track_ids = np.concatenate(all_track_ids, axis=0)
    
    # 强制垃圾回收
    import gc
    gc.collect()
    
    print(f"✅ 加载完成! 总样本数: {len(features)}, 特征维度: {features.shape}")
    return features, labels, track_ids

class MusicDataset(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        # 转换为张量
        feature = torch.FloatTensor(feature).unsqueeze(0)  # 添加通道维度
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label

def create_data_splits(features, labels, test_size=0.2, val_size=0.1, random_state=42):
    """
    创建训练、验证和测试集
    """
    print("🔄 划分数据集...")
    
    # 首先分离出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=test_size, 
        random_state=random_state, stratify=labels
    )
    
    # 从剩余数据中分离出验证集
    val_size_adjusted = val_size / (1 - test_size)  # 调整验证集比例
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print(f"✅ 数据集划分完成:")
    print(f"   训练集: {len(X_train)} 样本 ({len(X_train)/len(features)*100:.1f}%)")
    print(f"   验证集: {len(X_val)} 样本 ({len(X_val)/len(features)*100:.1f}%)")
    print(f"   测试集: {len(X_test)} 样本 ({len(X_test)/len(features)*100:.1f}%)")
    
    # 检查标签分布
    print(f"\n📊 各集合的标签分布:")
    print(f"   训练集: {np.bincount(y_train)}")
    print(f"   验证集: {np.bincount(y_val)}")
    print(f"   测试集: {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# 数据增强
class AudioTransform:
    def __init__(self, noise_factor=0.005, time_shift_factor=0.1):
        self.noise_factor = noise_factor
        self.time_shift_factor = time_shift_factor
    
    def __call__(self, x):
        # 添加噪声
        if random.random() > 0.5:
            noise = torch.randn_like(x) * self.noise_factor
            x = x + noise
        
        # 时间偏移
        if random.random() > 0.5:
            shift = int(x.shape[-1] * self.time_shift_factor * (random.random() - 0.5))
            if shift != 0:
                if shift > 0:
                    x = torch.cat([x[..., shift:], torch.zeros_like(x[..., :shift])], dim=-1)
                else:
                    x = torch.cat([torch.zeros_like(x[..., :abs(shift)]), x[..., :shift]], dim=-1)
        
        return x


    
class CNN(nn.Module):
    def __init__(self, num_classes=8):
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 第二层
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 第三层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # 第四层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
        )
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x




def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        with torch.no_grad():  
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(train_losses, label='Training Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_title('Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Training Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='red')
    ax2.set_title('Accuracy vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()
        
        
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs=50, device='cuda', patience=10):
    """完整的训练流程"""
    
    print(f"🚀 开始训练，共 {num_epochs} 个epoch...")
    
    # 初始化记录
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # 早停机制
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 调整学习率
        if scheduler:
            scheduler.step(val_loss)
        
        # 打印结果
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_music_model.pth')
            print(f"🎯 新的最佳验证准确率: {best_val_acc:.2f}%")
        
        # 早停检查
        if early_stopping(val_loss, model):
            print(f"⏹️  早停触发，在第 {epoch+1} 个epoch停止训练")
            break
    
    training_time = time.time() - start_time
    print(f"\n✅ 训练完成！")
    print(f"⏱️  训练时间: {training_time/60:.2f} 分钟")
    print(f"🏆 最佳验证准确率: {best_val_acc:.2f}%")
    
    return train_losses, val_losses, train_accs, val_accs




class ConfigurableCNN(nn.Module):
    """可配置的CNN模型，用于超参数调优"""
    
    def __init__(self, trial, num_classes=8):
        super(ConfigurableCNN, self).__init__()
        
        # 通过trial对象获取超参数
        self.n_layers = trial.suggest_int('n_layers', 2, 5)
        
        layers = []
        in_channels = 1
        
        for i in range(self.n_layers):
            # 每层的通道数
            out_channels = trial.suggest_categorical(f'n_units_l{i}', [16, 32, 64, 128, 256])
            
            # 卷积核大小
            kernel_size = trial.suggest_categorical(f'kernel_size_l{i}', [3, 5])
            
            # Dropout概率
            dropout_rate = trial.suggest_float(f'dropout_l{i}', 0.1, 0.5)
            
            # 构建卷积块
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout_rate)
            ])
            
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # 自适应池化
        pool_size = trial.suggest_categorical('pool_size', [2, 4, 8])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        
        # 分类器
        fc_input_size = in_channels * pool_size * pool_size
        hidden_size = trial.suggest_categorical('fc_hidden_size', [128, 256, 512])
        final_dropout = trial.suggest_float('final_dropout', 0.2, 0.7)
        
        self.classifier = nn.Sequential(
            nn.Dropout(final_dropout),
            nn.Linear(fc_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(final_dropout * 0.5),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    
def objective(trial):
    """Optuna的目标函数"""
    
    # 1. 模型超参数
    model = ConfigurableCNN(trial, num_classes=8).to(device)
    
    # 2. 训练超参数
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # 3. 优化器选择
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:  # SGD
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    
    # 4. 创建数据加载器（使用新的batch_size）
    if 'train_dataset' in globals():
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    else:
        # 如果数据集不存在，返回一个虚拟值用于演示
        return 0.5
    
    # 5. 训练配置
    criterion = nn.CrossEntropyLoss()
    n_epochs = 10  # 为了快速调优，减少epoch数
    
    # 6. 训练循环
    best_val_acc = 0.0
    
    for epoch in range(n_epochs):
        # 训练
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 50:  # 限制每个epoch的批次数，加速调优
                break
                
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx > 20:  # 限制验证批次数
                    break
                    
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)
        
        # 报告中间结果给Optuna
        trial.report(val_acc, epoch)
        
        # 如果效果不好，提前剪枝
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_acc

def run_hyperparameter_tuning(n_trials=50):
    """运行超参数调优"""
    
    print("🔍 开始超参数自动调优...")
    print(f"将尝试 {n_trials} 种不同的超参数组合")
    
    # 创建研究对象
    study = optuna.create_study(
        direction='maximize',  # 最大化验证准确率
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    )
    
    # 开始优化
    study.optimize(objective, n_trials=n_trials)
    
    # 输出结果
    print("🎯 超参数调优完成！")
    print(f"最佳验证准确率: {study.best_value:.4f}")
    print("最佳超参数组合:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study




def train_baseline():
    print("🏗️  创建CNN模型...")
    model = CNN(num_classes=8)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    model = model.to(device)

    print("\n📋 模型结构:")
    print(model)

    criterion = nn.CrossEntropyLoss()

    # 优化器 - 使用经典配置
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.001,           # 标准学习率
        weight_decay=1e-4   # L2正则化
    )

    # 学习率调度器(不调度)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    batch_size = 32
    num_workers = 0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,           
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,          
        num_workers=num_workers,
        pin_memory=True
    )
    

    print("准备基线模型")

    train_losses, val_losses, train_accs, val_accs = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler)
    
    
    

def manual_grid_search():
    """手动网格搜索"""
    
    if 'train_dataset' not in globals():
        print("⚠️  请先运行数据加载步骤！")
        return None
    
    print("🔍 开始手动网格搜索...")
    
    # 定义搜索空间
    param_grid = {
        'lr': [0.001, 0.0003, 0.0001],
        'batch_size': [32, 48, 64],              
        'weight_decay': [1e-4, 1e-5, 0],
        'dropout_rate': [0.2, 0.3, 0.5]
    }
    
    best_params = None
    best_acc = 0.0
    results = []
    
    # 遍历所有参数组合
    from itertools import product
    import gc
    
    param_combinations = list(product(*param_grid.values()))
    total_combinations = len(param_combinations)
    
    print(f"总共需要测试 {total_combinations} 种参数组合")
    
    for i, params in enumerate(param_combinations):
        lr, batch_size, weight_decay, dropout_rate = params
        
        print(f"\n{'='*50}")
        print(f"测试组合 {i+1}/{total_combinations}:")
        print(f"  lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}, dropout={dropout_rate}")

        try:
            # 1. 清理之前的计算图和缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # 2. 创建模型
            model = CNN(num_classes=8).to(device)
            
            # 修改dropout率
            for module in model.modules():
                if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                    module.p = dropout_rate
            
            # 3. 创建优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            
            # 4. 创建数据加载器
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                num_workers=0,
                pin_memory=True
            )
            
            # 5. 快速训练
            best_val_acc = 0.0
            for epoch in range(8):  # 只训练8个epoch
                # 训练阶段
                model.train()
                train_loss = 0.0
                train_batches = 0

                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    # 前向传播
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                    # 清理中间结果
                    del data, target, output, loss
                    
                    # 每10个batch清理一次缓存
                    if batch_idx % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # 验证阶段
                model.eval()
                correct = 0
                total = 0
                val_batches = 0
                
                with torch.no_grad():
                    for batch_idx, (data, target) in enumerate(val_loader):

                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        _, predicted = torch.max(output.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                        val_batches += 1
                        
                        # 清理中间结果
                        del data, target, output, predicted
                
                val_acc = correct / total if total > 0 else 0.0
                best_val_acc = max(best_val_acc, val_acc)

                avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
                print(f"    Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Acc={val_acc:.4f}")
            
            # 6. 保存结果
            result = {
                'params': {
                    'lr': lr, 
                    'batch_size': batch_size, 
                    'weight_decay': weight_decay, 
                    'dropout': dropout_rate
                },
                'accuracy': best_val_acc
            }
            results.append(result)
            
            print(f"  ✓ 最佳验证准确率: {best_val_acc:.4f}")
            
            # 更新最佳参数
            if best_val_acc > best_acc:
                best_acc = best_val_acc
                best_params = {
                    'lr': lr, 
                    'batch_size': batch_size, 
                    'weight_decay': weight_decay, 
                    'dropout': dropout_rate
                }
                print(f"  🎯 发现更好的参数组合！")
            
        except Exception as e:
            print(f"  ❌ 参数组合失败: {str(e)}")
            continue
        
        finally:
            # 7. 强制清理资源
            try:
                # 删除模型和优化器
                if 'model' in locals():
                    del model
                if 'optimizer' in locals():
                    del optimizer
                if 'criterion' in locals():
                    del criterion
                if 'train_loader' in locals():
                    del train_loader
                if 'val_loader' in locals():
                    del val_loader
                
                # 强制垃圾回收
                gc.collect()
                
                # 清空CUDA缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # 确保所有CUDA操作完成
                
                # 显示当前显存使用情况
                if torch.cuda.is_available():
                    current_memory = torch.cuda.memory_allocated(device) / 1024**2
                    print(f"  💾 当前显存使用: {current_memory:.2f} MB")
                
            except Exception as cleanup_error:
                print(f"  ⚠️ 清理资源时出错: {str(cleanup_error)}")
    
    print(f"\n✅ 网格搜索完成！")
    print(f"最佳准确率: {best_acc:.4f}")
    print("最佳参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    return best_params, results




if __name__ == "__main__":
    
    # 设置中文字体显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  
    plt.rcParams['axes.unicode_minus'] = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置图表样式
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    DATA_DIR = "fma_metadata"  
    AUDIO_DIR = "fma_small"   
    FEATURE_PATH = 'features'

    print(f"使用设备: {device}")
    
    # 加载特征数据
    features, labels, track_ids = load_all_features()

    if features is not None:
        # 划分数据集
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(features, labels)
        
        # 创建数据集
        train_transform = AudioTransform()  # 训练时使用数据增强
        
        train_dataset = MusicDataset(X_train, y_train, transform=train_transform)
        val_dataset = MusicDataset(X_val, y_val, transform=None)
        test_dataset = MusicDataset(X_test, y_test, transform=None)
        
        print("✅ 数据集创建完成！")
        
        

    # train_baseline()   # 已经完成，已保存
    
    # run_hyperparameter_tuning(n_trials=20) # 需要优化空间分配，显存不足

    manual_grid_search()
    