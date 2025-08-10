import pandas as pd
import numpy as np
import torch
import os
import gc
import glob
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import random
import json
from datetime import datetime
import itertools

def load_all_features():
    print("loading feature files...")

    feature_files = glob.glob(os.path.join(FEATURE_PATH, 'features_*.npz'))
    if not feature_files:
        print("fail to load")
        return None, None, None
    all_features = []
    all_labels = []
    all_track_ids = []
    
    for file in feature_files:
        with np.load(file) as data:
            # convert to float32 to save space
            all_features.append(data['features'].astype(np.float32))
            all_labels.append(data['labels'])
            all_track_ids.append(data['track_ids'])

    features = np.concatenate(all_features, axis=0)
    del all_features
    labels = np.concatenate(all_labels, axis=0)
    track_ids = np.concatenate(all_track_ids, axis=0)

    gc.collect()
    
    print(f"success")
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

        feature = torch.FloatTensor(feature).unsqueeze(0)
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label

def create_data_splits(features, labels, test_size=0.2, val_size=0.1, random_state=42):

    print("creating datasets")

    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=test_size, 
        random_state=random_state, stratify=labels
    )

    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )

    print(f"   Train: {len(X_train)} samples, ({len(X_train)/len(features)*100:.1f}%)")
    print(f"   Val : {len(X_val)} samples, ({len(X_val)/len(features)*100:.1f}%)")
    print(f"   Test: {len(X_test)} samples, ({len(X_test)/len(features)*100:.1f}%)")

    # print("Label distribution: ")
    # print(f"   Train: {np.bincount(y_train)}")
    # print(f"   Val : {np.bincount(y_val)}")
    # print(f"   Test: {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class AudioTransform:
    def __init__(self, noise_factor=0.005, time_shift_factor=0.1):
        self.noise_factor = noise_factor
        self.time_shift_factor = time_shift_factor
    
    def __call__(self, x):
        # noise
        if random.random() > 0.5:
            noise = torch.randn_like(x) * self.noise_factor
            x = x + noise
        
        # shift
        if random.random() > 0.5:
            shift = int(x.shape[-1] * self.time_shift_factor * (random.random() - 0.5))
            if shift != 0:
                if shift > 0:
                    x = torch.cat([x[..., shift:], torch.zeros_like(x[..., :shift])], dim=-1)
                else:
                    x = torch.cat([torch.zeros_like(x[..., :abs(shift)]), x[..., :shift]], dim=-1)
        
        return x

class CNN_GRU_Hybrid(nn.Module):
    def __init__(self, num_classes=8, cnn_dropout=0.3, classifier_dropout=0.3, 
                 gru_hidden_size=32, gru_num_layers=2):
        super(CNN_GRU_Hybrid, self).__init__()
        
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        
        # DSC
        def depthwise_separable_conv(in_channels, out_channels, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, 
                         padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.cnn_features = nn.Sequential(
            # standard conv
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # dsc
            depthwise_separable_conv(32, 64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(cnn_dropout * 0.5),  
            
            depthwise_separable_conv(64, 128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(cnn_dropout),  
            
            # adaptive pooling
            nn.AdaptiveAvgPool2d((None, 1))
        )
        
        # simple GRU
        self.gru = nn.GRU(
            input_size=128,  
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=0,  
            bidirectional=False  
        )
        
        self.residual_proj = nn.Linear(128, self.gru_hidden_size)
        
        # simple Attention
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden_size, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softmax(dim=1)
        )
        
        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout(classifier_dropout),  
            nn.Linear(self.gru_hidden_size, num_classes)
        )
        
        # Gated Fusion Mechanism
        # self.gate = nn.Sequential(
        #     nn.Linear(128 + self.gru_hidden_size, 1),
        #     nn.Sigmoid()
        # )
    
    def forward(self, x):
        batch_size = x.size(0)

        cnn_features = self.cnn_features(x)  
        
        # reshape
        batch_size, channels, h, w = cnn_features.shape
        cnn_features = cnn_features.view(batch_size, channels, h*w).transpose(1, 2)
        
        gru_output, _ = self.gru(cnn_features)  
        
        # residual
        residual = self.residual_proj(cnn_features)  
        combined = gru_output + residual
        
        # attention
        attention_weights = self.attention(combined) 

        attended_features = torch.sum(combined * attention_weights, dim=1)  

        output = self.classifier(attended_features)
        
        return output

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, 
                      epochs=20, device='cuda', patience=8, min_delta=1e-4):
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stopped = False

    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()

            del data, target, output, predicted, loss
            
            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_losses.append(avg_loss)
        
        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                del data, target, output, predicted
                
                if batch_idx % 15 == 0:
                    torch.cuda.empty_cache()
        
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_accuracies.append(val_acc)
        
        # early stopping
        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                early_stopped = True
                break

        torch.cuda.empty_cache()
        gc.collect()
    
    return {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'early_stopped': early_stopped,
        'total_epochs': epoch + 1,
        'final_train_acc': train_acc,
        'train_val_gap': abs(train_acc - best_val_acc),
        'train_losses': train_losses,
        'val_accuracies': val_accuracies
    }

def run_grid_search(X_train, X_val, y_train, y_val, device='cuda'):

    param_grid = {
        'learning_rate': [5e-5, 1e-4, 3e-4, 1e-3],
        'cnn_dropout': [0.3],
        'classifier_dropout': [0.3],
        'weight_decay': [1e-5],
    }
    
    print(f"Search Space: {len(param_grid['learning_rate']) * len(param_grid['cnn_dropout']) * len(param_grid['classifier_dropout']) * len(param_grid['weight_decay'])} combinations")
    
    # generate all combinations
    param_combinations = list(itertools.product(
        param_grid['learning_rate'],
        param_grid['cnn_dropout'], 
        param_grid['classifier_dropout'],
        param_grid['weight_decay'],
    ))
    
    results = []
    best_val_acc = 0.0
    best_config = None

    train_transform = AudioTransform()
    train_dataset = MusicDataset(X_train, y_train, transform=train_transform)
    val_dataset = MusicDataset(X_val, y_val, transform=None)
    batch_sz = 16
    
    for i, (lr, cnn_drop, cls_drop, wd) in enumerate(param_combinations):
        print(f"\n{'='*60}")
        print(f"{i+1}/{len(param_combinations)}")
        print(f"Config: lr={lr}, cnn_dropout={cnn_drop}, cls_dropout={cls_drop}, weight_decay={wd}")
        print(f"{'='*60}")

        train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)

        model = CNN_GRU_Hybrid(
            num_classes=8,
            cnn_dropout=cnn_drop,
            classifier_dropout=cls_drop,
            gru_hidden_size=32,
            gru_num_layers=2
        ).to(device)

        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        try:
            train_results = train_and_evaluate(
                model, train_loader, val_loader, criterion, optimizer,
                epochs= 30, device=device, patience=5
            )
            
            training_time = time.time() - start_time

            result = {
                'experiment_id': i + 1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'hyperparameters': {
                    'learning_rate': lr,
                    'cnn_dropout': cnn_drop,
                    'classifier_dropout': cls_drop,
                    'weight_decay': wd,
                    'batch_size': batch_sz,
                    'gru_hidden_size': 32,
                    'gru_num_layers': 2
                },
                'model_info': {
                    'parameter_count': param_count,
                    'model_size_mb': param_count * 4 / (1024 * 1024)
                },
                'training_results': train_results,
                'training_time_seconds': training_time,
                'training_time_minutes': training_time / 60
            }
            
            results.append(result)

            # update
            if train_results['best_val_acc'] > best_val_acc:
                best_val_acc = train_results['best_val_acc']
                best_config = result

            print(f"Best Val Acc: {train_results['best_val_acc']:.4f}")
            print(f"Time: {training_time/60:.1f} mins")
            print(f"Train Val Gap: {train_results['train_val_gap']:.4f}")
            print(f"Epochs: {train_results['total_epochs']}")
            
        except Exception as e:
            print(f"Training Failed: {str(e)}")
            result = {
                'experiment_id': i + 1,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'hyperparameters': {
                    'learning_rate': lr,
                    'cnn_dropout': cnn_drop,
                    'classifier_dropout': cls_drop,
                    'weight_decay': wd,
                    'batch_size': batch_sz
                },
                'error': str(e),
                'status': 'failed'
            }
            results.append(result)

        del model, optimizer, criterion, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()

    if best_config:
        print(f"Best Config : {best_config['hyperparameters']}")
        print(f"Best Val Acc: {best_val_acc:.4f}")
    
    return results, best_config

def save_results(results, filename='grid_search_results.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved to: {filename}")

def create_results_summary(results):
    summary_data = []
    
    for result in results:
        if 'error' not in result:
            summary_data.append({
                'experiment_id': result['experiment_id'],
                'learning_rate': result['hyperparameters']['learning_rate'],
                'cnn_dropout': result['hyperparameters']['cnn_dropout'],
                'classifier_dropout': result['hyperparameters']['classifier_dropout'],
                'weight_decay': result['hyperparameters']['weight_decay'],
                'batch_size': result['hyperparameters']['batch_size'],
                'best_val_acc': result['training_results']['best_val_acc'],
                'train_val_gap': result['training_results']['train_val_gap'],
                'early_stopped': result['training_results']['early_stopped'],
                'total_epochs': result['training_results']['total_epochs'],
                'training_time_min': result['training_time_minutes'],
                'parameter_count': result['model_info']['parameter_count']
            })
    
    df = pd.DataFrame(summary_data)
    return df


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATA_DIR = "fma_metadata"  
    AUDIO_DIR = "fma_small"   
    FEATURE_PATH = 'features'

    print(f"Device detected: {device}")

    features, labels, track_ids = load_all_features(FEATURE_PATH)

    if features is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(features, labels)

        results, best_config = run_grid_search(X_train, X_val, y_train, y_val, device)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_filename = f'grid_search_results_{timestamp}.json'
        save_results(results, results_filename)

        summary_df = create_results_summary(results)
        summary_filename = f'grid_search_summary_{timestamp}.csv'
        summary_df.to_csv(summary_filename, index=False)

        # show summary
        print(f"Summary")
        print(f"Tested combinations: {len(summary_df)}")
        print(f"Best Val Acc: {summary_df['best_val_acc'].max():.4f}")
        print(f"Avg Val Acc : {summary_df['best_val_acc'].mean():.4f}")
        
        # show Top 3
        print("Top 3:")
        top3 = summary_df.nlargest(3, 'best_val_acc')
        for i, (_, row) in enumerate(top3.iterrows()):
            print(f"   #{i+1}: Val Acc={row['best_val_acc']:.4f}, "
                  f"lr={row['learning_rate']}, "
                  f"cnn_drop={row['cnn_dropout']}, "
                  f"cls_drop={row['classifier_dropout']}, "
                  f"wd={row['weight_decay']}, "
                  f"batch_sz={row['batch_size']}")

        if best_config:
            best_config_filename = f'best_config_{timestamp}.json'
            with open(best_config_filename, 'w', encoding='utf-8') as f:
                json.dump(best_config, f, indent=2, ensure_ascii=False)
            print(f"Best Config saved to: {best_config_filename}")