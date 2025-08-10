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

def load_all_features(path):
    feature_files = glob.glob(os.path.join(path, 'features_*.npz'))
    if not feature_files:
        print("Features not exist")
        return None, None, None

    all_features = []
    all_labels = []
    all_track_ids = []
    
    for file in feature_files:
        print(f"loading: {file}")
        with np.load(file) as data:  
            # Convert to float32 to save memory
            all_features.append(data['features'].astype(np.float32))
            all_labels.append(data['labels'])
            all_track_ids.append(data['track_ids'])

    features = np.concatenate(all_features, axis=0)
    del all_features 
    labels = np.concatenate(all_labels, axis=0)
    track_ids = np.concatenate(all_track_ids, axis=0)

    gc.collect()
    
    print(f"Features file loaded successfully")
    
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
    X_temp, X_test, y_temp, y_test = train_test_split(
        features, labels, test_size=test_size, 
        random_state=random_state, stratify=labels
    )

    val_size_adjusted = val_size / (1 - test_size) 
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_temp
    )
    
    print("Datasets created")
    print(f"  training set: {np.bincount(y_train)}")
    print(f"validation set: {np.bincount(y_val)}")
    print(f"      test set: {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


class AudioTransform:
    def __init__(self, noise_factor=0.005, time_shift_factor=0.1):
        self.noise_factor = noise_factor
        self.time_shift_factor = time_shift_factor
    
    def __call__(self, x):
        # add noise
        if random.random() > 0.5:
            noise = torch.randn_like(x) * self.noise_factor
            x = x + noise
        
        # add offset
        if random.random() > 0.5:
            shift = int(x.shape[-1] * self.time_shift_factor * (random.random() - 0.5))
            if shift != 0:
                if shift > 0:
                    x = torch.cat([x[..., shift:], torch.zeros_like(x[..., :shift])], dim=-1)
                else:
                    x = torch.cat([torch.zeros_like(x[..., :abs(shift)]), x[..., :shift]], dim=-1)
        
        return x


class EfficientCNN(nn.Module):
    def __init__(self, num_classes=8, dropout_rate=0.3):
        super(EfficientCNN, self).__init__()
        
        # using depthwise separable convolutions to reduce parameters
        def depthwise_separable_conv(in_channels, out_channels, stride=1):
            return nn.Sequential(
                # depthwise convolution
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, 
                         padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                
                # pointwise convolution
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            depthwise_separable_conv(32, 64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5),
            
            depthwise_separable_conv(64, 128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def quick_train_optimized(model, train_loader, val_loader, criterion, optimizer, epochs=20, device='cuda'):
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}", end=': ')

        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # release memory
            del data, target, output, loss

            if batch_idx % 20 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Loss: {avg_loss:.4f}", end=', ')
        
        # validation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # release memory
                del data, target, output, predicted
                
                if batch_idx % 15 == 0:
                    torch.cuda.empty_cache()
        
        val_acc = correct / total if total > 0 else 0.0
        print(f"Val Acc: {val_acc:.4f}")
        best_val_acc = max(best_val_acc, val_acc)

        torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    return best_val_acc


class CNN_GRU_Hybrid(nn.Module):
    def __init__(self, num_classes=8, dropout_rate=0.3, gru_hidden_size=32, gru_num_layers=2):
        super(CNN_GRU_Hybrid, self).__init__()
        
        self.gru_hidden_size = gru_hidden_size
        self.gru_num_layers = gru_num_layers
        
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
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            depthwise_separable_conv(32, 64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5),
            
            depthwise_separable_conv(64, 128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # keep time dimension, pool only frequency
            nn.AdaptiveAvgPool2d((None, 1))
        )
        
        # simplified GRU
        self.gru = nn.GRU(
            input_size=128,  
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=0,  
            bidirectional=False  
        )
        
        self.residual_proj = nn.Linear(128, self.gru_hidden_size)
        
        # simplified attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(gru_hidden_size, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softmax(dim=1)
        )
        
        # simplified classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.gru_hidden_size, num_classes)
        )

    
    def forward(self, x):
        batch_size = x.size(0)

        cnn_features = self.cnn_features(x) 

        batch_size, channels, h, w = cnn_features.shape
        cnn_features = cnn_features.view(batch_size, channels, h*w).transpose(1, 2)
        
        gru_output, _ = self.gru(cnn_features)  

        residual = self.residual_proj(cnn_features)  
        combined = gru_output + residual

        attention_weights = self.attention(combined) 

        attended_features = torch.sum(combined * attention_weights, dim=1) 

        output = self.classifier(attended_features)
        
        return output
    
    
class CNN_LSTM_Hybrid(nn.Module):
    def __init__(self, num_classes=8, dropout_rate=0.3, lstm_hidden_size=32, lstm_num_layers=2):
        super(CNN_LSTM_Hybrid, self).__init__()
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers

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
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            depthwise_separable_conv(32, 64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5),
            
            depthwise_separable_conv(64, 128),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate),
            
            # keep time dimension, pool only frequency
            nn.AdaptiveAvgPool2d((None, 1))
        )
        
        # simplified LSTM
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )
        
        self.residual_proj = nn.Linear(128, self.lstm_hidden_size)
        
        # simplified attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Softmax(dim=1)
        )
        
        # simplified classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.lstm_hidden_size, num_classes)
        )


    def forward(self, x):
        batch_size = x.size(0)

        cnn_features = self.cnn_features(x)

        batch_size, channels, h, w = cnn_features.shape
        cnn_features = cnn_features.view(batch_size, channels, h*w).transpose(1, 2)
        
        lstm_output, _ = self.lstm(cnn_features)

        residual = self.residual_proj(cnn_features)
        combined = lstm_output + residual

        attention_weights = self.attention(combined)

        attended_features = torch.sum(combined * attention_weights, dim=1)  # [batch_size, hidden_size]

        output = self.classifier(attended_features)
        
        return output


if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DATA_DIR = "fma_metadata"  
    AUDIO_DIR = "fma_small"   
    FEATURE_PATH = 'features'

    features, labels, track_ids = load_all_features(FEATURE_PATH)

    if features is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(features, labels)

        train_transform = AudioTransform()
        
        train_dataset = MusicDataset(X_train, y_train, transform=train_transform)
        val_dataset = MusicDataset(X_val, y_val, transform=None)
        test_dataset = MusicDataset(X_test, y_test, transform=None)
        
        print("Dataset created")

    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    
    model = EfficientCNN(num_classes=8).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Efficient CNN\nNumber of parameters: {params:,}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    start_time = time.time()
    best_val_acc = quick_train_optimized(model, train_loader, val_loader, criterion, optimizer)
    end_time = time.time()
    print('Time cost: ', end_time-start_time)
    print('Best Val Acc: ', best_val_acc)
    
    hybrid_model = CNN_LSTM_Hybrid(num_classes=8).to(device)
    hybrid_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    print(f"CNN-LSTM Hybrid\nNumber of parameters: {hybrid_params:,}")
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=0.001, weight_decay=1e-4)
    start_time = time.time()
    best_val_acc = quick_train_optimized(hybrid_model, train_loader, val_loader, criterion, optimizer)
    end_time = time.time()
    print('Time cost: ', end_time-start_time)
    print('Best Val Acc: ', best_val_acc)
    
        
    hybrid_model = CNN_GRU_Hybrid(num_classes=8).to(device)
    hybrid_params = sum(p.numel() for p in hybrid_model.parameters() if p.requires_grad)
    print(f"CNN-GRU Hybrid\nNumber of parameters: {hybrid_params:,}")
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=0.001, weight_decay=1e-4)
    start_time = time.time()
    best_val_acc = quick_train_optimized(hybrid_model, train_loader, val_loader, criterion, optimizer)
    end_time = time.time()
    print('Time cost: ', end_time-start_time)
    print('Best Val Acc: ', best_val_acc)
    
    


