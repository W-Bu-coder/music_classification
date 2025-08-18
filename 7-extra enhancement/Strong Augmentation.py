import pandas as pd
import numpy as np
import torch
import os
import gc
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score, classification_report
import json
from datetime import datetime
warnings.filterwarnings('ignore')

def load_all_features(path):
    print("loading feature files...")

    feature_files = glob.glob(os.path.join(path, 'features_*.npz'))
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
    
    
class EnhancedAudioTransform:
    def __init__(self, noise_factor=0.005, time_shift_factor=0.1, 
                 strong_noise_factor=0.01, strong_time_shift_factor=0.2):
        self.noise_factor = noise_factor
        self.time_shift_factor = time_shift_factor
        self.strong_noise_factor = strong_noise_factor
        self.strong_time_shift_factor = strong_time_shift_factor
        
    
    def apply_augmentation(self, x, strong=False):
        noise_factor = self.strong_noise_factor if strong else self.noise_factor
        shift_factor = self.strong_time_shift_factor if strong else self.time_shift_factor
        
        # Apply noise
        if random.random() > 0.5:
            noise = torch.randn_like(x) * noise_factor
            x = x + noise
        
        # Apply time shift
        if random.random() > 0.5:
            shift = int(x.shape[-1] * shift_factor * (random.random() - 0.5))
            if shift != 0:
                if shift > 0:
                    x = torch.cat([x[..., shift:], torch.zeros_like(x[..., :shift])], dim=-1)
                else:
                    x = torch.cat([torch.zeros_like(x[..., :abs(shift)]), x[..., :shift]], dim=-1)
        
        return x
    
    def __call__(self, x):
        return self.apply_augmentation(x, strong=False)

class EnhancedMusicDataset(Dataset):
    def __init__(self, features, labels, transform=None, experimental_class_id=1, 
                 experimental_oversample_factor=2):
        self.original_features = features
        self.original_labels = labels
        self.experimental_class_id = experimental_class_id
        self.experimental_oversample_factor = experimental_oversample_factor
        self.transform = transform
        
        # Create enhanced audio transform for experimental samples
        self.enhanced_transform = EnhancedAudioTransform()
        
        # Build the enhanced dataset indices
        self._build_enhanced_indices()
    
    def _build_enhanced_indices(self):
        self.indices = []
        self.sample_types = []
        
        for idx in range(len(self.original_labels)):
            # Add original sample
            self.indices.append(idx)
            self.sample_types.append('original')
            
            # Experimental
            if self.original_labels[idx] == self.experimental_class_id:
                for _ in range(self.experimental_oversample_factor):
                    self.indices.append(idx)
                    aug_type = 'augmented_strong'
                    self.sample_types.append(aug_type)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        sample_type = self.sample_types[idx]
        
        feature = self.original_features[original_idx]
        label = self.original_labels[original_idx]
        
        feature = torch.FloatTensor(feature).unsqueeze(0)  # Add channel dimension
        label = torch.LongTensor([label])[0]
        
        # Apply appropriate augmentation based on sample type
        if sample_type == 'original':
            if self.transform:
                feature = self.transform(feature)
        else:
            feature = self.enhanced_transform.apply_augmentation(feature, strong=True)

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

# data augmentation
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

def save_model_checkpoint(model, optimizer, epoch, val_acc, loss, config, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'loss': loss,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)
    print(f"model saved to: {filepath}")

def load_model_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"model loaded from: {filepath}, timestamp: {checkpoint['timestamp']}")

    return checkpoint

def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler,
                      epochs=60, device='cuda', patience=10, min_delta=1e-4,
                      save_dir='models', config=None):
    # save model here
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"cnn_gru_model_{timestamp}"
    
    # use val loss as early stopping metric
    best_val_loss = float('inf')  
    best_val_acc = 0.0  
    best_epoch = 0
    patience_counter = 0
    early_stopped = False
    
    train_losses = []
    val_losses = []  
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
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
        train_acc = train_correct / train_total if train_total > 0 else 0.0
        train_losses.append(avg_train_loss)
        
        # validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_total = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)

                val_loss = criterion(output, target)
                val_loss_total += val_loss.item()
                val_batches += 1

                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                del data, target, output, predicted, val_loss
                
                if batch_idx % 15 == 0:
                    torch.cuda.empty_cache()
        
        avg_val_loss = val_loss_total / val_batches if val_batches > 0 else float('inf')
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        
        scheduler.step(val_acc)
        
        val_losses.append(avg_val_loss)  
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_path = os.path.join(save_dir, f"{model_name}_best.pth")
            save_model_checkpoint(model, optimizer, epoch+1, val_acc, avg_val_loss, config, best_model_path)
            print(f"New best Val Loss: {avg_val_loss:.4f} (Val Acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            print(f"patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                early_stopped = True
                print(f"Early Stopped on Epoch {epoch+1}")
                print(f"Best Val Loss: {best_val_loss:.4f}, Epoch: {best_epoch} ")
                break

        # regular save
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"{model_name}_epoch_{epoch+1}.pth")
            save_model_checkpoint(model, optimizer, epoch+1, val_acc, avg_val_loss, config, checkpoint_path)

        torch.cuda.empty_cache()
        gc.collect()

    final_model_path = os.path.join(save_dir, f"{model_name}_final.pth")
    save_model_checkpoint(model, optimizer, epoch+1, val_acc, avg_val_loss, config, final_model_path)
    
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,  
        'val_accuracies': val_accuracies,
        'best_val_loss': best_val_loss,  
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'config': config,
        'model_name': model_name
    }
    
    history_path = os.path.join(save_dir, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"Training history saved: {history_path}")

    # return all parameters that may be needed
    return {
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'early_stopped': early_stopped,
        'total_epochs': epoch + 1,
        'final_train_acc': train_acc,
        'final_val_loss': avg_val_loss,  
        'train_val_gap': abs(train_acc - best_val_acc),
        'train_losses': train_losses,
        'val_losses': val_losses,  
        'val_accuracies': val_accuracies,
        'model_name': model_name,
        'best_model_path': best_model_path,
        'final_model_path': final_model_path
    }
    
    
def evaluate_model(model, test_loader, device='cuda', class_names=None):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_predictions)
    
    # classification report
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(8)]
    
    report = classification_report(all_targets, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    print(f"Test Acc: {accuracy:.4f}")
    print(classification_report(all_targets, all_predictions, target_names=class_names))
    
    return accuracy, report, all_predictions, all_targets

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    DATA_DIR = "fma_metadata"  
    AUDIO_DIR = "fma_small"   
    FEATURE_PATH = 'features'

    print(f"Device detected: {device}")

    features, labels, track_ids = load_all_features(FEATURE_PATH)

    if features is not None:
        X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(features, labels)
        
        # train_transform = AudioTransform()
        # train_dataset = MusicDataset(X_train, y_train, transform=train_transform)
        # enhanced train dataset
        train_dataset = EnhancedMusicDataset(
            features=X_train, 
            labels=y_train, 
            experimental_class_id=1,
            experimental_oversample_factor=1  
        )
        val_dataset = MusicDataset(X_val, y_val, transform=None)
        test_dataset = MusicDataset(X_test, y_test, transform=None)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        config = {
            'num_classes': 8,
            'cnn_dropout': 0.2,
            'classifier_dropout': 0.4,
            'gru_hidden_size': 32,
            'gru_num_layers': 2,
            'learning_rate': 0.001,
            'beta2': 0.999,
            'weight_decay': 0.03,
            'batch_size': 16,
            'epochs': 60
        }

        model = CNN_GRU_Hybrid(
            num_classes=8,
            cnn_dropout=config['cnn_dropout'],
            classifier_dropout=config['classifier_dropout'],
            gru_hidden_size=32, # keep simple
            gru_num_layers=2
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            betas=(0.9, config['beta2']),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=5, 
            min_lr=1e-7, 
        )
        
        criterion = nn.CrossEntropyLoss()

        print("Training start")
        res = train_and_evaluate(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            epochs=60, config=config, save_dir='saved_models'
        )
        
        print("Training stop")
        print(f"Best Val Acc: {res['best_val_acc']:.4f}")
        print(f"Best Epoch: {res['best_epoch']}")
        print(f"Best Model: {res['best_model_path']}")

        print("Evaluate on test set...")
        checkpoint = load_model_checkpoint(res['best_model_path'], model)

        class_names = ['Electronic', 'Experimental', 'Folk', 'Hip-Hop', 
                      'Instrumental', 'International', 'Pop', 'Rock']
        
        test_accuracy, test_report, predictions, targets = evaluate_model(
            model, test_loader, device, class_names
        )

        print(f"Test Acc: {test_accuracy:.4f}")

        final_results = {
            'validation_accuracy': res['best_val_acc'],
            'test_accuracy': test_accuracy,
            'test_report': test_report,
            'config': config,
            'model_name': res['model_name']
        }
        
        results_path = f"saved_models/{res['model_name']}_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)
