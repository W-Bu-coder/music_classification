import pandas as pd
import numpy as np
import torch
import os
import librosa
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import time


def load_metadata():
    tracks = pd.read_csv(os.path.join(DATA_DIR, "tracks.csv"), header=[0, 1], index_col=0)
    genres = pd.read_csv(os.path.join(DATA_DIR, "genres.csv"), index_col=0)
    features = pd.read_csv(os.path.join(DATA_DIR, "features.csv"), header=[0, 1, 2], index_col=0)
    return tracks, genres, features

def process_tracks_data(tracks):
    track_data = pd.DataFrame({
        'track_id': tracks.index,
        'title': tracks[('track', 'title')],
        'duration': tracks[('track', 'duration')],
        'genre_top': tracks[('track', 'genre_top')],
        'genres': tracks[('track', 'genres')],
        'listens': tracks[('track', 'listens')],
        'bit_rate': tracks[('track', 'bit_rate')],
        'interest': tracks[('track', 'interest')]
    })
    
    print(f"csv: {len(track_data)} songs")
    
    # data cleaning
    track_data = track_data.dropna(subset=['genre_top'])
    track_data = track_data[track_data['genre_top'] != 0]
    
    # save key columns
    track_data['duration'] = pd.to_numeric(track_data['duration'], errors='coerce')
    track_data['listens'] = pd.to_numeric(track_data['listens'], errors='coerce')
    genre_name_to_id = {}
    for gid, row in genres.iterrows():
        genre_name_to_id[row['title']] = gid

    def map_genre_name_to_id(genre_name):
        if pd.isna(genre_name):
            return np.nan
        if isinstance(genre_name, str):
            return genre_name_to_id.get(genre_name, np.nan)
        else:
            return genre_name
    
    track_data['genre_top'] = track_data['genre_top'].apply(map_genre_name_to_id)

    track_data = track_data.dropna(subset=['genre_top'])
    
    print(f"after cleaning: {len(track_data)} songs")
    return track_data


def scan_audio_files(audio_dir):
    print(f"Scanning fma directory: {audio_dir}")
    
    if not os.path.exists(audio_dir):
        print("Error!")
        return {}
    
    audio_files = {}
    total_size = 0

    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.mp3'):
                file_path = os.path.join(root, file)

                try:
                    track_id = int(file.split('.')[0])
                    file_size = os.path.getsize(file_path)
                    
                    audio_files[track_id] = {
                        'path': file_path,
                        'size_bytes': file_size,
                        'size_kb': file_size / 1024,
                        'size_mb': file_size / (1024 * 1024)
                    }
                    total_size += file_size
                    
                except ValueError:
                    continue
    
    print(f"{len(audio_files)} audio file detected")
    
    return audio_files



def filter_valid_audio_files(audio_files_info, track_data, min_size_kb=10, max_size_kb=20000):
    valid_files = {}
    filtered_tracks = []
    
    for track_id, file_info in audio_files_info.items():
        if not (min_size_kb <= file_info['size_kb'] <= max_size_kb):
            continue

        if track_id not in track_data['track_id'].values:
            continue
            
        # collect data from csv
        track_row = track_data[track_data['track_id'] == track_id].iloc[0]

        combined_info = {
            'track_id': track_id,
            'title': track_row['title'],
            'genre_top': track_row['genre_top'],
            'duration': track_row['duration'],
            'listens': track_row['listens'],
            'audio_path': file_info['path'],
            'file_size_kb': file_info['size_kb'],
            'file_size_mb': file_info['size_mb']
        }
        
        valid_files[track_id] = file_info
        filtered_tracks.append(combined_info)

    valid_tracks_df = pd.DataFrame(filtered_tracks)

    print(f"Available songs: {len(valid_tracks_df)}")
    
    return valid_tracks_df, valid_files


def analyze_genre_distribution(valid_tracks_df, genres):
    genre_counts = valid_tracks_df['genre_top'].value_counts()

    genre_names = {}
    for genre_id in genre_counts.index:
        if genre_id in genres.index:
            genre_names[genre_id] = genres.loc[genre_id, 'title']
        else:
            genre_names[genre_id] = f"Unknown_{genre_id}"

    genre_stats_filtered = pd.DataFrame({
        'genre_id': genre_counts.index,
        'genre_name': [genre_names[gid] for gid in genre_counts.index],
        'track_count': genre_counts.values,
        'percentage': (genre_counts.values / len(valid_tracks_df) * 100).round(2)
    })
    
    print(f"Number of genres: {len(genre_stats_filtered)}")
    print(f"Number of songs: {len(valid_tracks_df)}")
    for i, row in genre_stats_filtered.head(10).iterrows():
        print(f"   {i+1:2d}. {row['genre_name']:15s}: {row['track_count']:4d} ({row['percentage']:5.1f}%)")
    
    return genre_stats_filtered


    
def extract_mel_spectrogram(audio_path, sr=22050, n_mels=128, hop_length=512, n_fft=2048, duration=30):
    try:
        # length: 30s
        y, sr = librosa.load(audio_path, sr=sr, duration=duration, offset=0)
        
        # pad 0 if too short
        target_length = sr * duration
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
        
    except Exception as e:
        print("Error occured during feature extraction")
        return np.zeros((n_mels, 1292))  # 30 s = 1292 frames

def batch_extract_features(valid_tracks_df, batch_size=100, save_interval=500):
    print("Batch extraction start")
 
    os.makedirs(FEATURE_PATH, exist_ok=True)
    
    features_list = []
    labels_list = []
    track_ids_list = []

    label_encoder = LabelEncoder()

    all_genres = valid_tracks_df['genre_top'].values
    encoded_labels = label_encoder.fit_transform(all_genres)

    for i, genre_id in enumerate(label_encoder.classes_):
        genre_name = genres.loc[genre_id, 'title'] if genre_id in genres.index else f"Unknown_{genre_id}"
        print(f"{i}: {genre_name} (ID: {genre_id})")

    indices = np.random.permutation(len(valid_tracks_df))
    
    processed_count = 0
    failed_count = 0
    
    for idx in tqdm(indices, desc="feature extraction"):
        row = valid_tracks_df.iloc[idx]

        mel_spec = extract_mel_spectrogram(row['audio_path'])
        
        if mel_spec is not None:
            features_list.append(mel_spec)
            labels_list.append(encoded_labels[idx])
            track_ids_list.append(row['track_id'])
            processed_count += 1
        else:
            failed_count += 1
        
        # free memory
        if len(features_list) >= save_interval:
            features_array = np.array(features_list)
            labels_array = np.array(labels_list)
            track_ids_array = np.array(track_ids_list)

            timestamp = int(time.time())
            np.savez_compressed(os.path.join(FEATURE_PATH, f'features_batch_{timestamp}.npz'), 
                              features=features_array, 
                              labels=labels_array,
                              track_ids=track_ids_array)
            
            features_list = []
            labels_list = []
            track_ids_list = []

    if features_list:
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        track_ids_array = np.array(track_ids_list)
        
        timestamp = int(time.time())
        np.savez_compressed(os.path.join(FEATURE_PATH, f'features_batch_{timestamp}.npz'), 
                          features=features_array, 
                          labels=labels_array,
                          track_ids=track_ids_array)
    
    print(f"Succeeded - {processed_count}; Failed: {failed_count}")
    
    return label_encoder


if __name__ == "__main__":
    DATA_DIR = "fma_metadata"  
    AUDIO_DIR = "fma_small"   
    FEATURE_PATH = 'features'

    audio_files_info = scan_audio_files(AUDIO_DIR)
    
    tracks, genres, features = load_metadata()
    track_data = process_tracks_data(tracks)
    
    valid_tracks_df, valid_audio_files = filter_valid_audio_files(
        audio_files_info, track_data, min_size_kb=10, max_size_kb=20000
    )
    
    if len(valid_tracks_df) > 0:
        final_genre_stats = analyze_genre_distribution(valid_tracks_df, genres)


    # test_data = valid_tracks_df.head(100)  # test on first 100 files
    # label_encoder = batch_extract_features(test_data, batch_size=50, save_interval=50)

    # extract all
    label_encoder = batch_extract_features(valid_tracks_df, batch_size=100, save_interval=500)