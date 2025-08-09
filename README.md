# music_classification
This is the music genre classification project on fma_small dataset using CNN-GRU Hybrid Model.

Basically, we processed following steps:

0 - download and processed data, then split it into 3 sets.

1 - trained CNN, Efficient CNN (using DSC), CNN-LSTM, CNN-GRU and finally selected CNN-GRU Hybrid model.

2 - determined the batch size (16)

3 - determined the learning rate (0.001)

4 - determined the optimizer (AdamW) and scheduler (ReduceLROnPlateau)

5 - hyperparameter tuning

6 - trained final model with optimized parameters and evaluate it on test set

Final Performance:
Best Val Acc: 61.06%​, Test Acc: 61.71%​.