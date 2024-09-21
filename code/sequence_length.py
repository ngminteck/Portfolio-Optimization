import torch

def create_lstm_train_sequences(X, y, sequence_length):
    sequences_X, sequences_y = [], []
    for i in range(len(X) - sequence_length + 1):
        sequences_X.append(X[i:i + sequence_length])
        sequences_y.append(y[i + sequence_length - 1])
    return torch.stack(sequences_X), torch.stack(sequences_y)

def create_lstm_predict_sequences(X, sequence_length):
    sequences = []
    for i in range(len(X) - sequence_length + 1):
        seq = X[i:i + sequence_length]
        sequences.append(seq)
    return torch.stack(sequences)
