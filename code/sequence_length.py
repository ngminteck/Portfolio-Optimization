import numpy as np

def create_lstm_train_sequences(X, y, sequence_length):
    sequences_X, sequences_y = [], []
    for i in range(len(X) - sequence_length + 1):
        sequences_X.append(X[i:i + sequence_length])
        sequences_y.append(y[i + sequence_length - 1])
    return np.array(sequences_X), np.array(sequences_y)

def create_lstm_predict_sequences(X, sequence_length):
    sequences = []
    for i in range(len(X) - sequence_length + 1):
        seq = X[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)