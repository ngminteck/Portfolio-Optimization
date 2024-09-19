import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class LSTMResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.5):
        super(LSTMResidualBlock, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)

        if input_size != hidden_size:
            self.residual_fc = nn.Linear(input_size, hidden_size)
        else:
            self.residual_fc = nn.Identity()

    def forward(self, x):
        residual = self.residual_fc(x)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.layer_norm(out + residual)
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks=1, num_layers=1, dropout_rate=0.5, classification=True):
        super(LSTMModel, self).__init__()
        self.blocks = nn.Sequential(
            LSTMResidualBlock(input_size, hidden_size, num_layers=num_layers, dropout_rate=dropout_rate),
            *[LSTMResidualBlock(hidden_size, hidden_size, num_layers=num_layers, dropout_rate=dropout_rate) for _ in
              range(num_blocks - 1)]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling for 1D
        self.fc = nn.Linear(hidden_size, 2 if classification else 1)
        self.classification = classification

    def forward(self, x):
        out = self.blocks(x)
        out = out.mean(dim=1)  # Global average pooling
        out = self.fc(out)
        if self.classification:
            out = F.log_softmax(out, dim=1)
        return out

def create_train_sequences(X, y, sequence_length):
    sequences_X, sequences_y = [], []
    for i in range(len(X) - sequence_length + 1):
        sequences_X.append(X[i:i + sequence_length])
        sequences_y.append(y[i + sequence_length - 1])
    return np.array(sequences_X), np.array(sequences_y)

def create_predict_sequences(X, sequence_length):
    sequences = []
    for i in range(len(X) - sequence_length + 1):
        seq = X[i:i + sequence_length]
        sequences.append(seq)
    return np.array(sequences)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        weights = F.softmax(self.attention(x), dim=1)
        out = torch.sum(weights * x, dim=1)
        return out

'''
self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True,
                    dropout=dropout_rate if num_layers > 1 else 0, bidirectional=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_blocks=1, num_layers=1, dropout_rate=0.5, classification=True):
        super(LSTMModel, self).__init__()
        self.blocks = nn.Sequential(
            LSTMResidualBlock(input_size, hidden_size, num_layers=num_layers, dropout_rate=dropout_rate),
            *[LSTMResidualBlock(hidden_size, hidden_size, num_layers=num_layers, dropout_rate=dropout_rate) for _ in
              range(num_blocks - 1)]
        )
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, 2 if classification else 1)
        self.classification = classification

    def forward(self, x):
        out = self.blocks(x)
        out = self.attention(out)
        out = self.fc(out)
        if self.classification:
            out = F.log_softmax(out, dim=1)
        return out
'''