from cov1d import *
class Conv1DLSTMModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_blocks=1, lstm_hidden_size=64, l2_lambda=0.01, dropout_rate=0.5,
                 classification=True):
        super(Conv1DLSTMModel, self).__init__()
        self.blocks = nn.Sequential(
            Conv1ResidualBlock(in_channels, out_channels, kernel_size, l2_lambda=l2_lambda, dropout_rate=dropout_rate),
            *[Conv1ResidualBlock(out_channels, out_channels, kernel_size, l2_lambda=l2_lambda,
                                 dropout_rate=dropout_rate) for _ in range(num_blocks - 1)]
        )
        self.lstm = nn.LSTM(out_channels, lstm_hidden_size, batch_first=True)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling for 1D
        self.fc = nn.Linear(lstm_hidden_size, 2 if classification else 1)
        self.classification = classification

    def forward(self, x):
        out = self.blocks(x)
        out = out.permute(0, 2, 1)  # Change shape to (batch_size, sequence_length, input_size) position
        out, _ = self.lstm(out)
        out = self.global_avg_pool(
            out.permute(0, 2, 1))  # Change shape back to (batch_size, input_size, sequence_length) position
        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = self.fc(out)
        if self.classification:
            out = F.log_softmax(out, dim=1)
        return out