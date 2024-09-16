import torch.nn as nn
import torch.nn.functional as F

class Conv1ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, l2_lambda=0.01, dropout_rate=0.5):
        super(Conv1ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding='same')
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(dropout_rate)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

        self.l2_lambda = l2_lambda

        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual  # Ensure no in-place operation
        out = self.relu(out)

        return out

class Conv1DModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_blocks=1, l2_lambda=0.01, dropout_rate=0.5,
                 classification=True):
        super(Conv1DModel, self).__init__()
        self.blocks = nn.Sequential(
            Conv1ResidualBlock(in_channels, out_channels, kernel_size, l2_lambda=l2_lambda, dropout_rate=dropout_rate),
            *[Conv1ResidualBlock(out_channels, out_channels, kernel_size, l2_lambda=l2_lambda,
                                 dropout_rate=dropout_rate) for _ in range(num_blocks - 1)]
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling for 1D
        self.fc = nn.Linear(out_channels, 2 if classification else 1)
        self.classification = classification

    def forward(self, x):
        out = self.blocks(x)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)  # Flatten the tensor
        out = self.fc(out)
        if self.classification:
            out = F.log_softmax(out, dim=1)
        return out