import torch
import torch.nn as nn
from torch.utils.data import Subset

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.position_embedding = nn.Embedding(input_dim, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads, dropout=dropout
            ),
            num_layers=num_layers,
        )

    def forward(self, x):
        # Add positional encoding
        x = x + self.position_embedding(torch.arange(x.size(1)).unsqueeze(0).to(x.device))
        # Pass through transformer encoder
        x = self.transformer_encoder(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, output_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=embed_dim, nhead=num_heads, dropout=dropout
            ),
            num_layers=num_layers,
        )
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, x, encoder_output):
        # Pass through transformer decoder
        x = self.transformer_decoder(x, encoder_output)
        # Linear layer for output
        x = self.fc(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, output_dim, dropout=0.1, is_classification=False):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(input_dim, embed_dim, num_heads, num_layers, dropout)
        self.is_classification = is_classification
        if is_classification:
            self.decoder = TransformerDecoder(embed_dim, output_dim, num_heads, num_layers, dropout)

    def forward(self, x):
        print(f'Type of x: {type(x)}')
        if isinstance(x, Subset):
            x = torch.stack([x[i] for i in range(len(x))])
        x = self.encoder(x)
        if self.is_classification:
            x = self.decoder(x, x)
            x = x[:, -1, :]  # Take the last token's output for classification
        return x