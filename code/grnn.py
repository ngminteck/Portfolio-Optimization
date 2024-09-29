import torch
import torch.nn as nn
import torch.nn.functional as F

class GRNN(nn.Module):
    def __init__(self, input_size, output_size, sigma, classification=False):
        super(GRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.sigma = sigma
        self.classification = classification

        # Adding a linear layer to introduce trainable parameters
        self.linear = nn.Linear(input_size, output_size)

        # Apply He initialization to the linear layer
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):

        dist = torch.cdist(x, x)
        weights = torch.exp(-dist ** 2 / (2 * self.sigma ** 2))
        # Calculate the sum along the specified dimension
        sums = weights.sum(dim=1, keepdim=True)

        # Check if sums are zero
        zero_sums_mask = (sums == 0).squeeze()
        if torch.any(zero_sums_mask):
            weights[zero_sums_mask, :] = 0
        else:
            weights = weights / sums

        output = torch.mm(weights, x)

        if self.classification:
            output = F.softmax(output, dim=1)
        else:
            output = self.linear(output)

        return output

