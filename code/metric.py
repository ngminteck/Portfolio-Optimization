import numpy as np
import torch


def profit_loss_np(y_true, y_pred):
    # Ensure the inputs are numpy arrays
    y_true = np.array(y_true, dtype=np.float32)
    y_pred = np.array(y_pred, dtype=np.float32)

    # Get the sign of y_true and y_pred
    sign_y_true = np.sign(y_true)
    sign_y_pred = np.sign(y_pred)


    # Calculate profit/loss based on sign correctness
    profit_loss = np.where(sign_y_true == sign_y_pred,
                           np.minimum(np.abs(y_true), np.abs(y_pred)),
                           -np.minimum(np.abs(y_true), np.abs(y_pred)))

    # Sum the profit/loss
    total_profit_loss = np.sum(profit_loss)

    return total_profit_loss

def profit_loss_torch(y_true, y_pred):

    # Get the sign of y_true and y_pred
    sign_y_true = torch.sign(y_true)
    sign_y_pred = torch.sign(y_pred)

    # Calculate profit/loss based on sign correctness
    profit_loss = torch.where(sign_y_true == sign_y_pred,
                              torch.minimum(torch.abs(y_true), torch.abs(y_pred)),
                              -torch.minimum(torch.abs(y_true), torch.abs(y_pred)))

    # Sum the profit/loss
    total_profit_loss = torch.sum(profit_loss)

    return total_profit_loss

