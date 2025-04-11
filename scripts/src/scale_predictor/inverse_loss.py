import torch
import torch.nn as nn

class MAELogMSEPenalizeLoss(nn.Module):
    """
    L = MAE + mean(log((|y_pred| + 1) / (y_true + 1)))^2) + under_penalization + negetive_penalization
    """
    def __init__(self, a=10.0, np=2.0, up=2.0):
        super().__init__()
        self.a = a
        self.negative_factor = np
        self.under_factor = up

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mae = torch.abs(y_pred - y_true)
        loss_elementwise = (torch.log((torch.abs(y_pred) + 1) / (y_true + 1))).pow(2)
        neg_mask = (y_pred < 0.0).float()
        under_mask = (y_pred < y_true).float()
        loss = 100 * (self.a  * loss_elementwise + mae)
        weighted_loss = loss * (1.0 + (self.negative_factor - 1.0) * neg_mask) * (1.0 + (self.under_factor - 1.0) * under_mask)
        return weighted_loss.mean()