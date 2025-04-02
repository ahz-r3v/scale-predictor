import torch
import torch.nn as nn
# import math

class InverseValueMSELoss(nn.Module):
    """
    L = mean( (1/(y + c)) * (y - y_pred)^2 )
    """
    def __init__(self, c=0.01):
        super().__init__()
        self.c = c  # smooth constant

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        weight = 1.0 / (y_true + self.c)
        loss = weight * (y_true - y_pred).pow(2)

        return loss.mean()
    
class InverseLogValueMSELoss(nn.Module):
    """
    L = mean( c/(log(y + 2))) * (y - y_pred)^2 )
    """
    def __init__(self, a=100.0, c=3.0):
        super().__init__()
        self.a = a
        self.c = c  # smooth constant

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        weight = self.a / torch.log(y_true + self.c)
        loss = weight * (y_true - y_pred).pow(2)

        return loss.mean()
    
class LogMSELoss(nn.Module):
    """
    L = MSE + mean(log((|y_pred| + 1) / (y_true + 1)))^2)
    """
    def __init__(self, a=10.0, n=2.0):
        """
        :param base_weight: 整体 loss 缩放系数
        :param neg_penalty: y_pred < 0 时的惩罚放大倍数
        :param eps: 防止除以0的数值下限
        """
        super().__init__()
        self.a = a

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse = (y_pred - y_true) ** 2
        loss_elementwise = (torch.log((torch.abs(y_pred) + 1) / (y_true + 1))) ** 2
        loss = self.a * (10 * loss_elementwise + mse)
        return loss.mean()

        # enlarge punishment for negative pred
        # negative_mask = (y_pred < 0).float()
        # weights = 1.0 + (self.penalty - 1.0) * negative_mask  # positive is 1.0, negative is penalty

        # weighted_loss = loss_elementwise * weights
        # return self.a * weighted_loss.mean()

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
    
class LogMSEPenalizeLoss(nn.Module):
    """
    L = MSE + mean(log((|y_pred| + 1) / (y_true + 1)))^2) + under_penalization + negetive_penalization
    """
    def __init__(self, a=10.0, np=2.0, up=2.0):
        super().__init__()
        self.a = a
        self.negative_factor = np
        self.under_factor = up

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse = torch.abs(y_pred - y_true).pwr(2)
        loss_elementwise = (torch.log((torch.abs(y_pred) + 1) / (y_true + 1))).pow(2)
        neg_mask = (y_pred < 0.0).float()
        under_mask = (y_pred < y_true).float()
        loss = 100 * (self.a  * loss_elementwise + mse)
        weighted_loss = loss * (1.0 + (self.negative_factor - 1.0) * neg_mask) * (1.0 + (self.under_factor - 1.0) * under_mask)
        return weighted_loss.mean()
    
class RelativeDiffLoss(nn.Module):
    """
    L = mean(|y - y_pred| / |y + c|)
    """
    def __init__(self, a=100, c=0.1):
        super().__init__()
        self.a = a
        self.c = c

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred may < 0, assume y_true >= 0 
        loss = torch.abs(y_pred - y_true) / (y_true + self.c)
        return self.a*loss.mean()
    
class GaussianWeightedMSELoss(nn.Module):
    """
    L = mean( a * exp( -((y - mu)^2) / (2 * sigma^2) ) * (y - y_pred)^2 )
    """
    def __init__(self, a=100.0, mu=0.0, sigma=50.0):
        super().__init__()
        self.a = a
        self.mu = mu
        self.sigma = sigma

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        weight = self.a * torch.exp(-((y_true - self.mu) ** 2) / (2 * self.sigma ** 2))
        loss = weight * (y_true - y_pred).pow(2)
        return loss.mean()

