import torch
from torch import nn, Tensor


class RankingLoss(nn.Module):
    """
    ref: https://arxiv.org/abs/2002.10857
    """

    def __init__(self, m: float, gamma: float) -> None:
        super(RankingLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, y_pred, y_true):
        y_pred = (1 - 2 * y_true) * y_pred   # 0: y_pred; 1: -y_pred
        y_pred_neg = y_pred - y_true * 1e12  # 0: y_pred; 1: -y_ypred-1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12  # 0: y_pred - 1e12; -y_pred

        ap = torch.clamp_min(y_pred_pos.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(y_pred_neg.detach() + self.m, min=0.)

        logit_p = y_pred_pos * self.gamma
        logit_n = (y_pred_neg - self.m) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=-1) + torch.logsumexp(logit_p, dim=-1))

        return loss.mean()
