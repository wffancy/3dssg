import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=0.2, reduction='mean'):
        super(CrossEntropyFocalLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        # logits: [N, C, H, W], target: [N, H, W]
        # loss = sum(-y_i * log(c_i))
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]
        logits = torch.sigmoid(logits)  # narrow the value scope of feature vector

        pt = F.softmax(logits, 1)
        pt = pt.gather(1, target).view(-1)  # [NHW]
        log_gt = torch.log(pt)

        if self.alpha is not None:
            # alpha: [C]
            alpha = self.alpha.gather(0, target.view(-1))  # [NHW]
            log_gt = log_gt * alpha

        loss = -1 * (1 - pt) ** self.gamma * log_gt

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

class PerClassBCEFocalLosswithLogits(nn.Module):    # namely multi-label classification
    def __init__(self, gamma=0.2, alpha=0.6, reduction='mean'):
        super(PerClassBCEFocalLosswithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        # logits: [N, F], target: [1, N]
        # logits = torch.clamp(logits, -5, 5)  # avoid overflow after sigmoid computation in the val phase, remains for observation
        if not self.training:
            logits = torch.clamp(logits, -5, 5)  # avoid overflow after sigmoid computation in the val phase, remains for observation
        logits = torch.sigmoid(logits)
        alpha = self.alpha
        gamma = self.gamma

        loss = -alpha * (1 - logits) ** gamma * target * torch.log(logits) - \
               (1 - alpha) * logits ** gamma * (1 - target) * torch.log(1 - logits)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss