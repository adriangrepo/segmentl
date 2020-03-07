import torch
from torch import nn
from torch.autograd import Variable
from torch import einsum
import numpy as np
from segmentl.distribution.distribution_based import CrossentropyND, TopKLoss, WeightedCrossEntropyLoss
from segmentl.utils import softmax_helper
from segmentl.region.region_based import SoftDiceLoss

class DiceTopKoss(nn.Module):
    # after https://github.com/JunMa11/SegLoss/tree/master/losses_pytorch
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(DiceTopKoss, self).__init__()
        self.aggregate = aggregate
        self.ce = TopKLoss(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("aggregate != sum is not implemented")
        return result

class CEDiceLoss(nn.Module):
    # after https://github.com/JunMa11/SegLoss/tree/master/losses_pytorch
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(CEDiceLoss, self).__init__()
        self.aggregate = aggregate
        self.ce = CrossentropyND(**ce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("aggregate != sum is not implemented")
        return result

class HausdorffDiceBinaryLoss(nn.Module):
    # after https://github.com/JunMa11/SegLoss/tree/master/losses_pytorch
    def __init__(self, soft_dice_kwargs, hd_kwargs, aggregate="sum", dc_wt=1.0, hd_wt=1.0):
        super(HausdorffDiceBinaryLoss, self).__init__()
        self.aggregate = aggregate
        self.dc = SoftDiceLoss(**soft_dice_kwargs)
        self.hd = HausdorffDiceBinaryLoss(**hd_kwargs)
        self.dc_wt=dc_wt
        self.hd_wt=hd_wt

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        hd_loss = self.hd(net_output, target)
        if self.aggregate == "sum":
            result = self.dc_wt*dc_loss + self.hd_wt*hd_loss
        else:
            raise NotImplementedError("aggregate != sum is not implemented")
        print(f'HausdorffDiceBinaryLoss: {result}')
        return result