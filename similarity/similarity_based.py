
import torch
from torch import nn
import torch.nn.functional as F



class AffinityFieldLoss(nn.Module):
    '''
        loss proposed in the paper: https://arxiv.org/abs/1803.10335
        used for segmentation tasks

        after: https://github.com/CoinCheung/pytorch-loss/blob/master/affinity_loss.py
    '''

    def __init__(self, kl_margin=3., lambda_edge=1., lambda_not_edge=1., ignore_lb=255):
        super(AffinityFieldLoss, self).__init__()
        self.kl_margin = kl_margin
        self.ignore_lb = ignore_lb
        self.lambda_edge = lambda_edge
        self.lambda_not_edge = lambda_not_edge
        self.kldiv = nn.KLDivLoss(reduction='none')

    def forward(self, logits, labels):
        if len(labels.shape) == 4 and labels.shape[1] == 1:
            # flatten first dimension for this Affinity loss implementation
            lbls = torch.flatten(labels, start_dim=1, end_dim=2)
        ignore_mask = lbls.cpu() == self.ignore_lb
        n_valid = ignore_mask.numel() - ignore_mask.sum().item()
        print(f'n_valid: {n_valid}')
        indices = [
            # center,               # edge
            ((1, None, None, None), (None, -1, None, None)),  # up
            ((None, -1, None, None), (1, None, None, None)),  # down
            ((None, None, 1, None), (None, None, None, -1)),  # left
            ((None, None, None, -1), (None, None, 1, None)),  # right
            ((1, None, 1, None), (None, -1, None, -1)),  # up-left
            ((1, None, None, -1), (None, -1, 1, None)),  # up-right
            ((None, -1, 1, None), (1, None, None, -1)),  # down-left
            ((None, -1, None, -1), (1, None, 1, None)),  # down-right
        ]

        losses = []
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)

        for idx_c, idx_e in indices:
            lbcenter = lbls[:, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]].detach()
            lbedge = lbls[:, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]].detach()
            igncenter = ignore_mask[:, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]].detach()
            ignedge = ignore_mask[:, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]].detach()
            lgp_center = probs[:, :, idx_c[0]:idx_c[1], idx_c[2]:idx_c[3]]
            lgp_edge = probs[:, :, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]]
            prob_edge = probs[:, :, idx_e[0]:idx_e[1], idx_e[2]:idx_e[3]]
            kldiv = (prob_edge * (lgp_edge - lgp_center)).sum(dim=1)
            kldiv[ignedge | igncenter] = 0

            loss = torch.where(
                lbcenter == lbedge,
                self.lambda_edge * kldiv,
                self.lambda_not_edge * F.relu(self.kl_margin - kldiv, inplace=True)
            ).sum() / n_valid
            losses.append(loss)
        print(f'loss: {sum(losses)}')
        return sum(losses)



class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        after https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch/blob/master/loss_functions.py
        Angular Penalty Softmax Loss
        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599
        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.eps = eps

    def forward(self, x, labels):
        '''
        input shape (N, in_features)
        '''
        assert len(x) == len(labels)
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.out_features

        for W in self.fc.parameters():
            W = F.normalize(W, p=2, dim=1)

        x = F.normalize(x, p=2, dim=1)

        wf = self.fc(x)
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        if self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        if self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

_PSP_AUX_WEIGHT = 0.4		# the weight of the auxiliary loss in PSPNet
class PSPNetRMILoss(nn.Module):
    '''
    Region Mutual Information Loss for SemanticSegmentation.  (Zhao et al., 2019)
    https://papers.nips.cc/paper/9291-region-mutual-information-loss-for-semantic-segmentation.pdf
    cade based on: https://github.com/ZJULearning/RMI

    for pspnet
    '''

    def __init__(self, global_step=0, loss_type=None, criterion=None):
        super(RMILoss, self).__init__()
        self.loss_type = loss_type
        self.criterion = criterion
        self.global_step = global_step

    def forward(self, output, target):
        """forward step"""
        # output of the model
        output = self.model(inputs)


        # PSPNet have auxilary branch
        if self.loss_type == 'rmi':
            # loss = self.criterion(output[0], target) + _PSP_AUX_WEIGHT * self.criterion(output[1], target)
            # loss = loss / (1.0 + _PSP_AUX_WEIGHT)
            loss = self.criterion(output[0], target) + _PSP_AUX_WEIGHT * F.cross_entropy(input=output[1],
                                                                                             target=target.long(),
                                                                                             ignore_index=255,
                                                                                             reduction='mean')
            output = output[0]
        elif self.loss_type == 'affinity':
            loss = (self.criterion(output[0], target, global_step=self.global_step) +
                        _PSP_AUX_WEIGHT * self.criterion(output[1], target, global_step=self.global_step))
            output = output[0]
        elif self.loss_type == 'pyramid':
            loss = self.criterion(output[0], target) + _PSP_AUX_WEIGHT * self.criterion(output[1], target)
            output = output[0]
        else:
            loss = (self.criterion(output[0], target.long()) + _PSP_AUX_WEIGHT * self.criterion(output[1],
                                                                                                    target.long()))
            output = output[0]
        # loss = loss.unsqueeze(dim=0)
        return output, loss



class RMILoss(nn.Module):
    '''
    Region Mutual Information Loss for SemanticSegmentation.  (Zhao et al., 2019)
    https://papers.nips.cc/paper/9291-region-mutual-information-loss-for-semantic-segmentation.pdf
    cade based on: https://github.com/ZJULearning/RMI
    '''

    def __init__(self, global_step=0, loss_type=None, criterion=None):
        super(RMILoss, self).__init__()
        self.loss_type = loss_type
        self.criterion = criterion
        self.global_step = global_step

    def forward(self, output, target):

        if self.loss_type == 'rmi':
            loss = self.criterion(output, target)
        elif self.loss_type == 'affinity':
            loss = self.criterion(output, target, global_step=self.global_step)
        elif self.loss_type == 'pyramid':
            loss = self.criterion(output, target)
        else:
            loss = self.criterion(output, target.long())
        # loss = loss.unsqueeze(dim=0)
        return output, loss