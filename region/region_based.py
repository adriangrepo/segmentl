import torch
from torch import nn
from torch.autograd import Variable
from torch import einsum
import torch.nn.functional as F
import numpy as np
from ..distribution.distribution_based import CrossentropyND, TopKLoss, WeightedCrossEntropyLoss
from ..utils import get_tp_fp_fn, get_intersection_union, flatten, sum_tensor, softmax_helper, \
    get_activation_fn, mean, flatten_probas, flatten_binary_scores

class IoULoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_run=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22
        :param apply_nonlin:
        :param batch_run: apply to batch
        :param do_bg:
        :param smooth:
        :param square: if True then fp, tp and fn will be squared before summation
        :return: -iou
        """
        super(IoULoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_run = batch_run
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        if self.batch_run:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            activation_fn = get_activation_fn(self.apply_nonlin)
            x = activation_fn(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        iou = (tp + self.smooth) / (tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_run:
                iou = iou[1:]
            else:
                iou = iou[:, 1:]
        iou = iou.mean()
        return 1-iou


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        paper: https://arxiv.org/pdf/1706.05721.pdf
        α and β control the magnitude of penalties for FPs (mask false, pred true) and FNs (mask true, but pred false) espectively
        """
        super(TverskyLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        #print(f'x: {x.shape}, max: {torch.max(x)}, min: {torch.min(x)}, y: {y.shape}, max: {torch.max(y)}, min: {torch.min(y)}')

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            activation_fn = get_activation_fn(self.apply_nonlin)
            x = activation_fn(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                tversky_index = tversky_index[1:]
            else:
                tversky_index = tversky_index[:, 1:]
        tversky_index = tversky_index.mean()
        return 1 - tversky_index

class AsymLoss(nn.Module):

    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False, beta = 1.5):
        """
        'To make a better adjustment of the weights of FPs and FNs (and achieve a better balance between precision
        and recall) in training fully convolutional deep networks for highly unbalanced data,
        where detecting small number of voxels in a class is crucial,
        we propose an asymmetric similarity loss function based on the Fβ score
        which is defined as:
            Fβ=(1+β2)precision×recall / (β2×precision+recall)

        by adjusting the hyperpa-rameter β we can control the trade-off between precision and recall (FPs and FNs).
        For better interpretability to choose β values, we rewrite Equation (3) as
        F(P,G;β)=|PG| / (|PG|+β2(1+β2)|G\P|+1(1+β2)|P\G|)
        It is notable that the Fβ index is a special case of Tversky index [35], where the constraint α+β=1 is preserved.'
        (Hashemi et al. (2018) https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8573779)

        β was allowed to vary between 1.0 and 3.0 with best result at β = 1.5.
        """
        super(AsymLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            activation_fn = get_activation_fn(self.apply_nonlin)
            x = activation_fn(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)# shape: (batch size, class num)
        weight = (self.beta**2)/(1+self.beta**2)
        asym = (tp + self.smooth) / (tp + weight*fn + (1-weight)*fp + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                asym = asym[1:]
            else:
                asym = asym[:, 1:]
        asym = asym.mean()
        return 1-asym


class FocalTverskyLoss(nn.Module):
    """
     'The focal loss function proposed in [4] reshapes the cross-entropy loss function with a modulating exponent
     to down-weight errors as-signed to well-classified examples.  In practice however it faces difficulty balancing
     precision and recall due to small regions-of-interest (ROI) found in medical images. Here (sic) we modulate
     the Tversky index to improve precision and recall balance'
     (Abraham and Khan, 2018, https://arxiv.org/pdf/1810.07842.pdf)

    after: https://github.com/nabsabraham/focal-tversky-unet/blob/347d39117c24540400dfe80d106d2fb06d2b99e1/losses.py#L65
    """
    def __init__(self, tversky_kwargs, gamma=4/3):
        super(FocalTverskyLoss, self).__init__()
        if gamma<=0:
            raise ValueError('FocalTverskyLoss: gamma must be greater than zero')
        self.gamma = gamma
        self.tversky = TverskyLoss(**tversky_kwargs)

    def forward(self, net_output, target):
        tversky_loss = self.tversky(net_output, target)
        focal_tversky = torch.pow(tversky_loss, 1/self.gamma)
        return focal_tversky

class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False):
        """
        apply_nonlin (str): An torch.nn activation applied to the outputs.
            Must be one of ["none", "Sigmoid", "Softmax2d"]
        paper: https://arxiv.org/pdf/1606.04797.pdf
        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth


    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            activation_fn = get_activation_fn(self.apply_nonlin)
            x = activation_fn(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dl = 1-dc.mean()
        print(f'type(dl): {type(dl)}, dl: {dl} shape: {dl.shape}')
        return dl


class GDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5, precision='half'):
        """
        Generalized Dice:
        'We also propose to use the class re-balancing properties of the Generalized Dice overlap,
        a known metric for segmentation assessment, as a robust and accurate deep-learning
        loss function for unbalanced tasks.' (Sudre et al. 2017. https://arxiv.org/pdf/1707.03237.pdf)
        code after: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLoss, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.precision=precision

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        print(f'net_output: {net_output.shape}, net_ouput max: {torch.max(net_output)}, gt: {gt.shape}, gt max: {torch.max(gt)} ')
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.precision=='half':
            y_onehot=y_onehot.half()

        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(net_output)
            pred = softmax_output
        else:
            pred = net_output

        print(f'pred: {pred.shape}, type: {pred.dtype}, y_onehot: {y_onehot.shape}, type: {y_onehot.dtype} max: {torch.max(y_onehot)}')
        # copy from https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L29
        intersection, union= get_intersection_union(pred, y_onehot, precision=self.precision)
        divided: torch.Tensor = 1 - 2 * (einsum("bc->b", intersection) + self.smooth) / (einsum("bc->b", union) + self.smooth)
        gdc = divided.mean()

        return gdc

class GDiceLossV2(nn.Module):
    def __init__(self, apply_nonlin=None, smooth=1e-5):
        """
        Generalized Dice;
        Copy from: https://github.com/wolny/pytorch-3dunet/blob/6e5a24b6438f8c631289c10638a17dea14d42051/unet3d/losses.py#L75
        paper: https://arxiv.org/pdf/1707.03237.pdf
        tf code: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py#L279
        """
        super(GDiceLossV2, self).__init__()

        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, net_output, gt):
        shp_x = net_output.shape # (batch size,class_num,x,y,z)
        shp_y = gt.shape # (batch size,1,x,y,z)
        # one hot code for gt
        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)


        if self.apply_nonlin is not None:
            softmax_output = self.apply_nonlin(net_output)
            input = flatten(softmax_output)
        else:
            input = flatten(net_output)

        target = flatten(y_onehot)
        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.smooth), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.smooth)

class PenaltyGDiceLoss(nn.Module):
    """
    paper: https://openreview.net/forum?id=H1lTh8unKN
    """
    def __init__(self, gdice_kwargs):
        super(PenaltyGDiceLoss, self).__init__()
        self.k = 2.5
        self.gdc = GDiceLoss(apply_nonlin=softmax_helper, **gdice_kwargs)

    def forward(self, net_output, target):
        gdc_loss = self.gdc(net_output, target)
        penalty_gdc = gdc_loss / (1 + self.k * (1 - gdc_loss))

        return penalty_gdc


class SensitivitySpecifityLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False, r = 0.1):
        """
        Sensitivity-Specifity loss

        'If one class contains vastly more samples, as is the case for lesion segmentation,
        the error measure is dominated by the majority class and consequently, the neural network would learn to
        completely ignore the minority class. To overcome this problem, we use a combination of sensitivity and specificity,
        which can be used together to measure classification performance even for vastly unbalanced problems.
        More precisely, the final error measure is a weighted sum of the mean squared difference of the
        lesion voxels(sensitivity) and non-lesion voxels (specificity)'

        'In terms of error the final error measure is a weighted sum of the mean squared difference of the
        lesion voxels (sensitivity) and non-lesion voxels (specificity):

        E = r( ∑p(S(p)−y(2)(p))2S(p) / ∑pS(p) ) + (1−r)( ∑p(S(p)−y(2)(p))^2 (1−S(p)) / ∑p(1−S(p)) )  (9)'

        'Due to the large number of non-lesion voxels, weighting the specificity error higher is important,
        but the algorithm is stable with respect to changes in r, which largely affects the threshold used to
        binarize the probabilistic output.
        In all our experiments, a sensitivity ratio between 0.10 and 0.01 yields very similar results.'
        (Brosch et al. (2015),
        http://www.rogertam.ca/Brosch_MICCAI_2015.pdf)

        tf code: https://github.com/NifTK/NiftyNet/blob/df0f86733357fdc92bbc191c8fec0dcf49aa5499/niftynet/layer/loss_segmentation.py#L392
        """
        super(SensitivitySpecifityLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.r = r

    def forward(self, net_output, gt, loss_mask=None):
        shp_x = net_output.shape
        shp_y = gt.shape
        # class_num = shp_x[1]

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            activation_fn = get_activation_fn(self.apply_nonlin)
            x = activation_fn(net_output)
            squared_error = (y_onehot - x) ** 2
        else:
            squared_error = (y_onehot - net_output) ** 2

        # non mask values
        bg_onehot = 1 - y_onehot
        specificity_part = sum_tensor(squared_error * y_onehot, axes) / (sum_tensor(y_onehot, axes) + self.smooth)
        sensitivity_part = sum_tensor(squared_error * bg_onehot, axes) / (sum_tensor(bg_onehot, axes) + self.smooth)

        ss = self.r * specificity_part + (1 - self.r) * sensitivity_part

        if not self.do_bg:
            if self.batch_dice:
                ss = ss[1:]
            else:
                ss = ss[:, 1:]
        ss = ss.mean()
        return 1-ss

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors

    after: https://github.com/bermanmaxim/LovaszSoftmax
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

class LovaszHinge(nn.Module):
    """
    'We present a method for direct optimization of the mean intersection-over-union loss in neural networks,
    in the context of semantic image segmentation, based on the convex Lov́asz extension of sub-modular losses.'
    (Berman et al. (2018) https://arxiv.org/pdf/1705.08790.pdf

    based on: https://github.com/bermanmaxim/LovaszSoftmax

    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """

    def __init__(self, batch_loss=True, ignore=None):
        super(LovaszHinge, self).__init__()
        self.batch_loss = batch_loss
        self.ignore = ignore

    def forward(self, logits, labels):
        print(f'logits: {logits.shape}, labels: {labels.shape}')
        if len(logits.shape)==4:
            logits =torch.flatten(logits, start_dim=1, end_dim=2)
        if len(labels.shape)==4:
            labels =torch.flatten(labels, start_dim=1, end_dim=2)
        if self.batch_loss:
            loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, self.ignore))
        else:
            loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), self.ignore))
                        for log, lab in zip(logits, labels))
        return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore

    after: https://github.com/bermanmaxim/LovaszSoftmax
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss

class LovaszSoftmax(nn.Module):
    """
    'We present a method for direct optimization of the mean intersection-over-union loss in neural networks,
    in the context of semantic image segmentation, based on the convex Lov́asz extension of sub-modular losses.'
    (Berman et al. (2018) https://arxiv.org/pdf/1705.08790.pdf

    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels

    based on: https://github.com/bermanmaxim/LovaszSoftmax
    """
    def __init__(self, classes='present', batch_loss=True, ignore=None):
        super(LovaszSoftmax, self).__init__()
        self.classes = classes
        self.batch_loss=batch_loss
        self.ignore=ignore

    def forward(self, probas, labels):
        if self.batch_loss:
            loss = lovasz_softmax_flat(*flatten_probas(probas, labels, self.ignore), classes=self.classes)
        else:
            loss = mean(
                lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), self.ignore), classes=self.classes)
                for prob, lab in zip(probas, labels))
        return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.

    after https://arxiv.org/abs/1705.08790

    based on: https://github.com/bermanmaxim/LovaszSoftmax
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


class ExpLogLoss(nn.Module):
    """
    paper: 3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes
    https://arxiv.org/pdf/1809.00076.pdf
    """
    def __init__(self, soft_dice_kwargs, wce_kwargs, gamma=0.3):
        super(ExpLogLoss, self).__init__()
        self.wce = WeightedCrossEntropyLoss(**wce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.gamma = gamma

    def forward(self, net_output, target):
        dc_loss = -self.dc(net_output, target) # weight=0.8
        wce_loss = self.wce(net_output, target) # weight=0.2
        # with torch.no_grad():
        #     print('dc loss:', dc_loss.cpu().numpy(), 'ce loss:', ce_loss.cpu().numpy())
        #     a = torch.pow(-torch.log(torch.clamp(dc_loss, 1e-6)), self.gamma)
        #     b = torch.pow(-torch.log(torch.clamp(ce_loss, 1e-6)), self.gamma)
        #     print('ExpLog dc loss:', a.cpu().numpy(), 'ExpLogce loss:', b.cpu().numpy())
        #     print('*'*20)
        explog_loss = 0.8*torch.pow(-torch.log(torch.clamp(dc_loss, 1e-6)), self.gamma) + \
            0.2*wce_loss

        return explog_loss


class TotalError(nn.Module):
    '''Inspired by https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
    FN + FP (e.g. 5% of the image's pixels were miscategorized).
    In the case where one is more important than the other,
    a weighted average may be used: c0FP + c1FN.'''

    def __init__(self, c0=0.5, c1=0.5, apply_nonlin='Sigmoid', batch_loss=False, do_bg=True):
        super(TotalError, self).__init__()
        self.c0=c0
        self.c1=c1
        assert (c0+c1)>0
        self.apply_nonlin=apply_nonlin
        self.batch_loss=batch_loss
        self.do_bg=do_bg
        #so dont divide by zero
        self.epsilon=1e-7

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_loss:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            activation_fn = get_activation_fn(self.apply_nonlin)
            x = activation_fn(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, False)

        wfp=self.c0*fp
        wfn=self.c1*fn

        loss=wfp+wfn

        if not self.do_bg:
            if self.batch_loss:
                loss = loss[1:]
            else:
                loss = loss[:, 1:]
        #loss here is a 1D tensor of length batch
        #convert loss as fraction of pixels per image
        loss=loss/((shp_x[2]*shp_x[3])+self.epsilon)
        #average over the batch
        loss = loss.mean()
        #print(f'type(loss): {type(loss)}, loss: {loss} shape: {loss.shape}')
        return loss

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

#testing code
def gen_sample():
    logits = torch.randn(4, 19, 768, 768).cuda()
    logits = torch.tensor(logits, requires_grad=True)
    scores = torch.softmax(logits, dim=1)
    # print(scores)
    labels = torch.randint(0, 1, (4, 1, 768, 768)).cuda()
    print(f'scores: {scores.shape}, labels: {labels.shape}')
    labels[0, 30:35, 40:45] = 1
    labels[1, 0:5, 40:45] = 1
    # print(labels)
    return scores, labels

def run_soft_dice(scores, labels):

    criteria = SoftDiceLoss()
    criteria.cuda()

    loss = criteria(scores, labels)
    print(f'SoftDiceLoss: {loss}')
    loss.backward()

def run_tversky(scores, labels):
    criteria = TverskyLoss(apply_nonlin=None, batch_dice=False, smooth=1e-5, do_bg=False)
    criteria.cuda()

    loss = criteria(scores, labels)
    print(f'TverskyLoss: {loss}')
    loss.backward()


if __name__ == '__main__':
    scores, labels=gen_sample()
    run_soft_dice(scores, labels)
    run_tversky(scores, labels)