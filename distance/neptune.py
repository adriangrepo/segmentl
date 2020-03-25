from functools import partial

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.functional as F
from segmentl.distance.neptune_utils import overlay_mask_batch

def _get_loss_variables(w0, sigma, imsize):
    # after https://github.com/neptune-ai/open-solution-mapping-challenge
    w0 = torch.Tensor([w0])
    sigma = torch.Tensor([sigma])
    C = torch.sqrt(torch.Tensor([imsize[0] * imsize[1]])) / 2
    return w0, sigma, C

def _get_size_weights(sizes, C):
    # after https://github.com/neptune-ai/open-solution-mapping-challenge
    sizes_ = sizes.clone()
    sizes_[sizes == 0] = 1
    size_weights = C / sizes_
    size_weights[sizes_ == 1] = 1
    return size_weights

def _get_distance_weights(d, w1, w0, sigma):
    '''
    :param d:
    :param w1:
    :param w0:
    :param sigma:
    :return:
    #after https://github.com/neptune-ai/open-solution-mapping-challenge
    '''
    weights = w1 + w0 * torch.exp(-(d ** 2) / (sigma ** 2))
    weights[d == 0] = 1
    return weights

def get_weights(distances, sizes, w0, sigma, imsize, device):
    '''
    distances: np array
    sizes: np array
    w1 is temporarily torch.ones - it should handle class imbalance for the whole dataset
    w0: 50
    sigma: 10
    image_h: 256
    image_channels: 3

    modified after https://github.com/neptune-ai/open-solution-mapping-challenge
    '''
    print(f'distances: {distances.shape}, sizes: {sizes.shape}')
    w0, sigma, C = _get_loss_variables(w0, sigma, imsize)
    if torch.cuda.is_available():
        w0 = w0.to(device)
        sigma = sigma.to(device)
        C = C.to(device)

    w1 = torch.ones(distances.shape)
    if torch.cuda.is_available():
        w1 = w1.to(device)
    size_weights = _get_size_weights(sizes, C)

    distance_weights = _get_distance_weights(distances, w1, w0, sigma)

    weights = distance_weights * size_weights

    return weights

class NDistWeightedBCELoss(nn.Module):
    '''
    Calculate weighted Cross Entropy loss for binary segmentation.

    This class calculates BCE, but each pixel loss is weighted.
    Target for weights is defined as a part of target, in target[:, 1:, :, :].
    If weights_function is not None weights are calculated by applying this function on target[:, 1:, :, :].
    If weights_function is None weights are taken from target[:, 1, :, :].

    Returns:
        torch.Tensor: Loss value.

    modified after https://github.com/neptune-ai/open-solution-mapping-challenge
    '''
    def __init__(self, w0, sigma, imsize, distance_type='nearest', border_width=0, n_distances=2,debug_path=None):
        '''
        Unlike original implementation, here we generate distances on the fly
        :param w0:
        :param sigma:
        :param imsize:
        :param distance_type: all, nearest or second_nearest
        :param border_width:
        :param n_distances: consider this number of nearest objects per image, 0 for all
        :param debug_path:
        '''
        super(NDistWeightedBCELoss, self).__init__()
        self.w0 = w0
        self.sigma = sigma
        self.imsize = imsize
        self.distance_type = distance_type
        self.border_width = border_width
        self.n_distances = n_distances
        self.debug_path=debug_path
        self.it_idx=0 #for debugging only


    def forward(self, output, target):
        '''
        output (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x (1 + K) x H x W). Where K is number of different weights.
        '''
        mask_overlayed, distances, nearest_distances, second_nearest_distances, sizes=overlay_mask_batch(self.imsize, target, border_width=self.border_width, n_distances=self.n_distances)
        sizes = torch.from_numpy(sizes.astype(np.int16)).to(target.device)
        if self.distance_type=='nearest':
            nearest_distances = torch.from_numpy(nearest_distances).to(target.device)
            dist_values = nearest_distances
        elif self.distance_type=='second_nearest':
            second_nearest_distances = torch.from_numpy(second_nearest_distances).to(target.device)
            dist_values = second_nearest_distances
        else:
            distances = torch.from_numpy(distances).to(target.device)
            dist_values=distances
        weights = get_weights(dist_values, sizes, self.w0, self.sigma, self.imsize, target.device)
        crit = torch.nn.BCEWithLogitsLoss()
        loss_per_pixel = crit(output, target)
        #loss_per_pixel = torch.nn.CrossEntropyLoss(reduce=False)(output, target)
        loss = torch.mean(loss_per_pixel * weights)
        if self.debug_path:
            np.save(str(self.debug_path) + f'/gt_{self.distance_type}_{self.n_distances}_{self.it_idx}', target.detach().cpu().numpy())
            np.save(str(self.debug_path) + f'/distances_{self.distance_type}_{self.n_distances}_{self.it_idx}', dist_values.detach().cpu().numpy())
            np.save(str(self.debug_path) + f'/sizes_{self.distance_type}_{self.n_distances}_{self.it_idx}', sizes.detach().cpu().numpy())
            np.save(str(self.debug_path) + f'/weights_{self.distance_type}_{self.n_distances}_{self.it_idx}', weights.detach().cpu().numpy())
            self.it_idx+=1
        return loss

class MixedDiceCELoss(nn.Module):
    '''
    Calculate mixed Dice and Cross Entropy Loss.

    Args:
        output (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor):
            Target of shape (N x (1 + K) x H x W).
            Where K is number of different weights for Cross Entropy.
        dice_weight (float, optional): Weight of Dice loss. Defaults to 0.5.
        dice_loss (function, optional): Dice loss function. If None multiclass_dice_loss() is being used.
        cross_entropy_weight (float, optional): Weight of Cross Entropy loss. Defaults to 0.5.
        cross_entropy_loss (function, optional):
            Cross Entropy loss function.
            If None torch.nn.CrossEntropyLoss() is being used.
        smooth (float, optional): Smoothing factor for Dice loss. Defaults to 0.
        dice_activation (string, optional):
            Name of the activation function for Dice loss, softmax or sigmoid.
            Defaults to 'softmax'.

    Returns:
        torch.Tensor: Loss value.
    after https://github.com/neptune-ai/open-solution-mapping-challenge
    '''
    def __init__(self, dice_weight=0.5, dice_loss=None,
                                  cross_entropy_weight=0.5, cross_entropy_loss=None, smooth=0,
                                  dice_activation='softmax'):
        super(MixedDiceCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.dice_loss = dice_loss
        self.cross_entropy_weight = cross_entropy_weight
        self.cross_entropy_loss = cross_entropy_loss
        self.smooth = smooth
        self.dice_activation = dice_activation

    def forward(self, output, target):
        dice_target = target[:, 0, :, :].long()
        cross_entropy_target = target
        if self.cross_entropy_loss is None:
            cross_entropy_loss = torch.nn.CrossEntropyLoss()
            cross_entropy_target = dice_target
        if self.dice_loss is None:
            dice_loss = multiclass_dice_loss
        return self.dice_weight * dice_loss(output, dice_target, self.smooth,
                                       self.dice_activation) + self.cross_entropy_weight * cross_entropy_loss(output,
                                                                                                    cross_entropy_target)


def multiclass_dice_loss(output, target, smooth=0, activation='softmax', excluded_classes=[]):
    """Calculate Dice Loss for multiple class output.

    Args:
        output (torch.Tensor): Model output of shape (N x C x H x W).
        target (torch.Tensor): Target of shape (N x H x W).
        smooth (float, optional): Smoothing factor. Defaults to 0.
        activation (string, optional): Name of the activation function, softmax or sigmoid. Defaults to 'softmax'.
        excluded_classes (list, optional):
            List of excluded classes numbers. Dice Loss won't be calculated
            against these classes. Often used on background when it has separate output class.
            Defaults to [].

    Returns:
        torch.Tensor: Loss value.
    after https://github.com/neptune-ai/open-solution-mapping-challenge
    """
    if activation == 'softmax':
        activation_nn = torch.nn.Softmax2d()
    elif activation == 'sigmoid':
        activation_nn = torch.nn.Sigmoid()
    else:
        raise NotImplementedError('only sigmoid and softmax are implemented')

    loss = 0
    dice = DiceLoss(smooth=smooth)
    output = activation_nn(output)
    for class_nr in range(output.size(1)):
        if class_nr in excluded_classes:
            continue
        class_target = (target == class_nr)
        class_target.data = class_target.data.float()
        loss += dice(output[:, class_nr, :, :], class_target)
    return loss

#from .steps.pytorch.validation
class DiceLoss(nn.Module):
    # after https://github.com/neptune-ai/open-solution-mapping-challenge
    def __init__(self, smooth=0, eps = 1e-7):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        return 1 - (2 * torch.sum(output * target) + self.smooth) / (
                    torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


def segmentation_loss(output, target, weight_bce=1.0, weight_dice=1.0):
    # after https://github.com/neptune-ai/open-solution-mapping-challenge
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    return weight_bce * bce(output, target) + weight_dice * dice(output, target)


def multiclass_segmentation_loss(output, target):
    # after https://github.com/neptune-ai/open-solution-mapping-challenge
    target = target.squeeze(1).long()
    cross_entropy = nn.CrossEntropyLoss()
    return cross_entropy(output, target)


def cross_entropy(output, target, squeeze=False):
    # after https://github.com/neptune-ai/open-solution-mapping-challenge
    if squeeze:
        target = target.squeeze(1)
    return F.nll_loss(output, target)


def mse(output, target, squeeze=False):
    # after https://github.com/neptune-ai/open-solution-mapping-challenge
    if squeeze:
        target = target.squeeze(1)
    return F.mse_loss(output, target)


def multi_output_cross_entropy(outputs, targets):
    # after https://github.com/neptune-ai/open-solution-mapping-challenge
    losses = []
    for output, target in zip(outputs, targets):
        loss = cross_entropy(output, target)
        losses.append(loss)
    return sum(losses) / len(losses)


