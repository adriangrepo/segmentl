from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
import torch
from torch import nn
from torch.autograd import Variable
from torch import einsum
from torch import Tensor
import numpy as np
from itertools import filterfalse as ifilterfalse
from scipy.spatial.distance import directed_hausdorff


def softmax_helper(x):
    # from pywick losses.py
    rpt = [1 for _ in range(len(x.size()))]
    print(f'softmax_helper() x: {x.shape}, rpt: {rpt}')
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    print(f'softmax_helper() rpt[1]: {rpt[1]}, x_max dim 1: {x_max}')
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def sum_tensor(inp, axes, keepdim=False):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return: tp, fp, fn
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape
    #print(f'get_tp_fp_fn() x: {shp_x}, y: {shp_y}, axes: {axes}')
    #get_tp_fp_fn() x: torch.Size([32, 1, 256, 256]), y: torch.Size([32, 1, 256, 256]), axes: [2, 3]

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))
            print(f'gt: {gt.shape}')

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)
            print(f'y_onehot: {y_onehot}')

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn

def get_intersection_union(pred, y_onehot, precision = 'half'):
    if len(y_onehot.shape)==4:
        if precision == 'half':
            w: torch.Tensor = 1 / (einsum("bcxy->bc", y_onehot).half()+ 1e-10) ** 2
        else:
            w: torch.Tensor = 1 / (einsum("bcxy->bc", y_onehot).type(torch.float32) + 1e-10) ** 2
        intersection: torch.Tensor = w * einsum("bcxy, bcxy->bc", pred, y_onehot)
        union: torch.Tensor = w * (einsum("bcxy->bc", pred) + einsum("bcxy->bc", y_onehot))
    elif len(y_onehot.shape)==5:
        if precision=='half':
            w: torch.Tensor = 1 / (einsum("bcxyz->bc", y_onehot).half() + 1e-10) ** 2
        else:
            w: torch.Tensor = 1 / (einsum("bcxyz->bc", y_onehot).type(torch.float32) + 1e-10) ** 2
        intersection: torch.Tensor = w * einsum("bcxyz, bcxyz->bc", pred, y_onehot)
        union: torch.Tensor = w * (einsum("bcxyz->bc", pred) + einsum("bcxyz->bc", y_onehot))
    else:
        raise ValueError(f'GDice not implemented for data dimension: {len(y_onehot.shape)}')
    return intersection, union

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order).contiguous()
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.view(C, -1)


def get_activation_fn(activation: str = None):
    """
    From Catalyst catalyst.utils.torch
    Returns the activation function from ``torch.nn`` by its name
    """
    if activation is None or activation.lower() == "none":
        activation_fn = lambda x: x  # noqa: E731
    else:
        activation_fn = torch.nn.__dict__[activation]()
    return activation_fn


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    after: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/LovaszSoftmax
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float:
    # after https://github.com/LIVIAETS/surface-loss/blob/master/utils.py
    assert len(pred.shape) == 2
    assert pred.shape == target.shape

    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])

def haussdorf(preds: Tensor, target: Tensor) -> Tensor:
    # after https://github.com/LIVIAETS/surface-loss/blob/master/utils.py
    assert preds.shape == target.shape
    assert one_hot(preds)
    assert one_hot(target)

    B, C, _, _ = preds.shape

    res = torch.zeros((B, C), dtype=torch.float32, device=preds.device)
    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()

    for b in range(B):
        if C == 2:
            res[b, :] = numpy_haussdorf(n_pred[b, 0], n_target[b, 0])
            continue

        for c in range(C):
            res[b, c] = numpy_haussdorf(n_pred[b, c], n_target[b, c])

    return res

