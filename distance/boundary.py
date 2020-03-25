from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from scipy.ndimage import distance_transform_edt as distance_transform

from ..utils import get_activation_fn
from ..region.region_based import SoftDiceLoss
from ..distance.utils import compute_edts_batch, generate_contours
from ..distance.utils import draw_poly


def boundary_map(pred, one_hot_gt, theta, theta0):
    # boundary map
    gt_b = F.max_pool2d(
        1 - one_hot_gt, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    gt_b -= 1 - one_hot_gt
    pred_b = F.max_pool2d(
        1 - pred, kernel_size=theta0, stride=1, padding=(theta0 - 1) // 2)
    pred_b -= 1 - pred

    # extended boundary map
    gt_b_ext = F.max_pool2d(
        gt_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)

    pred_b_ext = F.max_pool2d(
        pred_b, kernel_size=theta, stride=1, padding=(theta - 1) // 2)
    return gt_b_ext, pred_b_ext, gt_b, pred_b

class BoundarySKLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852

    after: https://github.com/yiskw713/boundary_loss_for_remote_sensing

    Using skimage measure to generate boundaries
    #TODO debug polygon drawing
    """

    def __init__(self, apply_nonlin='Sigmoid', theta0=3, theta=5, precision='half', image_size=[256,256],debug=False):
        super(BoundarySKLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        self.apply_nonlin = apply_nonlin
        self.precision = precision
        self.image_size=image_size
        self.debug=debug
        self.idx=0
        self.epsilon = 1e-7

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bath
        """

        n, c, _, _ = pred.shape

        #pred = torch.softmax(pred, dim=1)
        activation_fn = get_activation_fn(self.apply_nonlin)
        pred = activation_fn(pred)
        '''
        # one-hot vector of ground truth
        with torch.no_grad():
            if len(pred.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(pred.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(pred.shape)
                if pred.device.type == "cuda":
                    y_onehot = y_onehot.cuda(pred.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.precision=='half':
            y_onehot=y_onehot.half()
        '''
        pred_bs=[]
        gt_bs=[]
        dil_pred_bs=[]
        dil_gt_bs=[]
        for p, mask in zip(pred,gt):

            #gt_b, gt_dist=overlay_masks(mask.squeeze().detach().cpu().numpy(), self.image_size, distances=None)
            npm = mask.squeeze().detach().cpu().numpy()
            contours_gt = generate_contours(npm)
            bg = np.zeros_like(npm)
            border_gt=draw_poly(bg, contours_gt)
            dilated_gt = draw_poly(bg, contours_gt, thickness=5)

            npp=p.squeeze().detach().cpu().numpy()
            contours_pred = generate_contours(npp)
            border_pred=draw_poly(bg, contours_pred)
            dilated_pred = draw_poly(bg, contours_pred, thickness=5)

            #print(f'gt_b: {gt_b.shape}, dtype: {gt_b.dtype}, {gt_b}')
            #dilated_gt = ndimage.binary_dilation(gt_b).astype(gt_b.dtype)
            #pred_b, pred_dist = overlay_masks(p.squeeze().detach().cpu().numpy(), self.image_size, distances=None)
            #dilated_pred=ndimage.binary_dilation(pred_b).astype(pred_b.dtype)

            gt_bs.append(torch.from_numpy(border_gt))
            pred_bs.append(torch.from_numpy(border_pred))
            dil_gt_bs.append(torch.from_numpy(dilated_gt))
            dil_pred_bs.append(torch.from_numpy(dilated_pred))

        gt_b = torch.stack(gt_bs).to(device=pred.device)
        pred_b = torch.stack(pred_bs).to(device=pred.device)
        dil_gt_b = torch.stack(dil_gt_bs).to(device=pred.device)
        dil_pred_b = torch.stack(dil_pred_bs).to(device=pred.device)
        #print(f'gt_b: {gt_b.shape}, dil_gt_b: {dil_gt_b.shape}')
        #gt_b_ext, pred_b_ext, gt_b, pred_b = boundary_map(pred, y_onehot, self.theta, self.theta0)
        if self.debug:
            for j,k,l, m in zip(dil_gt_b, dil_pred_b, gt_b, pred_b):
                print(f'j: {j.shape}, k: {k.shape}, l: {l.shape}, m: {m.shape}')
                plt.imshow(j[:,:].detach().cpu().numpy())
                plt.savefig(f'images/BoundarySKLoss_gt_ex_b_{self.idx}.png')
                plt.imshow(k[:,:].detach().cpu().numpy())
                plt.savefig(f'images/BoundarySKLoss_pred_ex_b_{self.idx}.png')

                plt.imshow(l[:,:].detach().cpu().numpy())
                plt.savefig(f'images/BoundarySKLoss_gt_b_{self.idx}.png')
                plt.imshow(m[:,:].detach().cpu().numpy())
                plt.savefig(f'images/BoundarySKLoss_pred_b_{self.idx}.png')
                self.idx +=1

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        dil_gt_b = dil_gt_b.view(n, c, -1)
        dil_pred_b = dil_pred_b.view(n, c, -1)

        print(f'gt_b: {gt_b.shape}, pred_b: {pred_b.shape}, dil_gt_b: {dil_gt_b.shape}, dil_pred_b: {dil_pred_b.shape}')
        # Precision: TP/(TP+FP)
        P = torch.sum(pred_b * dil_gt_b, dim=2) / (torch.sum(pred_b, dim=2) + self.epsilon)
        # Recall: TP/(TP+FN)
        R = torch.sum(dil_pred_b * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + self.epsilon)

        # Boundary F1 Score ~ Dice
        BF1 = 2 * P * R / (P + R + self.epsilon)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss

    generate_contours

class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852

    after: https://github.com/yiskw713/boundary_loss_for_remote_sensing

    'To better account for boundary pixels we select BF1 metric (see original work [9] and its extension [12])
    to construct a differentiable surrogate and use it in training.
    The surrogate is not used alone for training, but as a weighted sum with IoU loss (from directIoUoptimization).
    We found that the impact of the boundary component of the loss function should be gradually increased during training,
    and so we proposed a policy for the weight update.'
    'Via the trial and error process we set θ0 to 3 and θ to 5-7 as a proper choice,
    because theses values deliver the most accurate boundaries in all experiments.'
    'for LBF1,IoUloss, it requires an additional procedure for mini grid-search:
    after the 8th epoch for every 30 epochs and for every weight w∈{0.1,0.3,0.5,0.7,0.9}
    in equation (BCE+wLBF1+ (1−w)LIoU) a network was trained.
    Then the best weight is chosen, and the process repeats'
    """

    def __init__(self, apply_nonlin='Sigmoid', theta0=3, theta=5, precision='half', debug=False):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        self.apply_nonlin = apply_nonlin
        self.precision = precision
        self.debug=debug
        self.epsilon = 1e-7
        #for debug only
        self.idx=0

    def forward(self, pred, gt):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bath
        """

        n, c, _, _ = pred.shape

        #pred = torch.softmax(pred, dim=1)
        activation_fn = get_activation_fn(self.apply_nonlin)
        pred = activation_fn(pred)

        # one-hot vector of ground truth
        with torch.no_grad():
            if len(pred.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(pred.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(pred.shape)
                if pred.device.type == "cuda":
                    y_onehot = y_onehot.cuda(pred.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.precision=='half':
            y_onehot=y_onehot.half()

        gt_b_ext, pred_b_ext, gt_b, pred_b = boundary_map(pred, y_onehot, self.theta, self.theta0)
        if self.debug:
            for j, k, l, m in zip(gt_b_ext, pred_b_ext, gt_b, pred_b):
                #print(f'j: {j.shape}, k: {k.shape}, l: {l.shape}, m: {m.shape}')
                plt.imshow(j[0,:,:].detach().cpu().numpy())
                plt.savefig(f'images/BoundaryLoss_gt_b_ext_{self.idx}.png')
                plt.imshow(k[0,:,:].detach().cpu().numpy())
                plt.savefig(f'images/BoundaryLoss_pred_b_ext_{self.idx}.png')
                plt.imshow(l[0,:,:].detach().cpu().numpy())
                plt.savefig(f'images/BoundaryLoss_gt_b_{self.idx}.png')
                plt.imshow(m[0,:,:].detach().cpu().numpy())
                plt.savefig(f'images/BoundaryLoss_pred_b_{self.idx}.png')
                self.idx+=1

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision: TP/(TP+FP)
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + self.epsilon)
        # Recall: TP/(TP+FN)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + self.epsilon)

        # Boundary F1 Score ~ Dice
        BF1 = 2 * P * R / (P + R + self.epsilon)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss



class BDLoss(nn.Module):
    def __init__(self, apply_nonlin='Sigmoid', debug=False):
        """
        Boundary loss computed on foreground
        ref: https://github.com/LIVIAETS/surface-loss/blob/108bd9892adca476e6cdf424124bc6268707498e/losses.py#L74
        """
        super(BDLoss, self).__init__()
        # default to Sigmoid for binary segmentation
        self.apply_nonlin = apply_nonlin
        self.debug=debug
        # self.do_bg = do_bg

    def forward(self, net_output, target, bound):
        """
        net_output: (batch_size, class, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        bound: precomputed distance map, shape (batch_size, class, x,y,z)
        """

        if self.apply_nonlin is not None:
            activation_fn = get_activation_fn(self.apply_nonlin)
            net_output = activation_fn(net_output)

        pc = net_output[:, 1:, ...].type(torch.float32)
        dc = bound[:, 1:, ...].type(torch.float32)

        # elementwise multiplication
        multipled = torch.einsum("bcxyz,bcxyz->bcxyz", pc, dc)
        bd_loss = multipled.mean()
        print(f'BDLoss: {bd_loss}')
        return bd_loss


class BDDiceLoss(nn.Module):
    def __init__(self, soft_dice_kwargs, bd_kwargs, aggregate="sum", bd_wt=1.0, dc_wt=1.0):
        super(BDDiceLoss, self).__init__()
        self.aggregate = aggregate
        self.bd = BDLoss(**bd_kwargs)
        self.dc = SoftDiceLoss(**soft_dice_kwargs)
        self.dc_wt=dc_wt
        self.bd_wt=bd_wt

    def forward(self, net_output, target, bound):
        dc_loss = self.dc(net_output, target)
        bd_loss = self.bd(net_output, target, bound)
        if self.aggregate == "sum":
            result = self.dc_wt*dc_loss + self.bd_wt*bd_loss
        else:
            raise NotImplementedError("aggregate != sum  is not implemented")
        print(f'BDDiceLoss: {result}')
        return result



def dist_dice(net_output, y_onehot, dist, smooth, debug_name=None):
    '''

    :param net_output:
    :param y_onehot:
    :param dist:
    :param smooth:
    :param dw_dice_wt: weighting for distance map
    :param debug_name: file postfix
    :return:
    '''
    # element-wise matrix multiplication (Hadamard product)
    tp = net_output * y_onehot
    t = tp[:, 0, ...] * dist

    tp = torch.sum(t, (0, 1, 2))

    # distance weighted prediction
    pdist = net_output[:, 0, ...] * dist

    # ground truth mask
    m = y_onehot[:, 0, ...]

    if debug_name:
        print(f'dist: {dist[0]}')
        print(f'pred: {net_output[:, 0, ...][0]}')
        print(f'pred*dist: {pdist[0]}')

        i = 0
        for di, pi, mi, ti in zip(dist.detach().cpu().numpy(), pdist.detach().cpu().numpy(), m.detach().cpu().numpy(), t.detach().cpu().numpy()):
            plt.imshow(di)
            plt.savefig(f'images/{debug_name}_dist_dice_dist_{i}.png')
            plt.imshow(pi)
            plt.savefig(f'images/{debug_name}_dist_dice_p_{i}.png')
            plt.imshow(mi)
            plt.savefig(f'images/{debug_name}_dist_dice_m_{i}.png')
            plt.imshow(ti)
            plt.savefig(f'images/{debug_name}_dist_dice_tp_{i}.png')
            i += 1

    dw_dice = (2 * tp + smooth) / (torch.sum(pdist, (0, 1, 2)) + torch.sum(m, (0, 1, 2)) + smooth)
    dw_dice = dw_dice.mean()
    return dw_dice

def dist_torch(net_output, gt, threshold, boundary_fill, dist_wt, debug_name=None):
    '''
    :param net_output: predictions
    :param gt: ground truth mask
    :param threshold: prediction threshold
    :param boundary_fill: penalise pred inside mask away from edges if False
    :return: dist, y_onehot: distance weight map and encoded preds
    '''
    # one hot code for gt

    with torch.no_grad():
        if len(net_output.shape) != len(gt.shape):
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(net_output.shape)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    gt_temp = gt[:, 0, ...].type(torch.float32)
    with torch.no_grad():
        # add 1.0 so don't multiply by zero, higher values (2.0) closer to boundaries
        masked_treshold = gt_temp.cpu().numpy() > threshold
        dist = compute_edts_batch(masked_treshold, boundary_fill) + 1.0
        nans = np.isnan(dist)
        assert nans.sum() == 0

    if debug_name:
        print('dist.shape: ', dist.shape)
        i = 0
        for di, ni, gi in zip(dist, net_output.detach().cpu().numpy(), gt.detach().cpu().numpy()):
            plt.imshow(di)
            plt.savefig(f'images/{debug_name}_dist_{i}.png')
            plt.imshow(ni.squeeze())
            plt.savefig(f'images/{debug_name}_pred_{i}.png')
            plt.imshow(gi.squeeze())
            plt.savefig(f'images/{debug_name}_gt_{i}.png')
            i += 1

    dist = torch.from_numpy(dist).to(net_output.device).type(torch.float32)
    return dist, y_onehot


class DistanceMapBinaryDiceLoss(nn.Module):
    """
    Distance map penalized Dice loss
    Modified after: https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/boundary_loss.py
    Original code based on: https://openreview.net/pdf?id=B1eIcvS45V
    Original paper used distance*CE loss

    'Error-penalizing distance maps were generated by computing the distance transform on the
    segmentation masks and then reverting them, by voxel-wise subtracting the binary segmentation
    from the mask overall max distance value.
    This  procedure  aims  to  compute  a  distance  mask  where  pixels  in proximity of the bones are weighted more,
    compared to those located far away.' (Distance Map Loss Penalty Term for Semantic Segmentation (Caliva et al. 2019))
    """
    def __init__(self, threshold=0.5, smooth=1e-5, dist_wt=1.0, boundary_fill=False, debug=False):
        '''
        :param threshold:
        :param smooth:
        :param dw_dice_wt: weighting for dice loss
        :param boundary_fill: original paper used distance inside boundary, set to True to fill with mask
        :param debug: use for plotting images to files
        '''
        super(DistanceMapBinaryDiceLoss, self).__init__()
        self.threshold = threshold
        self.smooth = smooth
        self.dist_wt = dist_wt
        self.boundary_fill=boundary_fill
        if debug:
            self.debug_name='DistanceMapBinaryDiceLoss'
        else:
            self.debug_name = ''

    def forward(self, net_output, gt):
        """
        target: ground truth, shape: (B, 1, x,y,z)
        binary net_output: (B, 1, x,y,z)
        """
        #print(f'>>forward pred: {net_output.shape}, gt: {gt.shape}')
        dist, y_onehot = dist_torch(net_output, gt, self.threshold, self.boundary_fill, self.dist_wt, self.debug_name)

        activation_fn = get_activation_fn('Sigmoid')
        net_activated = activation_fn(net_output)
        dw_dice=dist_dice(net_activated, y_onehot, dist, self.smooth, self.debug_name)
        return 1-dw_dice

def compute_edts_forhdloss(segmentation):
    res = np.zeros(segmentation.shape)
    # for each item in batch
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = distance_transform(posmask) + distance_transform(negmask)
    return res



class HausdorffBinaryLoss(nn.Module):
    def __init__(self, alpha=2.0, omega=256*256, threshold=0.5, debug=False):
        """
        compute hausdorff loss for binary segmentation
        https://arxiv.org/pdf/1904.10030v1.pdf

        α determines how strongly we penalize larger errors.
        """
        super(HausdorffBinaryLoss, self).__init__()
        self.debug=debug
        self.omega = omega
        self.alpha=alpha
        self.threshold=threshold


    def forward(self, net_output, target):
        """
        net_output: (batch_size, 1, x,y,z)
        (for nnUnet net_output: (batch_size, 2, x,y,z))
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """

        activation_fn = get_activation_fn('Sigmoid')
        net_output = activation_fn(net_output)

        # per batch
        pc = net_output[:, 0, ...].type(torch.float32)
        gt = target[:,0, ...].type(torch.float32)
        with torch.no_grad():
            pc_dist = compute_edts_forhdloss(pc.cpu().numpy()>self.threshold)
            gt_dist = compute_edts_forhdloss(gt.cpu().numpy()>self.threshold)

        if self.debug:
            # for each item in batch
            for i in range(pc_dist.shape[0]):
                # seems to somewhat work
                n = net_output.detach().cpu().numpy()[i]
                n=np.squeeze(n)
                u_elem, c_elem = np.unique(n, return_counts=True)

                #plt.imshow(n)
                #plt.savefig(f'images/HausdorffBinaryLoss_pred_{self.debug}_{i}.png')
                #plt.imshow(pc_dist[i])
                #plt.savefig(f'images/HausdorffBinaryLoss_pred_dist_{self.debug}_{i}.png')

                # works OK
                t = np.moveaxis(target.detach().cpu().numpy()[i], 0, -1)
                t = np.squeeze(t)
                #plt.imshow(t)
                #plt.savefig(f'images/HausdorffBinaryLoss_gt_{self.debug}_{i}_max{np.max(t)}.png')

                # works OK
                #plt.imshow(gt_dist[i])
                #plt.savefig(f'images/HausdorffBinaryLoss_gt_dist_{self.debug}_{i}.png')


        pred_error = (gt - pc)**2  # see eq(8), NB the paper uses p and q instead of the thresholded maps
        dist = pc_dist**self.alpha + gt_dist**self.alpha # see eq(8)

        dist = torch.from_numpy(dist)
        if dist.device != pred_error.device:
            dist = dist.to(pred_error.device).type(torch.float32)

        #elementwise multiplaction of two tensors
        if len(pred_error.shape)== 3:
            multipled = torch.einsum("xyz,xyz->xyz", pred_error, dist)
        else:
            multipled = torch.einsum("bxyz,bxyz->bxyz", pred_error, dist)

        hd_loss = multipled.mean()

        hd_loss=hd_loss*(1/self.omega)
        return hd_loss

class HausdorffDiceBinaryLoss(nn.Module):
    # after https://github.com/JunMa11/SegLoss/tree/master/losses_pytorch
    def __init__(self, soft_dice_kwargs, hd_kwargs, aggregate="sum", dc_wt=1.0, hd_wt=1.0):
        super(HausdorffDiceBinaryLoss, self).__init__()
        self.aggregate = aggregate
        self.dc = SoftDiceLoss(**soft_dice_kwargs)
        self.hd = HDDTBinaryLoss(**hd_kwargs)
        self.dc_wt = dc_wt
        self.hd_wt = hd_wt

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        hd_loss = self.hd(net_output, target)
        if self.aggregate == "sum":
            result = self.dc_wt * dc_loss + self.hd_wt * hd_loss
        else:
            raise NotImplementedError("aggregate != sum is not implemented")
        print(f'HausdorffDiceBinaryLoss: {result}')
        return result

    #testing code
def gen_sample():
    logits = torch.randn(4, 1, 768, 768).cuda()
    logits = torch.tensor(logits, requires_grad=True)
    scores = torch.softmax(logits, dim=1)
    # print(scores)
    labels = torch.randint(0, 1, (4, 1, 768, 768)).cuda()
    print(f'scores: {scores.shape}, labels: {labels.shape}')
    labels[0, 30:35, 40:45] = 1
    labels[1, 0:5, 40:45] = 1
    # print(labels)
    return scores, labels


if __name__ == '__main__':
    scores, labels = gen_sample()
    run_hdt(scores, labels)