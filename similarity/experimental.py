
import torch
from torch import nn
import torch.nn.functional as F
from math import exp
import numpy as np
from segmentl.utils import get_activation_fn
from skimage.feature import masked_register_translation
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage import transform as tf
from math import exp
import numpy as np
from segmentl.utils import get_activation_fn
from segmentl.distance.utils import get_edges, binary_mask, binarize_thresh, downsample, crop_center
from matplotlib import pyplot as plt

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None, pad=False):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    if pad:
        padding = window_size // 2
    else:
        padding = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    if torch.isnan(window).any():
        print(f'isnan(window): {torch.isnan(window).any()}, {window}')

    '''F.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) '''
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)

    if torch.isnan(mu1).any():
        print(f'--ssim() isnan(img1).any(): {torch.isnan(img1).any()}')
        print(f'--ssim() isnan(window).any(): {torch.isnan(window).any()}')
        print(f'--ssim() isnan(mu1): {torch.isnan(mu1).any()}, {mu1}')
    if torch.isnan(mu2).any():
        print(f'--ssim() isnan(img2).any(): {torch.isnan(img2).any()}')
        print(f'--ssim() isnan(window).any(): {torch.isnan(window).any()}')
        print(f'--ssim() isnan(mu2): {torch.isnan(mu2).any()}, {mu2}')

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    if torch.isnan(mu1_sq).any():
        print(f'isnan(mu1_sq): {torch.isnan(mu1_sq).any()}, {mu1_sq}')
    if torch.isnan(mu2_sq).any():
        print(f'isnan(mu2_sq): {torch.isnan(mu2_sq).any()}, {mu2_sq}')
    if torch.isnan(mu1_mu2).any():
        print(f'isnan(mu1_mu2): {torch.isnan(mu1_mu2).any()}, {mu1_mu2}')

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2

    if torch.isnan(sigma1_sq).any():
        print(f'--ssim() isnan(sigma1_sq): {torch.isnan(sigma1_sq).any()}, {sigma1_sq}')
        print(f'--ssim() isnan(sigma1_sq) shapes sigma1_sq: {sigma1_sq.shape}, window: {window.shape}, \
        img1: {img1.shape}, mu1_sq: {mu1_sq.shape}')
        print(f'--ssim() isnan(sigma1_sq) padding: {padding}, channel: {channel}')
        print(f'--ssim() isnan(sigma1_sq) isnan(img1): {torch.isnan(img1).any()}, {img1}')
        print(f'--ssim() isnan(sigma1_sq) isnan(window): {torch.isnan(window).any()}, {window}')
        print(f'--ssim() isnan(sigma1_sq) isnan(mu1_sq): {torch.isnan(mu1_sq).any()}, {mu1_sq}')
    if torch.isnan(sigma2_sq).any():
        print(f'--ssim() isnan(sigma2_sq): {torch.isnan(sigma2_sq).any()}, {sigma2_sq}')
    if torch.isnan(sigma12).any():
        print(f'--ssim() isnan(sigma12): {torch.isnan(sigma12).any()}, {sigma12}')

    K1 = 0.01   #default settings
    K2 = 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    epsilon = 1e-7
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    if torch.isnan(ssim_map).any():
        print(f'--ssim() isnan(ssim_map): {torch.isnan(ssim_map).any()}, {ssim_map}')
        #hack to set any nan to epsilon
        ssim_map[ssim_map != ssim_map] = epsilon

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    cs = torch.mean(v1 / v2)  # contrast sensitivity
    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=True, avg_pool=False, pad=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):

        if avg_pool:
            '''
            Applies 2D average-pooling operation in kH×kW regions by step size sH×sW steps. 
            input – input tensor (minibatch,in_channels,iH,iW)
            kernel_size – size of the pooling region. Can be a single number or a tuple (kH, kW)
            stride – stride of the pooling operation. Can be a single number or a tuple (sH, sW). Default: kernel_size
            '''
            img1 = F.avg_pool2d(img1, (2, 2))
            img2 = F.avg_pool2d(img2, (2, 2))
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range, pad=pad)

        if torch.isnan(sim).any():
            print(f'--mssim: sim nans: {sim}')
        if torch.isnan(cs).any():
            print(f'--mssim: cs nans: {cs}')

        mssim.append(sim)
        mcs.append(cs)



    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    if torch.isnan(mssim).any():
        print(f'--mssim: mssim nans: {mssim}')
    if torch.isnan(mcs).any():
        print(f'--mssim: mcs nans: {mcs}')

    mcsw = mcs ** weights
    mssimw = mssim ** weights

    if torch.isnan(mcsw).any():
        print(f'--mssim: mcsw nans: {mcsw}')
    if torch.isnan(mssimw).any():
        print(f'--mssim: mssimw nans: {mssimw}')

    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    #overall_mssim = prod(mcs_array(1:level - 1).^ weight(1: level - 1))*(mssim_array(level). ^ weight(level));
    #output = torch.prod(mcsw[:-1] * mssimw[-1])
    output = torch.prod(mcsw[1:-1] * mssimw)
    return output



class SSIM(torch.nn.Module):
    # after https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
    def __init__(self, apply_nonlin='Sigmoid', window_size=11, size_average=True, val_range=None, pad=False):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.apply_nonlin=apply_nonlin

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, pred, gt):
        if self.apply_nonlin:
            activation_fn = get_activation_fn(self.apply_nonlin)
            pred = activation_fn(pred)
        (_, channel, _, _) = pred.size()

        if channel == self.channel and self.window.dtype == pred.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(pred.device).type(pred.dtype)
            self.window = window
            self.channel = channel

        ssm, cs =ssim(pred, gt, window=window, window_size=self.window_size, size_average=self.size_average, \
                    full=True, val_range=None, pad=False)
        print(f'cs: {cs}, ssim: {ssm}, loss: {1-ssm}')
        return 1-ssm

class MSSSIM(torch.nn.Module):
    '''Multi-scale Structural Similarity Index (MS-SSIM)
        Z. Wang, E. P. Simoncelli and A. C. Bovik, "Multi-scale structural similarity
        for image quality assessment," Invited Paper, IEEE Asilomar Conference on
        Signals, Systems and Computers, Nov. 2003
    # after https://github.com/jorge-pessoa/pytorch-msssim/blob/master/pytorch_msssim/__init__.py
    '''
    def __init__(self, apply_nonlin='Sigmoid', window_size=11, size_average=True, channel=3, normalize=True, \
                 avg_pool=False, pad=False):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.normalize=normalize
        self.apply_nonlin = apply_nonlin
        self.avg_pool=avg_pool
        self.pad=pad

    def forward(self, pred, gt):
        if self.apply_nonlin:
            activation_fn = get_activation_fn(self.apply_nonlin)
            pred = activation_fn(pred)
        sim=msssim(pred, gt, window_size=self.window_size, size_average=self.size_average, \
                   normalize=self.normalize, avg_pool=self.avg_pool, pad=self.pad)
        print(f'<<MSSSIM mssim: {sim}, loss: {1-sim}')
        return 1-sim


class TransformEstimate(torch.nn.Module):
    '''Estimate 2D geometric transformation parameters.
    see https://scikit-image.org/docs/dev/api/skimage.transform.html?highlight=similarity#skimage.transform.estimate_transform
    Convert to np then back to  torch'''
    def __init__(self, apply_nonlin='Sigmoid'):
        super(TransformEstimate, self).__init__()
        self.apply_nonlin = apply_nonlin

    def forward(self, pred, gt):
        if self.apply_nonlin:
            activation_fn = get_activation_fn(self.apply_nonlin)
            pred = activation_fn(pred)
        print(f'pred: {pred.shape}, gt: {gt.shape}')
        predn=pred.detach().cpu().numpy()
        gtn=gt.detach().cpu().numpy()
        #delta=gtn-predn
        tform = tf.estimate_transform('euclidean', predn, gtn)
        tform=tform.type(torch.float32).to(device=gt.device)
        diff=tform(predn)-gtn
        mt=diff.mean()
        print(f'<<TransformEstimate difference: {mt}, loss: {1-mt}')
        return 1-mt


class SKMSSIM(torch.nn.Module):
    '''Mean structural Similarity with skimage
    When comparing images, the mean squared error (MSE)–while simple to implement–is not highly indicative of perceived similarity.
    Structural similarity aims to address this shortcoming by taking texture into account
    Convert to np then back to  torch'''

    '''
    ~/.virtualenvs/catalyst_base/lib/python3.7/site-packages/skimage/metrics/_structural_similarity.py 
    in structural_similarity(im1, im2, win_size, gradient, data_range, multichannel, gaussian_weights, full, **kwargs)
    151     if np.any((np.asarray(im1.shape) - win_size) < 0):
    152         raise ValueError(
--> 153             "win_size exceeds image extent.  If the input is a multichannel "
    154             "(color) image, set multichannel=True.")
    '''

    def __init__(self, apply_nonlin='Sigmoid'):
        super(SKMSSIM, self).__init__()
        self.apply_nonlin = apply_nonlin

    def forward(self, pred, gt):
        if self.apply_nonlin:
            activation_fn = get_activation_fn(self.apply_nonlin)
            pred = activation_fn(pred)
        predn=pred.detach().cpu().numpy()
        gtn=gt.detach().cpu().numpy()
        sim=ssim(predn, gtn, data_range=gtn.max() - gtn.min())
        sim=sim.type(torch.float32).to(device=gt.device)
        print(f'<<MSSSIM mssim: {sim}, loss: {1-sim}')
        return 1-sim

def ang(a, b):
    epsilon=1e-7
    nza=len(torch.nonzero(a))
    nzb=len(torch.nonzero(b))
    if nza < 1 and nzb < 1:
        #if both pred and gt are zeros want high correlation -
        # but for edges loss this encourages network to just pred nothing all the time - a dillemma I haven't overcome
        cosangle = 1.0
    else:
        cosangle = torch.dot(a, b) / ((torch.norm(a) * torch.norm(b))+epsilon)
    '''
    if nzb < 1 and nza >1 and cosangle > 0.2:
        #cosagle can be pretty unstable eg where get high correlation of full image to empty mask
        #here we use a total hack to give cosangle an arbitrary low value correlated to inverse of preds
        print(f'unstable - nzb: {nzb} nza: {nza}  cosangle: {cosangle}')
        cosangle = (a.shape[2]*a.shape[2]-nza)/(a.shape[2]*a.shape[2])
        print(f'unstable cosangle set to: {cosangle}')
    '''
    return cosangle

class CosEdgeEmbedding(torch.nn.Module):
    '''Measures the loss given inputs x1, x2, and a label tensor y containing values (1 or -1).
    This is used for measuring whether two inputs are similar or dissimilar, using the cosine distance,
    and is typically used for learning nonlinear embeddings or semi-supervised learning.

    Too unstable to use (using edges very sensitive to what to decide a loss value for expty pred==empty mask
    For non edges slightly more robust but gets unstable when mask (and preds) fill a large portion of the image
    '''
    def __init__(self, apply_nonlin='Sigmoid', batch=False, edges=True, debug_text=None):
        super(CosEdgeEmbedding, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.batch=batch
        self.edges=edges
        self.debug_text=debug_text

    def forward(self, pred, gt):
        if self.apply_nonlin:
            activation_fn = get_activation_fn(self.apply_nonlin)
            pred = activation_fn(pred)
        if self.batch:
            raise NotImplementedError('batch functionality not implemented yet')
        else:
            cels=[]
            i=0
            subsample=0.5
            trim_pix=0
            for pi, gi in zip(pred, gt):
                if self.edges:
                    #run the loss on edges of preds and mask
                    #cosine similary ias a poor loss function when preds (and mask) cover most of the image
                    g = gi.detach().cpu().numpy()
                    p = pi.detach().cpu().numpy()
                    p = p.squeeze()
                    g = g.squeeze()
                    #downsample to get thicker edges
                    lr_img_2x = downsample(p, fx=subsample, fy=subsample)
                    lr_m_2x = downsample(g, fx=subsample, fy=subsample)
                    if trim_pix>0:
                        lr_img_2x=crop_center(lr_img_2x, lr_img_2x.shape[0]-2,lr_img_2x.shape[0]-trim_pix)
                        lr_m_2x = crop_center(lr_m_2x, lr_img_2x.shape[0]-2,lr_img_2x.shape[0]-trim_pix)
                    p = binarize_thresh(lr_img_2x, thresh=0.5)
                    g = binarize_thresh(lr_m_2x, thresh=0.5)
                    edges_m = get_edges(g, 2)
                    edges_p = get_edges(p, 2)

                    p = np.where(edges_p > 0.1, 1, edges_p)
                    g = np.where(edges_m > 0.1, 1, edges_m)

                    p=torch.from_numpy(p).to(pred.device).type(torch.float32)
                    g = torch.from_numpy(g).to(pred.device).type(torch.float32)
                    p= p.squeeze().flatten()
                    g = g.squeeze().flatten()
                else:
                    p = pi.squeeze().flatten()
                    g = gi.squeeze().flatten()

                c = ang(p, g)

                if self.debug_text:
                    #for debugging qc images that are very similar
                    if c and c>0.5 and c<1.0:
                        s=int(g.shape[0]**(1/2.0))
                        g = g.view(s, -1).detach().cpu().numpy()
                        plt.imshow(g)
                        plt.savefig(f'images/CosEmbedding_{self.debug_text}_{i}_gt_{c}.png')
                        p = p.view(s, -1).detach().cpu().numpy()
                        plt.imshow(p)
                        plt.savefig(f'images/CosEmbedding_{self.debug_text}_{i}_pred_{c}.png')
                        i+=1
                cels.append(c)
            cel = sum(cels)/len(cels)
            cel=torch.tensor(cel).to(device=pred.device)
        return cel
