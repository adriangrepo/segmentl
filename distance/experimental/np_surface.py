import torch
from torch import nn
import numpy as np
from scipy import ndimage
from scipy.ndimage import morphology, distance_transform_edt
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,\
    generate_binary_structure
from scipy.ndimage.measurements import label, find_objects
from scipy.stats import pearsonr

from segmentl.distribution.distribution_based import CrossentropyND, TopKLoss

def surf(input1, input2, sampling, connectivity):
    '''
    Code is too slow to be feasible for use

    Surface distance metrics estimate the error between the outer surfaces S and S′ of the segmentations X and X′.
    The distance between a point p on surface S and the surface S′ is given by the minimum of the Euclidean norm:
        d(p,S′)=min(p′∈S′)||p−p′||2

    Doing this for all pixels in the surface gives the total surface distance between S and S′:
        d(S,S′)

    after https://mlnotebook.github.io/post/surface-distance-function/

    input1: the segmentation that has been created. It can be a multi-class segmentation, but this function will make the image binary.
    input2: the GT segmentation against which we wish to compare input1
    sampling: the pixel resolution or pixel size. This is entered as an n-vector where n is equal to the
    number of dimensions in the segmentation i.e. 2D or 3D.
    The default value is 1 which means pixels (or rather voxels) are 1 x 1 x 1 mm in size.
    connectivity: creates either a 2D (3 x 3) or 3D (3 x 3 x 3) matrix defining the neighbourhood around which
    the function looks for neighbouring pixels.
    Typically, this is defined as a six-neighbour kernel which is the default behaviour of this function.
    '''

    # Check size and make binary, any value greater than zero is made 1 (true).
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))

    # Create the kernel that will be used to detect the edges of the segmentations
    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    # Strip the outermost pixel from the edge of the segmentation.
    S = ((input_1.astype(np.float32)) - morphology.binary_erosion(input_1, conn).astype(np.float32)).astype(np.bool)
    # Subtract result from the segmentation itself to get the single-pixel-wide surface.
    Sprime = (input_2.astype(np.float32) - morphology.binary_erosion(input_2, conn).astype(np.float32)).astype(np.bool)

    # Give the distance_transform_edt function our pixel-size (sampling) and also the inverted surface-image.
    # The inversion is used such that the surface itself is given the value of zero i.e. any pixel at this location,
    # will have zero surface-distance.
    # The transform increases the value/error/penalty of the remaining pixels with increasing distance away from the surface.
    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    # Each pixel of the opposite segmentation-surface is then laid upon this ‘map’ of penalties and both results are
    # concatenated into a vector which is as long as the number of pixels in the surface of each segmentation.
    # This vector of surface distances is returned.
    # Note that this is technically the symmetric surface distance as we are not assuming that just doing this for
    # one of the surfaces is enough.
    # It may be that the distance between a pixel in A and in B is not the same as between the pixel in B and in A.
    # i.e. d(S,S′)≠d(S′,S)
    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return sds

class SurfaceDistanceMeanLoss(nn.Module):
    '''
    see https://mlnotebook.github.io/post/surface-distance-function/
    Variations:
    Mean Surface Distance (MSD) - the mean of the vector is taken.
    This tell us how much, on average, the surface varies between the segmentation and the GT.

    Residual Mean Square Distance (RMS) - the mean is taken from each of the points in the vector,
    these residuals are squared (to remove negative signs), summated, weighted by the mean and then the square-root is taken.

    Hausdorff Distance (HD) - the maximum of the vector.
    The largest difference between the surface distances.
    '''
    def __init__(self, sampling=1, connectivity=1, precision='half'):
        super(SurfaceDistanceMeanLoss, self).__init__()
        self.sampling = sampling
        self.connectivity = connectivity
        self.precision = precision

    def forward(self, pred, gt):
        sds = surf(pred.detach().cpu().numpy(), gt.detach().cpu().numpy(), self.sampling, self.connectivity)
        sdm=np.array(sds.mean())
        sdm=torch.from_numpy(sdm).to(pred.device)

        if self.precision=='half':
            sdm=sdm.half()

        return sdm

class SurfaceDistanceRMSLoss(nn.Module):
    def __init__(self, sampling=1, connectivity=1, precision='half'):
        super(SurfaceDistanceRMSLoss, self).__init__()
        self.sampling = sampling
        self.connectivity = connectivity
        self.precision = precision

    def forward(self, pred, gt):
        sds = surf(pred.detach().cpu().numpy(), gt.detach().cpu().numpy(), self.sampling, self.connectivity)
        rms=np.sqrt((sds ** 2).mean())
        rms = np.array(rms)
        rms=torch.from_numpy(rms).to(pred.device)

        if self.precision=='half':
            rms=rms.half()

        return rms

class SurfaceDistanceHausdorffLoss(nn.Module):
    def __init__(self, sampling=1, connectivity=1, precision='half'):
        super(SurfaceDistanceHausdorffLoss, self).__init__()
        self.sampling = sampling
        self.connectivity = connectivity
        self.precision = precision

    def forward(self, pred, gt):
        sds = surf(pred.detach().cpu().numpy(), gt.detach().cpu().numpy(), self.sampling, self.connectivity)
        hd=sds.max()
        hd = np.array(hd)
        hd=torch.from_numpy(hd).to(pred.device)

        if self.precision=='half':
            hd=hd.half()

        return hd