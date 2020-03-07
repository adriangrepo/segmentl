import numpy as np
import cv2

from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt

def compute_edts_image(posmask, boundary_fill):
    """
    posmask.shape = (x,y,(z))
    for binary segmentation
    """
    eps=1e-7 # not smaller as may be using half precision
    non_zero_count = np.count_nonzero(posmask)
    # if there is no positive mask in the gt, we want a zero distance map
    if non_zero_count > 0:
        pos_edt = distance_transform_edt(posmask)
        pos_edt = (np.max(pos_edt) - pos_edt) * posmask
        if boundary_fill:
            res = pos_edt / (np.max(pos_edt) + eps)
        else:
            negmask = ~posmask
            neg_edt = distance_transform_edt(negmask)
            neg_edt = (np.max(neg_edt) - neg_edt) * negmask
            res = pos_edt / (np.max(pos_edt) + eps) + neg_edt / (np.max(neg_edt) + eps)
    else:
        res = np.zeros(posmask.shape)
    return res

def compute_edts_batch(gt, boundary_fill=False):
    """
    gt.shape = (batch_size, x,y,(z))
    for binary segmentation
    """
    gts = np.squeeze(gt)
    res = np.zeros(gt.shape)
    for i in range(gt.shape[0]):
        posmask = gts[i]
        res[i] = compute_edts_image(i, posmask, res, boundary_fill=boundary_fill)
    return res

def compute_distances(gt, boundary_fill=False):
    """
    gt.shape = (batch_size, x,y,(z))
    for binary segmentation
    """
    gts = np.squeeze(gt)
    res = np.zeros(gt.shape)
    eps=1e-7 # not smaller as may be using half precision
    for i in range(gt.shape[0]):
        posmask = gts[i]
        non_zero_count=np.count_nonzero(posmask)
        #if there is no positive mask in the gt, we want a zero distance map
        if non_zero_count>0:
            pos_edt = distance_transform_edt(posmask)
            pos_edt = (np.max(pos_edt) - pos_edt) * posmask
            if boundary_fill:
                res[i] = pos_edt / (np.max(pos_edt) + eps)
            else:
                negmask = ~posmask
                neg_edt = distance_transform_edt(negmask)
                neg_edt = (np.max(neg_edt) - neg_edt) * negmask
                res[i] = pos_edt / (np.max(pos_edt)+eps) + neg_edt / (np.max(neg_edt)+eps)
        else:
            res[i] = np.zeros(posmask.shape)
    return res

def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

def downsample(img, fx=0.5, fy=0.5):
    '''after https://github.com/wqi/img-downsampler/blob/master/downsample.py'''''
    lr_img = cv2.resize(img, (0, 0), fx=fx, fy=fy,
                           interpolation=cv2.INTER_AREA)
    return lr_img

def upsample(img, fx=2, fy=2):
    '''after https://stackoverflow.com/questions/4195453/how-to-resize-an-image-with-opencv2-0-and-python2-6'''''
    hr_img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    return hr_img

def binary_mask(img, mask_im, thresh=127):
    im=img.copy()
    mask = mask_im >= thresh
    im[~mask] = 0
    return im, mask

def binarize_thresh(img, thresh=127):
    im = img >= thresh
    return im

def get_edges(mask, connectivity=2):
    # rank 2 structure with full connectivity
    struct = ndimage.generate_binary_structure(2, connectivity)
    #print(f'struct: {struct.shape}, mask: {mask.shape}')
    #print(f'mask: {mask}')
    #print(f'struct: {struct}')
    erode = ndimage.binary_erosion(mask, struct)
    edges = mask ^ erode
    return edges

def combine_edge(A, mask, edges):
    # the indices of the non-zero locations and their corresponding values
    nonzero_idx = np.vstack(np.where(mask)).T
    nonzero_vals = A[mask]

    # build a k-D tree
    tree = cKDTree(nonzero_idx)

    # use it to find the indices of all non-zero values that are at most 1 pixel
    # away from each edge pixel
    edge_idx = np.vstack(np.where(edges)).T
    neighbours = tree.query_ball_point(edge_idx, r=1, p=np.inf)

    # take the average value for each set of neighbours
    new_vals = np.hstack(np.mean(nonzero_vals[n]) for n in neighbours)

    # use these to replace the values of the edge pixels
    A_new = A.astype(np.double, copy=True)
    A_new[edges] = new_vals
    return A_new