import numpy as np
import uuid
from scipy.ndimage import distance_transform_edt
import joblib
from pycocotools import mask as cocomask
from scipy import ndimage as ndi
from segmentl.distance.utils import compute_edts_image, generate_contours
import cv2




def update_distances(dist, mask):
    '''
    :param dist:
    :param mask:
    :return: distances:
    after https://github.com/neptune-ai/open-solution-mapping-challenge
    '''
    if dist.sum() == 0:
        distances = distance_transform_edt(1 - mask)
    else:
        distances = np.dstack([dist, distance_transform_edt(1 - mask)])
    return distances

def clean_distances(distances, n_closest=2):
    '''
    :param distances: H x W x masked_objects where masked_objects can be stacks
    :param n_closest: number of closest objects to consider for distance calcs, use 0 for all objects
    :return:
    #modified after https://github.com/neptune-ai/open-solution-mapping-challenge
    '''
    if n_closest==0:
        #use all
        if len(distances.shape) < 3:
            n_closest=2
        else:
            n_closest=distances.shape[2]

    clean_distances=None
    if len(distances.shape) < 3:
        #Stack arrays in sequence depth wise (along third axis)
        distances = np.dstack([distances, distances])
    else:
        # --clean_distances distances.sort(axis=2)
        distances.sort(axis=2)
        # two closest objects
        distances = distances[:, :, :n_closest]
    second_nearest_distances = distances[:, :, 1]
    clean_distances = np.sum(distances, axis=2)
    return clean_distances, second_nearest_distances

def is_on_border(mask, border_width):
    # after https://github.com/neptune-ai/open-solution-mapping-challenge
    return not np.any(mask[border_width:-border_width, border_width:-border_width])

def label(mask):
    # after https://github.com/neptune-ai/open-solution-mapping-challenge
    labeled, nr_true = ndi.label(mask)
    return labeled

def get_size_matrix(mask):
    #after https://github.com/neptune-ai/open-solution-mapping-challenge
    sizes = np.ones_like(mask)
    labeled = label(mask)
    for label_nr in range(1, labeled.max() + 1):
        label_size = (labeled == label_nr).sum()
        sizes = np.where(labeled == label_nr, label_size, sizes)
    return sizes

def overlay_masks(gt, image_size, distances=None):
    #modified after https://github.com/neptune-ai/open-solution-mapping-challenge
    mask = np.zeros(image_size)
    # eg  len(annotations): 14, distances: (300, 300)
    #rle = cocomask.frPyObjects(ann['segmentation'], image_size[0], image_size[1])
    simplified_contours = generate_contours(gt)
    for sc in simplified_contours:
        sc=[sc.flatten().tolist()]
        try:
            rle = cocomask.frPyObjects(sc, image_size[0], image_size[1])
        except TypeError as e:
            print(f'--overlay_masks() casting sc to numpy array, type: {type(sc)}, type: {type(sc[0])}, sc: {sc}')
            sc = np.array(sc)
            rle = cocomask.frPyObjects(sc, image_size[0], image_size[1])
        m = cocomask.decode(rle)

        for i in range(m.shape[-1]):
            mi = m[:, :, i]
            mi = mi.reshape(image_size)

            if is_on_border(mi, 2):
                continue
            if distances is not None:
                distances = update_distances(distances, mi)
            mask += mi
    bin_mask= np.where(mask > 0, 1, 0).astype('uint8')
    return bin_mask, distances

def overlay_mask_one_image(image_size, gt, border_width=0, n_distances=2):
    '''
    :param image_size: [height,width]
    :param gt: single image ground truth mask
    :param border_width:
    :return:
    modified after https://github.com/neptune-ai/open-solution-mapping-challenge
    '''

    mask_overlayed = np.zeros(image_size).astype('uint8')
    distances = np.zeros(image_size)

    mask, distances = overlay_masks(gt, image_size=image_size, distances=distances)
    sizes = get_size_matrix(mask)
    nearest_distances, second_nearest_distances = clean_distances(distances, n_distances)
    if np.count_nonzero(gt)>1:
        distances = distance_transform_edt(1 - gt)
    else:
        # distance edt weights distances from zero if mask is empty, so make all distances zero
        distances=gt.copy()

    if border_width > 0:
        borders = (second_nearest_distances < border_width) & (~mask_overlayed)
        borders_class_id = mask_overlayed.max() + 1
        mask_overlayed = np.where(borders, borders_class_id, mask_overlayed)

    return mask_overlayed, distances, nearest_distances, second_nearest_distances, sizes

def overlay_mask_batch(image_size, gt, border_width=0, n_distances=2):
    # after https://github.com/neptune-ai/open-solution-mapping-challenge
    marrays=[]
    darrays = []
    ndarrays = []
    sndarrays = []
    sarrays = []
    for m in gt:
        m = m.detach().cpu().numpy()
        m=m.squeeze()
        mo, d, nd, snd, s = overlay_mask_one_image(image_size, m, border_width, n_distances)
        darrays.append(d)
        #transpose these matrices to match original mask
        marrays.append(mo.T)
        ndarrays.append(nd.T)
        sndarrays.append(snd.T)
        sarrays.append(s.T)

    #np.stack joins a sequence of arrays along a new axis.
    ma=np.stack(marrays, axis=0)
    da = np.stack(darrays, axis=0)
    nda = np.stack(ndarrays, axis=0)
    snda = np.stack(sndarrays, axis=0)
    sa = np.stack(sarrays, axis=0)
    return ma, da, nda, snda, sa