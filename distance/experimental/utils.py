import numpy as np
from scipy.ndimage import distance_transform_edt as distance

def pad(input, size=None, footprint=None, output=None, mode="reflect", cval=0.0):
    """
    Returns a copy of the input, padded by the supplied structuring element.

    In the case of odd dimensionality, the structure element will be centered as
    following on the currently processed position::

        [[T, Tx, T],
         [T, T , T]]

    , where Tx denotes the center of the structure element.
    Simulates the behaviour of scipy.ndimage filters.
    Parameters
    ----------
    input : array_like
        Input array to pad.
    size : scalar or tuple, optional
        See footprint, below
    footprint : array, optional
        Either `size` or `footprint` must be defined. `size` gives
        the shape that is taken from the input array, at every element
        position, to define the input to the filter function.
        `footprint` is a boolean array that specifies (implicitly) a
        shape, but also which of the elements within this shape will get
        passed to the filter function. Thus ``size=(n,m)`` is equivalent
        to ``footprint=np.ones((n,m))``. We adjust `size` to the number
        of dimensions of the input array, so that, if the input array is
        shape (10,10,10), and `size` is 2, then the actual size used is
        (2,2,2).
    output : array, optional
        The `output` parameter passes an array in which to store the
        filter output.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the array borders are
        handled, where `cval` is the value when mode is equal to
        'constant'. Default is 'reflect'.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'. Default
        is 0.0

    Returns
    -------
    output : ndarray
        The padded version of the input image.

    Notes
    -----
    Since version 1.7.0, numpy supplied a pad function `np.pad` that provides
    the same functionality and should be preferred.

    Raises
    ------
    ValueError
        If the provided footprint/size is more than double the image size.

    # !TODO: Utilise the np.pad function that is available since 1.7.0.
    #  The numpy version should go inside this function, since it does not support the supplying of a template/footprint on its own.
    after: https://github.com/doublechenching/brats_segmentation-pytorch/blob/master/utils/metric/utils.py
    """
    input = np.asarray(input)
    if footprint is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _ni_support._normalize_sequence(size, input.ndim)
        footprint = np.ones(sizes, dtype=bool)
    else:
        footprint = np.asarray(footprint, dtype=bool)
    fshape = [ii for ii in footprint.shape if ii > 0]
    if len(fshape) != input.ndim:
        raise RuntimeError('filter footprint array has incorrect shape.')

    if np.any([x > 2 * y for x, y in zip(footprint.shape, input.shape)]):
        raise ValueError(
            'The size of the padding element is not allowed to be more than double the size of the input array in any dimension.')

    padding_offset = [((s - 1) / 2, s / 2) for s in fshape]
    input_slicer = [slice(l, None if 0 == r else -1 * r) for l, r in padding_offset]
    output_shape = [s + sum(os) for s, os in zip(input.shape, padding_offset)]
    output = _ni_support._get_output(output, input, output_shape)

    if 'constant' == mode:
        output += cval
        output[input_slicer] = input
        return output
    elif 'nearest' == mode:
        output[input_slicer] = input
        dim_mult_slices = [(d, l, slice(None, l), slice(l, l + 1)) for d, (l, _) in
                           zip(list(range(output.ndim)), padding_offset) if not 0 == l]
        dim_mult_slices.extend([(d, r, slice(-1 * r, None), slice(-2 * r, -2 * r + 1)) for d, (_, r) in
                                zip(list(range(output.ndim)), padding_offset) if not 0 == r])
        for dim, mult, to_slice, from_slice in dim_mult_slices:
            slicer_to = [to_slice if d == dim else slice(None) for d in range(output.ndim)]
            slicer_from = [from_slice if d == dim else slice(None) for d in range(output.ndim)]
            if not 0 == mult:
                output[slicer_to] = np.concatenate([output[slicer_from]] * mult, dim)
        return output
    elif 'mirror' == mode:
        dim_slices = [(d, slice(None, l), slice(l + 1, 2 * l + 1)) for d, (l, _) in
                      zip(list(range(output.ndim)), padding_offset) if not 0 == l]
        dim_slices.extend([(d, slice(-1 * r, None), slice(-2 * r - 1, -1 * r - 1)) for d, (_, r) in
                           zip(list(range(output.ndim)), padding_offset) if not 0 == r])
        reverse_slice = slice(None, None, -1)
    elif 'reflect' == mode:
        dim_slices = [(d, slice(None, l), slice(l, 2 * l)) for d, (l, _) in
                      zip(list(range(output.ndim)), padding_offset) if not 0 == l]
        dim_slices.extend([(d, slice(-1 * r, None), slice(-2 * r, -1 * r)) for d, (_, r) in
                           zip(list(range(output.ndim)), padding_offset) if not 0 == r])
        reverse_slice = slice(None, None, -1)
    elif 'wrap' == mode:
        dim_slices = [(d, slice(None, l), slice(-1 * (l + r), -1 * r if not 0 == r else None)) for d, (l, r) in
                      zip(list(range(output.ndim)), padding_offset) if not 0 == l]
        dim_slices.extend(
            [(d, slice(-1 * r, None), slice(l, r + l)) for d, (l, r) in zip(list(range(output.ndim)), padding_offset) if
             not 0 == r])
        reverse_slice = slice(None)
    else:
        raise RuntimeError('boundary mode not supported')

    output[input_slicer] = input
    for dim, to_slice, from_slice in dim_slices:
        slicer_reverse = [reverse_slice if d == dim else slice(None) for d in range(output.ndim)]
        slicer_to = [to_slice if d == dim else slice(None) for d in range(output.ndim)]
        slicer_from = [from_slice if d == dim else slice(None) for d in range(output.ndim)]
        output[slicer_to] = output[slicer_from][slicer_reverse]

    return output


def __make_footprint(input, size, footprint):
    '''
    Creates a standard footprint element ala scipy.ndimage.
    after: https://github.com/doublechenching/brats_segmentation-pytorch/blob/master/utils/metric/utils.py
    '''
    if footprint is None:
        if size is None:
            raise RuntimeError("no footprint or filter size provided")
        sizes = _ni_support._normalize_sequence(size, input.ndim)
        footprint = np.ones(sizes, dtype=bool)
    else:
        footprint = np.asarray(footprint, dtype=bool)
    return footprint

def uniq(a: Tensor) -> Set:
    # after https://github.com/LIVIAETS/surface-loss/blob/master/utils.py
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    #after https://github.com/LIVIAETS/surface-loss/blob/master/utils.py
    return uniq(a).issubset(sub)

def simplex(t: Tensor, axis=1) -> bool:
    # after https://github.com/LIVIAETS/surface-loss/blob/master/utils.py
    #sum over columns
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: Tensor, axis=1) -> bool:
    # after https://github.com/LIVIAETS/surface-loss/blob/master/utils.py
    is_simplex= simplex(t, axis)
    is_sset = sset(t, [0, 1])
    return is_simplex and is_sset

def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    # after https://github.com/LIVIAETS/surface-loss/blob/master/utils.py
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def class2one_hot(seg: Tensor, C: int) -> Tensor:
    # after https://github.com/LIVIAETS/surface-loss/blob/master/utils.py
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    if len(seg.shape) == 4:
        seg = seg.squeeze()
    if C==1:
        assert sset(seg, [0,1])
    else:
        assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)
    return res