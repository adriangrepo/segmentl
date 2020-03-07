import numpy as np

def precision(result, reference):
    """
    Precison.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    precision : float
        The precision between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of retrieved instances that are relevant. The
        precision is not symmetric.

    See also
    --------
    :func:`recall`

    Notes
    -----
    Not symmetric. The inverse of the precision is :func:`recall`.
    High precision means that an algorithm returned substantially more relevant results than irrelevant.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Precision_and_recall
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    after: https://github.com/doublechenching/brats_segmentation-pytorch/blob/master/utils/metric/binary.py
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        precision = tp / float(tp + fp)
    except ZeroDivisionError:
        precision = 0.0

    return precision


def recall(result, reference):
    """
    Recall.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    recall : float
        The recall between two binary datasets, here mostly binary objects in images,
        which is defined as the fraction of relevant instances that are retrieved. The
        recall is not symmetric.

    See also
    --------
    :func:`precision`

    Notes
    -----
    Not symmetric. The inverse of the recall is :func:`precision`.
    High recall means that an algorithm returned most of the relevant results.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Precision_and_recall
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tp = np.count_nonzero(result & reference)
    fn = np.count_nonzero(~result & reference)

    try:
        recall = tp / float(tp + fn)
    except ZeroDivisionError:
        recall = 0.0

    return recall


def sensitivity(result, reference):
    """
    Sensitivity.
    Same as :func:`recall`, see there for a detailed description.

    See also
    --------
    :func:`specificity`
    after: https://github.com/doublechenching/brats_segmentation-pytorch/blob/master/utils/metric/binary.py
    """
    return recall(result, reference)


def specificity(result, reference):
    """
    Specificity.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.

    Returns
    -------
    specificity : float
        The specificity between two binary datasets, here mostly binary objects in images,
        which denotes the fraction of correctly returned negatives. The
        specificity is not symmetric.

    See also
    --------
    :func:`sensitivity`

    Notes
    -----
    Not symmetric. The completment of the specificity is :func:`sensitivity`.
    High recall means that an algorithm returned most of the irrelevant results.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Sensitivity_and_specificity
    .. [2] http://en.wikipedia.org/wiki/Confusion_matrix#Table_of_confusion
    after: https://github.com/doublechenching/brats_segmentation-pytorch/blob/master/utils/metric/binary.py
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    tn = np.count_nonzero(~result & ~reference)
    fp = np.count_nonzero(result & ~reference)

    try:
        specificity = tn / float(tn + fp)
    except ZeroDivisionError:
        specificity = 0.0

    return specificity


def true_negative_rate(result, reference):
    """
    True negative rate.
    Same as :func:`specificity`, see there for a detailed description.

    See also
    --------
    :func:`true_positive_rate`
    :func:`positive_predictive_value`
    after: https://github.com/doublechenching/brats_segmentation-pytorch/blob/master/utils/metric/binary.py
    """
    return specificity(result, reference)


def true_positive_rate(result, reference):
    """
    True positive rate.
    Same as :func:`recall` and :func:`sensitivity`, see there for a detailed description.

    See also
    --------
    :func:`positive_predictive_value`
    :func:`true_negative_rate`
    after: https://github.com/doublechenching/brats_segmentation-pytorch/blob/master/utils/metric/binary.py
    """
    return recall(result, reference)


def positive_predictive_value(result, reference):
    """
    Positive predictive value.
    Same as :func:`precision`, see there for a detailed description.

    See also
    --------
    :func:`true_positive_rate`
    :func:`true_negative_rate`
    after: https://github.com/doublechenching/brats_segmentation-pytorch/blob/master/utils/metric/binary.py
    """
    return precision(result, reference)