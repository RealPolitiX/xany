#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def riffle(*arr):
    """
    Interleave multiple arrays of the same number of elements.
    :Parameter:
        *arr : array
            A number of arrays
    :Return:
        riffarr : 1D array
            An array with interleaving elements from each input array.
    """

    arr = (map(np.ravel, arr))
    arrlen = np.array(map(len, arr))

    try:
        unique_length = np.unique(arrlen).item()
        riffarr = np.vstack(arr).reshape((-1,), order='F')
        return riffarr
    except:
        raise ValueError('Input arrays need to have the same number of elements!')


def arraybin(arr, bins, method='mean'):
    """
    Resize an nD array by binning.
    :Parameters:
        arr : nD array
            N-dimensional array for binning.
        bins : list/tuple of int
            Bins/Size shrinkage along every axis.
        method : str
            Method for binning, 'mean' or 'sum'.
    :Return:
        arrbinned : nD array
    """

    bins = np.asarray(bins)
    nb = len(bins)

    if nb != arr.ndim:
        raise ValueError('Need to specify bins for all dimensions, use 1 for the dimensions not to be resized.')
    else:
        shape = np.asarray(arr.shape)
        binnedshape = shape // bins
        binnedshape = binnedshape.astype('int')

        # Calculate intermediate array shape
        shape_tuple = tuple(riffle(binnedshape, bins))
        # Calculate binning axis
        bin_axis_tuple = tuple(range(1, 2*nb+1, 2))

        if method == 'mean':
            arrbinned = arr.reshape(shape_tuple).mean(axis=bin_axis_tuple)
        elif method == 'sum':
            arrbinned = arr.reshape(shape_tuple).sum(axis=bin_axis_tuple)

        return arrbinned