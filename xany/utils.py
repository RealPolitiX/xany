#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def angles(agstart, agend, step, seqrmv=[], to_radians=True):
    """ Generate a set of angles including missing values.
    
    :Parameters:
        agstart, agend, step : numeric, numeric, numeric
            Starting, ending and the interval of angles.
        seqrmv : list | []
            Sequence number of angles to remove.
        to_radians : bool | True
            Option to convert to radians.
    
    :Return:
        ags : 1d array
            Array of angles including missing values
    """
    
    ags = np.arange(agstart, agend, step)
    ags = np.delete(ags, seqrmv)
    
    if to_radians:
        ags = np.radians(ags)
        
    return ags

    
def riffle(*arr):
    """
    Interleave multiple arrays of the same number of elements.
    
    :Parameter:
        *arr : array
            A number of arrays.
    
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
            Array after binning.
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


def dsearchByName(dct, string):
    """ Searching dictionary entry by name.
    
    :Parameters:
        dct : dict
            Dictionary to search for entries.
        string : str
            Specific string to search in dictionary keys.
            
    :Return:
        knames : list
            A list of dictionary keys containing the specific string.
    """
    
    knames = []
    keys = [k for k in dct.keys()]
    for k in keys:
        if string in k:
            knames.append(k)
            
    return knames


def saveImstack(block, form='tiff', axis=0, fdir='./', fstring='', dtype=None, **kwds):
    """
    Output 3d image stack as single 2d images.

    :Parameters:
        block : 3d array
            3D stack of images.
        form : str | 'tiff'
            Format the images are saved into.
        axis : numeric | 0
            Axis index along which the images are stacked.
        fdir, fstring : str, str | './', ''
            Folder and file path string used in saving the image files.
        dtype : str | None
            Data type to convert the image matrices into.
        **kwds : keyword arguments 
    """
    
    if form == 'tiff':
        
        import tifffile as ti
        block = np.rollaxis(block, axis, 0)
        nim = block.shape[0]
        imseq = kwds.pop('seqstr', list(range(nim)))
        
        try:
            digitlen = len(list(np.max(np.abs(np.array(imseq)))))
        except:
            pass
        finally:
            digitlen = 3
        
        for i in range(nim):
            if dtype is not None:
                imslice = block[i,...]
            else:
                imslice = block[i,...].astype(dtype)
            
            ti.imsave(fdir + fstring + str(imseq[i]).zfill(digitlen) + '.tiff', data=imslice, **kwds)
            
    else:
        raise NotImplementedError