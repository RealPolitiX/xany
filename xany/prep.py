#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import glob as g
import natsort as nts
from skimage.filters.rank import median
from skimage.morphology import disk
import scipy.signal as ss


def findFiles(fdir, fstring='', ftype='h5', **kwds):
    """
    Retrieve files named in a similar way from a folder.
    
    :Parameters:
        fdir : str
            Folder name where the files are stored.
        fstring : str | ''
            Extra string in the filename.
        ftype : str | 'h5'
            Type of the files to retrieve.
        **kwds : keyword arguments
            Extra keywords for `natsorted()`.
    """
    
    files = nts.natsorted(g.glob(fdir + fstring + '.' + ftype), **kwds)
    
    return files

    
def f_multichannel(data, f, ch_index=0, ch_range=[None, None], **kwds):
    """
    Repeated data processing for multiple channels.
    
    :Parameters:
        data : ndarray
            N-dimensional array data.
        f : function
            Function to be mapped to the particular axis.
        ch_index : int | 0
            Index of the channel the function is applied to.
        ch_range : list | [None, None]
            Lower and upper bounds of the range for channel selection
        **kwds : keyword arguments
            Additional arguments passed to the function f.
    """
    
    mdata = np.rollaxis(data, ch_index)
    nch = mdata.shape[0]
    rlo, rhi = ch_range
    chs = range(nch)[rlo:rhi]
    
    mdata = np.asarray([f(mdata[ch,...], **kwds) for ch in chs])
    mdata = np.rollaxis(mdata, ch_index)
    
    return mdata


def remove_hotpixels(img, func='sk-median', **kwds):
    """ Remove hot pixels with a filter.
    """
    
    imsc = img.max()
    
    if func == 'sk-median':
        mask = kwds.pop('mask', disk(1))
        out = median(img/imsc, mask=mask, **kwds)
    elif func == 'ss-medfilt2d':
        ksize = kwds.pop('kernel_size', 3)
        out = ss.medfilt2d(img/imsc, kernel_size=ksize)
    
    return out*imsc