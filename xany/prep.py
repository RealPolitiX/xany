#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import glob as g
import natsort as nts
from skimage.filters.rank import median
from skimage.morphology import disk
import scipy.signal as ss
from scipy import interpolate


def findFiles(fdir, fstring='', ftype='h5', **kwds):
    """
    Retrieve files named in a similar way from a folder.
    
    :Parameters:
        fdir : str
            Folder name where the files are stored.
        fstring : str | ''
            Extra string in the filename.
        ftype : str | 'h5'
            The type of files to retrieve.
        **kwds : keyword arguments
            Extra keywords for `natsorted()`.
    """
    
    files = nts.natsorted(g.glob(fdir + fstring + '.' + ftype), **kwds)
    
    return files


def orderFiles(files, seqnum, nzf=0, verbose=False):
    """ 
    Order files according to a sequence string in the filename.
    
    :Parameters:
        files : list
            List of filenames.
        seqnum : list
            List of sequence numbers used to order file.
        nzf : int | 0
            Number of digits to fill to convert every element in the sequence to a string.
        verbose : bool | False
            Option to output checks for monitoring if all files are retrieved.
            
    :Return:
        fordered : list
            List of filenames as a subset of the original files ordered by the sequence number.
    """
    
    seqnum = np.array(seqnum, dtype='int')
    nseq = len(seqnum)
    
    seqstr = [str(num).zfill(nzf) for num in seqnum]
    fordered = [f for s in seqstr for f in files if s in f]
    nford = len(fordered)
    
    if verbose:
        if nseq > nford:
            print('A total of {} files are missing!'.format(nseq-nford))
        elif nseq == nford:
            print('All files are retrieved!')
    
    return fordered

    
def f_multichannel(data, f, ch_index=0, ch_range=[None, None], **kwds):
    """
    Repeated data processing for multiple channels.
    
    :Parameters:
        data : ndarray
            Multidimensional array data.
        f : function
            Function to be mapped to the particular axis.
        ch_index : int | 0
            Index of the channel the function is applied to.
        ch_range : list | [None, None]
            Lower and upper bounds of the range for channel selection.
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


def restore(img, extremes=['inf', 'nan'], upbound=None, debug=False, **kwds):
    """ 
    Restore an image with irregularly distributed extreme values, including infinities, NaNs
    and overly large values specified by an intensity threshhold.
    
    :Parameters:
        img : ndarray
            Multidimensional image data.
        extremes : list | ['inf', 'nan']
            Types of extreme values.
        upbound : numeric | None
            Upper bound of intensity value.
        debug : bool | False
            Option to go into debugging mode.
        **kwds : keyword arguments
            Keyword arguments for the interpolation function (``griddata``).
        
    :Return:
        imgcopy : ndarray
            Multidimensional image with restored intensity values.
    """
    
    imgcopy = img.copy()
    
    # Correct for infinity values
    if 'inf' in extremes:
        infpos = np.where(np.isinf(imgcopy))
        realpos = np.where(np.invert(np.isinf(imgcopy)))
        
        if debug:
            print(infpos)
        
        if len(infpos[0]) > 0:
            interpval = interpolate.griddata(realpos, imgcopy[realpos], infpos, **kwds)
            imgcopy[infpos] = interpval
    
    # Correct for NaN values
    if 'nan' in extremes:
        nanpos = np.where(np.isnan(imgcopy))
        realpos = np.where(np.invert(np.isnan(imgcopy)))
        
        if debug:
            print(nanpos)
            
        if len(nanpos[0]) > 0:
            interpval = interpolate.griddata(realpos, imgcopy[realpos], nanpos, **kwds)
            imgcopy[nanpos] = interpval
    
    # Correct for overly large intensity values specified by an upper bound
    if upbound is not None:
        uppos = np.where(imgcopy > upbound)
        realpos = np.where(imgcopy <= upbound)
        
        if debug:
            print(uppos)
        
        if len(uppos[0]) > 0:
            interpval = interpolate.griddata(realpos, imgcopy[realpos], uppos, **kwds)
            imgcopy[uppos] = interpval
    
    return imgcopy


def sequentialCleaning(img, method='deterministic', hot_pixel_bound=None, pct=99, **kwds):
    """
    Sequential cleaning of nonphysical values (extremes and hot pixels) from an image.
    """

    # Clean up the extreme values
    imgtmp = restore(img, extremes=['inf', 'nan'], upbound=None, **kwds)

    # Clean up the additional hot pixels
    if method == 'estimated':
        if hot_pixel_bound is None:
            hpub = np.percentile(imgtmp.ravel(), pct)
        imgseqclean = restore(imgtmp, extremes=None, upbound=hpub, **kwds)

    elif method == 'deterministic':
        imgseqclean = restore(imgtmp, extremes=[], upbound=hot_pixel_bound, **kwds)

    return imgseqclean