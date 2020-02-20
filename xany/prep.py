#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import glob as g
import natsort as nts
from skimage.filters.rank import median
from skimage.morphology import disk
import scipy.signal as ss
from scipy import interpolate
from h5py import File


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


def stringFinder(filenames, string, ret='index'):
    """
    Find the file and index among a list of filenames.

    :Parameters:
        filenames : list/tuple
            Collection of filenames.
        string : str
            A key string to look for in a filename.
        ret : str | 'index'
            Options for return values ('index' or 'filename').
    """
    
    if ret == 'index':
        nfiles = len(filenames)
        return [ind for ind in range(nfiles) if string in filenames[ind]]
    
    elif ret == 'filename':
        return [fn for fn in filenames if string in fn]


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


def loadH5Parts(filename, content, alias=None):
    """
    Load specified content from a single complex HDF5 file.
    
    :Parameters:
        filename : str
            Namestring of the file.
        content : list/tuple
            Collection of names for the content to retrieve.
        alias : list/tuple | None
            Collection of aliases to assign to each entry in content in the output dictionary.
    """
    
    with File(filename) as f:
        if alias is None:
            outdict = {k: f[k][:] for k in content}
        else:
            if len(content) != len(alias):
                raise ValueError('Not every content entry is assigned an alias!')
            else:
                outdict = {ka: f[k][:] for k in content for ka in alias}
    
    return outdict

    
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


def remove_patch(img, seqs=None, axis=0, method='to_nan'):
    """ Remove or nanify affected lines or patches in data.
    
    :Parameters:
        img : 2d array
        seqs : list/tuple | None
        axis : int | 0
        method : str | 'to_nan'
    
    :Return:
        imgout : 2d array
            Image with specified patch removed or nanified.
    """
    
    shape = img.shape
    
    if method == 'delete':
        imgout = np.delete(img, seqs, axis)
        
    elif method == 'to_nan':
        imgout = img.copy()
        
        for s in seqs:
            del_coords = np.stack(np.meshgrid([s], np.arange(shape[1-axis])), axis=2).squeeze()
            del_coords = np.roll(del_coords, shift=1*axis, axis=axis)
            imgout[del_coords[:,0], del_coords[:,1]] = np.nan
            
    return imgout


def restore(img, extremes=['inf', 'nan'], upbound=None, lobound=None, debug=False, **kwds):
    """ 
    Restore an image with irregularly distributed extreme values, including infinities, NaNs
    and overly large values specified by an intensity threshhold.
    
    :Parameters:
        img : ndarray
            Multidimensional image data.
        extremes : list | ['inf', 'nan']
            Types of extreme values.
        upbound, lobound : numeric, numeric | None, None
            Upper and lower bounds of the intensity value used to separate normal from abnormal pixels.
        debug : bool | False
            Option to go into debugging mode.
        **kwds : keyword arguments
            Keyword arguments for the interpolation function (``griddata``).
        
    :Return:
        imgcopy : ndarray
            Multidimensional image with restored intensity values.
    """
    
    imgcopy = img.copy()
    
    # Correct infinity values
    if 'inf' in extremes:
        infpos = np.where(np.isinf(imgcopy))
        realpos = np.where(np.invert(np.isinf(imgcopy)))
        
        if debug:
            print(infpos)
        
        if len(infpos[0]) > 0:
            interpval = interpolate.griddata(realpos, imgcopy[realpos], infpos, **kwds)
            imgcopy[infpos] = interpval
    
    # Correct NaN values
    if 'nan' in extremes:
        nanpos = np.where(np.isnan(imgcopy))
        realpos = np.where(np.invert(np.isnan(imgcopy)))
        
        if debug:
            print(nanpos)
            
        if len(nanpos[0]) > 0:
            interpval = interpolate.griddata(realpos, imgcopy[realpos], nanpos, **kwds)
            imgcopy[nanpos] = interpval
    
    # Correct overly large intensity values specified by an upper bound
    if upbound is not None:
        errpos = np.where(imgcopy > upbound)
        realpos = np.where(imgcopy <= upbound)

        if debug:
            print(errpos)
        
        if len(errpos[0]) > 0:
            interpval = interpolate.griddata(realpos, imgcopy[realpos], errpos, **kwds)
            imgcopy[errpos] = interpval

    # Correct overly small intensity values (i.e. near-zero) specified by an lower bound
    if lobound is not None:
        errpos = np.where(imgcopy < lobound)
        realpos = np.where(imgcopy >= lobound)
    
        if debug:
            print(errpos)
        
        if len(errpos[0]) > 0:
            interpval = interpolate.griddata(realpos, imgcopy[realpos], errpos, **kwds)
            imgcopy[errpos] = interpval
    
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
        imgseqclean = restore(imgtmp, extremes=[], upbound=hpub, **kwds)

    elif method == 'deterministic':
        imgseqclean = restore(imgtmp, extremes=[], upbound=hot_pixel_bound, **kwds)

    return imgseqclean


def recursiveCleaning(img, n=1, verbose=False, **kwds):
    """
    Recursively clean an image corrupted by extreme values.
    
    :Parameters:
        img : 2d array
            Image to clean.
        n : int | 1
            Number of iterations.
        verbose : bool | False
            Option to specify if printing out the number of iterations is needed.
        **kwds : keyword arguments
            Extra arguments for ``xany.prep.restore()``.
    
    :Return:
        imgcln : 2d array
            Image after cleaning.
    """
    
    imgcln = img.copy()
    n = int(n)
    nt = 0
    
    while nt < n:
        if verbose:
            print('Cleaning round #' + str(nt) + ' ...')
        imgcln = restore(imgcln, **kwds)
        nt += 1
        
    return imgcln


def fusionScaler(ima, imb, offset=5, axis=1, toscale='a'):
    """
    Global intensity scaling of images along one direction for image fusion.

    :Parameters:
        ima, imb : 2d array, 2d array
            Two images to fuse together.
        offset : int
            Pixel offset along the seam to fuse.
        axis : int | 1
            Axis along which to fuse the images (0 = row, 1 = column).
        toscale : str | 'a'
            Name of the image to apply the global scaling to ('a' or 'b').

    :Return:
        imasc, imbsc : 2d array, 2d array
            Two images after global intensity scaling (ready to fuse).
    """
    
    # Determine the intensity ratio
    if axis == 0:
        asum = ima[-offset:,:].sum()
        bsum = imb[:offset, :].sum()
    elif axis == 1:
        asum = ima[:, -offset:].sum()
        bsum = imb[:, :offset].sum()
    else:
        raise ValueError('Scaling factors can only be calculated along axis 0 or 1.')
        
    # Scale the image according to designation
    if toscale == 'a':
        imasc = ima * (bsum / asum)
        imbsc = imb.copy()
    elif toscale == 'b':
        imasc = ima.copy()
        imbsc = imb * (asum / bsum)
    else:
        raise ValueError('Scaling is designated to one image, a or b.')
        
    return imasc, imbsc


def fillBlock(stack, blocksize=None, mode='constant', constant_values=0, **kwds):
    """
    Combine 2D images with different sizes into a stack and fill the extra space with a constant.
    
    :Parameters:
        stack : list
            List of differently sized images.
        blocksize : list/tuple
            Size of the image in each subsection of the stack.
        mode : str | 'constant'
            Image padding method.
        constant_values : numeric | 0
            Values padded to the images.
            
    :Return:
        block : ndarray
            Padded images combined into a block.
    """
    
    block = []
    
    # Determine the minimum size needed to contain all arrays, estimate if not provided directly
    if blocksize is None:
        shapes = np.array([s.shape for s in stack])
        blkr, blkc = np.max(shapes, axis=0)
    else:
        blkr, blkc = blocksize
    
    for im in stack:
        imr, imc = im.shape
        block.append(np.pad(im, [[blkr-imr, 0], [blkc-imc, 0]], mode=mode, constant_values=constant_values, **kwds))
    
    block = np.asarray(block)
    
    return block