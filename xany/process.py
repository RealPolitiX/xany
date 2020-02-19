#! /usr/bin/env python
# -*- coding: utf-8 -*-

from skimage import transform as trf
from tomopy import *


def applyShifts(imgs, shifts=None, interp_order=5, **kwds):
    """
    Apply known shifts to images (e.g. tomograms).

    :Parameters:
        imgs : 3d array/list
            Stacked images in 3d or in a list. If a 3d array is provided, the first
            dimension is treated as the stacking direction.
        shifts : 2d array/list | None
            Pixel position shifts to images along each dimension.
        interp_order : int | 5
            Order of interpolation
        **kwds : keyword arguments
            Additional arguments for ``skimage.transform.warp()``.
    """
    
    # Determine the number of images
    try:
        if imgs.ndim == 3:
            imgs = list(imgs)
    except:
        pass
    
    nimg = len(imgs)
    
    # Determine shifts along each direction
    if shifts is None:
        try:
            xsh = kwds.pop('xshifts')
            ysh = kwds.pop('yshifts')
            shifts = np.stack((xsh, ysh), axis=1)
        except:
            raise ValueError('Need to specify shifts to apply them to images!')
            
    if shifts.shape[0] != nimg:
        raise ValueError('The number of specified shifts differ form the number of images!')
    else:
        for i in range(nimg):
            shift, img = shifts[i], imgs[i]
            tform = trf.SimilarityTransform(translation=(shift[1], shift[0]))
            img = trf.warp(img, tform, order=interp_order, **kwds)
        
    return imgs