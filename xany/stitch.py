#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from symmetrize import sym


def imextend(img, xtrans=0, ytrans=0, **kwds):
    """ Extend and shift an image.
    """
    
    xshift = kwds.pop('xshift', xtrans)
    yshift = kwds.pop('yshift', ytrans)
    
    hmat = sym.translation2D(xtrans, ytrans)
    oshape = tuple(np.array(img.T.shape) + np.abs(np.array([xshift, yshift])))
    imtrans = sym.imgWarping(img, hgmat=hmat, outshape=oshape)[0]
    
    return imtrans

    
def impairshift(ima, imb, ret='separate', **kwds):
    """ Shifting an image pair in a specified direction.
    
    :Parameters:
        ima, imb : numpy array, numpy array
            Image pair to be shifted.
        ret : str | 'separate'
            Return options ('separate' or 'combined').
        xsh, ysh : numeric, numeric
            Positive values mean they are moving apart, negative values mean
    """
    
    ashape, bshape = ima.shape, imb.shape
    xsh = kwds.pop('xsh', int(np.max([ashape[1], bshape[1]])))
    ysh = kwds.pop('ysh', int(np.max([ashape[0], bshape[0]])))
    
    imash = imextend(ima, xtrans=0, ytrans=0, xshift=xsh-ashape[1], yshift=ysh-ashape[0])
    imbsh = imextend(imb, xtrans=bshape[1]-xsh, ytrans=bshape[0]-ysh)    
    
    if ret == 'separate':
        return imash, imbsh
    
    elif ret == 'combined':
        xscal = np.linspace(1, 0, xsh, endpoint=True) # Boundary blending
        imash[:, ashape[1]-xsh:ashape[1]] *= xscal
        imbsh[:, ashape[1]-xsh:ashape[1]] *= xscal[::-1]
        imabsh = imash + imbsh
        return imabsh