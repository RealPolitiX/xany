#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from symmetrize import sym
from tqdm import notebook as nb


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


def stack_combine(stacka, stackb, axis=0, zsh=0, pbar=True, **kwds):
    """ Combining two stacks using alignment cues.
    
    :Parameters:
        stacka, stackb : numpy array, numpy array
            Image stacks (3D) to combine.
        axis : int | 0
            Axis of stacking.
        zsh : int | 0
            Shift of `stackb` along the stacking axis.
    """
    
    dmerg = np.min([stacka.shape[axis], stackb.shape[axis]]) # Determine the number of along the axis
    stackA, stackB = np.moveaxis(stacka, axis, 0), np.moveaxis(stackb, axis, 0)
    
    stackm = []
    for i in nb.tqdm(range(dmerg), disable=not(pbar)):
        abmerge = impairshift(stackA[nslice,...], stackB[nslice+zsh,...], ret='combined', **kwds)
        stackm.append(abmerge.tolist())
        
    stackm = np.asarray(stackm)
    
    return stackm