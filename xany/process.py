#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import numpy.fft as nfft
import scipy.ndimage as ndi
from skimage import transform as trf
import raster_geometry as rg
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


def sphere(rad, outsize, method='algebraic', anti_aliasing=5, smoothing=True):
    """ Generate a spherical mask (outside 0, inside 1).

    :Parameters:
        rad : numeric
            Radius of the sphere.
        outsize : list/tuple
            Total size of the volume.
        method : str | 'algebraic'
            Method for generating the sphere ('algebraic' or 'raster').
        anti_aliasing : int | 5
            Anti-aliasing parameter used for smoothing the boundary.
        smoothing : bool | True
            Option to smooth the boundary of the sphere.
    """

    outsize = np.array(outsize)

    if method == 'algebraic':
        nx, ny, nz = outsize
        rad = int(rad)
        x = np.linspace(-rad, rad, 2*rad, endpoint=True)
        y = np.linspace(-rad, rad, 2*rad, endpoint=True)
        z = np.linspace(-rad, rad, 2*rad, endpoint=True)
        xx, yy, zz = np.meshgrid(x, y, z)
        a = (xx**2 + yy**2 + zz**2 <= rad**2).astype('float')
        
        vol = np.zeros(outsize)
        vol[int(nx/2-rad):int(nx/2+rad),
            int(ny/2-rad):int(ny/2+rad),
            int(nz/2-rad):int(nz/2+rad)] = a

    elif method == 'raster':
        vol = rg.sphere(outsize, rad)
    
    if smoothing:
        vol = ndi.gaussian_filter(vol, 0.5 * anti_aliasing)
    
    return vol


def shell(rad, outsize, anti_aliasing=5):
    """
    Generate a 3D shell mask using the difference between two spheres.

    :Parameters:
        rad : numeric
            Radius of the shell.
        outsize : list/tuple
            Total size of the volume.
        anti_aliasing : int | 5
            See ``xany.process.sphere()``.
    """

    outsphere = sphere(rad + 0.5, outsize, anti_aliasing=anti_aliasing)
    insphere = sphere(rad - 0.5, outsize, anti_aliasing=anti_aliasing)
    
    return outsphere - insphere


def calculateFSC(vol, ref, step=1):
    """
    Calculation of the Fourier shell correlation (FSC) function of an image volume.
    """

    rad_max = int(min(vol.shape) / 2)
    f_vol = nfft.fftshift(nfft.fftn(vol))
    f_ref = nfft.fftshift(nfft.fftn(ref))
    f_prod = f_vol * np.conjugate(f_ref)
    
    f_vol_2 = np.real(f_vol * np.conjugate(f_vol))
    f_ref_2 = np.real(f_ref * np.conjugate(f_ref))
    rad_ls = np.arange(1, rad_max, step)
    fsc_ls = []

    for rad in rad_ls:
        mask = shell(rad, vol.shape, anti_aliasing=2)
        fsc = abs(np.sum(f_prod * mask))
        fsc /= np.sqrt(np.sum(f_vol_2 * mask) * np.sum(f_ref_2 * mask))
        fsc_ls.append(fsc)

    return np.array(fsc_ls)