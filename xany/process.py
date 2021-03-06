#! /usr/bin/env python
# -*- coding: utf-8 -*-

from . import utils as u, prep as pp
import numpy as np
import numpy.fft as nfft
import scipy.ndimage as ndi
from skimage import transform as trf
import raster_geometry as rg
from silx.io import dictdump
import scipy.io as sio
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
    """
    Generate a spherical mask (outside 0, inside 1).

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


class ImageStack(object):
    """ Class for processing a series of projection images in a tomography experiment.
    """
    
    def __init__(self, images=None, stack=None, axis=0):
        
        self.images = images
        self.stack = stack
        self.axis = axis
        
    def locate_files(self, fdir, fstring='', ftype='tiff', seqnum=None, **kwds):
        """ Locate and order tomogram files.
        """
        
        self.files = pp.findFiles(fdir=fdir, fstring=fstring, ftype=ftype)
        if seqnum is not None:
            self.ordfiles = pp.orderFiles(self.files, seqnum=seqnum, **kwds)
        else:
            self.ordfiles = list(self.files)
        
    def load_stack(self, fsource, ftype='tiff', selector=None, **kwds):
        """ Load tomograms one by one.
        """
        
        self.tomograms = []
        
        if ftype == 'tiff':
            
            import tifffile as ti
            for f in fsource:
                self.stack.append(ti.imread(f, **kwds))
                
        elif ftype == 'h5':
            
            for f in fsource:
                self.stack.append(pp.loadH5Parts(f, **kwds)[0][selector])     
            
        else:
            
            raise NotImplementedError


class TomoReconstructor(ImageStack):
    """
    Class for streamlined workflow of tomographic reconstruction including files preparation.
    """
    
    def __init__(self, tgrams=None, tblock=None, axis=0, angles=[]):

        super().__init__(images=tgrams, stack=tblock, axis=axis)
        
        self.tomograms = self.images
        self.tomoblock = self.stack
        self.angles = []
            
    def blocking(self, **kwds):
        """ Adjust the tomograms to have the same shape.
        """
        
        self.tomoblock = pp.fillBlock(**kwds)
        
    def generate_angles(self, **kwds):
        """ Generate angles associated with each tomogram.
        """
        
        self.angles = u.angles(**kwds)
        
    def delete_angles(self, **kwds):
        """ Delete a part of angles from the existing list.
        """
        
        self.angles = np.delete(self.angles, **kwds)
        
    @property
    def nangles(self):
        """ Number of angles.
        """
        
        return len(self.angles)
    
    @property
    def ngrams(self):
        """ Number of tomograms.
        """
        
        return self.tomoblock.shape[0]
        
    def delete_tomograms(self, ids, in_place=True, ret=False, **kwds):
        """ Remove one or more tomograms from an existing sequence (along with the paired angles).
        """
                
        temp = np.delete(self.tomograms, obj=ids, **kwds)
        try:
            tempa = np.delete(self.angles, obj=ids, **kwds)
        except:
            tempa = []
            
        if in_place:
            self.tomograms = temp
            self.angles = tempa
        
        if ret:
            return temp
        else:
            del temp
    
    def intensity_scale(self, axis=(1, 2), in_place=True, ret=False):
        """ Scale the intensity of tomograms along a specified axis.
        """
        
        temp = self.tomoblock / self.tomoblock.mean(axis=axis)

        if in_place:
            self.tomoblock = temp

        if ret:
            return temp
        else:
            del temp
        
    def pad_tomogram(self, in_place=True, ret=False, **kwds):
        """ Padding tomograms in various direction.
        """

        temp = []
        for itg, tg in enumerate(self.tomograms):
            temp.append(np.pad(tg, **kwds))
        
            if in_place:
                self.tomograms[itg] = temp[itg]
            
        if ret:
            return temp
        else:
            del temp
    
    def reconstruct(self, use_accelerated=False, ret=False, **kwds):
        """ Tomographic reconstruction using the algorithms within ``tomopy``.
        """
        
        if use_accelerated:
            self.recout = recon_accelerated(**kwds)
        else:
            self.recout = recon(**kwds)
        
        if ret:
            return self.recout
        
    def save_recon(self, save_addr, ftype='tiff', **kwds):
        """ Save the reconstruction output to file.
        """
        
        if ftype == 'tiff':
            import tifffile as ti
            ti.imsave(save_addr + '.' + ftype, data=self.recout, **kwds)

        else:
            raise NotImplementedError
    
    def save_internals(self, save_addr, form='h5'):
        """ Save the internal variables of the class
        """

        save_addr = save_addr + '.' + form

        if form == 'mat':
            sio.savemat(save_addr, self.__dict__)

        elif form in ('h5', 'hdf5'):
            try:
                dictdump.dicttoh5(self.__dict__, save_addr)
            except:
                import deepdish.io as dio
                dio.save(save_addr, self.__dict__, compression=None)


def ImtoVTK(imdir, form='tiff', outdir='./', **kwargs):
    """ Convert image files to VTK files.
    """
    
    # Load the image into array
    if form in ['tif', 'tiff']:
        import tifffile as ti
        imarray = ti.imread(imdir)
    
    typ = kwargs.pop('dtype', '')
    if typ:
        imarray = imarray.astype(typ)
    
    # imshape = imarray.shape
    pointVar = kwargs.pop('pointVar', 'image')
    cellVar = kwargs.pop('cellVar', 'spacing')
    pData = kwargs.pop('pointData', imarray)
    cData = kwargs.pop('cellData', np.array([]))
    
    import pyevtk.hl as hl
    if cData:
        hl.imageToVTK(outdir, cellData={cellVar:cData}, pointData={pointVar:pData})
    else:
        hl.imageToVTK(outdir, cellData=None, pointData={pointVar:pData})