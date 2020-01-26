import numpy as np
import glob as g
import natsort as nts
from skimage.filters.rank import median
from skimage.morphology import disk


def remove_hotpixels(img, rdisk):
    """ Remove hot pixels with a filter
    """
    
    imsc = img.max()
    out = median(img/imsc, disk(rdisk))
    
    return out*imsc