import numpy as np
from plots import adv_plt as plt
import seaborn as sns
import pandas as pd
from uncertainties import *
from uncertainties.unumpy import nominal_values as unp_n
from math import sqrt
from tqdm.auto import tqdm
from glob import glob
import re

def setPlotsScale( scale ):
    sns.set_context('paper', font_scale= scale )

@np.vectorize
def n( x ):
    if( isinstance( x, list ) ):
        return [ n( xi ) for xi in x ]
    return x.n if 'uncertainties' in str( type( x ) ) else x

@np.vectorize
def s( x ):
    if( isinstance( x, list ) ):
        return [ s( xi ) for xi in x ]
    return x.s if 'uncertainties' in str( type( x ) ) else 0

def between( y, x, xlim ):
    idx = ( x < np.max( xlim ) ) * ( x > np.min( xlim ) )

    return np.array(y)[idx]

def between_idx( x, xlim ):
    return ( x < np.max( xlim ) ) * ( x > np.min( xlim ) )

def mean( array, exclude_nan = False ):
    if( len( array ) == 1 ):
        return array[0]
    if( exclude_nan ):
        array = np.array( array )[ ~ np.isnan( array ) ]
    mean = np.mean( array )
    compatible = False
    for i1, a1 in enumerate( array ):
        for i2, a2 in enumerate( array ):
            if( i1 < i2 ):
                continue
            if( abs( n( a1 ) - n( a2 ) ) > 3 * sqrt( s( a1 )**2 + s( a2 )**2 ) ):
                compatible = False
                break
    if( compatible ):
        std = mean.s / sqrt( len( array ) )
    else:
        std = np.std( unp_n( array ) )

    return ufloat( n( mean ), std )

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
