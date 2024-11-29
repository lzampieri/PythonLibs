import numpy as np
from plots import adv_plt as plt
import seaborn as sns
import pandas as pd
from uncertainties import *
from uncertainties.unumpy import nominal_values as unp_n, uarray
from math import sqrt
from tqdm.auto import tqdm
from glob import glob
import re
from scipy.signal import savgol_filter

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

def force_ufloat( x ):
    return x if 'uncertainties' in str( type( x ) ) else ufloat( x, 0 )

def between( y, x, xlim, xandy = False, yandx = False ):
    idx = ( x < np.max( xlim ) ) * ( x > np.min( xlim ) )

    if( xandy ):
        return np.array(x)[idx], np.array(y)[idx]

    if( yandx ):
        return np.array(y)[idx], np.array(x)[idx]

    return np.array(y)[idx]

def between_idx( x, xlim ):
    return ( x < np.max( xlim ) ) * ( x > np.min( xlim ) )

def mean( array, exclude_nan = False ):
    if( len( array ) == 1 ):
        return force_ufloat( array[0] )
    if( exclude_nan ):
        array = np.array( array )[ ~ np.isnan( array ) ]
    mean = np.mean( array )
    compatible = True
    for i1, a1 in enumerate( array ):
        for i2, a2 in enumerate( array ):
            if( i1 < i2 ):
                continue
            if( abs( n( a1 ) - n( a2 ) ) > 3 * sqrt( s( a1 )**2 + s( a2 )**2 ) ):
                compatible = False
                break
    if( compatible ):
        std = sqrt( np.sum( s( array )**2 ) / len( array )**2 )
    else:
        std = np.std( n( array ), ddof = 1 )

    return ufloat( n( mean ), std )

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def filt( x, window = 15, degree = 3 ):
    return savgol_filter( x, window, degree )

def ival( i, xs ):
    return np.interp( i, np.arange( len( xs ) ), xs )

def cleanout_significative_digits( unc ):
    prec = 10 ** int( np.floor( np.log10( unc.s ) ) )
    return ufloat( np.round( unc.n / prec ) * prec, np.round( unc.s / prec ) * prec )

def summarize_data( xs, ys ):
    xss = np.unique( xs )
    xsnp = np.array( xs )
    ysnp = np.array( ys )
    yss = [ mean( ysnp[ xsnp == x ] ) for x in xss ]
    return xss, np.array( yss )