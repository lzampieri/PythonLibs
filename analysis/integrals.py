import numpy as np

def between( y, x, xlim ):
    idx = ( x < np.max( xlim ) ) * ( x > np.min( xlim ) )

    return np.trapz( y[idx], x[idx] )