import numpy as np
import matplotlib.pyplot as plt
from uncertainties import *

def unp_n( x ):
    return np.array( [ xx.n for xx in x ] if 'uncertainties' in str( type( x[0] ) ) else x )
def unp_s( x ):
    return np.array( [ xx.s for xx in x ] if 'uncertainties' in str( type( x[0] ) ) else x )

def sorted( x, y ):
    idx = np.argsort( x )
    return np.array(x)[idx], np.array(y)[idx]

def normalized( y ):
    return np.array( y ) / np.max( y )

def plot_with_optional_fmt( x, y, fmt = None, **plot_info ):
    if( fmt ):
        plt.plot( x, y, fmt, **plot_info )
        return
    plt.plot( x, y, **plot_info )

def unp_plot( x, y = [], as_area = False, avoid_errors = False, xy_sorted = False, **plot_info ):

    if( len( x ) == 0 and len( y ) == 0 ):
        return plt.plot( [], [], **plot_info )    
    
    if( len( y ) == 0 ):
        y = x
        x = list( range( len( y ) ) )

    if( xy_sorted ):
        x, y = sorted( x, y )

    if( as_area ):
        return unp_plot_area( x, y, **plot_info )
    
    is_y = 'uncertainties' in str( type( y[0] ) )
    is_x = 'uncertainties' in str( type( x[0] ) )

    if( avoid_errors ):
        plt.plot( unp_n(x), unp_n(y), **plot_info )
        return


    if( is_x and is_y ):
        plt.errorbar( unp_n(x), unp_n(y), unp_s(y), unp_s(x), **plot_info )
        return
    if( is_y ):
        plt.errorbar( x, unp_n(y), unp_s(y), **plot_info )
        return
    if( is_x ):
        plt.errorbar( unp_n(x), y, np.zeros_like(y), unp_s(x), **plot_info )
        return
    else:
        plot_with_optional_fmt( x, y, **plot_info )


def unp_plot_area( x, y, **plot_info ):
    is_y = 'uncertainties' in str( type( y[0] ) )
    is_x = 'uncertainties' in str( type( x[0] ) )

    x_n = unp_n( x ) if is_x else x
    y_n = unp_n( y ) if is_y else y
    y_s = unp_s( y ) if is_y else np.zeros_like( y )

    if( 'color' not in plot_info ):
        plot_info['color'] = next( plt.gca()._get_lines.prop_cycler )['color']

    plot_with_optional_fmt( x_n, y_n, **plot_info )
    plt.fill_between( x_n, y_n - y_s, y_n + y_s, color=plot_info['color'], alpha=0.2, label=None )