from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from uncertainties import unumpy as unp
import numpy as np
from matplotlib import cm
from math import ceil
import os

_plot_params = 0
_plot_tred = False
_plot_ca = None
_export_folder = "exports"
_factor = 1

def init( numrows = 1, numcols = 1, init = False, size = (6,4), tred = False ):
    plt.figure( figsize = ( numcols * size[0] / _factor, numrows * size[1] / _factor  ) )
    global _plot_params
    global _plot_tred
    _plot_params = [ int( numrows ), int( numcols ), 0 ]
    _plot_tred = tred
    if( init or ( numrows == 1 and numcols == 1 ) ):
        next()
def init_bytot( numtot, cols = 4, **args ):
    return init( ceil( numtot/cols), cols, **args )
def next():
    global _plot_params
    global _plot_tred
    global _plot_ca
    if( _plot_params[0] * _plot_params[1] == 1 and _plot_params[2] == 1 ):
        pass
    else:
        _plot_params[2] += 1
    if( _plot_tred ):
        _plot_ca = plt.subplot( *_plot_params, projection="3d" )
    else:
        _plot_ca = plt.subplot( *_plot_params )
    return _plot_ca
def surface_plot( x, y, z ):
    global _plot_ca
    _plot_ca.plot_surface( *np.meshgrid( x, y ), z, cmap=cm.jet, linewidth=0, antialiased=False )
def goto( i_row, i_col ):
    global _plot_params
    _plot_params[2] = i_col + i_row * _plot_params[1] - 1
    next()
def hline( y, format = ':k' ):
    xl = plt.xlim()
    plt.plot( xl, [y, y], format )
    plt.xlim( xl )
def vline( x, format = ':k' ):
    yl = plt.ylim()
    plt.plot( [x, x], yl, format )
    plt.ylim( yl )
def lacking( text="" ):
    plt.plot( [0, 1], [0, 1], 'r', lw=5)
    plt.plot( [0, 1], [1, 0], 'r', lw=5)
    plt.xlim( 0, 1 )
    plt.ylim( 0, 1 )
    plt.xticks([])
    plt.yticks([])
    plt.annotate( "Lacking", [0.5, 0.8], ha='center', fontsize=40)
    plt.annotate( "plot!", [0.5, 0.2], ha='center', fontsize=40)
    plt.annotate( text, [0.5, 0], ha='center', fontsize=15)
    plt.box(False)
def set_export_folder( folder ):
    global _export_folder
    _export_folder = folder
def set_scale_factor( new_factor ):
    global _factor
    _factor = new_factor
def export( filename, draft = False ):
    global _export_folder

    if not os.path.exists( _export_folder ):
       os.makedirs( _export_folder )

    if( draft ):
        plt.annotate( "DRAFT", [ 0.5, 0.7 ], ha='center', va='center', color='red', weight='bold', fontsize=40, xycoords= 'figure fraction' )
        plt.annotate( "All panels should be considered draft", [ 0.5, 0.5 ], ha='center', va='center', color='red', fontsize=20, xycoords= 'figure fraction' )

    plt.savefig( _export_folder + "/" + filename + ".png", bbox_inches="tight", dpi=600 )
    plt.savefig( _export_folder + "/" + filename + ".pdf", bbox_inches="tight" )

    print(_export_folder + "/" + filename + ".pdf")