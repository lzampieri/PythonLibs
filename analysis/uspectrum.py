from uncertainties import *
from uncertainties.unumpy import uarray, nominal_values as un, std_devs as us
import numpy as np
import matplotlib.pyplot as plt

def spectra_to_uspectrum( dataset, x_column = 'x', y_column = 'y' ):
    assert len( dataset ) > 0, "Dataset is empty!"

    uspectrum = {
        x_column: dataset[0][x_column]
    }

    for d in dataset:
        assert len( d[x_column] ) == len( uspectrum[x_column] ) and ( np.array( d[x_column] ) == np.array( uspectrum[ x_column ] ) ).all(), "Dataset's x axis are not compatible"

    ys = np.stack( [ np.array( d[y_column] ).flatten() for d in dataset ] )

    uspectrum[ y_column ] = uarray(
        np.mean( ys, axis = 0 ),
        np.std( ys, axis = 0 ) / np.sqrt( len( dataset ) )
    )

    return uspectrum



def plot( data, x_column = 'x', y_column = 'y', **plot_params ):
    plt.errorbar(
        data[x_column],
        un( data[y_column] ),
        us( data[y_column] ),
        **plot_params
    )

def plot_xy( xs, ys, **plot_params ):
    plt.errorbar(
        xs,
        un( ys ),
        us( ys ),
        **plot_params
    )