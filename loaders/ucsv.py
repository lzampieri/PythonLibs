from uncertainties import *
from uncertainties.unumpy import uarray
import numpy as np
from os.path import dirname
import pandas as pd

def load( filename ):
    if( filename.endswith( ".xlsx" ) ):
        data = pd.read_excel( filename ).to_dict( 'list' )
    else:
        if( not filename.endswith( ".ucsv" ) ):
            filename += ".ucsv"

        data = pd.read_csv( filename, index_col=False, encoding='utf-8' ).to_dict( 'list' )

    all_keys = list(data.keys())

    for k in all_keys:
        data[k] = np.array( data[k] )

    for k in all_keys:
        if( k.endswith('_n') and (k[:-1] + "s" in all_keys) ):
            data[ k[:-2] ] = uarray( data[ k ], data[ k[:-1] + "s" ] )
            data.pop( k )
            data.pop( k[:-1] + "s" )
    
    print( filename + " loaded!" )

    return data


def save(filename, arrays_dict):
    if( filename.endswith( ".ucsv" ) ):
        filename = filename[:-5]

    # Ensure arrays_dict is dict
    try:
        _ = arrays_dict.items()
    except (AttributeError, TypeError):
        arrays_dict = {'values': arrays_dict}

    splitted_dict = {}
    for k in arrays_dict.keys():
        # Use first row to check if is uncertainty
        if 'uncertainties' in str(type(arrays_dict[k][0])):
            splitted_dict[k + "_n"] = [i.n for i in arrays_dict[k]]
            splitted_dict[k + "_s"] = [i.s for i in arrays_dict[k]]
        else:
            splitted_dict[k] = arrays_dict[k]

    pd.DataFrame.from_dict( splitted_dict ).to_csv( filename + ".ucsv", index=False, encoding='utf-8' )

    print( filename + ".ucsv saved!" ) 