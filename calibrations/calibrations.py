import numpy as np
import re
from datetime import datetime
import pickle
from os.path import dirname, exists

def save(name, values, note = None ):
    name = name.lower()
    assert re.match( "^[a-z0-9_]+$", name ), "Only letters, numbers and underscores are allowed"

    filename = "c" + name + ".calib"

    content = {
        'name': name,
        'values': np.array( values ),
        'saved_on': datetime.now()
    }
    if( note ):
        content['note'] = note

    with open( dirname( __file__ ) + "/" + filename, 'wb') as file:
        pickle.dump( content, file )

    print( "Saved! -", filename )

def calibrate(name, data ):
    name = name.lower()
    assert re.match( "^[a-z0-9_]+$", name ), "Only letters, numbers and underscores are allowed"

    filename = "c" + name + ".calib"
    assert exists( dirname( __file__ ) + "/" + filename ), "File " + filename + " not found!"

    with open( dirname( __file__ ) + "/" + filename, 'rb') as file:
        pars = pickle.load( file )

    print( "Calibrating using file " + filename + " generated on " + pars['saved_on'].strftime("%m/%d/%Y, %H:%M:%S") )
    if( 'note' in pars.keys() ):
        print( "Note: ", pars['note'] )

    return np.polyval( pars['values'], data )