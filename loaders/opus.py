import opusFC
import numpy as np
from glob import glob

# Load data
def read_folder( folder, quiet = False, xfact = 1 ):
    output = {
        'wl': [],
        'A': [],
        'times': [],
        'folder': folder
    }
    items = []

    for f in np.hstack( [ glob( folder + "/*.[0-9]" ), glob( folder + "/*.[0-9][0-9]" ), glob( folder + "/*.[0-9][0-9][0-9]" ) ] ):
        datablocks = opusFC.listContents(f)
        for datablock in datablocks:
            if datablock[0] == 'AB':
                temp = opusFC.getOpusData( str(f), datablock)
                if( len( output['wl'] ) == 0 ):
                    output['wl'] = np.array( temp.x )
                else:
                    assert np.all( output['wl'] == np.array( temp.x ) ), f"Incoerency on spectra wavenumbers! ( {folder} )"
                items.append({
                    'id': int( f[ f.rfind('.')+1: ] ),
                    'A': np.array( temp.y ),
                })

    items.sort( key = lambda i: i['id'] )
    assert np.all( np.arange( len( items ) ) == [ i['id'] for i in items ] ), f"A progressive in missing! ( {folder} )"

    output['times'] = np.array( [ i['id'] * xfact for i in items ] )
    output['A'] = np.stack( [ i['A'] for i in items ] )

    if( not quiet ):
        print( "Loaded", len( items ), " spectra from ", folder )

    return output
