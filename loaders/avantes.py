from . import avantes_backend
import numpy as np

def load_raw8( filename, normalizeOnIntegrationTime = True ):
    """
    Load a raw8 file
    ---
    Parameters:
    filename: name of the file to be readed
    normalizeOnIntegrationTime: if the spectrum should be normalized on the integration time
    ---
    Return:
    file content, in a dictionary:{
        'wl': wavelenghts
        'int_unnormalized': spectrum, as readed from the file
        'int': spectrum, normalized if normalizeOnIntegrationTime is set, otherwise as readed from the file
        'header': content on the header of the file
    }
    """

    S = avantes_backend.Raw8( filename )

    data = {
        'wl': S.getWavelength(),
        'int_unnormalized': S.getScope(),
        'dark_unnormalized': S.getDark(),
        'label': S.header['comment'].decode('ascii')[:-1].rstrip('\x00'),
        'header': S.header
    }

    normalize = lambda d: d / S.header['IntTime'] if normalizeOnIntegrationTime else d

    data['int'] = normalize( data['int_unnormalized'] )

    if( data['header']['measMode'] > 0 ):
        print(fr"Avantes spectrum {filename}: dark substracted")
        data['int_withdark'] = np.copy( data['int'] )
        data['int_withdark_unnormalized'] = np.copy( data['int_unnormalized'] )
        data['dark'] = normalize( data['dark_unnormalized'] )
        data['int'] = data['int_withdark'] - data['dark']

    
    data['dark'] = data['dark_unnormalized'] / S.header['IntTime'] if normalizeOnIntegrationTime else data['dark_unnormalized']

    if( normalizeOnIntegrationTime ):
        print(fr"Avantes spectrum {filename} normalized by a factor of {S.header['IntTime']:.0f} μs")

    return data