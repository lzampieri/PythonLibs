from . import avantes_backend


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
        'label': S.header['comment'].decode('ascii')[:-1].rstrip('\x00'),
        'header': S.header
    }

    data['int'] = data['int_unnormalized'] / S.header['IntTime'] if normalizeOnIntegrationTime else data['int_unnormalized']

    if( normalizeOnIntegrationTime ):
        print(fr"Avantes spectrum {filename} normalized by a factor of {S.header['IntTime']:.0f} μs")

    return data