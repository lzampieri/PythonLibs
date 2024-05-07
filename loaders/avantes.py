from . import avantes_backend

def load_raw8( filename, normalizeOnIntegrationTime = True ):
    
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