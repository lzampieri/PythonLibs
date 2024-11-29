import numpy as np
from pathlib import Path
import re
from glob import glob
import pandas as pd

# Inspired by: https://github.com/Thorlabs/Light_Analysis_Examples/blob/51550408a935d70011864f1d0fdcf9b963e1e099/Python/Thorlabs%20OSA/pyOSA/spectrum_t.py#L3
# A script to read spectra from a thorlabs portable spectrometer,
# i.e. acquired using ThorSpectra software and saved as .spf2

def read_spf2(file, normalizeOnIntegrationTime = True ):

    with open(file, 'rb') as f:
        f.seek(24)
        Spectrum_Pts = np.fromfile(f, np.int32,1)[0]

        f.seek(0)
        data_asfloat = np.fromfile(f, np.single)

        start_wl = data_asfloat[11]
        end_wl = data_asfloat[12]
        
        intensity = data_asfloat[- Spectrum_Pts :]

        f.seek(908)
        model = "".join( [ chr(x) if x != 0 else "" for x in np.fromfile(f, np.uint8,24) ] )
        
        f.seek(900)
        intime = np.fromfile(f, np.double,1)[0]

        wl = np.linspace( start_wl, end_wl, Spectrum_Pts, endpoint=True )
        if( len( data_asfloat ) > 2 * Spectrum_Pts ):
            if( data_asfloat[ - Spectrum_Pts * 2 ] == start_wl and data_asfloat[ - Spectrum_Pts - 1 ] == end_wl ):
                wl = data_asfloat[ - Spectrum_Pts * 2 : - Spectrum_Pts ]
       
    output = {
        'wl': wl,
        'int_unnormalized': intensity,
        'filename': Path( file ).stem,
        'spectrometer': model,
        'integrationTime_ms': intime
    }
    
    normalize = lambda d: d / intime / 1e3 if normalizeOnIntegrationTime else d
    output['int'] = normalize( output['int_unnormalized'] )

    if( normalizeOnIntegrationTime ):
        print(fr"ThorSpectra spectrum {file} normalized by a factor of {intime:.3f} ms")
    
    return output