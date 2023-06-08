from spectrochempy import read_omnic
import numpy as np
import sys
from pathlib import Path
import re

filename = sys.argv[1]

spc_data = read_omnic( filename )

data = {
    'wl': spc_data.x.data,
    'A': np.squeeze( spc_data.data ),
    'filename': Path( filename ).stem,
    'version': float( sys.argv[2] )
}

match = re.match( '.*_(\d+)', data['filename'] )
if( match ):
    data['id'] = match.groups()[0]

np.save( filename, data )