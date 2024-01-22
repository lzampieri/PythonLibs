import numpy as np
import pathlib
import subprocess
from os.path import exists
from os import getcwd
from pathlib import Path
import re
from glob import glob
import pandas as pd

nicolet_convert_script_version = 0.3

# def load( filename ):
#     return read_spa( filename )

# def read_spa_subprocess( filename ):
#     subprocess.call([
#         'python',
#         str( pathlib.Path(__file__).parent.resolve() ) + "\\nicolet_convert_script.py",
#         getcwd() + "\\" + filename,
#         str( nicolet_convert_script_version )
#         ])
#     print('File converted: ' + filename)

# def read_spa( filename ):
#     # return getcwd() + "\\" + filename
#     if( not exists( filename + ".npy" ) ):
#         read_spa_subprocess( filename )
    
#     datafile = np.load( filename + ".npy", allow_pickle=True ).item()
#     if( datafile['version'] != nicolet_convert_script_version ):
#         read_spa_subprocess( filename )
#         datafile = np.load( filename + ".npy", allow_pickle=True ).item()
    
#     return datafile



# Source: https://github.com/lerkoah/spa-on-python/blob/master/LoadSpectrum.py
def read_spa(filepath):
    '''
    Input
    Read a file (string) *.spa
    ----------
    Output
    Return spectra, wavelenght (nm), titles
    '''
    with open(filepath, 'rb') as f:
        f.seek(564)
        Spectrum_Pts = np.fromfile(f, np.int32,1)[0]
        f.seek(30)
        SpectraTitles = np.fromfile(f, np.uint8,255)
        SpectraTitles = ''.join([chr(x) for x in SpectraTitles if x!=0])

        f.seek(576)
        Max_Wavenum=np.fromfile(f, np.single, 1)[0]
        Min_Wavenum=np.fromfile(f, np.single, 1)[0]
        # print(Min_Wavenum, Max_Wavenum, Spectrum_Pts)
        Wavenumbers = np.flip(np.linspace(Min_Wavenum, Max_Wavenum, Spectrum_Pts))

        f.seek(288);

        Flag=0
        while Flag != 3:
            Flag = np.fromfile(f, np.uint16, 1)

        DataPosition=np.fromfile(f,np.uint16, 1)
        f.seek(DataPosition[0])

        Spectra = np.fromfile(f, np.single, Spectrum_Pts)
    return {
        'wl': Wavenumbers,
        'A': Spectra,
        'title': SpectraTitles
    }

def spa_to_csv( file ):
    data = read_spa( file )

    df = pd.DataFrame( data = data['A'], index = data['wl'], columns = ['A'] )
    df.to_csv( file + ".csv" )

    print("Produced file", file + ".csv")

def read_srs(file):
    with open( file, 'rb') as f:
        data = np.fromfile(f, np.single, -1)
    with open( file, 'rb') as f:
        data_asint = np.fromfile(f, np.int32,-1)
    with open( file, 'rb') as f:
        data_aschar = np.fromfile(f, np.uint8,-1)

    # Data positioning
    length = data_asint[ 3509 ]

    if( ( length > 10000 ) or ( length < 100 ) ):
        print( f"File {file} discarded as probably corrupted")
        return read_srs_splitted( file )

    start = 4838 + length
    header = 25
    pos_time = 3

    # Wavelengths
    max_wl = data[ 3512 ]
    min_wl = data[ 3513 ]

    # return data_asint

    # print( max_wl, min_wl, length )

    wl = np.flip( np.linspace( min_wl, max_wl, length ) )
    spectra = []
    times = []

    # Spectra
    while( 1 ):
        new_time = data_asint[ start + pos_time ] / 6000
        # print( new_time )

        if( len( times ) > 0 and new_time < times[-1] ):
            break

        times.append( new_time )

        if( len( times ) == 1 ):
            spectra = data[ start+header: start + header + length]
        else:
            spectra = np.vstack( [ spectra, data[ start+header: start + header + length] ])

        start = start + header + length

        if( start + header + length > len( data ) ):
            break

    # return spectra

    spectra = np.flip( spectra, axis=1 )

    title_length = np.where( data_aschar[14970:] == 0 )[0][0]
    title = ''.join([ chr( x ) for x in data_aschar[ 14970 : 14970 + title_length ] ])

    output = {
        'wl': wl,
        'times': np.array( times ),
        'A': spectra,
        'filename': Path( file ).stem,
        'title': title
    }

    match = re.match( '.*_(\d+)', output['filename'] )
    if( match ):
        output['id'] = match.groups()[0]

    return output

def srs_to_csv( file ):
    data = read_srs( file )

    df = pd.DataFrame( data = data['A'].T, index = data['wl'], columns = data['times'] )
    df.to_csv( file + ".csv" )

    print("Produced file", file + ".csv")

def read_srs_splitted( file ):
    folder = file[:-4]

    spa_files = list( glob( folder + "/*.spa" ) )
    pattern = ".*[/\\\](\d+)\.spa"

    output = {
        'wl': [],
        'times': [],
        'A': [],
        'filename': file,
        'title': file
    }

    for spa_file in spa_files:
        matches = re.match( pattern, spa_file )
        if( not matches ): continue

        id = int( matches.groups()[0] )
        data = read_spa( spa_file )
        output['wl'] = data['wl']
        output['A'].append( list( data['A'] ) )
        output['times'].append( id )

    if( len( output['times'] ) > 0 ):

        output['A'] = np.array( output['A'] )
        output['times'] = np.array( output['times'] )

        print("Successfully read splitted files")
        return output
    
    print("Please provide a deserialized dataset in folder {folder}")