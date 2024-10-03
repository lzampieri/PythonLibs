import pandas as pd
from . import rspro, nicolet, opus, gc, images, evt, avantes, lecroy, metex, EPR, thorspectra
from glob import glob
import re
from os.path import exists, dirname
from standard_imports import *
from uncertainties import *
from uncertainties.unumpy import uarray

default_toload = [ 'rspro','nicolet_SPA','nicolet_SRS','opus','gc','gc_series','jpg','evt','avantes','thorspectra','lecroy','metex', 'teledyne', 'EPR', 'mycsv']

def load_index( folder, filename = "index.xlsx" ):
    dataset = pd.read_excel( folder + "/" + filename ).rename( columns = { 'id': 'ID', 'Id': 'ID' } )

    # Support for ufloat
    all_cols = dataset.columns
    for c in all_cols:
        if( c[-2:] == '_n' and c[:-2] + '_s' in all_cols ):
            dataset[ c[:-2] ] = uarray( dataset[c], dataset[ c[:-2] + '_s' ] )
            dataset.drop( columns = [ c, c[:-2] + '_s'], inplace= True )
            print( f"Cols {c} and {c[:-2] + '_s'} merged in col. {c[:-2]}")

    return dataset.to_dict('records')

def write_index( data, folder, columns, filename = "index.xlsx" ):
    if( len( data ) == 0 ):
        pd.DataFrame.from_records( [] ).to_excel( folder + '/' + index_filename, index=False )
        return

    # Support for ufloat
    normal_columns = []
    unc_columns = []
    for c in columns:
        assert c in data[0].keys(), f"Column {c} not found in data"
        if( 'uncertainties' in str( type( data[0][c] ) ) ):
            unc_columns.append( c )
        else:
            normal_columns.append( c )

    index_towrite = []

    for d in data:
        index_towrite.append({
            k: ( d[k] if k in d.keys() else "" ) for k in normal_columns
        })
        for unc_col in unc_columns:
            index_towrite[-1][unc_col + '_n'] = n( d[unc_col] )
            index_towrite[-1][unc_col + '_s'] = s( d[unc_col] )

    pd.DataFrame.from_records( index_towrite ).to_excel( folder + '/' + filename, index=False )

def load_data( dataset, folder, load_all = False, to_load = default_toload ):
    ids = [ d['ID'] for d in dataset ]

    # RSPRO
    if( 'rspro' in to_load ):

        pattern = ".*DSO(\d+)\.CSV"
        for file in glob( folder + "/*" ):

            matches = re.match( pattern, file )
            if( not matches ): continue

            id = int( matches.groups()[0] )
            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            dataset[ ids.index( id ) ][ 'rspro' ] = rspro.load( file )

    # Nicolet SPA
    if( 'nicolet_SPA' in to_load ):

        pattern = ".*[_/\\\\](\d+)\.SPA"
        pattern2= ".*[_/\\\\](\d+)_[^/\\\\]*\.SPA"
        for file in glob( folder + "/*" ):

            matches = re.match( pattern, file )
            if( not matches ): matches = re.match( pattern2, file )
            if( not matches ): continue
            if( "ackground" in file ): continue

            id = int( matches.groups()[0] )
            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            dataset[ ids.index( id ) ][ 'nicolet_SPA' ] = nicolet.read_spa( file )

    # Nicolet SRS
    if( 'nicolet_SRS' in to_load ):

        pattern = ".*[_/\\\\](\d+)\.srs$"
        pattern2= ".*[_/\\\\](\d+)_[^/\\\\]*\.srs$"
        for file in glob( folder + "/*" ):

            matches = re.match( pattern, file )
            if( not matches ): matches = re.match( pattern2, file )
            if( not matches ): continue

            id = int( matches.groups()[0] )
            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            dataset[ ids.index( id ) ][ 'nicolet_SRS' ] = nicolet.read_srs( file )


    # Lausanne FTIR (OPUS)
    if( 'opus' in to_load ):

        pattern = ".*[_/\\\\](\d+)_[^/\\\\]+[/\\\\](.*\.0)"
        pattern2= ".*[_/\\\\](\d+)[/\\\\](.*\.0)"
        # pattern = ".*[^/\\\\][/\\\\].*\.0"
        for subfile in glob( folder + "/**/*.0" ):

            matches = re.match( pattern, subfile )
            if( not matches ): matches = re.match( pattern2, subfile )
            if( not matches ): continue

            id = int( matches.groups()[0] )

            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            subfolder = subfile.replace( matches.groups()[1], '' )[:-1]

            dataset[ ids.index( id ) ][ 'opus' ] = opus.read_folder( subfolder, xfact = ( dataset[ ids.index( id ) ][ 'xfact' ] if 'xfact' in dataset[ ids.index( id ) ] else 1 ) )

    # GC AXY
    if( 'gc' in to_load ):

        pattern = ".*[_/\\\\](\d+)\.AXY"
        pattern2= ".*[_/\\\\](\d+)_[^/\\\\]*\.AXY"
        for file in glob( folder + "/*" ):

            matches = re.match( pattern, file )
            if( not matches ): matches = re.match( pattern2, file )
            if( not matches ): continue

            id = int( matches.groups()[0] )
            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            dataset[ ids.index( id ) ][ 'gc' ] = gc.read_axy( file )

    # GC AXY series
    gc_series_toimport = {}
    if( 'gc_series' in to_load ):

        pattern = ".*[_/\\\\](\d+)[_/\\\\][^/\\\\]*_\d+\.AXY"
        pattern2= ".*[_/\\\\](\d+)_[^/\\\\]*[_/\\\\][^/\\\\]*_\d+\.AXY"
        for file in glob( folder + "/*/*" ):

            matches = re.match( pattern, file )
            if( not matches ): matches = re.match( pattern2, file )
            if( not matches ): continue

            id = int( matches.groups()[0] )
            if( id not in ids ):
                if( not load_all ):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            gc_series_toimport[ id ] = dirname( file  )
    if( len( gc_series_toimport ) > 0 ):
        for id, path in gc_series_toimport.items():
            dataset[ ids.index( id ) ][ 'gc_series' ] = gc.read_axy_series( path )

    # JPG images
    if( 'jpg' in to_load ):

        pattern = ".*[_/\\\\](\d+)\.jpg"
        pattern2= ".*[_/\\\\](\d+)_[^/\\\\]*\.jpg"
        for file in glob( folder + "/*" ):

            matches = re.match( pattern, file )
            if( not matches ): matches = re.match( pattern2, file )
            if( not matches ): continue

            id = int( matches.groups()[0] )
            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            dataset[ ids.index( id ) ][ 'jpg' ] = images.read_jpg( file )

    # Events indices
    if( 'evt' in to_load ):

        pattern = ".*[_/\\\\](\d+)\.evt.xlsx"
        pattern2= ".*[_/\\\\](\d+)_[^/\\\\]*\.evt.xlsx"
        for file in glob( folder + "/*" ):

            matches = re.match( pattern, file )
            if( not matches ): matches = re.match( pattern2, file )
            if( not matches ): continue

            id = int( matches.groups()[0] )
            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            dataset[ ids.index( id ) ][ 'evt' ] = evt.read_evt( file )

    # Spectroscopy avantes
    if( 'avantes' in to_load ):

        pattern = ".*[_/\\\\](\d+)\.R[aAwW][wWdD]8$"
        pattern2= ".*[_/\\\\](\d+)_[^/\\\\]*\.R[aAwW][wWdD]8$"
        pattern3= ".*[_/\\\\]New Experiment(\d+)\.R[aAwW][wWdD]8$"

        for file in glob( folder + "/*" ):

            matches = re.match( pattern, file )
            if( not matches ): matches = re.match( pattern2, file )
            if( not matches ): matches = re.match( pattern3, file )
            if( not matches ): continue

            id = int( matches.groups()[0] )
            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            dataset[ ids.index( id ) ][ 'avantes' ] = avantes.load_raw8( file )
            dataset[ ids.index( id ) ][ 'evt' ] = evt.read_evt( file )

    # Spectroscopy thorspectra (thorlabs)
    if( 'thorspectra' in to_load ):

        pattern = ".*[_/\\\\](\d+)\.spf2"
        pattern2= ".*[_/\\\\](\d+)_[^/\\\\]*\.spf2"

        for file in glob( folder + "/*" ):

            matches = re.match( pattern, file )
            if( not matches ): matches = re.match( pattern2, file )
            if( not matches ): continue

            id = int( matches.groups()[0] )
            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            dataset[ ids.index( id ) ][ 'thorspectra' ] = thorspectra.read_spf2( file )

    # Lecroy waveforms
    if( 'lecroy' in to_load ):

        pattern = ".*[/\\\\]S[CD](.)(\d+).TXT"

        for file in glob( folder + "/*" ):

            matches = re.match( pattern, file )
            if( not matches ): continue

            id = int( matches.groups()[1] )
            channel = matches.groups()[0]
            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            if( 'lecroy' not in dataset[ ids.index( id ) ].keys() ):
                dataset[ ids.index( id ) ][ 'lecroy' ] = {}
            dataset[ ids.index( id ) ][ 'lecroy' ][ channel ] = lecroy.load( file )

    # Metex data
    if( 'metex' in to_load ):

        pattern = ".*[/\\\\]M(\d+)\.metex\.csv"

        for file in glob( folder + "/*" ):
            matches = re.match( pattern, file )
            if( not matches ): continue

            id = int( matches.groups()[0] )
            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            dataset[ ids.index( id ) ][ 'metex' ] = metex.load( file )

    # Lecroy waveforms
    if( 'teledyne' in to_load ):

        pattern = ".*[/\\\\]C(\d+)-[^/\\\\]+-(\d+).csv"

        for file in glob( folder + "/*" ):

            matches = re.match( pattern, file )
            if( not matches ): continue

            id = int( matches.groups()[1] )
            channel = int( matches.groups()[0] )
            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            if( 'teledyne' not in dataset[ ids.index( id ) ].keys() ):
                dataset[ ids.index( id ) ][ 'teledyne' ] = {}
            dataset[ ids.index( id ) ][ 'teledyne' ][ channel ] = lecroy.load( file )

    # EPR raw data
    if( 'EPR' in to_load ):

        pattern = ".*[/\\\\](\d+)([^/\\\\\d][^/\\\\]*)?.EPR"

        for file in glob( folder + "/*" ):

            matches = re.match( pattern, file )
            if( not matches ): continue

            id = int( matches.groups()[0] )
            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            dataset[ ids.index( id ) ][ 'EPR' ] = EPR.load( file )

    # Custom made csv, expected to be clear and plain
    if( 'mycsv' in to_load ):

        pattern = ".*[/\\\\]MC_(\d+)(_[^/\\\\]*)?.my.csv"

        for file in glob( folder + "/*" ):
            # print(file)

            matches = re.match( pattern, file )
            if( not matches ): continue

            id = int( matches.groups()[0] )
            if( id not in ids ):
                if( not load_all):
                    continue
                dataset.append( { 'ID': id } )
                ids.append( id )

            data = pd.read_csv( file, index_col=None).to_dict('list')
            dataset[ ids.index( id ) ][ 'mycsv' ] = { k: np.array( v ) for k, v in data.items() }


    return dataset

def load_index_and_data( folder, index_filename = "index.xlsx", load_all = False, update_index = False, to_load = default_toload, load_only = [] ):
    if( exists( folder + "/" + index_filename ) ):
        index = load_index( folder, index_filename )
    else:
        if( load_all or update_index ):
            index = []
        else:
            raise FileNotFoundError(f"File {folder}/{index_filename} not found!")

    if( load_only and len( load_only ) > 0 ):
        index = list( filter( lambda d: d['ID'] in load_only, index ) )

    prevs_length = len( index )
    prevs_keys = list( index[0].keys() ) if prevs_length > 0 else [ 'ID' ]

    load_data(
        index,
        folder,
        load_all,
        to_load
    )

    if( update_index and len( index ) > prevs_length ):
        write_index( index, folder, prevs_keys, index_filename )
        print(f"Index uploaded, { len( index ) - prevs_length } rows added, total of { len( index ) } rows loaded")
    else:
        print(f"{ len( index ) } rows loaded")

    return index

def add_cols_to_index( data, folder, cols_to_add = [], force_update = False, index_filename = "index.xlsx", verbose = False ):
    old_index = load_index( folder, index_filename,  )

    assert len( data ) == len( old_index ), f"The dataset provided have a length different from the actual index! {len( data )} vs {len( old_index )} "

    keys = list( old_index[0].keys() )
    to_update = force_update

    for k in cols_to_add:
        if( k not in keys and k in data[0].keys() ):
            keys.append( k )
            to_update = True
    
    if( not to_update ):
        if( verbose ):
            print("Nothing to add")
        return
    
    if( verbose ):
        print( "Adding", len( keys ) - len( list( old_index[0].keys() ) ), "cols" )

    write_index( data, folder, keys, index_filename )
        
    if( verbose ):
        print( "Index updated" )


def calibrate_row( dataset, instrument, channel, item_to_factor ):

    for i in range( len( dataset ) ):
        dataset[ i ][ instrument ][ channel ] = dataset[ i ][ instrument ][ channel ] * item_to_factor( dataset[ i ] )

def export_columns_to_file( dataset, folder, dict_creator, filename = 'index.xlsx' ):
    records = []

    for d in dataset:
        records.append( dict_creator( d ) )

    pd.DataFrame.from_records( records ).to_excel( folder + '/' + filename, index=False )