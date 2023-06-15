import pandas as pd
from . import rspro, nicolet, opus, gc, images
from glob import glob
import re
from os.path import exists
from standard_imports import *

def load_index( folder, filename = "index.xlsx" ):
    dataset = pd.read_excel( folder + "/" + filename ).rename( columns = { 'id': 'ID', 'Id': 'ID' } ).to_dict('records')

    return dataset

def load_data( dataset, folder, load_all = False, to_load = ['rspro','nicolet_SPA','nicolet_SRS','opus','gc','jpg'] ):
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

    return dataset

def load_index_and_data( folder, index_filename = "index.xlsx", load_all = False, update_index = False, to_load = ['rspro','nicolet_SPA','nicolet_SRS','opus','gc','jpg'], load_only = [] ):
    if( exists( folder + "/" + index_filename ) ):
        index = load_index( folder, index_filename )
    else:
        if( update_index ):
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
        index_towrite = []
        for i in index:
            index_towrite.append({
                k: ( i[k] if k in i.keys() else "" ) for k in prevs_keys
            })
        pd.DataFrame.from_records( index_towrite ).to_excel( folder + '/' + index_filename, index=False )
        print(f"Index uploaded, { len( index ) - prevs_length } rows added, total of { len( index ) } rows loaded")
    else:
        print(f"{ len( index ) } rows loaded")

    return index

def add_cols_to_index( data, folder, cols_to_add = [], index_filename = "index.xlsx", verbose = False ):
    old_index = load_index( folder, index_filename )

    assert len( data ) == len( old_index ), "The dataset provided have a length different from the actual index!"

    keys = list( old_index[0].keys() )
    to_update = False

    for k in cols_to_add:
        if( k not in keys ):
            keys.append( k )
            to_update = True
    
    if( not to_update ):
        if( verbose ):
            print("Nothing to add")
        return
    
    if( verbose ):
        print( "Adding", len( keys ) - len( list( old_index[0].keys() ) ), "cols" )

    index_towrite = []
    for d in data:
        index_towrite.append({
            k: n( d[k] ) for k in keys
        })

    pd.DataFrame.from_records( index_towrite ).to_excel( folder + '/' + index_filename, index=False )
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