from picoscope import ps2000a
import time
import numpy as np
from uncertainties import ufloat

class Picoscope:
    ps = None

    availableRanges = []

    def __init__( self ):
        self.data = {}

        if( Picoscope.ps ):
            Picoscope.ps.close()
        Picoscope.ps = ps2000a.PS2000a()

        # Flash led as a feedback
        self.flashLed()

        # Init channels
        self.setSamplingFrequency(250E6, 4096)
        Picoscope.ps.setChannel("A", "DC")
        Picoscope.ps.setChannel("B", "DC")
        
        self.acquire()

    def setSamplingFrequency( self, sampleFrequency, noSamples ):
        Picoscope.ps.setSamplingFrequency( sampleFrequency, noSamples )
        self.data.update( {
            "Time": np.arange( noSamples ) / sampleFrequency
        } )
        self.acquire()

    def setVoltageRange( self, channel, range ):
        Picoscope.ps.setChannel( channel, "DC", range)
        self.acquire()

    def acquire( self ):
        Picoscope.ps.runBlock()
        Picoscope.ps.waitReady()

        self.data.update( {
            "A": Picoscope.ps.getDataV( 'A', 4096 ),
            "B": Picoscope.ps.getDataV( 'B', 4096 )
        } )

    def flashLed( self ):
        Picoscope.ps.flashLed(10)

    def runExampleWaveform( self, amplitude ):
        x = self.data["Time"]
        y = amplitude * np.sin( x / 1e-6 )
        Picoscope.ps.setAWGSimple( y, np.ptp( x ), shots = 1000 )

    def getChNum( self, channel ):
        if not isinstance(channel, int):
            return Picoscope.ps.CHANNELS[channel]
        else:
            return channel
        
    def getChLet( self, channel ):
        letters = [ "A", "B" ]
        if isinstance(channel, int):
            return letters[channel]
        else:
            return channel
        
    def safelySetRange( self, channel, range ):
        try:
            Picoscope.ps.setChannel( channel, VRange= range - 1e-4 )
        except OSError:
            print( f"Unable to set range {range}" )
            Picoscope.availableRanges = np.delete( Picoscope.availableRanges, np.where( Picoscope.availableRanges == range )[0] )

    def getAvailableRanges( self ):
        if( len( Picoscope.availableRanges ) == 0 ):
            Picoscope.availableRanges = np.array( [ d['rangeV'] for d in Picoscope.ps.CHANNEL_RANGE ] )
        return Picoscope.availableRanges

    def autorange( self, channel ):
        
        # Current range infos
        channel = self.getChNum( channel )
        actual_range = Picoscope.ps.CHRange[ channel ]
        available_ranges = self.getAvailableRanges()
        actual_range_idx = np.argmin( np.abs( available_ranges - actual_range ) )

        # Check current data
        self.acquire()
        maxval = np.max( np.abs( self.data[ self.getChLet( channel ) ] ) )

        # Try to rise range
        if( actual_range_idx < len( available_ranges ) - 1 and maxval > 0.9 * actual_range ):
            self.safelySetRange( channel, available_ranges[ actual_range_idx + 1 ] )
            return self.autorange( channel )

        # Try to lower range
        if( actual_range_idx > 0 and maxval < 0.9 * available_ranges[ actual_range_idx - 1 ] ):
            self.safelySetRange( channel, available_ranges[ actual_range_idx - 1 ] )
            return self.autorange( channel )

        return actual_range

    def autoRangeAndAvg( self, channel ):
        ar = self.autorange( channel )
        
        self.acquire()
        return ufloat(
            np.mean( self.data[ self.getChLet( channel ) ] ),
            np.std( self.data[ self.getChLet( channel ) ] )
        )