from picoscope import ps2000a
import time
import numpy as np

class Picoscope:
    ps = None

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