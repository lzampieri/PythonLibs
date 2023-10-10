from .spectrum import Spectrum
from numpy import array, genfromtxt, abs
#from pylab import *
import warnings

def find_nearest(array,value):
    idx = (abs(array-value)).argmin()
    return idx

class FHRSpectra(object):
    def __init__(self, **kwargs):
        self.wavs = kwargs.pop('wavs',None)
        self.intensities = kwargs.pop('intensities',None)
        self.delays = kwargs.pop('delays',None)

    @staticmethod
    def readwavs(filename):
        with open(filename) as f:
            wavs = f.readline()
        wavs=wavs.split('\t')
        del wavs[0]
        return array([float(w) for w in wavs])


    @classmethod
    def from_file(cls, filename):
        wavs = FHRSpectra.readwavs(filename)
        nums = genfromtxt(filename, skip_header=1)
        ret = cls(wavs=wavs, intensities=nums[:,1:], delays=nums[:,0].astype(str))
        return ret

    def get_spectrum(self,**kwargs):
        index = kwargs.pop('index', None)
        delay = kwargs.pop('delay', None)
        if index is None and delay is None:
            return None
        if index is None and delay is not None:
            index = find_nearest(self.delays, delay)
            return Spectrum(x=self.wavs, y=self.intensities[index,:]), self.delays[index]
        if index is not None and delay is None:
            return Spectrum(x=self.wavs, y=self.intensities[index,:]), self.delays[index]
        if index is not None and delay is not None:
            warnings.warn('FHRSpectra.get_spectrum: given both, index and delay.\n\
Returning two Spectra, first for the matchig delay, second for the given index!')
            match_index = find_nearest(self.delays, delay)
            return (Spectrum(x=self.wavs, y=self.intensities[match_index, :]), self.delays[match_index]), (Spectrum(x=self.wavs, y=self.intensities[index,:]),self.delays[index])

    # def movie(self):
    #     from matplotlib.pyplot import  subplots
    #     fig, ax = subplots(1,1)
    #     for i in range(self.intensities.shape[0]):
    #         ax.plot(self.wavs, self.intensities[i,:])
    #         plt.pause(0.1)
    #         ax.title(self.delays[i])
    #         plt.draw()
    #         ax.clear()
