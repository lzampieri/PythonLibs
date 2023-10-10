# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4

import sqlite3 as sqlite
import pandas
from . import spectrum
import os
import warnings
import numpy
from scipy.constants import physical_constants, h, c
from copy import copy, deepcopy


path = os.path.dirname(spectrum.__file__)

##############
##for pyinstaller
#import sys
#path = sys._MEIPASS
#####################


kB = physical_constants['Boltzmann constant in inverse meters per kelvin'][0]
kB /=100 #inverse cm


class SpecDB(object):
    """
    Class for working with spectral databases, using pandas
    """
    def __init__(self, filename, **kwargs):
        """
args:
-----
filename: name of the database file. Just the file, not the path. The files
will be ALWAYS searched for exclusively in massiveOES_location/data
directory. Providing full path will result in error.

kwargs:
-------
kind: either 'emission' (default) or 'absorption'. Others are not implemented
        """
        name_stop = filename.find('.')
        self.specie_name = copy(filename[:name_stop])
        self.filename = filename
        to_open = os.path.join(path,'data',filename)
        self.kind = kwargs.pop('kind', 'emission')
        self.uorl = {'emission':'upper', 'absorption':'lower'}[self.kind] 
        if not SpecDB.isSQLite3(to_open):
            warnings.warn('The file '+to_open+
                        ' is not a valid database!')
            self.conn = None
            return
        else:
            self.conn = sqlite.connect(to_open)
        if self.kind == 'emission':
            self.states = pandas.read_sql_query("select J,E_J,E_v from upper_states",
                self.conn)
        elif self.kind == 'absorption':
            self.states = pandas.read_sql_query("select J,E_J,E_v from lower_states",
                self.conn)
        else:
            warnings.warn('massiveOES.SpecDB: Kind '+self.kind+
                ' unknown. Spectral simulations for '+
                self.specie_name + ' will not work.', Warning)
        self.last_Trot = None
        self.last_Tvib = None
        self.norm = None
        self.spec = None
        self.last_wmin = 0
        self.last_wmax = numpy.inf
        self.table = None


    @staticmethod
    def isSQLite3(filename):
        from os.path import isfile, getsize
        if not isfile(filename):
            return False
        if getsize(filename) < 100: # SQLite database file header is 100 bytes
            return False

        with open(filename, 'rb') as fd:
            header = fd.read(100)
        return header[:16] == b'SQLite format 3\x00'


    def __getstate__(self):
        return self.filename, self.kind

    def __setstate__(self, state):
        if hasattr(state, 'split'): #older files have only a filename
                                    #string
            filename = state
            ret = self.__init__(filename)
        elif len(state)==2:
            filename, kind = state
            ret = self.__init__(filename, kind=kind)
        else:
            raise RuntimeError('Cannot load, incompatible file format!'
                               +str(state))

        return ret

    def calculate_norm(self, Trot, Tvib, ):
        #print(Trot, Tvib)
        parts = (2*self.states.J+1)*numpy.exp(-self.states.E_J /
                (kB*Trot) - self.states.E_v / (kB*Tvib))
        return numpy.sum(parts)


    def get_spectrum(self, Trot, Tvib, wmin=None, wmax = None, **kwargs):
        """
kwargs:
   wavelength: either 'vacuum' or 'air', returns the wavelength in vacuum or air (default is 'air')

   as_spectrum: bool, if True object massiveOES.Spectrum is returned, otherwise a numpy array is returned

    y: either 'photon_flux' (default), i.e. the y*axis is population * (emission coefficient)
      or 'intensity', i.e. the y-axis is population * (emission coefficient) * wavenumber
        """
        as_spectrum = kwargs.pop('as_spectrum', True)
        wav_reserve = kwargs.pop('wav_reserve', 2)
        deep = kwargs.pop('wav_reserve', False)
        # print '****************************'
        # print Trot, self.last_Trot
        # print Tvib, self.last_Tvib
        # print wmin, self.last_wmin
        # print wmax, self.last_wmax

        y = kwargs.pop('y', 'photon_flux')
        wav = kwargs.pop('wavelength', 'air')
        wav += '_wavelength'
        recalculate_pops = False

        if (wmin < self.last_wmin or wmax > self.last_wmax or
                self.table is None):
            self.last_wmin = wmin - wav_reserve
            self.last_wmax = wmax + wav_reserve
            self.table = self.get_table_from_DB(self.last_wmin,
                    self.last_wmax, wav=wav,**kwargs)
            recalculate_pops = True

        if (self.last_Trot != Trot or self.last_Tvib != Tvib or
                recalculate_pops):
            self.norm = self.calculate_norm(Trot, Tvib)
            self.last_Trot = Trot
            self.last_Tvib = Tvib
            #print(Trot, Tvib)
            self.table['pops'] = (2*self.table['J']+1)*numpy.exp(-self.table['E_v']/(kB*Tvib) -
                    self.table['E_J']/(kB*Trot)) / self.norm

            #print('norm =', self.norm)
        if self.kind=='absorption':
            self.table['y'] = self.table['pops'] * self.table['B']
        elif self.kind=='emission':
            self.table['y'] = self.table['pops'] * self.table['A']
        else:
            return
        if y == 'intensity':
            self.table['y'] *= self.table['wavenumber']
        if as_spectrum:
            self.spec = spectrum.Spectrum(x = self.table[wav],
                    y = self.table['y'])
        else:
            self.spec = numpy.array([self.table[wav], self.table['y']]).T
        if deep:
            return deepcopy(self.spec)
        else:
            return copy(self.spec)    


    def get_table_from_DB(self, wmin=None, wmax=None, **kwargs):
        """
        """
        wav = kwargs.pop('wav', 'air_wavelength')
        q = 'SELECT air_wavelength,vacuum_wavelength,'
        if self.kind=='absorption':
            q+= 'B,'
        elif self.kind=='emission':
            q+='A,'
        q+='J,E_J,E_v,wavenumber'
        q+= ' FROM '
        if wmin != 0  and wmax != numpy.inf:
            q += " (SELECT * FROM lines WHERE " + wav + " BETWEEN  "
            q += str(wmin) + " AND " + str(wmax) + ")"
        else:
            q += " lines "
        q += " INNER JOIN "
        if self.kind=='absorption':
            q+= " lower_states on lower_state=lower_states.id"
        elif self.kind=='emission':
            q+=" upper_states on upper_state=upper_states.id"
        q+= " ORDER BY " + wav
        #print('******pandas************')
        #print(q)
        table = pandas.read_sql_query(q,self.conn)
        return table

    def get_lines_by_states(self, wmin, wmax, **kwargs):
        minlines = kwargs.pop('minlines', 1)
        wav = kwargs.pop('wavelength', 'air')
        wav += '_wavelength'
        max_J = kwargs.pop('max_J', None)
        max_v = kwargs.pop('max_v', None)
        singlet_like = kwargs.pop('singlet_like', False)

        q = 'select '
        q += 'lines.id, A, '+wav+', upper_state, branch, wavenumber, lower_state, '
        q += 'E_J, J, component, E_v, v'
        q += ' from lines inner join '+self.uorl+'_states on '+self.uorl+'_state='+self.uorl+'_states.id'
        q += ' where lines.'+wav+' between '+str(wmin) + ' and '+str(wmax)
        if max_v is not None:
            q+= ' and v <= ' + str(max_v)
        if max_J is not None:
            q+= ' and J <= ' + str(max_J)
        big_table = pandas.read_sql_query(q, self.conn)
        
        if singlet_like:
            gr = big_table.groupby(['v', 'J'])
            states = big_table.loc[:,['E_J','J','E_v','v']].drop_duplicates()
            states = states.groupby(['v', 'J']).mean()
        else:
            gr = big_table.groupby([self.uorl+'_state'])
            states = big_table.loc[:,[self.uorl+'_state','E_J','J','component','E_v','v']].drop_duplicates()
            states.set_index(self.uorl+'_state', inplace=True)

        specs = []
        numlines = []
        state_names = []
        for name, group in gr:
            if len(group) >= minlines:
                numlines.append(len(group))
                specs.append(group.loc[:,[wav, 'A']])
                #specs.append(group)                
                state_names.append(name)
        if singlet_like:
            states_ordered = states.loc[state_names].reset_index()
        else:
            states_ordered = states.loc[state_names,:] #reorder
        print(len(numlines), len(states_ordered))
        states_ordered['numlines'] = numlines
        
        return {'specs':specs, 'states':states_ordered.loc[states_ordered['numlines']>=minlines]}

def puke_spectrum(params, **kwargs):
    step = kwargs.pop('step', params['wav_step'].value)
    wmin = kwargs.pop('wmin', params['wav_start'].value)
    wmax = kwargs.pop('wmax', None)
    sims = kwargs.pop('sims', {})
    points_density = kwargs.pop('points_per_nm', 1000)

    if wmax is None:
        wmax = (params['wav_start'].value +
                params['wav_step'].value * params.number_of_pixels +
                params['wav_2nd'].value * params.number_of_pixels**2)

    spectra = []
    #print(wmax)
    for specie in params.info['species']:
        temp_spec = sims[specie].get_spectrum(params[specie+'_Trot'].value,
                                  params[specie+'_Tvib'].value,
                                  wmin = wmin, wmax = wmax,
                                  as_spectrum = False)
        temp_spec[:,1] *= params[specie+'_intensity'].value
        # print 'intensity: ', params[specie+'_intensity'].value
        # print 'from DB: '
        # print temp_spec[:,1]
        spectra.append(temp_spec)
    if len(spectra) == 0:
        warnings.warn('No simulation files given, returning empty spectrum!', Warning)
        return spectrum.Spectrum(x = [], y = [])

    spec = numpy.concatenate(spectra)
    spec = spec[spec[:,0].argsort()]
    spec = spectrum.Spectrum(x=spec[:,0], y = spec[:,1])
    spec.refine_mesh(points_per_nm = points_density)
    #print "after refine: "
    #print max(spec.y)
    spec.convolve_with_slit_function(gauss = params['slitf_gauss'].value,
                                          lorentz = params['slitf_lorentz'].value,
                                          step = step)
    #print 'after convolve'
    #print max(spec.y)
    if len(spec.y)>0:
        spec.y += params['baseline'].value
        spec.y += params['baseline_slope'].value*(spec.x-params['wav_start'].value)

    #print spec.y
    return spec
