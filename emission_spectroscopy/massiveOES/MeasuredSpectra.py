# -*- coding: utf-8 -*-
"""
Created on Mon Sep 01 10:36:08 2014

@author: Petr
"""
#import image_functions
import copy
import numpy
from numpy.linalg import lstsq
from scipy.optimize import nnls
#import matplotlib.pyplot as plt
import pickle
#import simulated_spectra
from . import SpecDB, puke_spectrum, spectrum
import logging
from .FHRSpectra import FHRSpectra
import warnings
from numpy.linalg import inv
import lmfit
import copy
import os.path
from collections import OrderedDict
import json
from asteval import valid_symbol_name


class Parameters(object):
    """Class containing the parameters of the fit. Contains also instance
    of lmfit.Parameters class (as self.prms). The respective parameters can be
    accessed via __getitem__() method (i.e. brackets), just like in a
    dictionary. Apart from that contains also some extra information
    necessary for massiveOES to run properly, such as number of
    pixels and list of relevant species.

    It is usually not necessary to explicitly call any class methods,
    as this is accomplished from MeasuredSpectra objects.
    """

    def __init__(self, *args, **kwargs):
        """
        wavelengths:  *iterable* with three numbers. The wavelength axis is then calculated from
                      pixel position (pos) as
                      lambda = wavelengths[0] + wavelengths[1]*pos + wavelengths[2]*pos**2

        slitf_gauss: gaussian HWHM of the slit function
        slitf_lorentz: lorentzian HWHM of the slit function

        baseline: *float* constant offset of the spectrum from zero

        baseline_slope: *float* accounts for non-constant baseline (necessary on some older spectrometers,
                        otherwise should be kept fixed at 0)

        simulations: list of specDB objects, or simply empty list []
"""
        self.number_of_pixels = kwargs.pop('number_of_pixels', 1024)
        self.prms = lmfit.Parameters()
        self.info = {'species':[]}
        wavs = kwargs.pop('wavelengths', (0,1,0))
        self.prms.add('wav_start', value=wavs[0])
        self.prms.add('wav_step', value=wavs[1])
        self.prms.add('wav_2nd', value=wavs[2])

        gauss = kwargs.pop('slitf_gauss', 1e-9)
        lorentz = kwargs.pop('slitf_lorentz', 1e-9)
        self.prms.add('slitf_gauss', value=gauss)
        self.prms.add('slitf_lorentz', value=lorentz)

        baseline = kwargs.pop('baseline', 0)
        self.prms.add('baseline', value=baseline)

        baseline_slope = kwargs.pop('baseline_slope', 0)
        self.prms.add('baseline_slope', value=baseline_slope)

        simulations = kwargs.pop('simulations', None)
        if simulations is not None:
            for sim in simulations:
                self.add_specie(sim)

    def __getitem__(self, key):
        return self.prms.__getitem__(key)

    def keys(self):
        return self.prms.keys()

    def add_specie(self, specie, **kwargs):
        """
        specie: specDB object
        """
        Trot = kwargs.pop('Trot', 1e3)
        Tvib = kwargs.pop('Tvib', 1e3)
        intensity = kwargs.pop('intensity', 1)
        specie_name = specie.specie_name
        if valid_symbol_name(specie_name) and specie_name not in self.info['species']:
            self.info['species'].append(specie_name)
            #self.info[specie_name+'_sim'] = specie # specDB object
            self.prms.add(specie_name+'_Trot', value=Trot)
            self.prms[specie_name+'_Trot'].min = 300
            self.prms[specie_name+'_Trot'].max = 10000
            self.prms.add(specie_name+'_Tvib', value=Tvib)
            self.prms[specie_name+'_Tvib'].min = 300
            self.prms[specie_name+'_Tvib'].max = 10000
            self.prms.add(specie_name+'_intensity', value=intensity)
            self.prms[specie_name+'_intensity'].min = 0
        else:
            msg = 'Specie \''+ specie_name +'\' not added: Invalid name or already added! See lmfit.Parameters for details.'
            warnings.warn(msg,Warning)

    def rm_species(self, specie):
        """
        Remove simulated spectrum of given specie.

        specie: string, name of the specie (eg. \'OH\')
        """
        self.info['species'].remove(specie)
        #self.info.pop(specie+'_sim', 0)
        #print specie
        self.prms.pop(specie+'_Trot', 0)
        self.prms.pop(specie+'_Tvib', 0)
        self.prms.pop(specie+'_intensity',0)


    def save(self, filename):
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load(filename):
        with open(filename, 'rb') as input:
            return_val = pickle.load(input)
        return return_val


class MeasuredSpectra():
    """Class containing the measured data. Suitable for storing number of
    spectra (high numbers are possible, but it is not optimized to be
    memory efficient - all spectra will be loaded into the memory at
    initiating an instance of MeasuredSpectra). A reasonable strategy
    is to divide large measurements into several MeasuredSpectra
    objects.

    It also keeps references to spectral simulations and contains
    methods for least squares fitting.

    """


    def __init__(self, **kwargs):
        """Most of the kwargs are never used in the class methods and serve
        only for future reference - to allow the experimenter to keep
        track of the important metadata.

        The only exception is \'spectra'\ that can be used to fill in
        the measured data. The format of this should be a list of
        iterables (tuples or lists) containing [identificator, data],
        where identificator can be a string or a number (can contain
        e.g. spatial position or delay after the trigger) and data
        should be 1D numpy array (a vector) of \'intensity\'
        values. The wavelengths are not contained here, but are always
        calculated from Parameters (see class Parameters).

        kwargs:

        spectra: a nested OrderedDict with keys defined by the user (typically used
        for experimental info,  such as physical coordinates or experimental
        conditions). Each value contains again a dictionary with 'params' and 'spectrum'.
        'params' are massiveOES.MeasuredSpectra.Parameters,
        'spectrum' are either 1D numpy arrays (y-axis of spectra)
        or massiveOES.Spectrum objects. In case of 1D arrays , the wavelength
        axis is calculated from parameters [wav_start, wav_step, wav_2nd]. In
        case of massiveOES.Spectrum object, the wavelengths are believed to be
        right and are not allowed to change.

        spectra (OrderedDict)
          {spectrum_id: {'params':massiveOES.Parameters,
                        'spectrum': 1D array or massiveOES.Spectrum}}

        at the time of init, only 'data' of each dict should be defined,
        the 'params' will be added by automatically by
        MeasuredSpectra.create_fit_parameters()

        filename: *string* - name of the file with the source data. Only for future reference, can be omitted.

        filenameSPE: *string* obsolete, used to keep track of the original SPE files from PIMAX ICCD camera

        date: defaults to 0, but can be used to keep the date of the measurement. Format is arbitrary.

        time: defaults to 0, but can be used to keep the time of the measurement. Format is arbitrary.

        accumulations: to keep track of the number of accumulations.

        """

        self.filename = kwargs.pop('filename', '')
        self.filenameSPE = kwargs.pop('filenameSPE','')
        self.date = kwargs.pop('date',0)
        self.time = kwargs.pop('time',0)
        self.accumulations = kwargs.pop('accumulations',0)
        self.gatewidth = kwargs.pop('gatewidth',0)
        self.regionofinterest_y = kwargs.pop('ROI_y',None)
        self.regionofinterest_x = kwargs.pop('ROI_x',None)

        #self.fiberposition = 0
        #self.fibersteepnes = 0
        #self.fiberverticalstep = 0

        self.spectra = kwargs.pop('spectra',OrderedDict())
        self.image_analysis_values = []
        self.image_analysis_values_used = []
        self.minimizer=None
        self.minimizer_result = None
        self.simulations = {}
        self.create_fit_parameters(**kwargs)

    def __getstate__(self):
        ret = (self.filename, self.filenameSPE, self.date, self.time, self.accumulations,
               self.gatewidth, self.regionofinterest_x, self.regionofinterest_y,
               self.spectra, self.simulations)
        return ret

    def __setstate__(self, state):
        self.filename, self.filenameSPE, self.date, self.time, self.accumulations, self.gatewidth, self.regionofinterest_x, self.regionofinterest_y,  self.spectra,self.simulations = state


    @classmethod
    def from_FHRSpectra(cls, FHRSpec, **kwargs):
        """
        for internal use only
        """
        #print( FHRSpec.wavs)
        #print( FHRSpec.intensities)
        #print( FHRSpec.delays)
        spectra = OrderedDict()
        wavs = [numpy.min(FHRSpec.wavs), numpy.mean(numpy.diff(FHRSpec.wavs)), 0]
        for i in range(len(FHRSpec.delays)):
            if wavs[1] < 0:
                #reverse order of intensities if wavelength ordered descendingly
                spectra[FHRSpec.delays[i]] = {'spectrum' : FHRSpec.intensities[i][::-1]}
            else:
                spectra[FHRSpec.delays[i]] = {'spectrum' : FHRSpec.intensities[i]}

        if wavs[1] < 0:
            wavs[1] *= -1

        ret = cls(spectra=spectra, wavelengths=wavs)

        return ret

    @classmethod
    def from_CSV(cls, filename, **genfromtxtargs):
        """Used to extract measured data from ASCII files containing the
        wavelengths in the first column and and intensity vectors of
        y-values in other columns. Delimiter must be coma:

        w1,y11,y21
        w2,y12,y22
        w3,y12,y23
        
        Other Parameters
        ----------------
        **genfromtxtargs: forwarded to :py:func:`numpy.genfromtxt`
            default : {"delimiter":','}
        ...

        """
        # loads data from csv, expects first column of wavelengts and coma as separator
        delimiter = genfromtxtargs.pop("delimiter", ",")
        dataarray = numpy.genfromtxt(filename, delimiter = delimiter, **genfromtxtargs)
        wavs = [dataarray[0,0], dataarray[1,0] - dataarray[0,0],0]
        spec = OrderedDict()
        for i in range(1, dataarray.shape[1]):
            spec[i] = {'spectrum':dataarray[:,i]}

        ret = MeasuredSpectra(filename = filename, spectra = spec, wavelengths = wavs)

        return ret

    @staticmethod
    def from_FHRfile(filename):
        """used to extract measured data from ASCII files containing the
        wavelengths in the first row and ID and intensity vector in
        each other row, i.e.:
            \t w1  \t w2  \t w3...
        ID1 \t y11 \t y12 \t y13...
        ID2 \t y21 \t y22 \t y23...
        ...

        """
        fhrs = FHRSpectra.from_file(filename = filename)
        ret = MeasuredSpectra.from_FHRSpectra(fhrs)
        return ret

    def add_specie(self, specie, specname, **kwargs):
        """use this spectral simulation for comparison with measured data.
        The reference to simulation object will be added
        to self.simulations and a string describing the specie will be
        added to Parameters.info dictionary.

args:
   specie: specDB object
   specname: to which Parameters object the specie should be added

        """
        #specie_name = specie.specie_name
        if specie.specie_name not in self.simulations:
            self.simulations[specie.specie_name] = specie
        self.spectra[specname]['params'].add_specie(specie, **kwargs)


    def create_fit_parameters(self, **kwargs):
        """
        **kwargs:
    simulated_spectra: *list* of SpecDB objects

    wavelengths: tuple/list with meaning [0]: wavelength_start, [1]:
    wavelength_step (difference of wavelength position of two
    neiboughring pixels), [2]: 2nd order correction. The resulting
    wavelegth axis is computed from pixel index 'pos' as x = wav[0] +
    wav[1]*pos + wav[2]*pos**2

    other kwargs are passed directly to Parameters.__init__()
        """
        simulated_spectra = kwargs.pop('simulated_spectra', [])
        #print(simulated_spectra)
        for spec in self.spectra:
            if 'params' not in self.spectra[spec]:
                self.spectra[spec]['params'] = Parameters(number_of_pixels=len(self.spectra[spec]['spectrum']), **kwargs)
                for specie in simulated_spectra:
                    self.add_specie(specie,spec, **kwargs)


    def plot(self, **kwargs):
        """use matplotlib to visualise measured data. The x-axis is not
        calculated, the pixel number is used instead.

        kwargs:

        plotrange: [from, to] (defaults to [0, len(self.spectra)].
                   Which spectra will be show in the plot.

        """
        import matplotlib.pyplot as plt
        plot_range = kwargs.pop('plotrange', [0,len(self.spectra)])
        for i in range(plot_range[0],plot_range[1]):
            plt.plot(self.spectra[i][1])
        plt.show()

    def save(self, filename):
        self.minimizer=None
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)



    def prepare_for_nnls(self, specname, species, **kwargs):
        #import matplotlib.pyplot as plt
        prms = self.spectra[specname]['params']
        wav = kwargs.pop('wavelength', 'air')
        kwargs['wavelength'] = wav
        wav += '_wavelength'
        numpoints = len(self.spectra[specname]['spectrum'])

        end = kwargs.pop('wmax', None)
        to_match = self.get_measured_spectrum(specname)
        xx = to_match.x
        start = min(xx)
        end = max(xx)
        info = OrderedDict()
        A = []
        for specie in species:
            if specie in prms.info['species']:
                if specie in kwargs:
                    print (specie, kwargs[specie])
                    info[specie] = self.simulations[specie].get_lines_by_states(start, end, **kwargs[specie])
                else:
                    print(kwargs)
                    info[specie] = self.simulations[specie].get_lines_by_states(start, end, **kwargs)

                for spec in info[specie]['specs']:
                    spec = numpy.vstack(([min(to_match.x), 0],
                                         spec,
                                         [max(to_match.x), 0]))
                    spec = spectrum.Spectrum(x = spec[:,0], y = spec[:,1])
                    spec.refine_mesh()
                    spec.convolve_with_slit_function(gauss=prms['slitf_gauss'].value,
                                                   lorentz=prms['slitf_lorentz'].value,
                                                     step=prms['wav_step'].value)
                    spec = spectrum.match_spectra(spec, to_match)[0]
                    A.append(spec.y)
            info[specie].pop('specs')

        return numpy.array(A).T, info



    def fit_nnls(self, specname, species, baseline=False, baseline_slope=False, **kwargs):
        """State-by-state fit of molecular simulation to measured data in
        order to construct Boltzmann plot.

        args:
        -----
        specname: identificator of the measured spectrum is of interest

        species: *list of strings* identifying the desired
                 species. The species need to be added to MeasuredSpectra
                 object before calling this method.

        baseline: *bool* if True, the constant baseline will be fitted
                   as well. If False, the baseline from
                   self.spectra[specname]['params']['baseline'], i.e. the background
                   determined by fitting will be subtracted
                   from the measured data before performing the fit

        baseline_slope: *bool* if True, the tilted background will be fitted
                        as well. If False, the tilted background from
                        self.spectra[specname]['params']['baseline'], i.e.
                        the background slope determined by fitting will be subtracted
                        from the measured data before performing the fit

        kwargs:
        -------
        A dictionary of additional parameters for each fitted specie.
        These are:

        max_v: *number* the maximal vibrational quantum number allowed

        max_J: *number* the maximal rotational quantum number allowed

        minlines: *number* the minimal number of lines emitted by transition
                   from given upper state in the observed wavelength region.
                   This is recommended to be at least 2 to suppress the
                   influence of noise.

        singlet_like: *bool* if True, the spin-orbit or spin-rotational
                      components of the upper states will not be distinguished

        returns:
        --------
        frame, specs
        frame: pandas data frame with information about the states, including
               their relative populations and error estimates
        specs: numpy matrix. With numpy.dot(specs, frame.pops), the fitted
               spectrum can be easily reconstructed for plots.

        """
        specs, info = self.prepare_for_nnls(specname, species, **kwargs)
        if baseline:
            info['baseline'] = True
            tostack = numpy.ones((specs.shape[0],1))
            specs = numpy.hstack((specs, tostack))
            if hasattr(self.spectra[specname]['spectrum'], 'y'):
                tofit = self.spectra[specname]['spectrum'].y
            else:
                tofit = self.spectra[specname]['spectrum']
        else:
            info['baseline'] = False
            if hasattr(self.spectra[specname]['spectrum'], 'y'):
                tofit = self.spectra[specname]['spectrum'].y - self.spectra[specname]['params']['baseline'].value
            else:
                tofit = self.spectra[specname]['spectrum'] - self.spectra[specname]['params']['baseline'].value


        if baseline_slope:
            #tostack = numpy.arange(specs.shape[0])
            params = self.spectra[specname]['params']
            tostack = self.get_wavs(specs.shape[0],
                                    params['wav_start'].value,
                                    params['wav_step'].value,
                                    params['wav_2nd'].value)
            specs = numpy.hstack((specs, tostack.reshape((specs.shape[0], 1))))
            info['baseline_slope'] = True
        else:
            info['baseline_slope'] = False
            tofit -= self.spectra[specname]['params']['baseline_slope']*numpy.arange(len(tofit))

        pops, rnorm = scipy.optimize.nnls(specs,tofit)
        if baseline:
            print('baseline = ', pops[-1])
        errs = MeasuredSpectra.nnls_errors(specs, rnorm)
        frame = MeasuredSpectra.reorganize_nnls_to_pandas(pops, info, errs)
        return frame, specs
        #return pops, info, rnorm


    @staticmethod
    def reorganize_nnls_to_pandas(pops, info, errs):
        """
        internal method for creating convenient pandas dataframes
        from original output of MeasuredSpectra.fit_nnls()
        """
        import pandas
        frames = []
        slice_start = 0
        for specie in info:
            if specie.startswith('baseline'):
                continue
            #print 'reorganize: len of state = ',  len(info[specie]['states'][0])

            #v = [state[0] for state in info[specie]['states']]
            #J = [state[1] for state in info[specie]['states']]
            #if len(info[specie]['states'][0]) == 3:
            #    component = [state[2] for state in info[specie]['states']]
            #else:
            #    component = [numpy.nan for state in info[specie]['states']]
            df = info[specie]['states']
            slice_stop = slice_start+len(df)
            df['pops'] = pops[slice_start:slice_stop]
            df['errors'] = errs[slice_start:slice_stop]
            df['specie'] = specie
            frames.append(df)
            slice_start = slice_stop
        ret = pandas.concat(frames)
        ret.reset_index(drop=True, inplace = True)
        return ret

    @staticmethod
    def nnls_errors(jac, rnorm):
        """
        An internal function for calculating errors of the fit_nnls result.

        args:
        jac: Jacobian matrix

        rnorm: sum of squared residuals

        pops: results of the fit
        """
        print(jac.shape)
        hess = numpy.dot(jac.T, jac)
        dof = jac.shape[0] - jac.shape[1]
        print('dof = ', dof)
        covmat = rnorm/(dof)*numpy.linalg.inv(hess)
        return abs(numpy.diag(covmat))**0.5


    def measured_Spectra_for_fit(self, specname, params = None):
        warnings.warn('This method has been finally renamed to get_measured_spectrum().',
                      DeprecationWarning)
        return self.get_measured_spectrum(specname, params=params)
        

    def get_measured_spectrum(self, specname, params = None):
        """
        Return the measured spectrum identified by specname, including the wavelength axis.

        args:
        specname: which spectrum to return

        params: *instance of massiveOES.Parameters*, defaults to None,
                in which case the self.spectra[specname]['params'] is used

        return:

        an instance of massiveOES.Spectrum() with attributes x and y
        """

        meas_spectra_y = self.spectra[specname]['spectrum'] #+

        #the MeasuredSpectra objects can newly contain spectra including
        #the x-axis. In that case, the wav_... params are ignored
        #used for cases when the existing wavelength calibration is more correct
        #than the 2nd order polynomial approximation (typically spectra glued from
        #more windows and carefully pre-processed to fit
        if hasattr(meas_spectra_y, 'x') and hasattr(meas_spectra_y, 'x'):
            return meas_spectra_y

        if params is None:
            params = self.spectra[specname]['params']

        wavs = MeasuredSpectra.get_wavs(len(self.spectra[specname]['spectrum']),
                                        params['wav_start'].value,
                                        params['wav_step'].value,
                                        params['wav_2nd'].value)

        meas_spectra =  spectrum.Spectrum(y=meas_spectra_y ,
                                          x = wavs)
        logging.debug('meas_spectra: ')
        logging.debug('x = [%f ... %f]', meas_spectra.x[0], meas_spectra.x[-1])
        logging.debug('y = [%f ... %f]', meas_spectra.y[0], meas_spectra.y[-1])
        return meas_spectra

    def get_simulated_spec(self, specname):
        """
Return simulated spectrum according to the parameters and
simulation databases stored for given specname. Works only
after the simulated spectra database(s) has(ve) been added and
gives reasonable results only after a successful fit.

Args:
        specname: identificator of the spot to which the simulates spectrum should be returned

return:
        simulated spectrum: massiveOES.Spectrum object

        """
        sim = puke_spectrum(self.spectra[specname]['params'],
                            sims = self.simulations)
        return sim

    @staticmethod
    def get_wavs(numpoints, start, step, second):
        pos = numpy.arange(numpoints)
        ret = start + pos*step + pos**2*second
        return ret


    def show(self, specname, save_to_file = None, **kwargs):
        """
Show the agreement of the measured data with the simulation with current parameters.
Args:
        specname: identificator of the current measured spectrum
        save_to_file: *string* containing the filename that the plot will be saved to. If None (default), the plot will be shown.

**kwargs:
        convolve: *bool*, Goes directly to simulated_spectra.puke_spectrum()
        """
        import matplotlib.pyplot as plt
        convolve = kwargs.pop('convolve', True)
        mtpl = self.get_measured_spectrum(specname)
        simtpl =  puke_spectrum(self.spectra[specname]['params'], convolve = convolve, sims=self.simulations)
        plt.plot(mtpl.x, mtpl.y, label = 'Measured')
        plt.plot(simtpl.x, simtpl.y, label = 'Fitted')
        plt.xlim(min(mtpl.x), max(mtpl.x))
        plt.legend()
        plt.xlabel('wavelength [nm]')
        plt.ylabel('intensity [arb. units]')
        if save_to_file is None:
            plt.show()
        else:
            plt.savefig(save_to_file)
        return

    def to_json(self, filename):
        spectra = OrderedDict()
        params = OrderedDict()
        for specname in self.spectra:
            if hasattr(self.spectra[specname]['spectrum'], 'x'):
                spectra[str(specname)] = {'x':list(self.spectra[specname]['spectrum'].x),
                                    'y':list(self.spectra[specname]['spectrum'].y)}
            else:
                spectra[str(specname)] = list(self.spectra[specname]['spectrum'])

            par = self.spectra[specname]['params']
            params[str(specname)] = {'number_of_pixels':par.number_of_pixels,
                    'info':par.info,
                    'prms':par.prms.dumps()}

        #simulations = list(self.simulations.keys())
        simulations = OrderedDict()
        for simkey in self.simulations:
            simulations[simkey] = {'kind':self.simulations[simkey].kind}
        to_save = {'spectra':spectra,
                'params':params,
                'simulations':simulations
                }

        with open(filename,'w') as fp:
            json.dump(to_save, fp)

    @staticmethod
    def from_json(filename):
        with open(filename, 'r') as fp:
            loaded = json.load(fp, object_pairs_hook=OrderedDict)
        spectra = loaded['spectra']
        #spec = []
        spec = OrderedDict()
        for s in spectra:
            if hasattr(spectra[s], 'keys'):
                spec[s] = {'spectrum':spectrum.Spectrum(x = numpy.array(spectra[s]['x']),
                               y = numpy.array(spectra[s]['y']))}
            else:
                spec[s] = {'spectrum':numpy.array(spectra[s])}

        sims = {}
        for sim in loaded['simulations']:
            try:
                sims[sim] = SpecDB(sim + '.db', kind=loaded['simulations'][sim]['kind'])
            except TypeError:
                sims[sim] = SpecDB(sim + '.db')

        for param, s in zip(loaded['params'], list(spec.keys())):
            to_app = Parameters(number_of_pixels=loaded['params'][param]['number_of_pixels'])
            to_app.info = loaded['params'][param]['info']
            try:
                to_app.prms.loads(loaded['params'][param]['prms'])
            except TypeError:
                for entry in json.loads(loaded['params'][param]['prms']):
                    to_app.prms.add(entry[0], value=entry[1],
                            vary=entry[2], min=entry[4], max=entry[5])
            spec[s]['params'] = to_app

        ret = MeasuredSpectra(spectra = spec)

        ret.simulations = sims
        return ret


    @staticmethod
    def load(filename):
        """
        load previously saved MassiveOES object
        """
        print(":::::::::::::::::::::::::::::::::::")
        print(filename)
        with open(filename, 'rb') as input:
            loaded = pickle.load(input)
        # invalid_paths = return_val.check_simulations()
        # for ip in invalid_paths:
        #     msg = 'The simulation '+ ip + ' was not found!'
        #     warnings.warn(msg, Warning)
        ret = MeasuredSpectra()
        ret.filename =  loaded.filename
        ret.filenameSPE = loaded.filenameSPE
        ret.date = loaded.date
        ret.time = loaded.time
        ret.accumulations = loaded.accumulations
        ret.gatewidth = loaded.gatewidth
        ret.regionofinterest_x = loaded.regionofinterest_x
        ret.regionofinterest_y = loaded.regionofinterest_y
        ret.spectra =  loaded.spectra
        #ret.simulations = loaded.simulations
        for key in loaded.simulations:
            print(key, loaded.simulations[key].kind)
            ret.simulations[key] = SpecDB(key+'.db', kind=loaded.simulations[key].kind)
        return ret


    def _dummy(self, params, specname, **kwargs):
        """
        method with the desired signature for lmfit.minimize
        """
        by_peaks = kwargs.pop('by_peaks', False)
        convolve = kwargs.pop('convolve', True)
        weighted = kwargs.pop('weighted', False)
        step = params['wav_step'].value


        self.spectra[specname]['params'].prms = params
        params = self.spectra[specname]['params']
        if by_peaks: #and reduced_sumsq:
            return spectrum.compare_spectra_by_peaks_reduced_sumsq\
                   (self.get_measured_spectrum(specname, params=params),
                    puke_spectrum(params,
                                  convolve=convolve,
                                  step = step,
                                  sims = self.simulations),
                    **kwargs)

        if not by_peaks and not weighted:
            return spectrum.compare_spectra\
                   (self.get_measured_spectrum(specname, params = params),
                    puke_spectrum(params,
                                  convolve=convolve, step = step, sims = self.simulations), **kwargs)
        if not by_peaks and weighted:
            return spectrum.compare_spectra_weighted\
                   (self.get_measured_spectrum(specname, params = params),
                    puke_spectrum(params,
                                  convolve=convolve, step = step, sims = self.simulations), **kwargs)



    def fit(self, specname, **kwargs):
        """Find optimal values of the fit parameters for spectrum identified by specname. The optimal values are then stored in self.spectra[specname]['params'], not returned!

        args:
        -----
        specname: identificator of spectrum to fit

        **kwargs:
        ---------
        maxiter: *int* maximal number of itertions, handy with slower optimization methods

        method: *string* see lmfit documentation for available methods

        by_peaks: *bool* defaults to False. If you own echelle spectrometer,
                  play around with enabling this option. Otherwise, leave it at false.
                  Never properly tested.

        return:
        -------
        result: *bool*, True if the fit converged successfully, False otherwise

        """

        print('********* specname = ', specname, ' ************')
        kwargs['number_of_pixels'] = self.spectra[specname]['params'].number_of_pixels
        maxiter = kwargs.pop('maxiter', 2000)
        #options = {'maxiter':maxiter}
        method = kwargs.pop('method', 'leastsq')
        if method == 'leastsq':
            self.minimizer = lmfit.Minimizer(self._dummy,
                                             self.spectra[specname]['params'].prms,
                                             fcn_args=(specname,),
                                         # fcn_kws={'by_peaks':by_peaks,
                                         #          'reduced_sumsq':sumsq,
                                         #          'convolve':convolve}
                                             fcn_kws = kwargs,
                                             maxfev=maxiter)
        else:
            self.minimizer = lmfit.Minimizer(self._dummy,
                                             self.spectra[specname]['params'].prms,
                                             fcn_args=(specname,),
                                         # fcn_kws={'by_peaks':by_peaks,
                                         #          'reduced_sumsq':sumsq,
                                         #          'convolve':convolve}
                                             fcn_kws = kwargs,
                                             options = {'maxiter':maxiter, 'xtol':0.05})

        # if kwargs['by_peaks']:
        #     method = kwargs.pop('method', 'L-BFGS-B')
        #     self.minimizer.scalar_minimize(method=method)
        # if not kwargs['by_peaks']:


        #print "FITUJU METODOU:   ", method
        self.minimizer_result = self.minimizer.minimize(method=method)
        self.spectra[specname]['params'].prms = self.minimizer_result.params
        return self.minimizer_result

    def export_results(self, filename):
        """
        Save the results of the optimisation as csv file. Uses pandas.

        args:
        -----
        filename: *string* a name of the file, the data will be exported to.
                  Does NOT ask for confirmation before overwritng!
        """

        import pandas

        #result = numpy.zeros((len(self.spectra), 2+6*len(all_species)))
        result = {'spectrum':[],
                  'reduced_sumsq':[]}
        #result[:,:] = numpy.nan

        for specie in self.simulations:
            result[specie+'_Trot'] = []
            result[specie+'_Trot_dev'] = []
            result[specie+'_Tvib'] = []
            result[specie+'_Tvib_dev'] = []
            result[specie+'_intensity'] = []
            result[specie+'_intensity_dev'] = []

        for specname in self.spectra:
            result['spectrum'].append(specname)
            # if the list of simulations is empty, do not calculate residuals
            if not self.spectra[specname]['params'].info['species']:
                sumsq = numpy.nan
            else:
                residuals = self._dummy(self.spectra[specname]['params'].prms, specname)

                sumsq = numpy.sum(residuals[~numpy.isinf(residuals)]**2)
                if hasattr(self.spectra[specname]['spectrum'], 'y'):
                    y = self.spectra[specname]['spectrum'].y
                else:
                    y = self.spectra[specname]['spectrum']
                sumsq /= numpy.sum(y - self.spectra[specname]['params']['baseline'].value -
                        self.spectra[specname]['params']['baseline_slope']*numpy.arange(len(self.spectra[specname]['spectrum'])))
            result['reduced_sumsq'].append(sumsq)
            for specie in self.simulations:
                if specie in self.spectra[specname]['params'].info['species']:
                    result[specie+'_Trot'].append(
                        self.spectra[specname]['params'][specie + '_Trot'].value)

                    result[specie+'_Trot_dev'].append(
                        self.spectra[specname]['params'][specie + '_Trot'].stderr)

                    result[specie+'_Tvib'].append(
                        self.spectra[specname]['params'][specie + '_Tvib'].value)

                    result[specie+'_Tvib_dev'].append(
                        self.spectra[specname]['params'][specie + '_Tvib'].stderr)

                    result[specie+'_intensity'].append(
                        self.spectra[specname]['params'][specie + '_intensity'].value)

                    result[specie+'_intensity_dev'].append(
                        self.spectra[specname]['params'][specie + '_intensity'].stderr)
                else:
                    result[specie+'_Trot'].append(numpy.nan)
                    result[specie+'_Tvib'].append(numpy.nan)
                    result[specie+'_Trot_dev'].append(numpy.nan)
                    result[specie+'_Tvib_dev'].append(numpy.nan)
                    result[specie+'_intensity'].append(numpy.nan)
                    result[specie+'_intensity_dev'].append(numpy.nan)

        out = pandas.DataFrame(result)
        out.to_csv(filename)
        return out
