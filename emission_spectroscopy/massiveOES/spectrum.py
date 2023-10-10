import numpy as np
import warnings
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import logging
#from LIF import Voigt
from scipy.interpolate import interp1d
from scipy.special import wofz




class Spectrum(object):
    """An object holding x and y axis and implememnting some methods
    often used in processing of spectroscopic data.
    """
    def __init__(self, **kwargs):
        """

        kwargs:
        -------
        x: 1D numpy array of wavelengths (or wavenumbers, if you prefffer)

        y: 1D numpy array of intensities (or fluxes or signal strengths, whatever)

        x and y should be equally long (though it is not strictly
        forbidden, most of the methods will fail otherwise)

        """
        self.x = kwargs.pop('x', [])
        self.y = kwargs.pop('y', [])
        if len(self.y) > 0:
            self.maximum = np.max(self.y)
        else:
            self.maximum = None
        normalize = kwargs.pop('normalize', False)
        if normalize and self.maximum is not None:
            self.y /= self.maximum #normalize intensities
        if len(self.x) != len(self.y):
            warnings.warn("Spectrum: length of x and y mismatch!", Warning)
            
    def __len__(self):
        lx = len(self.x)
        ly = len(self.y)
        if lx == ly:
            return lx
        else:
            warnings.warn("Spectrum: length of x and y mismatch!", Warning)
            return lx, ly


    def find_peaks(self, only_strong = False, only_strong_factor = 0.1, **kwargs):
        """Function takes array of data and peaks and valleys by finding
    second numerical derivative and returns 2 Spectrum objects: one
    consisiting only of peaks, the other only of valleys

        """
#list of arrays in this order [[peak position],
#    [peak height], [valley position], [valley height]]
        first_derivative = np.diff(self.y)
        result = np.zeros(len(self.y))
        
        first_derivative /= np.abs(first_derivative)
        second_derivative = np.diff(first_derivative)
        second_derivative /= -np.abs(second_derivative)
        result[1:-1] = second_derivative
        
        peak_position = self.x[result == 1]
        peak_height = self.y[result == 1]
        
        if only_strong:
            condition = peak_height > only_strong_factor*np.max(peak_height)
            peak_height = peak_height[condition]
            peak_position = peak_position[condition]

        valley_position = self.x[result == -1]
        valley_depth = self.y[result == -1]
            
        valleys = Spectrum(x = valley_position, y = valley_depth, normalize=False)
        peaks = Spectrum(x = peak_position, y = peak_height, normalize=False)

        return peaks, valleys

    @staticmethod
    def match_peaks(first, second, **kwargs):
        """
        Find the peaks at similar positions and make them into pairs 
        for eventual comparison of their heights

        args:
        -----
        first: massiveOES.Spectrum object containting peaks

        second: massiveOES.Spectrum object containign peaks
        
        **kwargs:
        ---------
        tolerance:  in the units of your spectra's x-axis. The recomended 
                    value is 2 x (the spetral distance of measured points). 
                    Defaults to 0.032. Note, that this may not be what you want!
        """
        tolerance = kwargs.pop('tolerance', 0.032)
        # if tolerance is None:
        #     warnings.warn("Spectrum.match_peaks(): please, specify the tolerance in the units of your spectra's x-axis. The recomended value is 2 x (the spetral distance of measured points).", Warning)
        #     return None
        result = []
        for i, first_x in enumerate(first.x):
            testpeaks_idcs = np.where(first.x < first_x + tolerance)
            testpeaks_idcs = np.where(first.x[testpeaks_idcs] < first_x - tolerance)
            if len(testpeaks_idcs) > 1:
                continue
            near_peak_idcs = np.where(second.x < first_x + tolerance)
            near_peak_idcs = np.where(second.x[near_peak_idcs] > first_x - tolerance)
            if len(near_peak_idcs[0]) == 1:
                # print('first_x = ',type(first_x))
                # print('first.y[i] = ',type(first.y[i]))
                # print('second.x[n] = ',second.x[near_peak_idcs[0][0]])
                # print('seconf.y[n] = ',second.y[near_peak_idcs[0][0]])
                result.append([first_x, first.y[i], second.x[near_peak_idcs[0][0]], second.y[near_peak_idcs[0][0]]])
        if not result:
            return (None, None)
        result = np.array(result)
        matched_first = Spectrum(x = result[:,0], y = result[:,1], normalize=False)
        matched_second = Spectrum(x = result[:,2], y = result[:,3], normalize=False)
        return matched_first, matched_second


    def ratio_of_integrals(self, first_range, second_range):
        """return ratio of integrals under two specified spectral regions
        (first_range and second_range). Can be used to estimate
        rotational or vibrational temperature.

        return: ration (float)

        """
        firstx = self.x[self.x > first_range[0]]
        firsty = self.y[self.x > first_range[0]]
        firsty = firsty[firstx < first_range[1]]
        firstx = firstx[firstx < first_range[1]]

        secondx = self.x[self.x > second_range[0]]
        secondy = self.y[self.x > second_range[0]]
        secondy = secondy[secondx < second_range[1]]
        secondx = secondx[secondx < second_range[1]]

        ratio = np.trapz(firsty, firstx) / np.trapz(secondy, secondx)

        return ratio
    
    @staticmethod
    def find_nearest(array,value):
        idx = (np.abs(array-value)).argmin()
        return idx

    def guess_temperature(self, first_range, second_range, lookup_table):
        """Guess temperature from ratio of integrals of two spectral
        regions. Lookup table for this, resulting from preceding
        calculation has to be provided.
        """
        ratio = self.ratio_of_integrals(first_range, second_range)
        idx = self.find_nearest(lookup_table[:,1], ratio)
        temp = lookup_table[idx, 0]
        return temp

    def convolve_with_slit_function(self, **kwargs):
        """
        Broaden the peaks in the spectrum by Voigt profile and by a rectangle of given width. 
        Changes state of the instance, returns nothing.
        
        **kwargs:
        ---------
        gauss: *float* gaussian HWHM, defaults to 0.1
        lorentz: *float* lorentzian HWHM, defaults to 1e-9
        step: *float* distance between pixels in nm
        """
        #logging.debug('====////****convolve_with_slit_function****////====')
        gauss = kwargs.pop('gauss', 0.1)
        lorentz = kwargs.pop('lorentz', 1e-9)
        #logging.debug('gauss = %s', gauss)
        #logging.debug('lorentz =%f ', lorentz)
        slit = Voigt(self.x, gauss, lorentz, np.mean(self.x), 1)
        #slit /= np.trapz(slit, self.x) asi spatne
        slit /= np.sum(slit)
        
        #np.save('slit.npy', slit)
        instrumental_step = kwargs.pop('step', None)
        numpoints = len(self.y)
        if instrumental_step is None:
            convolution_profile = slit[slit>max(slit)/1000.]
        else:
            # convolvuju s obdelnikem o delce jednoho pixelu na spektometru 
            # aby nedochazelo ke ztrate tenkych car pri interpolaci na realne spektrum
            # tady normalizuju ctvercovy profil na jednicku
            simulated_step = self.x[1]-self.x[0]
            if instrumental_step/simulated_step < 1:
                msg = 'Your simulated spectra resolution is more rough than experimental data.'
                warnings.warn(msg, UserWarning)
                convolution_profile = slit[slit>max(slit)/1000.]
            else:
                instrumetal_step_profile = np.ones(int(instrumental_step/simulated_step) + 1)
                #instrumetal_step_profile /= np.sum(instrumetal_step_profile)
                if len(slit) >= len(instrumetal_step_profile):
                    convolution_profile_uncut = fftconvolve(slit,instrumetal_step_profile,mode='same')
                else:
                    convolution_profile_uncut = fftconvolve(instrumetal_step_profile,slit,mode='same')
                convolution_profile = convolution_profile_uncut[convolution_profile_uncut>max(convolution_profile_uncut)/1000]
        
        #np.save('conv_profile.npy', convolution_profile_uncut)
        #print 'integral = ', np.trapz(self.y, self.x)
        self.y = fftconvolve(self.y, convolution_profile, mode = 'same')
        #print 'integral after fftconv= ', np.trapz(self.y, self.x)
        #np.save('yconv.npy', self.y)

        if len(self.y)>0:
            pass #normalization is not good at this point
            #self.y /= np.max(self.y) 
        else:
            self.y = np.ones(numpoints) * 1e100 # if the array gets destroyed by fftconvolve, 
                                                # set array to ridiculously huge values
        #np.save('y_normed.npy', self.y)        
        if any(np.isnan(self.y)):
            self.y[:] = 1e100
            logging.debug('conv = %s', self.y)
        return 

    def refine_mesh(self, points_per_nm = None):
        """
        adds artificial zeros in between lines. Usually used after creating a simulated spectrum before 
        convolution with slit function. Necessary for later comparing simulation with measurement.

        return:
        Spectrum objects with pretty many points (or fine mesh, if you preffer)
        """
        
            

        start_spec = np.min(self.x) - 2 #prevent lines from falling to edges
        end_spec = np.max(self.x) + 2
        if points_per_nm is None:
            points_per_nm = 3000
            #points_per_nm = int(10000./np.abs(end_spec-start_spec))
            
        
        no_of_points = int(np.abs(end_spec-start_spec)*points_per_nm)
        
        #print(no_of_points)
        
        spec = np.zeros((no_of_points, 2))
        
        spec[:,0] = np.linspace(start_spec, end_spec, no_of_points)

        for i in range(len(self.x)):
            index = int((self.x[i] - start_spec)*points_per_nm + 0.5)
            spec[index,1] += self.y[i]
        self.x = spec[:,0]
        self.y = spec[:,1]
        return spec



def match_spectra(sim_spec, exp_spec):
    """
    Take two Spectrum objects with different x-axes (the ranges must partially overlap) 
    and return a tuple of Spectrum objects defined at the same x-points. This enables comparing.
    Args:
    ----
    exp_spec: *Spectrum object*, experimental spectrum
    sim_spec: *Spectrum object*, simulated spectrum. 

    Returns:
    (Spectrum_simulated, Spectrum_experimental): a tuple of Spectrum objects, with identical x-axes.
    """
    # ## crop
    # if np.min(exp_spec.x) < np.min(sim_spec.x):
    #     #print 'cropping to ', np.min(exp_spec.x)
    #     exp_w = exp_spec.x[exp_spec.x>np.min(sim_spec.x)]
    #     exp_i = exp_spec.y[exp_spec.x>np.min(sim_spec.x)]
    # else:
    #     #print 'not cropping...'
    #     exp_w = exp_spec.x
    #     exp_i = exp_spec.y
    # if np.max(exp_spec.x) > np.max(sim_spec.x):
    #     #print 'cropping to ', np.max(exp_spec.x)
    #     exp_w = exp_w[exp_w<np.max(sim_spec.x)]
    #     exp_i = exp_i[exp_w<np.max(sim_spec.x)]
    ## crop end
    ######################################################
    #logging.debug('sim_w cropped = %f ... %f', sim_w[0], sim_w[-1])
    #logging.debug('sim_i cropped = %f ... %f', sim_i[0], sim_i[-1])
    ######################################################

    if len(sim_spec.x) == 0:
        return ( Spectrum(x = exp_spec.x, y=np.zeros_like(exp_spec.x)), exp_spec)

    if np.min(exp_spec.x) < np.min(sim_spec.x):
        sim_spec.x = np.concatenate([[min(exp_spec.x)],[min(sim_spec.x)-1e-3], sim_spec.x])
        sim_spec.y = np.concatenate([[0],[0], sim_spec.y])
    if np.max(exp_spec.x) > np.max(sim_spec.x):
        sim_spec.x = np.concatenate([sim_spec.x,[max(sim_spec.x)+1e-3], [max(exp_spec.x)]])
        sim_spec.y = np.concatenate([sim_spec.y,[0], [0]])
    interp = interp1d(sim_spec.x, sim_spec.y)
    y_interp = interp(exp_spec.x)

    #ret = (Spectrum(x = exp_w, y = y_interp, normalize = False),  Spectrum(x = exp_w, y = exp_i, normalize = False))
    ret = ( Spectrum(x = exp_spec.x, y=y_interp), exp_spec)

    return ret

def compare_spectra(spectrum_exp, spectrum_sim, **kwargs):
    """
    spectrum_exp: Spectrum object
    spectrum_sim: Spectrum object
    The wavelength axes are expected to differ, but overlap
    
    plot_it: boolean. if True, the matched spectra will be plotted to 
             'plots/comp_spectra'+suff+'.svg'
    suff: string. Needed when this function is called in a loop to prevent 
          overwriting of the output. 

    returns: sqrt of (sum of squares of differences of the spectra divided 
             by (the number of points)**2 )
    """
    plot_it = kwargs.pop('plot_it', False)
    suff = kwargs.pop('suff', '')
    show_it = kwargs.pop('show_it', False)

    matched = match_spectra(spectrum_sim, spectrum_exp)
    dif = matched[0].y - matched[1].y
    logging.debug('len(dif) = %i', len(dif))
    logging.debug('dif = ')
    logging.debug('%s', dif)

    if plot_it:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1,1)
        ax.plot(matched[1].x, matched[1].y, label='simulation')
        ax.plot(matched[0].x, matched[0].y, label='experiment')
        fig.text(0.5,0.02, 'sumsq = {:,.5e}'.format(ret) , ha = 'center')
        fig.savefig('plots/comp_spectra'+suff, format = 'svg')
        plt.close()
    if show_it:
        import matplotlib.pyplot as plt
        plt.gca().clear()
        plt.plot(matched[1].x, matched[1].y, label='simulation')
        plt.plot(matched[0].x, matched[0].y, label='experiment')
        plt.legend()
        plt.draw()
        plt.pause(0.001)
    #print(np.dot(dif, dif) / len(dif)**2)
    #print len(dif)
    return dif

def compare_spectra_weighted(spectrum_exp, spectrum_sim, **kwargs):
    """
    weight the sum of squared residuals by the measured intensity at given position.
    """
    dif = compare_spectra(spectrum_exp, spectrum_sim, **kwargs)
    dif *= spectrum_exp.y
    return dif

def compare_spectra_reduced_susmq(spectrum_exp, spectrum_sim, **kwargs):
    """
    compare spectra, take the residuals and divide them by (len of the measured spectrum)**2,
    usefull, if the length of the measurement is expected to vary during the minimization process.
    """
    dif = compare_spectra(spectrum_exp, spectrum_sim, **kwargs)
    ret = np.dot(dif, dif) / len(dif)**2
    logging.debug('sumsq = {:.8e}'.format(ret))
    if np.isnan(ret):
        ret = 1e100
    print('sumsq = ', ret)
    return ret
    
    
        
# def add_spectra(simulated_spectra, amplitudes):
#     added_spectra = np.zeros((len(simulated_spectra[0].x),2))
#     added_spectra[:,0] = simulated_spectra[0].x
#     for amplitude, spectrum in zip(amplitudes,simulated_spectra):
#         added_spectra[:,1] += amplitude*spectrum.y 
#     return added_spectra

def add_spectra(simulated_spectra, amplitudes):
    """
    calculate linear combination of several Spectrum objects

    args:
    -----
    simulated_spectra: list of massiveOES.Spectrum objects

    amplitudes: list of floats, coefficients in the linear combination

    return:
    ------
    massiveOES.Spectrum object - linear combination of several Spectrum objects

    
    """
    if not simulated_spectra:
        return Spectrum()
    added_spectra = np.zeros((len(simulated_spectra[0].x),2))
    added_spectra[:,0] = simulated_spectra[0].x
    for amplitude, spectrum in zip(amplitudes,simulated_spectra):
        added_spectra[:,1] += amplitude*spectrum.y 
    ret = Spectrum(x = added_spectra[:,0], y = added_spectra[:,1], normalize = False)
    return ret


#def temp_spectra(large_spectra, temperatures):# temperatures = list ((Tv , Tr) (Tv,Tr)...
#    temp_spec_list = []
#    for temperature, spectrum in zip(temperatures,large_spectra):
#        temp_spec = interpolate_spectra(spectrum, temperature[0], temperature[1])
#        temp_spec_list.append(Spectrum(x=temp_spec[:,0], y=temp_spec[:,1]))
#    return temp_spec_list
    
def voigt(x, y):
    """
Taken from `astro.rug.nl <http://www.astro.rug.nl/software/kapteyn-beta/kmpfittutorial.html?highlight=voigt#voigt-profiles/>`_

The Voigt function is also the real part of
`w(z) = exp(-z^2) erfc(iz)`, the complex probability function,
which is also known as the Faddeeva function. Scipy has
implemented this function under the name `wofz()`
    """
    z = x + 1j*y
    I = wofz(z).real
    return I

def Voigt(nu, alphaD, alphaL, nu_0, A, a=0, b=0):
    """
Taken from `astro.rug.nl <http://www.astro.rug.nl/software/kapteyn-beta/kmpfittutorial.html?highlight=voigt#voigt-profiles/>`_

The Voigt line shape in terms of its physical parameters

Args:
  **nu**:  light frequency axis

  **alphaD**:  Doppler broadening HWHM

  **alphaL**:  Lorentzian broadening HWHM

  **nu_0**:  center of the line

  **A**:  integral under the line

  **a**:  constant background

  **b**:  slope of linearly changing background (bg = a + b*nu)

Returns:
  **V**: The voigt profile on the nu axis
    """
    if alphaD == 0:
        alphaD = 1e-10
    if alphaL == 0:
        alphaL = 1e-10
    f = np.sqrt(np.log(2))
    x = (nu-nu_0)/alphaD * f
    y = alphaL/alphaD * f
    backg = a + b*nu
    V = A*f/(alphaD*np.sqrt(np.pi)) * voigt(x, y) + backg
    return V
    
    
                
        
#def fit_ratios(measured_spectrum, simulated_spectra):
#    bounds = []
#    for spectrum in simulated_spectra:
#        bounds.append((0,1.5))
#            
#    p_init = np.ones(len(simulated_spectra))
#    to_min = lambda p: 1e10*spect_fit.compare_spectra(np.array([measured_spectrum.x, measured_spectrum.y]).T, add_spectra(simulated_spectra, p), show_it=True)
#    result = minimize(to_min, p_init ,bounds = bounds, method = 'L-BFGS-B')
#    return result
#    
#def fit_temperatures(measured_spectrum,simulated_spectra):
#    
#    
#    
#def fit_all(measured_spectrum,simulated_spectra):
#    
#    to_min = lambda p_t: 1e10*spect_fit.compare_spectra(np.array([measured_spectrum.x, measured_spectrum.y]).T, model_spectra(simulated_spectra, p_t), show_it=True)
#    result = minimize(to_min, p_init ,bounds = bounds, method = 'L-BFGS-B')
    
    
def compare_spectra_by_peaks(spectrum1, spectrum2, **kwargs):
    """
    kwargs:
       show_it: *bool* if True, result is plotted and shown

       rest of kwargs goes to Spectrum.match_peaks()
    """
    peaks1, valleys1 = spectrum1.find_peaks(**kwargs)
    peaks2, valleys2 = spectrum2.find_peaks(**kwargs)
    show_it = kwargs.pop('show_it', False)
    matched1, matched2 = Spectrum.match_peaks(peaks1, peaks2, **kwargs)
    if matched1 is None:
        raise RuntimeError('No matching peaks found!')

    dif = -(matched1.y - matched2.y)
    resid_y = compare_spectra(spectrum1, spectrum2)
    
    if show_it:
        import matplotlib.pyplot as plt
        plt.plot(spectrum1.x, spectrum1.y, label = '1st')
        plt.plot(matched1.x, matched1.y, 'o',label = '1st peaks')
        plt.plot(spectrum2.x, spectrum2.y, label = '2nd')
        plt.plot(matched2.x, matched2.y, '^',label = '2nd peaks')
        plt.plot(matched1.x, dif, 's' ,label = 'residuals')
        plt.plot(spectrum1.x, resid_y, color = '0.25', label = 'line residuals')
        plt.xlim(min(matched1.x), max(matched1.x))
        plt.legend()
        #plt.draw()
        #plt.pause(.0001)
        #plt.gca().clear()        

    return dif,matched1.y
    

def compare_spectra_by_peaks_reduced_sumsq(spectrum1, spectrum2, **kwargs):
    dif,intensity = compare_spectra_by_peaks(spectrum1, spectrum2, **kwargs)
    dif *= intensity
    return dif**2 / len(dif)**2
    #result = np.sum(np.dot(dif, dif)) / len(dif)**2 #weight is prop to intensity
    #if np.isnan(result):
    #    result = 1e100
    #return result
