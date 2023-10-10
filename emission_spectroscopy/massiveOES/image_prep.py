import massiveOES, lmfit
import numpy
from collections import OrderedDict
#import matplotlib.pyplot as plt

def add_spectra(first, second):
   matched2, matched1 = massiveOES.spectrum.match_spectra(second, first)
   matched1.y += matched2.y
   return matched1

class Preparator(object):

    def __init__(self, codename, image, dark, sensitivity = None, wavs = None, **kwargs):
        """
        sensitivity: massiveOES.Spectrum object, y is the spectral sensitivity
        """
        self.raw_image = image
        self.dark = dark
        self.sensitivity = sensitivity #ocekavame massiveOES spektrum objekt
        self.codename = codename
        self.preprocess_dark_done = False
        self.preprocess_sens_done = False
        self.preprocess_wavs_done = False
        self.wavs = wavs
        self.slitf = kwargs.pop('slitf', [1e-9, 1e-9])

    def preprocess_dark(self):
        self.image = self.raw_image - self.dark
        self.preprocess_dark_done = True

    def preprocess_wavs(self):
        if self.wavs is None:
            try:
                self.measured_spectra_instance = massiveOES.MeasuredSpectra.from_json(self.codename + '_initial_wavs' + '.json')
            except:
                spec_dict = OrderedDict()
                spec_dict['full_chip'] = {'spectrum' : numpy.sum(self.image, axis=0)}
                self.measured_spectra_instance = massiveOES.MeasuredSpectra(spectra = spec_dict)
                self.measured_spectra_instance.to_json(self.codename + '_initial_wavs' + '.json')
                print('Generating inital calibration file as no wavs were iputed, please go and do some work to fix this!')
                return False

            self.wavs = {'wav_start' : self.measured_spectra_instance.spectra['full_chip']['params']['wav_start'],
                         'wav_step' : self.measured_spectra_instance.spectra['full_chip']['params']['wav_step'],
                         'wav_2nd' : self.measured_spectra_instance.spectra['full_chip']['params']['wav_2nd']}
            if self.wavs['wav_start'] == 0:
                print('Fited spectra starting at 0 nm??? I think you are doing it wrong!')
                self.wavs = None
                return False

        else:
            try:
                self.measured_spectra_instance = massiveOES.MeasuredSpectra.from_json(self.codename + '_initial_wavs' + '.json')
            except:
                spec_dict = OrderedDict()
                spec_dict['full_chip'] = {'spectrum' : numpy.sum(self.image, axis=0)}
                self.measured_spectra_instance = massiveOES.MeasuredSpectra(spectra = spec_dict, wavelengths = [self.wavs[0],self.wavs[1],self.wavs[2]])
                self.measured_spectra_instance.to_json(self.codename + '_initial_wavs' + '.json')
                print('Generating inital calibration file as no wavs were iputed, please go and do some work to fix this!')
                return False
            self.wavs = {'wav_start' : self.measured_spectra_instance.spectra['full_chip']['params']['wav_start'],
             'wav_step' : self.measured_spectra_instance.spectra['full_chip']['params']['wav_step'],
             'wav_2nd' : self.measured_spectra_instance.spectra['full_chip']['params']['wav_2nd']}
            if self.wavs['wav_start'] == 0:
                print('Fited spectra starting at 0 nm??? I think you are doing it wrong!')
                self.wavs = None
                return False
        self.preprocess_wavs_done = True

        return True


    def divide_et_impera(self, division_size):
        if self.wavs is None:
            print('Hey I need some wavelengths to start with!')
            return False
        spec_dict = OrderedDict()
        for i in range(int(self.image.shape[0]/division_size)):
            spec_dict[str((i*division_size + (i+1)*division_size)/2.)] = {'spectrum' : numpy.sum(self.image[i*division_size:(i+1)*division_size,:], axis=0)}
        self.measured_spectra_instance = massiveOES.MeasuredSpectra(spectra = spec_dict,
                        wavelengths = [self.wavs['wav_start'].value,
                                       self.wavs['wav_step'].value,
                                       self.wavs['wav_2nd'].value],
                        slitf_gauss=self.slitf[0], slitf_lorentz=self.slitf[1])
        self.measured_spectra_instance.to_json(self.codename + '_divided_wavs' + '.json')
        print('Image divided to spectra for wav calibration. Now fit them so I can fit the profile!')
        return True

    def find_the_bend(self, plot_it = True, ignore = None, polyorder = 2):
        try:
            self.measured_spectra_instance = massiveOES.MeasuredSpectra.from_json(self.codename + '_divided_wavs' + '.json')
            # for i, spec in enumerate(self.measured_spectra_instance.spectra):
            #     print(i, self.measured_spectra_instance.spectra[spec]['params']['wav_start'], self.measured_spectra_instance.spectra[spec]['params']['wav_start'].stderr)
        except:
            print('I have no divided image wavs file to work with!')
            return False
        wav_start = []
        wav_start_error = []
        image_position = []
        for spectrum in self.measured_spectra_instance.spectra:
            wav_start.append(self.measured_spectra_instance.spectra[spectrum]['params']['wav_start'].value)
            wav_start_error.append(self.measured_spectra_instance.spectra[spectrum]['params']['wav_start'].stderr)
            image_position.append(float(spectrum))
        wav_start = numpy.array(wav_start)
        image_position = numpy.array(image_position)
        wav_start_error = numpy.array(wav_start_error)

        mask = (image_position >= ignore[0]) * (image_position < ignore[1])
        if ignore is None:
            self.polynomial_of_wavs_to_rows = numpy.polyfit(image_position,wav_start, polyorder, rcond=None, full=False, w=1./(0.0001+wav_start_error), cov=False)
        else:
            self.polynomial_of_wavs_to_rows = numpy.polyfit(image_position[mask],
            wav_start[mask], polyorder, rcond=None, full=False, w=1./(0.0001+wav_start_error[mask]), cov=False)

        plt.errorbar(image_position,wav_start, yerr=wav_start_error)
        self.p = numpy.poly1d(self.polynomial_of_wavs_to_rows)
        plt.plot(image_position, self.p(image_position))
        plt.show()

    def fix_the_bend(self):
        pass

    def final_cut(self, treshold, limit=1024):
        self.row_max = numpy.percentile(self.image, 99, axis = 1)
        spec_dict = OrderedDict()
        wav_dict = OrderedDict()

        for i in range(self.image.shape[0]):
            summ = 0
            for j in range(self.image.shape[0] - i):

                summ += self.row_max[j+i]
                if summ >= treshold:
                    break

            if j > limit:
                continue

            if summ < treshold:
                break

            if j == 0:
                position = str(i) +'_'+str(i) +'_'+str(i)
                spec_dict[position] = {'spectrum' : self.image[i,:]}
                wav_dict[position] = self.p(i)


            if j > 0:
                position = str(numpy.average(numpy.arange(i,i+j), weights= self.row_max[i:i+j]))+'_'+str(i) +'_'+str(i+j)
                core_x = massiveOES.MeasuredSpectra.get_wavs(len(self.image[i,:]), self.p(i),self.wavs['wav_step'],self.wavs['wav_2nd'])
                core_spec = massiveOES.spectrum.Spectrum(x = core_x , y = self.image[i,:])
                wav_dict[position] = self.p(i)
                for k in range(i+1,i+j+1):
                    temp_x =  massiveOES.MeasuredSpectra.get_wavs(len(self.image[k,:]), self.p(k),self.wavs['wav_step'],self.wavs['wav_2nd'])
                    temp_spec =  massiveOES.spectrum.Spectrum(x = temp_x , y = self.image[k,:])
                    core_spec = add_spectra(core_spec, temp_spec)
                spec_dict[position] = {'spectrum' : self.image[i,:]}

        self.measured_spectra_instance = massiveOES.MeasuredSpectra(spectra = spec_dict,
                                            wavelengths = [self.wavs['wav_start'].value,
                                            self.wavs['wav_step'].value,self.wavs['wav_2nd'].value],
                                            slitf_gauss=self.slitf[0], slitf_lorentz=self.slitf[1])

        for spec_name in self.measured_spectra_instance.spectra:
            self.measured_spectra_instance.spectra[spec_name]['params']['wav_start'].value = wav_dict[spec_name]

        self.measured_spectra_instance.to_json(self.codename + '_final' + '.json')


    def preprocess_sens(self):
        if not self.preprocess_dark_done:
            print('Substract dark first!!!!')
            return False
        if not self.preprocess_wavs_done:
            print('No calibration of wavs present!!!!')
            return False
        x = massiveOES.MeasuredSpectra.get_wavs(self.image.shape[0],
                                     self.wavs['wav_start'],
                                     self.wavs['wav_step'],
                                     self.wavs['wav_2nd'])
        calibration = numpy.interp(x, self.sensitivity.x, self.sensitivity.y, left=None, right=None, period=None)
        self.image = self.image/calibration[:,None]
