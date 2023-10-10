import inspect
from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'qt4'

from traits.api import HasTraits, List, Str, Float, File, Button,\
Instance, Int, Array, on_trait_change, Bool, Enum
from traitsui.api import View, Item, HGroup, Handler,\
EnumEditor, TextEditor, TableEditor, VGroup, HSplit, FileEditor, Spring,\
StatusItem, TextEditor, ListStrEditor


from traitsui.menu import OKButton, CancelButton

from traitsui.table_column \
    import ObjectColumn

from traitsui.extras.checkbox_column \
    import CheckboxColumn

# from traitsui.key_bindings \
#     import KeyBinding, KeyBindings

from massiveOES import MeasuredSpectra, Parameters,\
 puke_spectrum, Spectrum, SpecDB#, LargeSimulatedSpectra
from massiveOES.spectrum import compare_spectra, match_spectra
#from traitsui.tabular_adapter import TabularAdapter
from chaco.api import Plot, ArrayPlotData
from chaco.tools.api import PanTool, BetterZoom
from enable.api import ComponentEditor
import lmfit
from numpy import inf, trapz, array, savetxt
from datetime import datetime

from pyface.api import FileDialog, OK, CANCEL, MessageDialog

from copy import copy, deepcopy

from massiveOES.linearization import Coefficients
from massiveOES.bp_inspect import BoltzmannInspector
from pyface.api import ImageResource
###################################
## Path to icons
import sys
import os
import glob
from pyface.resource_manager import resource_manager

MASSIVEOES_PATH = os.path.dirname(__file__)

def set_pyface_resource_path():
    """Set Pyface image/icon path to local application folder"""
    if hasattr(sys, 'frozen'):  # If we're in an .exe file
        path = os.path.dirname(sys.executable)
        for root_path in ['pyface', 'traitsui']:
            for root, dirs, _files in os.walk(root_path):
                if 'images' in dirs:
                    resource_manager.extra_paths.append(os.path.join(path, root))
                    print(path, root)


def set_traitsui_resource_path():
    """Set Traitsui image/icon path to local application folder"""
    if hasattr(sys, 'frozen'):  # If we're in an .exe file
        path = os.path.dirname(sys.executable)
        os.environ['TRAITS_IMAGES'] = os.path.join(path, r'traitsui\image\library')

set_pyface_resource_path()
set_traitsui_resource_path()
##################################
###end of path to icons


# def vypis_zpravu(obj):
#     print obj.specie, obj.Trot, obj.Tvib

# key_bindings = KeyBindings(
#     KeyBinding( binding1    = 'Ctrl-s',
#                 description = 'Save to a file',
#                 method_name = 'save_file' ))


# class KBHandler(Handler):
#     def save_file(self):
#         print 'Handler is saving the file, pyco!'

simulations_editor = TableEditor(
    sortable     = False,
    configurable = True,
    auto_size    = False,
    deletable    = True,
    selection_mode = 'row',
    selected  = 'simulation',
    #on_select = vypis_zpravu,
    columns  = [ ObjectColumn( name  = 'specie',  label = 'Specie'),
                 #ObjectColumn( name = 'filename', editable = False, width  = 0.24,
                 #              horizontal_alignment = 'left' ),
                 ObjectColumn( name   = 'intensity',     label  = 'Intensity' ),
                 ObjectColumn( name   = 'intensity_min', label  = 'I min' ),
                 ObjectColumn( name   = 'intensity_max', label  = 'I max' ),
                 CheckboxColumn( name   = 'intensity_vary', label  = 'fit I' ),
                 ObjectColumn( name   = 'Trot',     label  = 'Trot' , width = 50),
                 ObjectColumn( name   = 'Trot_min',   label  = 'Trot min' ),
                 ObjectColumn( name   = 'Trot_max',       label  = 'Trot max' ),
                 CheckboxColumn( name   =  'Trot_vary',    label  = 'fit Trot'),
                 ObjectColumn(name = 'Tvib', label = 'Tvib', width = 50),
                 ObjectColumn(name = 'Tvib_min', label = 'Tvib min'),
                 ObjectColumn(name = 'Tvib_max', label = 'Tvib max'),
                 CheckboxColumn(name = 'Tvib_vary', label = 'fit Tvib')] )

class TrSimulation(HasTraits):
    #filename = Str('')
    specie = Str('')

    intensity = Float(1.)
    intensity_min = Float(0)
    intensity_max = Float(2.)
    intensity_vary = Bool(True)

    Trot = Float(2e3)
    Trot_min = Float(500)
    Trot_max = Float(10000)
    Trot_vary = Bool(True)

    Tvib = Float(2e3)
    Tvib_min = Float(1000)
    Tvib_max = Float(10000)
    Tvib_vary = Bool(True)


    def __init__(self):
        """
simSpec: specDB object, to be filled by Plotter
        """
        self.simSpec = None

class TrParameters(HasTraits):
    number_of_pixels = Int
    wav_start = Float
    wav_step = Float
    wav_2nd = Float
    gauss = Float(1e-9)
    lorentz = Float(1e-9)
    baseline = Float
    baseline_slope = Float

    wav_start_min = Float(-inf)
    wav_start_max = Float(inf)
    wav_start_vary = Bool(True)

    wav_step_min = Float(-inf)
    wav_step_max = Float(inf)
    wav_step_vary = Bool(True)


    wav_2nd_min = Float(-inf)
    wav_2nd_max = Float(inf)
    wav_2nd_vary = Bool(True)

    gauss_min = Float(-inf)
    gauss_max = Float(inf)
    gauss_vary = Bool(True)

    lorentz_min = Float(-inf)
    lorentz_max = Float(inf)
    lorentz_vary = Bool(True)

    baseline_min = Float(-inf)
    baseline_max = Float(inf)
    baseline_vary = Bool(True)

    baseline_slope_min = Float(-inf)
    baseline_slope_max = Float(inf)
    baseline_slope_vary = Bool(True)


    simulation = Instance(TrSimulation)
    # simulations = List(TrSimulation,
    #                    editor = TabularEditor(adapter=SimulationAdapter(),
    #                                           selected='simulation',
    #                                           auto_update=True,
    #                                           editable=True,
    #                                           operations=['edit', 'delete']))
    simulations = List(TrSimulation, editor = simulations_editor)


def TrParamsToParams(trParams):
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    print( 'TrParamsToParams called from:', calframe[1][3])

    ret = Parameters(slitf_gauss = trParams.gauss, slitf_lorentz=trParams.lorentz,
                     baseline=trParams.baseline,  baseline_slope = trParams.baseline_slope, simulations=[],
                     wavelengths = (trParams.wav_start, trParams.wav_step,
                                     trParams.wav_2nd), number_of_pixels = trParams.number_of_pixels)

    for param in ['gauss', 'lorentz']:
        for suff in ['min', 'max', 'vary']:
            expr = 'ret[\'slitf_' + param + '\'].'+suff+'='+'trParams.'+param+'_'+suff
            exec(expr)

    for param in ['wav_start', 'wav_step', 'wav_2nd', 'baseline', 'baseline_slope']:
        for suff in ['min', 'max', 'vary']:
            exec('ret[\'' + param + '\'].'+suff+'='+'trParams.'+param+'_'+suff)

    for sim in trParams.simulations:
        #print "id(simSpec): ", id(sim.simSpec)
        ret.add_specie(sim.simSpec, Trot = sim.Trot, Tvib=sim.Tvib,
                       intensity = sim.intensity)
        specie = sim.simSpec.specie_name
        ret[specie+'_Trot'].min = sim.Trot_min
        ret[specie+'_Trot'].max = sim.Trot_max
        ret[specie+'_Trot'].vary = sim.Trot_vary

        ret[specie+'_Tvib'].min = sim.Tvib_min
        ret[specie+'_Tvib'].max = sim.Tvib_max
        ret[specie+'_Tvib'].vary = sim.Tvib_vary

        ret[specie+'_intensity'].min = sim.intensity_min
        ret[specie+'_intensity'].max = sim.intensity_max
        ret[specie+'_intensity'].vary = sim.intensity_vary
    return ret


def ParamsToTrParams(params, tr_params, measSpec=None):
    ret = tr_params

    tr_params.number_of_pixels = params.number_of_pixels
    for param in ['wav_start',
                  'wav_step',
                  'wav_2nd',
                  'baseline',
                  'baseline_slope']:
        exec('ret.'+param +'=params[\''+param+'\'].value' )
        for suff in ['min', 'max', 'vary']:
            exec('ret.' + param + '_' +suff + '=params[\''+param+'\'].'+suff)

    for param in ['gauss', 'lorentz']:
        exec('ret.'+param +'=params[\'slitf_'+param+'\'].value' )
        for suff in ['min', 'max', 'vary']:
            exec('ret.' + param + '_' + suff + '=params[\'slitf_'+param + '\'].'+ suff)

    while ret.simulations:
        ret.simulations.pop()
    for specie in params.info['species']:
        #ret_sim = TrSimulation(params.info[specie+'_sim'])
        ret_sim = TrSimulation()
        ret_sim.specie = specie
        if measSpec is not None:
            ret_sim.simSpec = measSpec.simulations[specie]
        # try:
        #     ret_sim.filename = params.info[specie+'_sim'].status['sim_filename']
        # except KeyError:
        #     pass
        ret_sim.intensity = params[specie+'_intensity'].value
        ret_sim.intensity_min = params[specie+'_intensity'].min
        ret_sim.intensity_max = params[specie+'_intensity'].max
        ret_sim.intensity_vary = params[specie+'_intensity'].vary

        ret_sim.Trot = params[specie+'_Trot'].value
        ret_sim.Trot_min = params[specie+'_Trot'].min
        ret_sim.Trot_max = params[specie+'_Trot'].max
        ret_sim.Trot_vary = params[specie+'_Trot'].vary

        ret_sim.Tvib = params[specie+'_Tvib'].value
        ret_sim.Tvib_min = params[specie+'_Tvib'].min
        ret_sim.Tvib_max = params[specie+'_Tvib'].max
        ret_sim.Tvib_vary = params[specie+'_Tvib'].vary

        ret.simulations.append(ret_sim)
    return


class MeasSpecHandler(Handler):
    """Handler class to enable dynamic changes of the dropdown list to
    select a particular measured spectrum. Is used by MeasSpecLoader class each
    time, when new file with measured spectrum is loaded.
    """

    specs = List(Int)

    def object_open_button_changed(self, info):
        if info.object.measSpec is not None:
            self.specs = list(range(len(info.object.measSpec.spectra)))
            info.object.which_measured=0

class Info(HasTraits):
    msg = Str('')

    warning = View(Item('msg',show_label=False,style='readonly'),
                   title='Warning',kind='modal',buttons = ['OK'])

class MeasSpecLoader(HasTraits):
    which_measured = Int
    meas_x=Array
    meas_y=Array
    reports = List(Str)
    report = Str('')
    open_button = Button('Open')
    save_as_button = Button('Save as')
    #save_button = Button('Save')
    #invalid_species = Str('')

    view = View(
        HGroup(Item('open_button', show_label = False),
               Item('which_measured',
                    editor=EnumEditor(name='handler.specs')),
               Item('save_as_button', show_label = False)),
               #Item('save_button', show_label=False)),
        #Item('invalid_species', style='readonly'),
        handler = MeasSpecHandler
    )

    def __init__(self, **kwargs):
        self.measSpec = kwargs.pop('measSpec', None)
        self.filename_out = ''
        self.filename = ''



    def _save_as_button_changed(self):
        wildcard = 'Shareable MassiveOES projects in json (*.json)|*.json| Internal OES format  (*.oes)|*.oes | All files (*.*)|*.*'
        dlg = FileDialog(action='save as', title='Save project as',
                wildcard=wildcard)
        if dlg.open() == OK:
            self.filename_out = dlg.path
            filename, extension = os.path.splitext(dlg.path)
            if (extension == '.oes') or (extension == '.OES'):
                self.measSpec.save(dlg.path)
            elif (extension == '.json') or (extension == '.json'):
                print(dlg.path)
                self.measSpec.to_json(dlg.path)

    # def _save_button_changed(self):
    #     if self.filename_out != '':
    #         self.measSpec.save(self.filename_out)
    #         print 'saved...'

    def _open_button_changed(self):
        wildcard = 'Shareable MassiveOES projects in json (*.json)|*.json|Tab separated, spectra in rows (*.txt)|*.txt|Comma separated values, spectra in columns (*.csv)|*.csv| Internal OES format  (*.oes)|*.oes | All files (*.*)|*.*'
        dlg = FileDialog(action='open', title='Select measurement',
                wildcard = wildcard)
        if dlg.open() == OK:
            #print dir(dlg)
            #print(self.filename)
            #self.filename = dlg.path.encode('utf8')
            self.filename = dlg.path
            filename, extension = os.path.splitext(dlg.path)
            if (extension == '.csv') or (extension=='.CSV'):
                self.measSpec = MeasuredSpectra.from_CSV(dlg.path)
            elif extension == '.txt' or (extension=='.TXT'):
                self.measSpec = MeasuredSpectra.from_FHRfile(dlg.path)
            elif (extension == '.oes') or (extension == '.OES'):
                self.measSpec = MeasuredSpectra.load(dlg.path)
            elif (extension == '.json') or (extension == '.JSON'):
                self.measSpec = MeasuredSpectra.from_json(dlg.path)



            # self.invalid_species =  str(self.measSpec.check_simulations())

            # while self.invalid_species != '[]':
            #     msg = 'Warning! The following simulation files have not been found:\n\n'
            #     for name in self.measSpec.check_simulations():
            #         msg += name + '\n'
            #     msg += '\nAfter pressing OK, you will be asked to select valid files '
            #     msg += '(more at once, if needed).'
            #     info = Info(msg=msg)
            #     info.edit_traits()
            #     dlg = FileDialog(action='open files',
            #                      title='Select simulation file to replace invalid ones...')
            #     dlgret = dlg.open()
            #     if dlgret == OK:
            #         print dlg.paths
            #         sims = []
            #         for fname in dlg.paths:
            #             #sims.append(LargeSimulatedSpectra.load(fname))
            #             sims.append(specDB(fname))
            #         self.measSpec.replace_simulations(sims)
            #     elif dlgret == CANCEL:
            #         msg = MessageDialog()
            #         msg.warning('Some simulations will not be shown!')
            #         break
            #     self.invalid_species = str(self.measSpec.check_simulations())
                    #self.meas.measSpec.export_results(dlg.path)


            self.reports = []
            for i in range(len(self.measSpec.spectra)):
                self.reports.append('')
            self._which_measured_changed()


    def _readjust_meas(self):
        if self.measSpec is not None:
            specname = list(self.measSpec.spectra.keys())[self.which_measured]
            spec = self.measSpec.measured_Spectra_for_fit(specname)

            self.meas_x = array(spec.x)
            self.meas_y = array(spec.y)

    def _which_measured_changed(self):
        self._readjust_meas()
        if self.measSpec is not None:
            if self.reports[self.which_measured] == '':
                specname = list(self.measSpec.spectra.keys())[self.which_measured]
                self.reports[self.which_measured] += self.filename + '\n'+\
                                str(datetime.now()) + '\n' +\
                                'ID: ' + str(specname)+\
                                lmfit.fit_report(self.measSpec.spectra[specname]['params'].prms)
                #print self.reports[]
            self.report = self.reports[self.which_measured]

class select_sim(HasTraits):
    selected = Str
    available = List(Str)

    traits_view = View(Item('available',
        editor=ListStrEditor(title='Select specie', editable=False,
            selected = 'selected'),),
        buttons = [OKButton, CancelButton], kind = 'modal')

class SetName(HasTraits):
    name = Str
    available = List(Str)

    traits_view = View(Item('name', editor = EnumEditor(name = 'available'),
                            label = 'Copy parameters from spectrum no.'),
                       buttons = [OKButton, CancelButton], kind = 'modal')

class IntAndBool(HasTraits):
    num = Int
    use = Bool(True)


to_fit_editor = TableEditor(
    sortable     = False,
    configurable = True,
    auto_size    = False,
    deletable    = False,
    selection_mode = 'row',
    selected  = 'simulation',
    #on_select = vypis_zpravu,
    columns  = [ ObjectColumn( name  = 'num',  label = 'Measured spectrum no.', editable = False),
                 CheckboxColumn( name   = 'use',     label  = 'fit?', editable = True )])


class SelectInts(HasTraits):
    selected = Instance(IntAndBool)
    available = List(IntAndBool)
    select_all = Bool(False)
    select_by_expr = Bool(False)
    expr_to_select = Str('')
    take_init_from = Str('num-1')
    label_for_take_from = Str('Take initial values from expression:')

    traits_view = View(
        HGroup(
            Item('select_all'),
        ),
        Item('available', editor = to_fit_editor, show_label=False),
        HGroup(
            Item('select_by_expr', label = 'Select by expression'),
            Item('expr_to_select', show_label = False, visible_when='select_by_expr',
                 editor = TextEditor(auto_set=False, enter_set=True, evaluate=str)),
        ),
        Item('label_for_take_from', show_label=False, style='readonly'),
        Item('take_init_from', show_label=False),
        kind = 'modal',
        title = 'Select spectra to fit',
        resizable = True,
        buttons = [OKButton, CancelButton],
        width = 220
    )

    def __init__(self, number_of_entries):
        for i in range(number_of_entries):
            self.available.append(IntAndBool(num=i, use = True))

    def _select_all_changed(self):
        for item in self.available:
            item.use = self.select_all

    def _expr_to_select_changed(self):
        try:
            select = eval(self.expr_to_select)
            print ('selected: ',)
            for item in self.available:
                if item.num in select:
                    item.use = True
                    print (item.num,)
                else:
                    item.use = False
            print( '\n')
        except Exception as exc:
            print( exc)



class Plotter(HasTraits):
    plot = Instance(Plot)

    print_sims = Button('Print sims')
    meas = Instance(MeasSpecLoader)
    #sim = Instance(SimSpecLoader)
    permanent_message = Str('The measured spectrum should have photon flux (photons/s/m2) as the y axis, not intensity (not J/s/m2)!')
    ######################################
    #simulation and parameters
    tr_params = Instance(TrParameters, required=True)
    sim_x = Array
    sim_y = Array
    add_simulation = Button('Add simulation')
    export_sim = Button('Current simulation to CSV')
    #filename = File
    unclick_all = Bool(True)
    sims = {}



    fit = Button('Fit')
    minimize_method = Enum('leastsq', 'nelder', 'lbfgsb')
    from_which = Button('Copy parameters from another spectrum')
    set_to_fit = Button('Run fit for more measured spectra')
    export_results = Button('Export results')
    update_simulations = Button('Update simulations') #workaround for
                                                      #plot update
                                                      #after changing
                                                      #values for
                                                      #simulations in
                                                      #the TableEditor
    linearize = Button('Linearize x-axis')
    bp_insp = Button('Boltzmann plot')
    undo = Button('Undo')

    maxiter = Int(2000)

    residual_y = Array

    stop_button = Button('Stop fit')

    traits_view = View(
        #Spring(),
        Item('permanent_message',style = 'readonly' , show_label=False),
        HGroup(
            ### Left panel
            VGroup(
                #Item('print_sims'),
                HGroup(
                    Item('meas', style='custom', show_label=False,),
                    Item('add_simulation', show_label = False),
                    Item('export_sim', show_label=False)
                ),
                #Item('object.meas.invalid_species', style='readonly'),
                HGroup(
                    Item('fit', show_label=False),
                    Item('undo', show_label=False),
                    Item('stop_button', show_label=False),
                    Item('minimize_method', show_label=False),
                    Item('maxiter', label = 'Max. iterations')
                ),
                Item('plot', editor=ComponentEditor(), show_label=False,
                     resizable=True),
            ),
            ### Right panel
            VGroup(
                HGroup(
                    Item('from_which', show_label = False),
                    Item('set_to_fit', show_label = False),
                    Item('export_results', show_label = False),
                    Item('linearize', show_label=False),
                    Item('bp_insp', show_label=False)
                ),
                HGroup(
                    Spring(),
                    Item('unclick_all', label = 'Select all'),
                ),
                HGroup(
                    Item('object.tr_params.baseline', label='value', editor=TextEditor(auto_set=False, evaluate=float),),
                    Item('object.tr_params.baseline_min', label='min', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.baseline_max', label='max', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.baseline_vary', label='fit'),
                    label='Baseline'
                ),
                HGroup(
                    Item('object.tr_params.baseline_slope', label='value', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.baseline_slope_min', label='min', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.baseline_slope_max', label='max', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.baseline_slope_vary', label='fit'),
                    label='Baseline slope'
                ),
                HGroup(
                    Item('object.tr_params.wav_start', label='value', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.wav_start_min', label='min', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.wav_start_max', label='max', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.wav_start_vary', label='fit'),
                    label='starting wavelength'
                ),
                HGroup(
                    Item('object.tr_params.wav_step', label='value', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.wav_step_min', label='min', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.wav_step_max', label='max', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.wav_step_vary', label='fit'),
                    label='wavelength step'
                ),
                HGroup(
                    Item('object.tr_params.wav_2nd', label='value', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.wav_2nd_min', label='min', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.wav_2nd_max', label='max', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.wav_2nd_vary', label='fit'),
                    label='quadratic correction'
                ),
                HGroup(
                    Item('object.tr_params.gauss', label='value', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.gauss_min', label='min', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.gauss_max', label='max', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.gauss_vary', label='fit'),
                    label='Slit fn (Gaussian HWHM)'
                ),
                HGroup(
                    Item('object.tr_params.lorentz', label='value', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.lorentz_min', label='min', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.lorentz_max', label='max', editor=TextEditor(auto_set=False, evaluate=float)),
                    Item('object.tr_params.lorentz_vary', label='fit'),
                    label='Slit fn (Lorentzian HWHM)'
                ),
                Item('object.tr_params.simulations', show_label=False),
                Item('update_simulations', show_label=False),
                Item('object.meas.report', label='Parameters:', style='custom', show_label=False)
            )
        ),
        title = 'Massive OES GUI',
        resizable=True,
        icon=ImageResource('images/massiveOES_icon.png'),
        #key_bindings = key_bindings,
        #handler = KBHandler(),
    )


    def __init__(self):
        self.meas = MeasSpecLoader()
        self.setting_in_progress = False
        self.adding_simulation=False
        self.fit_which = []

        self.tr_params = TrParameters()


        self.on_trait_change(self._dummy_replot,'sim_y')
        self.meas.on_trait_change(self._set_tr_params, 'meas_y')
        strparams = []
        for p in ['lorentz', 'gauss', 'baseline', 'baseline_slope', 'wav_start', 'wav_step', 'wav_2nd']:
            for suff in ['', '_min', '_max', '_vary']:
                strparams.append(p+suff)
        self.tr_params.on_trait_change(self._set_params, strparams)
        self.tr_params.on_trait_change(self._set_params, 'simulations[]')
        self.tr_params.on_trait_change(self._readjust_sims, strparams)
        self.tr_params.on_trait_change(self.meas._readjust_meas, ['wav_start', 'wav_step', 'wav_2nd'])
        self.lin = None
        self.bkp_params = None
        self.fits_running = False
        self.stop_fit = False



    def _unclick_all_changed(self):
        self.adding_simulation = True
        for param in ['wav_start', 'wav_step', 'wav_2nd',
                      'baseline', 'baseline_slope', 'gauss', 'lorentz']:
            #print param+'_vary = '+ str(self.unclick_all)
            exec('self.tr_params.'+param+'_vary = '+ str(self.unclick_all))
        self.adding_simulation = False
        self._readjust_sims()

    def _add_simulation_changed(self):
        self.adding_simulation = True
        sims = glob.glob(os.path.join(MASSIVEOES_PATH,'data','*db'))
        sims = [os.path.split(f)[-1] for f in sims]
        ss = select_sim(available=sims)
        ss.configure_traits()
        if ss.selected != '':
            sim = SpecDB(ss.selected)
            specname = list(self.meas.measSpec.spectra.keys())[self.meas.which_measured]
            self.meas.measSpec.add_specie(sim, specname)

            ParamsToTrParams(self.meas.measSpec.spectra[specname]['params'],
                             self.tr_params, measSpec=self.meas.measSpec)

            #print (self.tr_params.simulations)
            self.tr_params.simulations[-1].simSpec = sim

        self.adding_simulation = False
        self._readjust_sims()

    def _export_sim_changed(self):
        self._readjust_sims()
        dlg = FileDialog(action='save as', title='Export simulated spectrum as CSV')
        if dlg.open() == OK:
            savetxt(dlg.path, array([self.sim_x, self.sim_y]).T, delimiter = ',')

    def _bp_insp_changed(self):
        meas_to_bpi = MeasuredSpectra()
        specname = list(self.meas.measSpec.spectra.keys())[self.meas.which_measured]
        meas_to_bpi.spectra[specname] = deepcopy(self.meas.measSpec.spectra[specname])
        meas_to_bpi.simulations = self.meas.measSpec.simulations

        bp = BoltzmannInspector(which_measured = 0, measSpec = meas_to_bpi)
        bp.configure_traits()
        del meas_to_bpi

    def _get_current_sims(self):
        sims = {}
        for spec in self.tr_params.simulations:
            sims[spec.simSpec.specie_name] = spec.simSpec
        return sims

    def _readjust_sims(self):
        if self.adding_simulation:
            return
        print ('READJUSTING SIMULATIONS')
        params = TrParamsToParams(self.tr_params)
        #print params.prms.keys()
        print (params.prms)
        sims = self._get_current_sims()
        sim = puke_spectrum(params, convolve=True,
                            step = self.tr_params.wav_step, sims = sims)

        self.sim_x = sim.x
        self.sim_y = sim.y
        self._replot(params=params)

    def _print_sims_changed(self):
        print (self.meas.measSpec.simulations)

    def _linearize_changed(self):
        self.lin = Coefficients(plotter = self, meas=self.meas)
        self.lin.configure_traits()

    def _set_to_fit_changed(self):
        #from PySide import QtCore
        if self.meas.measSpec is not None:
            to_fit = SelectInts(number_of_entries = len(self.meas.measSpec.spectra))
            result = to_fit.configure_traits()
            self.fit_which = to_fit.available
            if not result:
                return
            self.fits_running = True
            keys = list(self.meas.measSpec.spectra.keys())
            for num in range(len(keys)):
                specname = keys[num]
                if to_fit.available[num].use:
                    take_from = eval(to_fit.take_init_from)

                    ## This prevents users from frequently encountered situation of
                    ## selecting -1st element of list
                    ## Also prevents from falling if accidentaly selected index out of range
                    if take_from not in range(len(keys)):
                        take_from = 0

                    if num != take_from:
                        self.meas.measSpec.spectra[specname]['params'] =\
                            deepcopy(self.meas.measSpec.spectra[keys[take_from]]['params'])
                    self.meas.which_measured = num
                    self._fit_changed()
                    if self.stop_fit:
                        self.stop_fit = False
                        self.fits_running = False
                        break
                    #QtCore.QCoreApplication.processEvents()
                    #self._replot()
            self.fits_running = False


    def _export_results_changed(self):
        dlg = FileDialog(action='save as', title='Export as CSV')
        if dlg.open() == OK:
            self.meas.measSpec.export_results(dlg.path)

    def _from_which_changed(self):
        specname = list(self.meas.measSpec.spectra.keys())[self.meas.which_measured]
        from_which = SetName(available = list(self.meas.measSpec.spectra.keys()),
                            name = specname)
        from_which.edit_traits()
        if specname != from_which.name:
            self.meas.measSpec.spectra[specname]['params'] =\
                  deepcopy(self.meas.measSpec.spectra[from_which.name]['params'])
            self._set_tr_params()
            self.meas._which_measured_changed()
            self._readjust_meas_x()


    def _set_tr_params(self):
        if self.meas.measSpec is not None:
            self.setting_in_progress = True
            print ('STARTING: set_tr_params!')
            specname = list(self.meas.measSpec.spectra.keys())[self.meas.which_measured]
            ParamsToTrParams(self.meas.measSpec.spectra[specname]['params'],
                             self.tr_params, measSpec=self.meas.measSpec)
            self.setting_in_progress = False
            self.plot.x_mapper.range.low = min(self.meas.meas_x)
            self.plot.x_mapper.range.high = max(self.meas.meas_x)
            self.plot.y_mapper.range.low = -0.1
            self.plot.y_mapper.range.high = max(self.meas.meas_y)
            print ('Stopping: set_tr_params!')
            self._replot()

    # def _add_sim(self):
    #     print "********* ADDING SIMULATIONS *************"
    #     sims = self.sim._get_current_sims()
    #     for specie in sims:
    #         if specie not in self.meas.measSpec.simulations:
    #             self.meas.measSpec.simulations[specie] = sims[specie]
    #     self._give_measSpec_to_sim()

    def _set_params(self):
        print ('setting meas.simulations')
        print (self.setting_in_progress, self.adding_simulation)
        if not self.setting_in_progress and not self.adding_simulation:
            if self.meas.measSpec is None:
                params = TrParamsToParams(self.tr_params)
                self._replot(params = params)
            else:
                specname = list(self.meas.measSpec.spectra.keys())[self.meas.which_measured]
                self.meas.measSpec.spectra[specname]['params'] =\
                TrParamsToParams(self.tr_params)
                self.meas._readjust_meas()
                self._replot()


    def _plot_default(self):
        """
        Create the plot object and plot initial data. Add pan and zoom tools.
        """
        self.plotdata = ArrayPlotData(sim_x=self.sim_x, sim_y=self.sim_y,
                                      meas_x=self.meas.meas_x, meas_y=self.meas.meas_y,
                                      residual_x=self.meas.meas_x, residual_y=self.residual_y)
        plot = Plot(self.plotdata)
        plot.plot(('sim_x', 'sim_y'), type='line', color='blue', name='simulation')
        plot.plot(('meas_x', 'meas_y'), type='line', color='green', name='measurement')
        plot.plot(('meas_x', 'residual_y'), type='line', color='grey',
                  line_width=2.0, name = 'residuals')
        plot.tools.append(PanTool(plot))
        plot.tools.append(BetterZoom(plot))
        plot.legend.visible = True
        return plot

    def _dummy_replot(self):
        """
        Function that takes no parameters to work properly with watching for trait changes
        """
        self._replot(params = None)

    def _replot(self, params = None):
        print ('_replot')
        specname = list(self.meas.measSpec.spectra.keys())[self.meas.which_measured]
        if self.meas.measSpec is None:
            self.plotdata.set_data('sim_x', self.sim_x)
            self.plotdata.set_data('sim_y', self.sim_y)
            return

        if params is None:

            params = self.meas.measSpec.spectra[specname]['params']

        self.plotdata.set_data('meas_x', self.meas.meas_x)
        self.plotdata.set_data('meas_y', self.meas.meas_y)

        if (len(self.meas.meas_x) > 0 and len(self.sim_x) > 0
            and len(self.meas.meas_y) > 0 and len(self.sim_y) > 0
            and self.tr_params.wav_step != 0):
            spec_meas = self.meas.measSpec.measured_Spectra_for_fit(specname,
                                                                    params = params)
            spec_sim = puke_spectrum(params = params,
                                     step = self.tr_params.wav_step,
                                     sims = self.meas.measSpec.simulations)
            self.residual_y = array(compare_spectra(spec_meas, spec_sim))
            self.plotdata.set_data('residual_x', self.meas.meas_x)
            self.plotdata.set_data('residual_y', self.residual_y)
            self.plot.x_mapper.range.low = min(self.meas.meas_x)
            self.plot.x_mapper.range.high = max(self.meas.meas_x)

            #show only interpolated simulation points
            matched_sim, matched_exp = match_spectra(spec_sim, spec_meas)
            print ("Integral z y = ", trapz(matched_sim.y, matched_sim.x))

            self.plotdata.set_data('sim_x', matched_sim.x)
            self.plotdata.set_data('sim_y', matched_sim.y)
        else:
            self.residual_x = []
            self.residual_y = []
            self.plotdata.set_data('residual_x', self.residual_x)
            self.plotdata.set_data('residual_y', self.residual_y)
            self.plotdata.set_data('sim_x', self.sim_x)
            self.plotdata.set_data('sim_y', self.sim_y)


    def _stop_button_changed(self):
        if self.fits_running:
            self.stop_fit = True

    def _fit_changed(self):
        self.bkp_params = TrParamsToParams(self.tr_params)
        specname = list(self.meas.measSpec.spectra.keys())[self.meas.which_measured]
        if True:
        #try:
            self.meas.measSpec.fit(specname, method = self.minimize_method,
                               by_peaks=False, show_it=False,
                               only_strong  = True, weighted=False, maxiter = self.maxiter)
            self.meas.reports[self.meas.which_measured] = str(datetime.now()) + '\n'+\
                           'file name: ' + str(self.meas.filename) + '\n' +\
                           'for spectrum no.: ' + str(self.meas.which_measured) + '\n' +\
                           'ID: ' + str(specname)\
                            + '\n'+str(self.meas.measSpec.minimizer_result.message) + '\n' +\
                            str(self.meas.measSpec.minimizer_result.lmdif_message) + '\n' +\
                            'chisqr = ' + str(self.meas.measSpec.minimizer_result.chisqr) + '\n' +\
                           lmfit.fit_report(
                    self.meas.measSpec.spectra[specname]['params'].prms) \
            +'\n\n' + self.meas.reports[self.meas.which_measured] #the newest comes on top
            ## the line above updates the text in the memory
            ## the line below displays the updated text
            self.meas.report = self.meas.reports[self.meas.which_measured]
            self.adding_simulation=True
            self._set_tr_params()
            self.adding_simulation=False
        # except Exception as exc:
        #     print 'Fit failed!'
        #     print type(exc)

    def _undo_changed(self):
        if self.bkp_params is not None:
            ParamsToTrParams(self.bkp_params,self.tr_params, self.meas.measSpec)
            self._set_params()
            self._replot()


    def _readjust_meas_x(self):
        numpoints = len(self.meas.meas_x)
        self.meas.meas_x = MeasuredSpectra.get_wavs(numpoints,
                                                    self.tr_params.wav_start,
                                                    self.tr_params.wav_step,
                                                    self.tr_params.wav_2nd)
        self._replot()

    def _update_simulations_changed(self):
        #self._add_sim()
        self._readjust_sims()
        if self.meas.measSpec is not None:
            specname = list(self.meas.measSpec.spectra.keys())[self.meas.which_measured]
            self.meas.measSpec.spectra[specname]['params'] =\
                TrParamsToParams(self.tr_params)
        self._replot(TrParamsToParams(self.tr_params))

if __name__ == '__main__':
    #import cProfile
    p = Plotter()
    #cProfile.run("""
    p.configure_traits()
    #""")
