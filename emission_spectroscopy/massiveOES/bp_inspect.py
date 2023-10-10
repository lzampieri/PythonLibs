"""
Scattergram inspector tool
Allows the user to highlight and/or select individual points of a scattergram.
When the mouse hovers over a scatter point, it changes temporarily. If you click
on a point, you select and mark (or unselect and unmark) the point.
"""

# Major library imports
import numpy
from scipy.stats import linregress
from scipy.constants import physical_constants

kB = physical_constants['Boltzmann constant in inverse meters per kelvin'][0]/100 #in 1/(cm*K)

# Enthought library imports
from massiveOES import spectrum, puke_spectrum
from enable.api import Component, ComponentEditor,  BaseTool, KeySpec
from traits.api import Button, HasTraits, Instance, List, Str, Float, File,\
Int, Array, on_trait_change, Bool, Enum
from traitsui.api import UItem, Group, View, HGroup

from traitsui.table_column \
    import ObjectColumn

from traitsui.extras.checkbox_column \
    import CheckboxColumn

from pyface.api import FileDialog, OK, CANCEL, MessageDialog

# Chaco imports
from chaco.api import ArrayPlotData, Plot, ScatterInspectorOverlay, LassoOverlay
from chaco.tools.api import PanTool, ZoomTool, ScatterInspector, LassoSelection
from traitsui.api import View, Item, HGroup, Handler,\
EnumEditor, TextEditor, TableEditor, VGroup, HSplit, FileEditor, Spring,\
StatusItem, SetEditor
import pandas
from pyface.api import ImageResource



#===============================================================================
# # Create the Chaco plot.
#===============================================================================]

class Viblevel(HasTraits):

    specie_bp = Str('')
    vibration_level = Int()
    show_vib = Bool(True)

    def __init__(self, species_name, vib_level, show):
        self.specie_bp = species_name
        self.vibration_level = vib_level
        self.show_vib = True


simulations_editor = TableEditor(
    sortable     = False,
    configurable = True,
    auto_size    = False,
    deletable    = True,
    selection_mode = 'row',
    selected  = 'viblevel',
    #on_select = vypis_zpravu,
    columns  = [ ObjectColumn( name  = 'specie_bp',  label = 'Specie'),
                 ObjectColumn( name   = 'vibration_level',     label  = 'Vibrational state' ),
                 CheckboxColumn( name   = 'show_vib',     label  = 'plot?', editable = True )
                 ])

species_editor = TableEditor(
    sortable = False,
    configurable = True,
    auto_size    = False,
    deletable    = False,
    selection_mode = 'row',
    selected  = 'specie',
    columns = [ ObjectColumn(name = 'name', label = 'Electronic Transtion', editable=False),
                CheckboxColumn(name = 'use', label = 'Use for fit'),
                ObjectColumn(name = 'vmax', label = 'vmax', editable = True),
                ObjectColumn(name = 'Jmax', label = 'Jmax', editable = True),
                ObjectColumn(name = 'minlines', label = 'Lines at least', editable = True),
                CheckboxColumn(name='singlet_like', label = 'singlet-like')
                ])


def PrepDB(database):

    #for key in database.keys():
    database['E'] = database['E_J'] + database['E_v']
    database['degeneracy'] = 2*database['J'] + 1
    #database[key]['specie'] = key

    return database

def dbdict2db(dbdict):
    db = pandas.concat(dbdict.values())
    return db

# def db2dbdict(db):
#     dbdict = {}
#     for specie in db['specie'].unique():
#         sub_db = db.loc[(db.specie == specie)]
#         dbdict[specie] = sub_db
#     return dbdict

#===============================================================================
# Attributes to use for the plot view.
size=(900,500)
title="BP Inspector"

class SimPlotter(HasTraits):
    plot = Instance(Component)
    pd = ArrayPlotData()

    def __init__(self, **kwargs):
        """
kwargs:
  params: massiveOES.Parameters object

  measured: massiveOES.Spectrum object containing the measured
            spectrum (e.g. from MeasuredSpecta.measured_Spectra_for_fit())

  to_simulate: dictionary of pandas DataFrames, one for each species

  simulations: dictionary of massiveOES.SpecDB objects, keys should be
               identical with the keys of to_simulate
        """

        #massiveOES.Spectrum object
        self.params = kwargs.pop('params', None)
        self.measured = kwargs.pop('measured',spectrum.Spectrum(x=[], y=[]))
        #dict of pandas dataframes with lines to simulate
        self.df_to_sim = kwargs.pop('to_simulate', {})
        #dit of massiveOES.SpecDB onjects
        self.sims = kwargs.pop('simulations', [])
        self.simulated = spectrum.Spectrum(x = [], y = [])
        #self.simulated = self.calculate_simspec(self.df_to_sim, **kwargs)

    def calculate_simspec(self, df, specs, **kwargs):
        """
        df: pandas DataFrame
        """
        y = numpy.dot(specs[:,df.index], df.pops)
        return spectrum.Spectrum(x = self.measured.x, y = y)


    def _plot_default(self):
        self.pd = ArrayPlotData(meas_x = numpy.array(self.measured.x), meas_y = numpy.array(self.measured.y),
                                      sim_x = numpy.array(self.simulated.x), sim_y = numpy.array(self.simulated.y))
        plot = Plot(self.pd, border_visible=True, overlay_border=True)
        plot.plot(('meas_x', 'meas_y'), type='line',color='green', name='measurement')
        plot.plot(('sim_x', 'sim_y'), type='line', color='blue', name='simulation')
        plot.tools.append(PanTool(plot))
        plot.overlays.append(ZoomTool(plot))
        plot.legend.visible=True
        return plot

    def replot(self, df, specs, **kwargs):
        self.simulated = self.calculate_simspec(df, specs, **kwargs)
        self.pd.set_data('sim_x', numpy.array(self.simulated.x))
        self.pd.set_data('sim_y', numpy.array(self.simulated.y))
        return

class SpecieForFit(HasTraits):
    use = Bool(False)
    name = Str()
    vmax = Int(-1)
    Jmax = Int(-1)
    minlines = Int(2)
    singlet_like = Bool(False)

class ClickTool(BaseTool):

    first_key = Instance(KeySpec, args=("s",), ignore=['shift'])
    second_key = Instance(KeySpec, args=("c",), ignore=['shift'])
    third_key = Instance(KeySpec, args=("a",), ignore=['shift'])

    adder1 = Int(0)
    adder2 = Int(0)
    adder3 = Int(0)

    def normal_key_pressed(self, event):
        if self.first_key.match(event):
            self.adder1 += 1
        elif self.second_key.match(event):
            self.adder2 += 1
        elif self.third_key .match(event):
            self.adder3 += 1


class BoltzmannInspector(HasTraits):
    plot = Instance(Component)
    print_them = Button("(S)how Selection")
    clear = Button("(C)lear Selection")
    select_all = Button("Select (A)ll")
    click_tool = Instance(ClickTool)


    pd = ArrayPlotData()
    report = Str('')
    T_report = Str('')
    simplotter = Instance(SimPlotter)
    specie = Instance(SpecieForFit)
    species = List(SpecieForFit, editor = species_editor)
    viblevel = Instance(Viblevel)
    species_bp = List(Viblevel, editor = simulations_editor)

    fit = Button('fit')
    trot_b = Button('Calculate Trot')
    tvib_b = Button('Calculate Tvib')

    save_selection = Button('Save selection as csv')

    traits_view = View(
                    Group(
                        HGroup(UItem('print_them'),
                               UItem('clear'),
                               UItem('select_all'),
                               UItem('trot_b'),
                               UItem('tvib_b'),
                               UItem('fit'),
                               UItem('save_selection')
                           ),
                        HGroup( UItem('plot', editor=ComponentEditor(size=size),
                                      show_label=False),
                                UItem('object.simplotter.plot',
                                      editor=ComponentEditor(size=size))
                                ),


                        HGroup(Item('species_bp', show_label=False),
                               Item('species', show_label=False),),
                        #Item('update_simulations', show_label=False),
                        HGroup(
                            Item('report', label='Selection:', style='custom', show_label=False),
                            Item('T_report', label='fitted temperature',
                                 style='custom', show_label=True) ),
                        orientation = "vertical"),
        resizable=True, title=title, icon=ImageResource('images/bp_inspect.png'),
                    )

    def __init__(self,**kwargs):

        self.which_measured = kwargs.pop('which_measured', 0)
        self.meas = kwargs.pop('measSpec', None)
        #self.species = self.meas.simulations.keys()
        self.which_measured = list(self.meas.spectra.keys())[self.which_measured]


        for key in self.meas.simulations:
            self.species.append(SpecieForFit(name = key))

        self.plot_list = []
        self.on_trait_change(self._species_bp_changed, 'viblevel.show_vib')
        self.specs = None

        self.trot_fit_exists = False

        #pandas database - replace with loader
        db = kwargs.pop('db', {})

        # db = self.meas.fit_nnls(self.which_measured, self.meas.simulations.keys() ,
        #                         **kwargs)
        empty_df = pandas.DataFrame({'pops':[],
                                   'errors':[],
                                   'v':[], 'J':[],
                                   'component' : [],
                                   'E_J':[],
                                   'E_v':[],
                                     'numlines':[],
                                     'specie':[]})

        #pandas database preparation
        self.database_full = empty_df


        #database_full = pandas.DataFrame() #
        self.database_for_plot = pandas.DataFrame()

        self.selection_df = self.database_full.loc[self.database_full.v==-1]
        self._prep_table()
        self._species_bp_changed()

        self.on_trait_change(self._print_them_changed,"click_tool:adder1")
        self.on_trait_change(self._clear_changed,"click_tool:adder2")
        self.on_trait_change(self._select_all_changed,"click_tool:adder3")
        measured_spectrum =  self.meas.measured_Spectra_for_fit(self.which_measured)
        measured_spectrum.y -= self.meas.spectra[self.which_measured]['params']['baseline'].value
        measured_spectrum.y -= self.meas.spectra[self.which_measured]['params']['baseline_slope'].value*(measured_spectrum.x-measured_spectrum.x[0])
        #TODO: odnekud vzit x0!
        self.meas.spectra[self.which_measured]['params']['baseline'].value = 0
        self.meas.spectra[self.which_measured]['params']['baseline_slope'].value = 0

        self.simplotter = SimPlotter(to_simulate=db,
                                     simulations = self.meas.simulations,
                                     measured = measured_spectrum,
                                     params = self.meas.spectra[self.which_measured]['params'],
                                     **kwargs)



    def _save_selection_changed(self):
        self._print_them_changed()
        dlg = FileDialog(action='save as', title = 'Save selection as CSV')
        if dlg.open() == OK:
            self.selection_df.to_csv(dlg.path)

    def _fit_changed(self):
        species_for_fit = [s.name for s in self.species if s.use]
        kwargs = {}
        for specie in species_for_fit:
            kwargs[specie] = {}
            spec = [spec for spec in self.species if (spec.name == specie)][0]
            print(spec.name, spec.vmax, spec.Jmax, spec.minlines)
            if spec.vmax > -1:
                kwargs[specie]['max_v'] = spec.vmax
            if spec.Jmax > -1:
                    kwargs[specie]['max_J'] = spec.Jmax
            if spec.minlines > 0:
                kwargs[specie]['minlines'] = spec.minlines
            kwargs[specie]['singlet_like'] = spec.singlet_like
        print (species_for_fit)
        print (kwargs)
        db, specs = self.meas.fit_nnls(self.which_measured, species_for_fit, **kwargs)
        self.specs = specs
        self.database_full = PrepDB(db)
        self._prep_table()
        self.simplotter.replot(self.database_full, specs)
        self._species_bp_changed()
        self._replot()

    def _plot_default(self):
         #self.plot.set(title="Boltzmann plot inspector", padding=50)
        plot = Plot(self.pd, border_visible=True, overlay_border=True)

        self.click_tool = ClickTool(plot)
        plot.tools.append(self.click_tool)

        # Attach some tools to the plot
        #plot.tools.append(PanTool(plot))
        #plot.overlays.append(ZoomTool(plot))



        plot.y_mapper.range.low = 0
        plot.y_mapper.range.high = 1
        plot.x_mapper.range.low = 0
        plot.x_mapper.range.high = 1


        return plot

    def _trot_b_changed(self):
        self._print_them_changed()

        y = numpy.log(self.selection_df.pops / (numpy.array(self.selection_df.degeneracy, dtype=float)))

        slope, intercept, r_value, p_value, std_err = linregress(self.selection_df.E,
                                                                 y)
        Trot = -1/(kB*slope)
        Trot_dev = abs(std_err/slope) * Trot
        self.T_report = '\nTrot(selection) = ('+str(Trot) + u'\u00B1 '+ str(Trot_dev) + ') K'
        self.pd['trot_x'] = self.selection_df.E
        self.pd['trot_y'] = numpy.exp(self.selection_df.E*slope + intercept)
        self.plot.plot(('trot_x', 'trot_y'), type='line', color='black', name='trot_fit',
                       value_scale='log', line_width=2.0)
        self.trot_fit_exists = True



    def _tvib_b_changed(self):
        x = self.database_for_plot.E_v.unique()
        y = []
        for ev in x:
            y.append(numpy.log(self.database_for_plot.pops[self.database_for_plot.E_v==ev].sum()))
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        Tvib = -1/(kB*slope)
        Tvib_dev = abs(std_err/slope) * Tvib
        self.T_report = '\nTvib(selection) = ('+str(Tvib) + u'\u00B1 '+ str(Tvib_dev) + ') K'



    def _prep_table(self):
        self.species_bp = []
        for key in self.database_full['specie'].unique():
            vibration_levels = self.database_full['v'].unique()
            for level in vibration_levels:
                specie_bp = Viblevel(key, level, True)
                #print(specie_bp.specie_bp)
                #print(self.species_bp)
                self.species_bp.append(specie_bp)



    def _print_them_changed(self):
        print('printing selections')
        scatter_plot = self.scatter
        try:
            selections = scatter_plot.index.metadata.get('selections', [])
        except:
            return

        if scatter_plot is None:
            print('No data to inspect!')
            self.report = ('No data to inspect!')
            return
        elif selections == []:
            print('Empty selection!')
            self.report = ('Empty selection!')
            return
        else:
        #scatter_plot = self.plot.plots['background'][0]
            #print(selections)
            x_data = scatter_plot.index.get_data()
            y_data = scatter_plot.value.get_data()

            data = zip(x_data[selections], y_data[selections])
            self.selection_df = self._find_state(data)
            self.report = str(self.selection_df[['specie','v','J','E','pops']])
            self.simplotter.replot(self.selection_df, self.specs)

    def _clear_changed(self):
        print('clearing selections')
        plot = self.scatter

        for name in ('index', 'value'):
            if not hasattr(plot, name):
                continue
            md = getattr(plot, name).metadata
            selection = md.get('selections', None)

            # If no existing selection
            if selection is None:
                pass
            # check for list-like object supporting append
            else:
                md['selections'] = []
                    # Manually trigger the metadata_changed event on
                    # the datasource.  Datasources only automatically
                    # fire notifications when the values inside the
                    # metadata dict change, but they do not listen
                    # for further changes on those values.
                getattr(plot, name).metadata_changed = True

        return

    def _find_state(self, data):
        select_frame = pandas.DataFrame(columns = self.database_for_plot.columns)
        for coords in data:
            row = self.database_for_plot.loc[(self.database_for_plot.E == coords[0]) & ((self.database_for_plot.pops/self.database_for_plot.degeneracy) == coords[1])]
            if row.empty:
                print('Data not found WTF!!!!')
                return
            select_frame = select_frame.append(row)
        return select_frame


    def _replot(self):

        #print(type(self.database_for_plot['specie'].unique()))

        if self.trot_fit_exists:
            self.plot.delplot('trot_fit')
            self.trot_fit_exists = False

        if self.database_for_plot is None:
            self.scatter = None
            print('Prazdna databaze')
            if self.plot_list != []:
                for plot in self.plot_list:
                    self.plot.delplot(plot)

                self.plot_list = []

            return

        callsign = []

        for key in self.database_for_plot['specie'].unique():
            vibration_levels = self.database_for_plot['v'].unique()
            for level in vibration_levels:
                callsign.append([key +' '+ str(level), key, level])



        callsign = numpy.array(callsign)

        pd = self.pd
        colors = [ 'green', 'blue', 'red', 'darkgray', 'lightgreen', 'lightblue', 'pink', 'silver' ]

        for cs in callsign:
            selection = (self.database_for_plot.v == int(cs[2])) & (self.database_for_plot.specie == cs[1])
            pd[cs[0]+'_x'] = self.database_for_plot.loc[selection]['E']
            pd[cs[0]+'_y'] = self.database_for_plot.loc[selection]['pops']/self.database_for_plot.loc[selection]['degeneracy']

        to_deplot = set(self.plot_list).difference(callsign[:,0])
        to_plot = set(callsign[:,0]).difference(self.plot_list)

        pd['whole_x'] = self.database_for_plot['E']
        pd['whole_y'] = self.database_for_plot['pops']/self.database_for_plot['degeneracy']


        self.plot.y_mapper.range.low = max(pd['whole_y'])*1e-4
        self.plot.y_mapper.range.high = max(pd['whole_y'])*1.1
        self.plot.x_mapper.range.low = min(pd['whole_x'])*0.975
        self.plot.x_mapper.range.high = max(pd['whole_x'])*1.025

        if hasattr(self, 'scatter'):
            if hasattr(self, 'index_datasource'):
                del(self.index_datasource)
            if hasattr(self.scatter, 'overlays'):
                del(self.scatter.overlays)
            del(self.scatter)


        self.scatter = self.plot.plot(("whole_x","whole_y"), type="scatter",marker = 'dot', color = (0,0,0,0), value_scale = "log", name='background', xbounds = (1e-7,1))[0]

        for cs in to_deplot:
            self.plot.delplot(cs)

        for cs in to_plot:
            graph = self.plot.plot((cs+'_x', cs+'_y'),  type="scatter",marker = 'circle', value_scale = "log", name= cs, color = 'auto')[0]



        self.plot_list = set(set(self.plot_list).union(callsign[:,0])).difference(set(self.plot_list).difference(callsign[:,0]))

        self.scatter.tools.append(self.click_tool)

        # Attach the inspector and its overlay
        self.scatter.tools.append(ScatterInspector(self.scatter))
        overlay = ScatterInspectorOverlay(self.scatter,
                        hover_color="red",
                        hover_marker_size=6,
                        selection_marker_size=6,
                        selection_color="yellow",
                        selection_outline_color="purple",
                        selection_line_width=3)
        self.scatter.overlays.append(overlay)
        self.scatter.index.metadata['selections'] = []
        #self.scatter.index.on_trait_change(self._print_them_changed)

        lasso_selection = LassoSelection(component=self.scatter,
                                         selection_datasource=self.scatter.index,
                                         #drag_button="left")
                                         drag_button="right")
        self.scatter.tools.append(lasso_selection)
        #self.scatter.active_tool = lasso_selection
        #self.scatter.tools.append(ScatterInspector(my_plot))
        lasso_overlay = LassoOverlay(lasso_selection=lasso_selection,
                                     component=self.scatter)
        self.scatter.overlays.append(lasso_overlay)
        self.index_datasource = self.scatter.index
        lasso_selection.on_trait_change(self._selection_changed,
                                        'selection_changed')

    def _species_bp_changed(self):

        print('Replotiing')


        partial_frames = []

        for specie_vib in self.species_bp:
            if specie_vib.show_vib:
                partial_frames.append(self.database_full.loc[(self.database_full.v == specie_vib.vibration_level) & (self.database_full.specie == specie_vib.specie_bp)])

        if partial_frames != []:
            self.database_for_plot =  pandas.concat(partial_frames)
            self.database_for_plot = self.database_for_plot[self.database_for_plot['pops'] != 0]
        else:
            self.database_for_plot = None

        self._replot()

    def _selection_changed(self):
        mask = self.index_datasource.metadata['selection']
        #print "New selection: "
        index = numpy.compress(mask, numpy.arange(len(mask)))
        #selection = self.scatter.index.metadata.get('selections', [])
        # Ensure that the points are printed immediately:
        #sys.stdout.flush()
        plot = self.scatter

        for name in ('index', 'value'):
            if not hasattr(plot, name):
                continue
            md = getattr(plot, name).metadata
            selection = md.get('selections', None)

            # If no existing selection
            if selection is None:
                pass
            # check for list-like object supporting append
            else:

                new_list = list(set(md['selections']).union(index))
                md['selections'] = new_list
                # for indice in index:
                    # # print(md['selections'])
                    # # print(indice)
                    # if indice not in md['selections']:
                            # new_list = md['selections'] + [indice]
                            # md['selections'] = new_list
                            # # Manually trigger the metadata_changed event on
                            # # the datasource.  Datasources only automatically
                            # # fire notifications when the values inside the
                            # # metadata dict change, but they do not listen
                            # # for further changes on those values.
                getattr(plot, name).metadata_changed = True
                    # else:
                        # pass
        return

    def _select_all_changed(self):

        plot = self.scatter
        try:
            selections = plot.index.metadata.get('selections', [])
        except:
            returns

        if plot is None:
            print('No data to inspect!')
            self.report = ('No data to inspect!')
            return
        else:
        #scatter_plot = self.plot.plots['background'][0]
            #print(selections)
            x_data = plot.index.get_data()
            #y_data = scatter_plot.value.get_data()

            max_index = len(x_data)

            for name in ('index', 'value'):
                if not hasattr(plot, name):
                    continue
                md = getattr(plot, name).metadata
                selection = md.get('selections', None)

                md['selections'] = range(max_index)
                # If no existing selection


        return








if __name__ == "__main__":
    bp.configure_traits()
