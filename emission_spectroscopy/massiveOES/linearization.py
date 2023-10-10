# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:16:35 2015

@author: Petr
"""

from traits.api import HasTraits, Float,\
Int, Button, Array, Instance,\
on_trait_change, List, Str
from traitsui.api import View, Item, Group,\
VGroup, HGroup, TabularEditor

from enable.api import BaseTool, KeySpec
from enable.component_editor import ComponentEditor

from chaco.api import Plot, ArrayPlotData
from chaco.tools.api import PanTool, BetterZoom
from pyface.api import ImageResource

import numpy as np
import warnings
from operator import itemgetter

from traitsui.tabular_adapter import TabularAdapter
from massiveOES.FHRSpectra import find_nearest    

class Pair(HasTraits):

    def __init__(self, left=0, right = 1e-92):
        self.left=left
        self.right=right

    def __getitem__(self, index):
        if index == 0:
            return self.left
        elif index == 1:
            return self.right
        else: 
            return None

    left = Int
    right = Float

class ClickTool(BaseTool):
    
    click_coordsL = Float
    click_coordsR = Float
    adder = Int(0)

    first_key = Instance(KeySpec, args=("a",), ignore=['shift'])
    second_key = Instance(KeySpec, args=("s",), ignore=['shift'])
    add_key = Instance(KeySpec, args=("d",), ignore=['shift'])

    def normal_key_pressed(self, event):
        if self.first_key.match(event):
            print( "Mouse went down at", event.x, event.y)
            self.click_coordsL = self.component.map_data((event.x, event.y))[0]
            print ("In data coord:", self.click_coordsL    )
        elif self.second_key.match(event):
            print ("Mouse went down at", event.x, event.y)
            self.click_coordsR = self.component.map_data((event.x, event.y))[0]
            print ("In data coord:", self.click_coordsR)
        elif self.add_key.match(event):
            self.adder += 1 
            

class PairAdapter(TabularAdapter):
     columns = [('Measurement','left'), ('Simulation','right')]


class Coefficients(HasTraits):

#Vobal
######################################################################################
    
     info = Str('Use \'a\' and \'s\' to select matching positions in measured and simulated\nspectrum, respectively, and \'d\' to add this pair of values to the list.')
     debug = True
     start = Float(1)   
     linear = Float(1)
     quadratic = Float(0)
     nop = Int(100)
     #load_meas_button = Button("Get measured spectra")
     #load_sim_button = Button("Get simulated spectra")
     plot_button = Button("Plot!")
     click_tool = Instance(ClickTool)

     printit = Button('printit')
     add_pair = Button('Add current values')

     linbutt = Button('Linearize!')

     send_vals = Button('Send result')

     left = Int(0)
     right = Float(0)
     current_pair = Pair()
     pairs = List(Pair)
     
     meas_x = Array
     meas_y = Array
     sim_x = Array
     sim_y = Array
     
     plot = Instance(Plot)
     
     view = View(
         HGroup(
             VGroup(
                 Item(name='start',label='Wavelenght of first point in spectra'),
                 Item(name='linear',label='Waveleght step per point of spectra'),
                 Item(name='quadratic',label='Quadratic corection of spectra x-axis'),
                 HGroup(Item(name='left', style='readonly'),
                        Item(name='right', style='readonly')
                    ),
                 HGroup(Item(name='add_pair', show_label=False),
                        Item(name='linbutt', show_label=False),
                        Item(name='send_vals', show_label=False)),),
             Item('pairs', editor=TabularEditor(adapter=PairAdapter(),
                                                operations=['delete']), 
                  show_label=False),),
                 #Group(Item(name='load_meas_button',show_label=True)),
                 #Group(Item(name='load_spec_button',show_label=True)),
                 #Item(name='printit', show_label=False),
         Item('info', style='readonly'),
         Item('plot',
              editor=ComponentEditor(),
              resizable=True, height = 700,
              show_label=False),
         resizable=True,
         kind = 'live',
         title = 'Linearize',
         icon = ImageResource('images/linearization.png') 
     )

     def __init__(self,  plotter=None, meas=None):
        """
        plotter: instance of massiveOES.GUI.Plotter
        """
        #HasTraits.__init__(self)
        self.not_ready = True
        self.sim = plotter
        self.sim_x = plotter.sim_x
        self.sim_y = plotter.sim_y
        self.meas_y = meas.meas_y

        self.start = plotter.tr_params.wav_start
        self.linear = plotter.tr_params.wav_step
        self.quadratic = plotter.tr_params.wav_2nd

        self.on_trait_change(self._replot,'meas_x')
        self.on_trait_change(self._replot,'meas_y')
        self.on_trait_change(self._replot,'sim_x')
        self.on_trait_change(self._replot,'sim_y')
        self.on_trait_change(self.callbackL,"click_tool:click_coordsL")
        self.on_trait_change(self.callbackR,"click_tool:click_coordsR")
        self.on_trait_change(self._add_pair_changed, 'click_tool:adder')
        self.on_trait_change(self.recalculate_meas,'start,linear,quadratic')
        self.plot.x_mapper.range.low = min(self.sim_x)
        self.recalculate_meas()
        self.set_ranges()
        #import os
        #cwd = os.getcwd()	
        #print('*********xxxx*****', cwd)	
     

     def _printit_changed(self):
         print ('pair', self.current_pair.left, self.current_pair.right)
         print ('meas_y', self.meas_y)
         print ('meas_x', self.meas_x)

     def _send_vals_changed(self):
        self.sim.tr_params.wav_start = self.start
        self.sim.tr_params.wav_step = self.linear
        self.sim.tr_params.wav_2nd = self.quadratic
         
    
                 
     def _add_pair_changed(self):
         self.current_pair = Pair(left=self.left, right=self.right)
         self.pairs.append(self.current_pair)

     def _plot_default(self):
         self.plotdata = ArrayPlotData(meas_x = self.meas_x, meas_y = self.meas_y, sim_x = self.sim_x, sim_y = self.sim_y, click_meas = [0,0], click_sim = [0,0], click_y = [0,1])
         plot = Plot(self.plotdata)
         plot.plot(('sim_x', 'sim_y'), type='line', color='blue', name='simulation')
         plot.plot(('meas_x', 'meas_y'), type='line', color='green', name='measurement')
         plot.plot(('click_sim', 'click_y'), color='blue')
         plot.plot(('click_meas', 'click_y'), color='green')

         self.click_tool = ClickTool(plot)
         plot.tools.append(self.click_tool)
         plot.tools.append(PanTool(plot))
         plot.tools.append(BetterZoom(plot))

         #self.plotdata.set_data('click_sim', [min(self.sim_x), min(self.sim_x)])

         plot.legend.visible = True
         self.not_ready = False
         return plot

     def set_ranges(self):
         self.plot.y_mapper.range.high = max(self.meas_y) 
         self.plot.y_mapper.range.low = min(self.meas_y)
         self.plot.x_mapper.range.high = max(self.meas_x) 
         self.plot.x_mapper.range.low = min(self.meas_x)
         
     
     def _load_meas_button_changed(self):
         self.get_measured()
         
     def _load_sim_button_changed(self):
         self.get_sim()
    
     def _replot(self):
        if self.not_ready:
            return
         
        self.debugmesage(self.sim_y) 
        self.plotdata.set_data('sim_x', self.sim_x)
        self.plotdata.set_data('sim_y', self.sim_y)
        self.plotdata.set_data('meas_x', self.meas_x)
        self.plotdata.set_data('meas_y', self.meas_y)
        self.plotdata.set_data('click_meas', [self.meas_x[self.left], 
                                              self.meas_x[self.left]])        
        self.plotdata.set_data('click_sim', [self.right, self.right])
        self.plotdata.set_data('click_y', [self.plot.y_mapper.range.low,
                                           self.plot.y_mapper.range.high])

    
    

     def debugmesage(self,string):
         if self.debug is True:
             print(string)
#Streva
######################################################################################

     
     # nevim jak tady
     def get_from_params(self, number_of_points, **kwargs):
         
         self.start = kwargs.pop('start', 0)
         self.linear = kwargs.pop('linear', 1)
         self.quadratic = kwargs.pop('quadratic', 0)
         self.nop = number_of_points

### tady se bude podavat realne a nejake modelove spektrum takze to se musi spravit     
     def get_measured(self):
        self.meas_x = np.linspace(1,100,num=100)
        #testovaci data
        self.meas_y =  np.exp(-((np.array(self.meas_x)-25)**2)/4) + np.exp(-((np.array(self.meas_x)-75)**2)/4)
        self.debugmesage('Spectrum loaded')
        
     def get_sim(self):
        self.sim_x = np.linspace(1,100,num=100)
        #testovaci data
        self.sim_y = np.exp(-((np.array(self.sim_x)-37.5)**2)/2) + np.exp(-((np.array(self.sim_x)-62.5)**2)/2)
        self.debugmesage('Spectrum loaded')
    
         
     def point_value(self, coordinate):
         return self.start + self.linear*coordinate + self.quadratic*(coordinate**2)
         
     def recalculate_meas(self):
         print ('RECALCULATING MEAS!')
         tmp = []
         for i in range(len(self.meas_y)):
             tmp.append(self.point_value(i))
         self.meas_x = tmp
         print (self.meas_x)
    
     def recalculate_sim(self):
         tmp = Array
         for i in range(len(self.sim_y)):
             tmp.append(point_value(i))
         self.sim_x = tmp     
             

     def calculate(self,pairs):
         print ('KALKULUJEM!')
         # take list of pairs (position in measured list, new value for x axis for this position)
         if len(pairs) == 0:
             warnings.WarningMessage('Empty input!')
             return False
     
     
         if sorted(pairs, key=itemgetter(0)) != sorted(pairs, key=itemgetter(1)):
             warnings.WarningMessage('Input does not provide correct data to estimate parameters! (Has to be growing function)')
             return False
     
         pairs = sorted(pairs, key = itemgetter(0))
     
         if len(pairs) == 1:
             self.start += pairs[0].right -self.point_value(pairs[0].left)
             return True
         
         # if len(pairs) == 2:
         #     self.linear = np.abs(pairs[0].right-pairs[1].right/np.abs(pairs[0].left - pairs[1].left))
         #     self.start += pairs[0].right -self.point_value(pairs[0].left)
         #     return True
         
         
         x = []; y =[]
         for pair in pairs:
             x.append(pair.left)
             y.append(pair.right)
             if len(pairs) == 2:
                  self.linear, self.start = np.polyfit(x,y,1)
             else:
                 self.quadratic, self.linear,self.start = np.polyfit(x,y,2)
             

     def _linbutt_changed(self):
         self.calculate(self.pairs)

             
     #@on_trait_change("click_tool:click_coordsL")
     def callbackL(self):
         self.left = int(find_nearest(self.meas_x, self.click_tool.click_coordsL))
         print ("Demo got coordinates left: ", self.click_tool.click_coordsL)
         self._replot()
        
     #@on_trait_change("click_tool:click_coordsR")
     def callbackR(self):
         self.right = self.click_tool.click_coordsR
         print ("Demo got coordinates left: ", self.click_tool.click_coordsR )
         self._replot()
             

     @staticmethod
     def exists(string):
         try:
             eval(string)
         except NameError:
             return False
         else:
             return True
             

 
    
             
#mojevokno = Coefficients() 
  
#mojevokno.configure_traits()
         
         
