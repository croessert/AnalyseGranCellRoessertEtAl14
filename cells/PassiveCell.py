# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 20:06:39 2011

@author: chris
"""

from __future__ import division
import numpy as np
from neuron import h

import sys
mymodules = '../'
sys.path.append(mymodules)    
from units import *
from Cell import *


class PassiveCell(Cell):
    
    def __init__(self, GS=100, CS=1000):
        
        self.soma = h.Section(name='soma', cell=self)
        CM = CS * 1e-6                              # uF
        cm = 1
        Area = CM/cm                              # (uF/(uF/cm2)) = cm2     
        l = np.sqrt(Area/np.pi)                     # cm
        self.soma.L = self.soma.diam = l * 1e4      # um
        self.soma.nseg = 1
        self.soma.insert('pas')
        #GS =  100                                  # nS
        self.soma(0.5).pas.g = GS * 1e-9 / Area
        self.soma(0.5).pas.e = -70
        
        self.thresh = 0
        self.v_init = self.soma(0.5).pas.e # supply own v_init 
        
        Cell.__init__(self)
        
        
# test code
if __name__ == '__main__': 
    
   
    from Stimulation import *
    from Plotter import *
    from Stimhelp import * 
    
    cell = PassiveCell()
        
    sim = Stimulation(cell, temperature = 0)
    
    sim.set_IClamp(amp = 0.13, delay = 100*ms, dur = 500*ms)  # 0.004
    
    tstop = 600*ms
    sim.run(do_freq = 0, tstop = tstop)
    
    t1, voltage, current, g_in = sim.get()
        
    plot(t1, voltage)
    
    show()  
        