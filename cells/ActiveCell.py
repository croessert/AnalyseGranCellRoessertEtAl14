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


class ActiveCell(Cell):
    
    def __init__(self, L_diam = 100):
        self.soma = h.Section(name='soma', cell=self)
        self.soma.L = L_diam     # um, default: 100
        self.soma.diam = L_diam  # um, default: 500
        self.soma.nseg = 1
        self.soma.insert('hh')
        
        self.thresh = -20
        self.v_init = -64.9736783327 # supply own v_init
        
        Cell.__init__(self)
        
        
# test code
if __name__ == '__main__': 
    

    from Stimulation import *
    from Plotter import *
    from Stimhelp import *
    from ActiveCell import *
    
    cell = ActiveCell()
    #cell.add_resonance(tau_r = 5*ms, gr = 0.1*uS)
    
    sim = Stimulation(cell)
    
    sim.set_IClamp(amp = 4, delay = 100*ms, dur = 500*ms)  # 0.004
    
    tstop = 600*ms
    sim.run(1, tstop)
    
    t1, voltage, current, g_in = sim.get()
    freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = sim.if_get(compute_mean = 0)
    
    print "freq_times: " + str(spike_freq)
    
    plot(t1, voltage)
    
    show()  
        
    
