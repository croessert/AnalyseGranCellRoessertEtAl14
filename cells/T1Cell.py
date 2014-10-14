# -*- coding: utf-8 -*-
"""
Type 1 - Spiking neuron

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


class T1Cell(Cell):
    
    def __init__(self, GS=10, CS=500):

        self.soma = h.Section(name='soma', cell=self)
        CM = CS * 1e-6                              # uF
        cm = 1
        Area = CM/cm                              # (uF/(uF/cm2)) = cm2     
        l = sqrt(Area/pi)                     # cm
        self.soma.L = self.soma.diam = l * 1e4      # um
        self.soma.nseg = 1
        
        self.soma.insert('fh')         
        self.soma(0.5).fh.gnabar = 0.1
        self.soma(0.5).fh.gl = GS * 1e-9 / Area        
        self.soma(0.5).fh.el = -70
        
        self.soma.insert('kml')  
        self.soma(0.5).kml.gbar = 0.005
        self.soma(0.5).kml.bn = -20
        self.soma(0.5).kml.gn = 10
        self.soma(0.5).kml.tn = 3
        
        self.soma.insert('kml2')  
        self.soma(0.5).kml2.gbar = 5e-4
        self.soma(0.5).kml2.bn = -50
        self.soma(0.5).kml2.gn = 15
        self.soma(0.5).kml2.tn = 50
        
        self.thresh = 0
        self.v_init = self.soma(0.5).fh.el # supply own v_init 
        
        Cell.__init__(self)
        

    def g_record(self):
        
        self.rec_gres = h.Vector()
        self.rec_gres.record(self.soma(0.5).kml2._ref_g)
        
        
    def g_get(self):
        
        self.gres = array(self.rec_gres)
        return self.gres


# test code
if __name__ == '__main__': 
    
   
    from Stimulation import *
    from Plotter import *
    from Stimhelp import * 
    
    cell = T1Cell()
    #cell.add_resonance(tau_r = 50*ms, gr = 0.01*uS)
    
    sim = Stimulation(cell, temperature = 14)
    
    #sim.cell.soma(0.5).kml2.gbar = 1e-4
    #sim.cell.soma(0.5).kml2.bn = -60
    #sim.cell.soma(0.5).kml2.gn = 10
    #sim.cell.soma(0.5).kml2.tn = 50
        
    sim.set_IClamp(amp = 1.5, delay = 100*ms, dur = 500*ms)  # 0.004
    
    tstop = 600*ms
    
    cell.g_record()
        
    sim.run(1, tstop)
    
    gres = cell.g_get()    
    t1, voltage, current, g_in = sim.get()
    freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = sim.if_get(compute_mean = 0)
    
    #print "freq_times: " + str(spike_freq)
    
    subplot(3,1,1)
    plot(t1, voltage)
    subplot(3,1,2)
    plot(freq_times, spike_freq)
    subplot(3,1,3)
    plot(t1, gres)
    
    show()  
    
    
#    v=arange(-80,20,0.1)
#    bn=-50
#    gn=15
#    tn=50
#    ninf = (1/2) * (1 + tanh( (v-bn) / gn ) )
#    ninf2 = (1 / (1 + exp((bn - v) / (gn/2))))
#    ntau = tn / cosh( (v-bn) / (2*gn) )
#    ntau2 = (2*tn) / ( exp((v-bn)/(2*gn)) + exp((-v+bn)/(2*gn)) )
#    plot(v, ninf, v, ninf2, 'r--')
#    plot(v, ntau, v, ntau2, 'r--')
