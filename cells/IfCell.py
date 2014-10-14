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

              
class IfCell(Cell):
    
    def __init__(self, C = 0.0001*uF, R = 200*MOhm, e = -74*mV, thresh = -54*mV, vrefrac = -60*mV, dgk = 0, egk = -90*mV, ctau = 10*ms, dgk2 = 0, dtau = 1*ms, sigma_C = 0, sigma_R = 0, gid = 0, s_reset_noise = 0*mV, refrac = 0.00001*ms):        

        self.gid = gid
        
        if sigma_C == 0:
            self.CM = C                 # uF
        else:
            np.random.seed(self.gid*40)
            self.CM = np.random.normal(C, C*sigma_C, 1).clip(min=C*sigma_C)   
            
        if sigma_R == 0:
            self.RM = R
        else:
            np.random.seed(self.gid*40)
            self.RM = np.random.normal(R, R*sigma_R, 1).clip(min=R*sigma_R)   
            
        self.mtau = self.RM / MOhm * self.CM / uF  # s
        #print "tau=",self.mtau*1e3
             
        # refractory period [ms]
        grefrac = 10000*uS           # clamp to refractory period

        self.thresh = thresh
        self.soma = h.Section(name='soma', cell=self)

        cm = 1 # uF/cm2
        Area = self.CM / cm    # (uF/(uF/cm2)) = cm2     
        l = np.sqrt(Area / np.pi) * 1e4   # um
        
        self.soma.L = l         # um
        self.soma.diam = l      # um
        self.soma.nseg = 1
        
        self.soma.insert('pas')

        GM = 1 / (self.RM*1e6)    #  S
        g = GM / Area                # S/cm2        
        self.soma(0.5).pas.g = g
        self.soma(0.5).pas.e = e/mV
        
        if s_reset_noise > 0: # add noise model B (Gerstner 2000)
            
            self.spkout = h.SpikeOutRandreset(self.soma(0.5))
            self.spkout.thresh = thresh/mV      # (mV)
            self.spkout.refrac = refrac/ms      # (ms)
            self.spkout.vrefrac = vrefrac/mV    # (mV) reset potential
            self.spkout.grefrac = grefrac/uS    # (uS) clamped at reset
            self.spkout.mtau = self.mtau/ms      # (ms)
            
            noiseRandObj = h.Random()  # provides NOISE with random stream
            self.noise = noiseRandObj  # has to be set here not inside the nmodl function!!   
            self.spkout.noiseFromRandom(self.noise)  # connect random generator!
            self.noise.MCellRan4(2, self.gid+1)  # set lowindex to gid+1, set highindex to > 0
            self.noise.normal(0,s_reset_noise) # /ms
            
        elif (dgk == 0) and (dgk2 == 0):
            
            self.spkout = h.SpikeOut(self.soma(0.5))
            self.spkout.thresh = thresh/mV      # (mV)
            self.spkout.refrac = refrac/ms      # (ms)
            self.spkout.vrefrac = vrefrac/mV    # (mV) reset potential
            self.spkout.grefrac = grefrac/uS    # (uS) clamped at reset
            
        elif dgk2 == 0:
        
            self.spkout = h.SpikeOutAdapt(self.soma(0.5))
            self.spkout.thresh = thresh/mV      # (mV)
            self.spkout.refrac = refrac/ms      # (ms)
            self.spkout.vrefrac = vrefrac/mV    # (mV) reset potential
            self.spkout.grefrac = grefrac/uS    # (uS) clamped at reset
            
            dgkbar = dgk*1e-3 / Area         # dgk in (uS), * 1e-3 = (mS), / cm2 = (mS/cm2)
            
            self.spkout.dgkbar = dgkbar      # (mS/cm2) AHP conductance
            self.spkout.egk   = egk/mV          # (mV) AHP reversal potential  
            self.spkout.ctau =  ctau/ms         # (ms) AHP time constant
            
            #self.spkout.idrive =  0          # (nA) continuous drive not used 
            
        else:
            
            self.spkout = h.SpikeOutAdapt2(self.soma(0.5))
            self.spkout.thresh = thresh/mV      # (mV)
            self.spkout.refrac = refrac/ms      # (ms)
            self.spkout.vrefrac = vrefrac/mV    # (mV) reset potential
            self.spkout.grefrac = grefrac/uS    # (uS) clamped at reset
            
            dgkbar = dgk*1e-3 / Area         # dgk in (uS), * 1e-3 = (mS), / cm2 = (mS/cm2)
            
            self.spkout.dgkbar = dgkbar      # (mS/cm2) AHP conductance
            self.spkout.egk   = egk/mV          # (mV) AHP reversal potential  
            self.spkout.ctau =  ctau/ms         # (ms) AHP time constant
            
            dgkbar2 = dgk2*1e-3 / Area         # dgk in (uS), * 1e-3 = (mS), / cm2 = (mS/cm2)
            
            self.spkout.dgkbar2 = dgkbar2      # (mS/cm2) AHP conductance 
            self.spkout.dtau =  dtau/ms         # (ms) AHP time constant


        self.v_init = e # supply own v_init
        
        Cell.__init__(self) 
        
        
    def add_resonance(self, tau_r = 5*ms, gr = 0*uS):
        
        self.gr = gr
        self.tau_r = tau_r
        self.resonance = h.ResonantCurrent(self.soma(0.5))
        self.resonance.tau = tau_r/ms
        self.resonance.g = gr/uS   
        self.resonance.vrest = self.soma(0.5).pas.e/mV
        
        
    def add_resonance_spike(self, tau_r = 5*ms, gr = 0*uS):
        
        self.gr = gr
        self.tau_r = tau_r
        self.resonance = h.KCa(self.soma(0.5))
        self.resonance.ctau = tau_r/ms
        self.resonance.dgkbar = gr/uS   
        self.resonance.egk = self.soma(0.5).pas.e/mV
        self.resonance.thresh = self.thresh
        
  
# test code
if __name__ == '__main__': 
    
   
    from Stimulation import *
    from Plotter import *
    from Stimhelp import * 
    from IfCell import *
    
    #cell = IfCell()
    #cell.add_resonance(tau_r = 5*ms, gr = 0.1*uS)
    #cell = IfCell(C = 3.1*pF, R = 1/(0.43*nS), e = -58*mV, thresh = -35*mV, vrefrac = -65*mV)
    #cell = IfCell(C = 28*pF, R = 1/(2.3*nS), e = -55*mV, thresh = -52*mV, vrefrac = -65*mV)
    
    #cell = IfCell(C = 2.985e-06*uF, R = 9045*MOhm)

    thresh = -20.13*mV
    tau_r = 50*ms 
    gr = 0.0001*uS # -0.0012463*uS
    R = 9045*MOhm # 737*MOhm
        
    celltype = "IfCell"
    cell_exe = "cell = IfCell(C = 2.985e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -60*mV, dgk =" + str(gr) + ", egk = -71.5*mV, ctau =" + str(tau_r) + ")"#; cell.add_resonance_spike(tau_r =" + str(tau_r) + ", gr =" + str(gr) + ")" 
    exec cell_exe
    
    cell = IfCell(C = 1, R = 1e14, e = 0, thresh = 1, vrefrac = 0)   
    #cell.v_init = 0        
    
    #cell = IfCell(C = 2.985e-06*uF, R = 9045*MOhm, e = -71.5*mV, thresh = -20.13*mV, vrefrac = -60*mV)
    #cell.add_resonance(tau_r = 1*ms, gr = -0.0012463*uS)
    #cell.add_resonance_spike(tau_r = 100*ms, gr = 0.008*uS)

    sim = Stimulation(cell, temperature = 0)
    sim.spikes_from_neuron = True
    
    sim.set_IClamp(amp = 10, delay = 500*ms, dur = 2000*ms)  # 0.00568*nA
    
    tstop = 2000*ms
    sim.run(1, tstop)
    
    t1, voltage, current, g_in = sim.get()
    freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = sim.if_get(compute_mean = 0)
    
    print "freq_times: " + str(spike_freq)
    
    figure()
    plot(t1, voltage)
    
    figure()
    plot(freq_times, spike_freq)
    
    show()  
    
    