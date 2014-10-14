# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 15:48:04 2011

@author: chris
"""
from __future__ import with_statement
from __future__ import division

import sys
sys.path.append('../NET/sheff/weasel/')
sys.path.append('../NET/sheffprk/')

import os

from mpi4py import MPI

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', action='store', dest='opt')
    parser.add_argument('--noplot', action='store_true') # IF OPTIONAL ARG: nargs='?', default=False
    parser.add_argument('--norun', action='store_true')
    parser.add_argument('--noconst', action='store_true')
    parser.add_argument('--noqual', action='store_true')
    results = parser.parse_args()
    
    import matplotlib
    if MPI.COMM_WORLD.rank == 0: 
        matplotlib.use('Tkagg', warn=True)    
    else:    
        matplotlib.use('Agg', warn=True)
    
    do_plot = 1
    if results.noplot:  # do not plot to windows
        matplotlib.use('Agg', warn=True)
        if MPI.COMM_WORLD.rank == 0: print "- No plotting"
        do_plot = 0

import numpy as np
from neuron import h

from Plotter import *
from Stimhelp import *
from cells.PassiveCell import *

from units import *

try:
    import cPickle as pickle
except:
    import pickle

import gzip
import multiprocessing


class Stimulation:
    """
    Set, control and start a simulation with this class.
    >>> sim = Stimulation(cell)
    >>> sim.set_IClamp()
    >>> sim.run()
    >>> show()
    """

    def __init__(self, cell = None, celltype = None, cell_exe = 0, temperature = 6.3, do_run = 1, pickle_prefix = "", give_freq = False, istart = 0, istop = 0.07, di = 0.001, use_multisplit=False):
        
        self.cell = cell
        self.celltype = celltype
        self.cell_exe = cell_exe  
        
        self.temperature = temperature
        self.stim = None
        self.syn1 = self.syn2 = self.syn3 = self.syn4 = None
        
        self.fluct = None
                
        self.tmax = 10*s   # maximum length of plot that should be plotted!!
        self.synout_tau1 = 5*ms
        self.synout_tau2 = 5*ms
        self.spikes_from_neuron = False # set true for synaptic interpolation!
        
        self.dt = 0.025*ms
        self.ihold = 0*nA 
        self.amp = 0.1*nA
        self.amp_noise = 0*nA
        self.tau_noise = 10*ms

        self.amod = None
        self.anoise = None
        
        self.give_freq = give_freq # RUN self.set_i()
        
        self.do_run = do_run
        self.pickle_prefix = pickle_prefix
        self.color_vec = None
        self.del_freq = array([])
        
        self.jitter = 0*ms
        self.seed = None
        
        self.istart = istart
        self.istop = istop
        self.di = di
        
        self.syn_n = 0
        self.syn_g = 0.002
        self.syn_noise = 1
        self.syn_tau1 = 0*ms
        self.syn_tau2 = 10*ms
        
        self.icloc = "soma(0.5)"
        
        self.vinit = -71
        
        # Inhibition
        self.inh_n = 4
        self.inh_hold = 10
        self.inh_g = 1
        self.inh_noise = 0
        
        self.linewidth = 2
        
        #### Make a new ParallelContext object
        self.pc = h.ParallelContext()
        self.id = self.pc.id()
        self.nhost = self.pc.nhost()
                
        sme = "mpi4py thinks I am %d of %d on %s, NEURON thinks I am %d of %d\n"
        processorname = MPI.Get_processor_name()
        self.comm = MPI.COMM_WORLD
        
        if self.id == 0:
            print sme % (self.comm.rank, self.comm.size, processorname, self.pc.id(), self.pc.nhost())
            
        self.run_pp = False
        self.delta_t = 0*ms
        self.use_multisplit = use_multisplit
    
        if self.use_multisplit: 
            self.set_multisplit()
        else:
            # run the simulation
            h.load_file("stdrun.hoc")
            
        self.data_dir = "./data"
        self.minimal_dir = False

    def set_multisplit(self):
        
        #Hines = h.CVode()
        #Hines.active(0)
        h.load_file("parcom.hoc")
        p = h.ParallelComputeTool()
        cpus = 32
            
        p.change_nthread(cpus,1)    
        
        p.multisplit(1)
        print "Using multisplit, cpus:", cpus
            
    def set_IClamp(self, amp = 1*nA, delay = 100*ms, dur = 800*ms, stims = 1):
        """
        Initializes values for current clamp.
        """
        if hasattr(self.cell, 'input_vec'):
            
            #self.stim = []
            if self.stim is not None:
                for i in self.stim:
                    del i
            
            #if isinstance(self.stim, list):
            #    for i in self.stim:
            #        i.amp = amp / self.cell.n_input_spiny / nA  
            #else:
            self.stim = []
            for vec in self.cell.input_vec:
                for i in vec:
                    self.stim.append(h.IClamp(i(0.5)))
                    self.stim[-1].amp = self.cell.ifac * amp / self.cell.n_input_spiny / nA
                    self.stim[-1].delay = delay / ms
                    self.stim[-1].dur = dur / ms
                
        else:
         
            exec("stim = h.IClamp(self.cell." + self.icloc + ")") # stim = h.IClamp(self.cell.soma(0.5))
            stim.amp = amp/nA
            stim.delay = delay/ms
            stim.dur = dur/ms
            
            if stims == 2:
                self. stim2 = stim           
            else:    
                self.stim = stim


    def set_IPlay(self, iarray, tarray):
        """
        Initializes values for current clamp to play signal.
        """
        stim = None  # delete IClamp that might has been set, don't know if needed?
        exec("stim = h.IClamp(self.cell." + self.icloc + ")") # stim = h.IClamp(self.cell.soma(0.5))
        stim.delay = 0
        stim.dur = 1e9

        tvec = h.Vector(tarray/ms)
        ivec = h.Vector(iarray/nA)
        
        ivec.play(stim._ref_amp, tvec, 1)

        self.stim = stim
        self.iarray = ivec
        self.tarray = tvec


    def set_SynPlay(self, iarray, tarray, n_syn=1, g_s=0, tau1=0*ms, tau2=10*ms, noise=1):
        
        self.vecstim = []
        self.nc_vectim = []
        self.spike_vec = []
        self.nc = []
        
        for i in range(n_syn):
            
            if self.celltype == 'Grc':
                        
                self.cell.createsyn(nmf = 1, ngoc = 0, weight = g_s) 
                                
                self.vecstim.append(h.VecStim(.5))
                self.nc_vectim.append(h.NetCon(self.vecstim[-1],None))
                
                print "-- generate train"      
                train = mod_spike_train(modulation = (tarray, iarray), noise = noise, seed = i*66)
                print "-- generate train done"      
                
                self.spike_vec.append(h.Vector(train))
                self.vecstim[-1].play(self.spike_vec[-1]) 
                
                self.nc.append(h.NetCon(self.vecstim[-1], self.cell.MF_L[-1].input))
                
                #nc = [h.NetCon(self.vecstim[-1],self.cell.input,0,0,1) for goc in gocs.MF_L]
                
                self.nc[-1].weight[0] = 1
                self.nc[-1].delay = 0                
                
                
            else:

                self.cell.create_synapses(n_ex=1, tau1=tau1, tau2=tau2)
            
                self.vecstim.append(h.VecStim(.5))
                self.nc_vectim.append(h.NetCon(self.vecstim[-1],None))
                
                print "-- generate train"      
                train = mod_spike_train(modulation = (tarray, iarray), noise = noise, seed = i*66)
                print "-- generate train done"      
                
                #plt.figure(1001)
                #plt.plot(tarray, iarray)     
                #plt.plot(train[0:-1]*ms,1/diff(train*ms))
                #plt.show()
                
                self.spike_vec.append(h.Vector(train))
                self.vecstim[-1].play(self.spike_vec[-1]) 
                
                self.nc.append(h.NetCon(self.vecstim[-1], self.cell.synlist[-1]))
            
                self.nc[-1].weight[0] = g_s/n_syn
                self.nc[-1].delay = 0
                
                
    def set_InhPlay(self, iarray, tarray, n_inh=4, g_i=1, noise=1):
        
        for i in range(n_inh):
            
            if self.celltype == 'Grc':
                        
                self.cell.createsyn(nmf = 0, ngoc = 1, weight = g_i) 
                                
                self.vecstim.append(h.VecStim(.5))
                self.nc_vectim.append(h.NetCon(self.vecstim[-1],None))
                
                print "-- generate train"                
                train = mod_spike_train(modulation = (tarray, iarray), noise = noise, seed = i*67)
                print "-- generate train done"          
                
                self.spike_vec.append(h.Vector(train))
                self.vecstim[-1].play(self.spike_vec[-1]) 
                
                self.nc.append(h.NetCon(self.vecstim[-1], self.cell.GOC_L[-1].input))  
                
                self.nc[-1].weight[0] = 1
                self.nc[-1].delay = 0            
        

    def set_VClamp(self, amp0 = -90*mV, amp1 = 20*mV, amp2 = -90*mV,
                   dur0 = 100*ms, dur1 = 800*ms, dur2 = 100*ms):
        """
        Initializes values for voltage clamp.
        """

        exec("stim = h.VClamp(self.cell." + self.icloc + ")") # stim = h.VClamp(self.cell.soma(0.5))
        stim.dur[0] = dur0/ms
        stim.dur[1] = dur1/ms
        stim.dur[2] = dur2/ms
        stim.amp[0] = amp0/mV
        stim.amp[1] = amp1/mV
        stim.amp[2] = amp2/mV

        self.stim = stim

        self.syn1 = None


    def set_alphaStim(self, freq = 500*Hz, gmax = 0.005*uS, tau = 0.1*ms, onset = 100*ms, e = 0*mV):
        """
        Stimulate with four consecutive EPSCs.
        """

        
        exec("syn1 = h.AlphaSynapse(self.cell." + self.icloc + ")") # syn1 = h.AlphaSynapse(self.cell.soma(0.5))  # first EPSC
        syn1.onset = onset/ms
        syn1.tau = tau/ms
        syn1.gmax = gmax/uS
        syn1.e = e/mV
        self.syn1 = syn1

        tnext = 1 / freq

        exec("syn2 = h.AlphaSynapse(self.cell." + self.icloc + ")") # syn2 = h.AlphaSynapse(self.cell.soma(0.5))  # second EPSC
        syn2.onset = (onset + tnext)/ms
        syn2.tau = tau/ms
        syn2.gmax = gmax/uS
        syn2.e = e/mV
        self.syn2 = syn2

        exec("syn3 = h.AlphaSynapse(self.cell." + self.icloc + ")") # syn3 = h.AlphaSynapse(self.cell.soma(0.5))   # second EPSC
        syn3.onset = (onset + tnext + tnext)/ms
        syn3.tau = tau/ms
        syn3.gmax = gmax/uS
        syn3.e = e/mV
        self.syn3 = syn3

        exec("syn4 = h.AlphaSynapse(self.cell." + self.icloc + ")") # syn4 = h.AlphaSynapse(self.cell.soma(0.5))  # second EPSC
        syn4.onset = (onset + tnext + tnext + tnext)/ms
        syn4.tau = tau/ms
        syn4.gmax = gmax/uS
        syn4.e = e/mV
        self.syn4 = syn4

        self.stim = None
        

    def set_Ifluct(self, sigma = 1*nA, tau = 2*ms, m = 0*nA):
        """
        Initializes values for current clamp.
        """
        
        if (sigma > 0) | (m > 0):
            exec("fluct = h.Ifluct1(self.cell." + self.icloc + ")") # fluct = h.Ifluct1(self.cell.soma(0.5))
            fluct.m = m/nA # [nA]
            fluct.s = sigma/nA # [nA]
            fluct.tau = tau/ms  # [ms]
            
            self.fluct = fluct
        
    
    def set_Gfluct(self, E_e = 0*mV, E_i = -75*mV, g_e0 = 0.0121*uS, g_i0 = 0.0573*uS, std_e = 0.0030*uS, std_i = 0.0066*uS, tau_e = 2.728*ms, tau_i = 10.49*ms):
        """
        Initializes values for current clamp.
        """

        
        exec("fluct = h.Gfluct3(self.cell." + self.icloc + ")") # fluct = h.Gfluct3(self.cell.soma(0.5))
        fluct.E_e = E_e/mV  # [mV]
        fluct.E_i = E_i/mV  # [mV]
        fluct.g_e0 = g_e0/uS  # [uS]
        fluct.g_i0 = g_i0/uS  # [uS]
        fluct.std_e = std_e/uS  # [uS] 
        fluct.std_i = std_i/uS  # [uS] 
        fluct.tau_e = tau_e/ms  # [ms] 
        fluct.tau_i = tau_i/ms  # [ms]    

        self.fluct = fluct
        
        # connect noise source!
        self.noiseRandObj = h.Random()  # provides NOISE with random stream
        self.fluct.noiseFromRandom(self.noiseRandObj)  # connect random generator!
        self.noiseRandObj.MCellRan4(1, 1)  # set lowindex to gid+1, set highindex to > 0 
        self.noiseRandObj.normal(0,1)
        
        
    def get_i(self, a, do_plot = True):

        import md5
        m = md5.new()
        m.update(self.cell_exe)
        filename = self.data_dir + '/if_' + self.celltype + '_' + m.hexdigest() + '.p'
        print filename
        
        if self.run_pp:
            if self.id == 0:
                is_there = os.path.isfile(filename)
            else:
                is_there = None
            
            is_there = MPI.COMM_WORLD.bcast(is_there, root=0)
        else:
            is_there = os.path.isfile(filename)    
        
        if is_there is not True: # run i/f estimation  
        
            if self.id == 0: print '- running i/f estimation for ', self.celltype, ' id: ' , m.hexdigest() 
            
            current_vector, freq_vector, freq_onset_vector = self.get_if(istart = self.istart, istop = self.istop, di = self.di) 
            
            if self.id == 0:
                
                if do_plot:
                    plt.figure(99)
                    plt.plot(current_vector, freq_vector, '*-')
                    plt.savefig("./figs/dump/latest_if_" + self.celltype +".pdf", dpi = 300)  # save it 
                    plt.clf()
                    #plt.show()
                
                ifv = {'i':current_vector,'f':freq_vector}
                pickle.dump(ifv, gzip.GzipFile(filename, "wb" ))
                
            if self.run_pp: self.comm.Barrier() 
            
        else:
            
            if self.id == 0:            
                ifv = pickle.load(gzip.GzipFile(filename, "rb" ))
        
        if self.run_pp: MPI.COMM_WORLD.Barrier()
        
        if self.id == 0:
            
            f =  ifv.get('f') 
            i =  ifv.get('i')
        
            i = i[~isnan(f)]
            f = f[~isnan(f)]
        
            iin = if_extrap(a, f, i)
        
        else:
            
            iin = None
        
        if self.run_pp:
            iin = MPI.COMM_WORLD.bcast(iin, root=0)
            MPI.COMM_WORLD.Barrier()
        
        
        return iin
        

    def set_i(self):
        
        ihold = np.copy(self.ihold)
        
        # Ihold given as frequency, convert to current
        if self.give_freq is True:
            
            ihold = self.get_i(ihold)
            if self.id == 0: print '- ihold: ', self.ihold, 'Hz, => ihold: ', ihold, 'nA' 
        
        amp, amp_noise = self.set_amp(self.ihold, ihold)   
    
        return ihold, amp, amp_noise
        
        
    def set_amp(self, ihold, iholdi):
        
        amp = np.copy(self.amp)                
        # Modulation depth given, not always applied to current!
        if self.amod is not None:
            
            if self.give_freq is True:
                # Apply to amplitude:
                a = ihold + self.amod*ihold
                amp = self.get_i(a)-iholdi
                
            else:
                amp = self.amod * ihold
            
            if self.id == 0: print '- amp: ihold: ', ihold, 'nA , amod: ', self.amod, ', => amp: ', amp, 'nA' 
            
        
        amp_noise = np.copy(self.amp_noise)
        if self.anoise is not None:
            
            if self.give_freq is True:
                # Apply to amplitude:
                a = ihold + self.anoise*ihold
                amp_noise = self.get_i(a)-iholdi

            else:
                amp_noise = self.anoise * ihold
            
            if self.id == 0: print '- noise: ihold: ', ihold, 'nA , anoise: ', self.anoise, ', => amp_noise: ', amp_noise, 'nA' 
            
            
        return amp, amp_noise
        
        
    def record(self):
        """
        Initializes recording vectors. Internal function
        """

        # Time
        self.rec_t = h.Vector()
        self.rec_t.record(h._ref_t)
        
        # Voltage
        self.rec_v = h.Vector()
        exec("self.rec_v.record(self.cell." + self.icloc + "._ref_v)") 
        
        # Stimulus
        self.rec_i = h.Vector()
        
        if isinstance(self.stim, list):
            pass
        elif self.stim is not None:  # onyl one of these
            self.rec_i.record(self.stim._ref_i)
        elif self.syn1 is not None: 
            self.rec_i.record(self.syn1._ref_i)

        # Synapses
        if hasattr(self.cell, 'synlist'):
            if len(self.cell.synlist) > 0:  # record synaptic current!!
                self.rec_syn_list = h.List()
                self.rec_gin_list = h.List()
                for i in range(len(self.cell.synlist)):
                    rec_syn = h.Vector()
                    rec_syn.record(self.cell.synlist[i]._ref_i)
                    self.rec_syn_list.append(rec_syn)
                    
                    rec_gin = h.Vector()
                    rec_gin.record(self.cell.synlist[i]._ref_g)
                    self.rec_gin_list.append(rec_gin)
        
        # Noise        
        if self.fluct is not None:
            self.rec_n = h.Vector() 
            self.rec_n.record(self.fluct._ref_i)


    def if_record(self):
        """
        Initializes recording vectors to record "zero" crossings. Internal function
        """

        # Spiketimes
        self.rec_s = h.Vector()
        nc = self.cell.connect_target(None)  # threshold is set in neuron definition, or here!
        nc.record(self.rec_s)  # record indexes of the positive zero crossings
        
        # Synaptic integrated response
        self.rec_g = h.Vector()
        self.passive_target = PassiveCell()
        syn = self.passive_target.create_synapses(tau1 = self.synout_tau1, tau2 = self.synout_tau2)  # if tau1=tau2: alpha synapse!
        self.nc_p = self.cell.connect_target(syn)  # threshold is set in neuron definition, or here!
        self.rec_g.record(syn._ref_g)
        
    
    def run(self, do_freq = 1, tstop = 1000*ms):
        """
        Starts the stimulation.
        """
        self.record()
        
        self.rec_g = []
        if do_freq and self.spikes_from_neuron:
            self.if_record()

        h.celsius = self.temperature              
        h.init()
        h.tstop = tstop/ms
        h.dt = self.dt/ms
        h.steps_per_ms = 1 / (self.dt/ms)
        
        if hasattr(self.cell, 'v_init'):
            h.v_init = self.cell.v_init  # v_init is supplied by cell itself!
        else:
            h.v_init = self.vinit
                
        h.run()        
        
        
    def run_steps(self, do_freq = 1, tstop = 1000*ms, simstep = 1*ms, do_loadstate = True):
        """
        Starts the stimulation. Simulates in steps
        """
        
        self.record()
        
        self.rec_g = []
        if do_freq and self.spikes_from_neuron:
            self.if_record()
        
#        if self.use_multisplit:
#            
#            #Hines = h.CVode()
#            #Hines.active(0)
#            
#            h.load_file("parcom.hoc")
#            p = h.ParallelComputeTool()
#            cpus = 32
#                
#            p.change_nthread(cpus,1)    
#            
#            p.multisplit(1)
#            print "Using multisplit, cpus:", cpus
#            
#        else:
#            
#            # run the simulation
#            h.load_file("stdrun.hoc")
        
        #cpu = multiprocessing.cpu_count()
        #h.load_file("parcom.hoc")
        #p = h.ParallelComputeTool()
        #p.change_nthread(cpu,1)
        #p.multisplit(1)
        #print 'cpus:', cpu
        
        # run the simulation
            
        h.celsius = self.temperature              
        h.init()
        h.tstop = tstop/ms
        h.dt = self.dt/ms
        h.steps_per_ms = 1 / (self.dt/ms)
        
        if hasattr(self.cell, 'v_init'):
            h.v_init = self.cell.v_init  # v_init is supplied by cell itself!
        else:
            h.v_init = self.vinit     
        
        h.stdinit()  
        import time
        t0 = time.time()
        
        h.finitialize() 
        
        filename = './states_' + self.celltype + '_Stimulation.b'
        if hasattr(self.cell, 'load_states') and do_loadstate:
            self.cell.load_states(filename)
            
        cnt = 1
        while h.t < tstop/ms:           
            h.continuerun(cnt*simstep/ms)
            print "Simulated time =",h.t*ms, "s"
            cnt += 1

        print "psolve took ", time.time() - t0, "seconds"
        self.tstop = tstop  
        
        if hasattr(self.cell, 'get_states') and (do_loadstate is False):
            self.cell.get_states(filename)
        

    def get(self):
        """
        Gets the recordings. Transform time back to s
        """

        t1 = array(self.rec_t)*ms
        voltage = array(self.rec_v)*mV
        g_in = zeros(len(voltage)) 
                
        if (self.stim is not None) or (self.syn1 is not None):
            current = array(self.rec_i)*nA
        else:
            current = zeros(len(voltage))
        
        if hasattr(self.cell, 'synlist'):
            if self.cell.synlist > 0:  # record synaptic current!!
                for i in range(len(self.cell.synlist)):
                    current = current + -1*array(self.rec_syn_list[i])*nA  # switch current, so same sign as electrode current
                    g_in = g_in + array(self.rec_gin_list[i])*nA
                
        if self.fluct is not None: # if there is a fluctuating current present combine with input current
            current = current + array(self.rec_n)*nA
            
        return t1, voltage, current, g_in 


    def if_get(self, change_factor = 0.3, compute_mean = 1):
        """
        Gets the spike times/frequency recordings.
        """
        
        t1 = array(self.rec_t)*ms
               
        if self.spikes_from_neuron:
            spike_times = array(self.rec_s)*ms  # use this here for default, so threshold can be set for each cell individually in the neuron framework
        else:
            if hasattr(self.cell, 'thresh'):
                vthres = self.cell.thresh  # voltage threshold, careful for I&F Neurons where vthres = spike_threshold 
            else:  # no attribute set, use 0
                vthres = -20   
            voltage = array(self.rec_v)*mV
            s, spike_times = get_spikes(voltage, vthres, t1)

        gsyn = array(self.rec_g)  # integrated synaptic response
        
        if (self.jitter > 0 or self.spikes_from_neuron == False) and len(spike_times) > 1:
            
            spike_times = spike_times + self.delta_t
            
            if self.jitter > 0:
                np.random.seed(66)
                x = np.random.normal(0, self.jitter, len(spike_times))             
                spike_times = spike_times + x
            
            import neuronpy.util.spiketrain
            [resp, _] = neuronpy.util.spiketrain.get_histogram(spike_times, bins = t1)
            resp = concatenate((zeros(1),resp))
            
            Ksyn = syn_kernel(arange(0, 10*self.synout_tau2, self.dt), self.synout_tau1, self.synout_tau2) 
            Ksyn = concatenate((zeros(len(Ksyn)-1), Ksyn))
            gsyn = convolve(Ksyn, resp, mode='same')

        freq_times, spike_freq, freq_mean, freq_onset = get_spikefreq(spike_times, stimlength = compute_mean, compute_mean = compute_mean, change_factor = change_factor)
        

        return freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn 
        
        
    def do_ZAP_Stim(self, t_stim = 20*s, fmax = 50*Hz, ex = 1, do_freq = 1):

        ihold, amp, amp_noise = self.set_i()
        t, zap, f = create_ZAP(dt = self.dt, ihold = ihold, amp = amp, fmax = fmax, t_stim = t_stim, ex=ex)    
        fs = 1 / self.dt # sampling rate 
        
        delay_baseline = 2 * fs  # 2 s baseline delay
        
        stimulus = concatenate([self.ihold*ones(delay_baseline), zap])  # construct stimulus
        t = arange(0, size(stimulus) * self.dt,self.dt)
        tstop = t[-1]
        
        print "- starting ZAP stimulation! with amp = " + str(amp) + ", ihold = " + str(ihold) + ", dt = " + str(self.dt) + " => maximum frequency = " + str(fmax) + "\r"    

        self.set_IPlay(stimulus, t)          
        self.run(do_freq, tstop)
   
        t1, voltage, current, g_in = self.get() # time in ms
        self.spikes_from_neuron = do_freq
        freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = self.if_get(compute_mean = 0)  
        
        subplot(3,1,1)
        plot(t, stimulus)
        subplot(3,1,2)
        plot(t1, voltage)
        subplot(3,1,3)
        plot(freq_times,spike_freq)
        
    
    def fun_ssine_Stim(self, freq_used = array([1, 10, 100, 1000])*Hz, method_interpol = np.array(["none", "linear", "quadratic", "syn"])):
        """
        Compute impedance and/or transfer function using Single sine stimulation
        Only compute transfer function if there is a steady state (resting) firing rate!
        """
        
        if self.id == 0: 
            
            filename = str(self.pickle_prefix) + "_results_sim_ssine.p"
            filepath = self.data_dir + "/" + filename
            print filepath
            
            if self.do_run or (os.path.isfile(filepath) is False):
                
                from scipy.interpolate import InterpolatedUnivariateSpline 
                import neuronpy.util.spiketrain
                
                for i, m in enumerate(method_interpol):
                        if "syn" in m: 
                            method_interpol[i] = "syn " + str(self.synout_tau1/ms) + "/" + str(self.synout_tau2/ms) + "ms"
                       
                fs = 1 / self.dt # sampling rate 
                fmax = fs / 2 # maximum frequency (nyquist)
                
                if self.syn_n == 0:
                    
                    ihold, self.amp, self.amp_noise = self.set_i() 
                    if self.id == 0: print "- (CURRENT INPUT) starting single sine impedance/transfer function estimation! with amp = " + str(self.amp) + ", ihold = " + str(ihold) + ", dt = " + str(self.dt) + " => maximum frequency = " + str(fmax) + "\r"    
    
                else:
                    
                    ihold = self.ihold
                    self.amp, self.amp_noise = self.set_amp(ihold, 0)  
                    if self.id == 0: print "- (SYNAPTIC INPUT) starting single sine impedance/transfer function estimation! with amp = " + str(self.amp) + ", ihold = " + str(ihold) + ", dt = " + str(self.dt) + " => maximum frequency = " + str(fmax) + "\r"    
    
                
                for i, fu in enumerate(freq_used):
                    
                    t, stimulus, i_startstop, t_startstop = create_singlesine(fu = fu, amp = self.amp, ihold = ihold, dt = self.dt, periods = 10, minlength = 6*s, t_prestim = 1*s)
                    tstop = t[-1]
                    
                    if self.syn_n == 0:
                        
                        self.set_IPlay(stimulus, t)
    
                    else:
                        
                        if self.id == 0: print "- setting excitation"  
                        self.set_SynPlay(stimulus, t, n_syn=self.syn_n, g_s=self.syn_g, tau1=self.syn_tau1, tau2=self.syn_tau2, noise=self.syn_noise)
                        if self.id == 0: print "- setting excitation done" 
                        
                        if self.inh_n > 0:
                            # Inhibition
                            t, stimulus_inh, _, _ = create_singlesine(fu = fu, amp = 0, ihold = self.inh_hold, dt = self.dt, periods = 10, minlength = 6*s, t_prestim = 1*s)
                            
                            if self.id == 0: print "- setting inhibition"                    
                            self.set_InhPlay(stimulus_inh, t, n_inh=self.inh_n, g_i=self.inh_g, noise=self.inh_noise)
                            if self.id == 0: print "- setting inhibition done"    
                        
                    if self.id == 0: print "- single sine processing frequency = " + str(fu)
                    
                    self.set_Ifluct(sigma = self.amp_noise, tau = self.tau_noise) 
                    self.run(1, tstop)
        
                    t1, voltage, current, g_in = self.get() # time in ms
                    freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = self.if_get(compute_mean = 0)  # time in ms
                    is_impedance = len(freq_times) <= 2  # get impedance if less than tree spikes!!
                    
                    if i == 0:   
                        
                        # select first sinusoidal to plot
                        stimulus_plot = stimulus[i_startstop[0]:i_startstop[1]]
                        time_plot2 = t1[i_startstop[0]:i_startstop[1]]-t1[i_startstop[0]]
                        voltage_plot = voltage
                        current_plot = current
                        time_plot = t1
                        freq_times_plot = freq_times
                        spike_freq_plot = spike_freq
                        VAF_vec = None
                        
                        # construct vectors
                        if is_impedance:  # get impedance
                            amp_vec = zeros(len(freq_used)) # amplitude vector
                            mag_vec = zeros(len(freq_used))  # impedance magnitude vector
                            pha_vec = zeros(len(freq_used)) # imedance phase vector
                            fmean = zeros(len(freq_used)) # mean firing frequency
                            ca = zeros(len(freq_used), dtype=complex)
                        else:
                            amp_vec = zeros(len(freq_used)) # amplitude vector
                            fmean = zeros(len(freq_used)) # mean firing frequency
                            ca = zeros(len(freq_used), dtype=complex)
                            # create matrix to hold all different interpolation methods:
                            mag_vec = zeros((len(method_interpol),len(freq_used)))  # transfer magnitude vector
                            pha_vec = zeros((len(method_interpol),len(freq_used))) # transfer phase vector  
                            VAF_vec = zeros((len(method_interpol),len(freq_used))) # transfer phase vector 
                            
                            freq_out_signal_interp_mat = zeros((len(method_interpol),len(time_plot2)))
                                                        
                            
                    response = voltage
                    # DON'T USE THIS, USE ORIGINAL SIGNAL
                    #stimulus_out = current  
                    #stimulus_out_interp = interp(t, t1[:-1],stimulus_out[:-1]) # interpolate (downsample) to be eqivalent with input
                    
                    if is_impedance:  # get impedance
                    
                        # Get impedance function
                        response_interp = interp(t, t1[:-1], response[:-1]) # interpolate (downsample) to be eqivalent with input
            
                        output_signal = response_interp[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                        stimulus_signal = stimulus[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                        t_input_signal = t[i_startstop[0]:i_startstop[1]] - t[i_startstop[0]]
                        
                        amp_vec[i], mag_vec[i], pha_vec[i], fm, _ = get_magphase(stimulus_signal, t_input_signal, output_signal, t_input_signal, method = "fft", f = fu)
                        ca[i] = 1 / (mag_vec[i] * exp(np.pi / 180 * 1j * pha_vec[i]))
                        
                    else:  # get transfer function
                    
                        for l in range(len(method_interpol)):
                            
                            stimulus_signal = stimulus[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                            t_input_signal = t[i_startstop[0]:i_startstop[1]] - t[i_startstop[0]]
                                
                            if "none" in method_interpol[l]:
                            
                                # no interpolation
                                                        
                                f_start = find(freq_times >= t_startstop[0])[0]
                                f_stop = find(freq_times <= t_startstop[1])[-1]
                                freq_out_signal = spike_freq[f_start:f_stop] # cut out relevant signal
                                freq_times_out = freq_times[f_start:f_stop] - t[i_startstop[0]] # cut out relevant signal      
                                
                                vamp, mag_vec[l,i], pha_vec[l,i], fm, freq_out_signal_interp  = get_magphase(stimulus_signal, t_input_signal, freq_out_signal, freq_times_out, method = "fit", f = fu)
                                
                                VAF_vec[l,i] = float("nan")
                                
                                if i == 0:   
                                    freq_times_plot = freq_times_out
                                    spike_freq_plot = freq_out_signal
                                                                                   
                            
                            if "linear" in method_interpol[l]:
                                
                                # linear interpolation
                                spike_freq_interp = interp(t, freq_times, spike_freq, left=0, right=0) # interpolate to be eqivalent with input, set zero at beginning and end!
                                                        
                                freq_out_signal_interp = spike_freq_interp[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                                vamp, mag_vec[l,i], pha_vec[l,i], fmean[i], _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_interp, t_input_signal, method = "fft", f = fu)
                            
                                results = est_quality(t_input_signal, fu, freq_out_signal_interp, self.amp*mag_vec[l,i], pha_vec[l,i]/ (180 / np.pi), fmean[i])  
                                NI, VAF_vec[l,i] = results.get('NI'), results.get('VAF')
                                print "- NI: " + str(NI) + ", VAF: " + str(VAF_vec[l,i]) 
                                                            
                            
                            if "dt" in method_interpol[l]:
                                # binary spike code using dt as bin size!
                                [response, _] = neuronpy.util.spiketrain.get_histogram(freq_times, bins = t)
                                spike_freq_interp = concatenate((zeros(1),response))
                                
                                freq_out_signal_interp = spike_freq_interp[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                                vamp, mag_vec[l,i], pha_vec[l,i], fm, _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_interp, t_input_signal, method = "fft", f = fu)
                            
                                results = est_quality(t_input_signal, fu, freq_out_signal_interp, self.amp*mag_vec[l,i], pha_vec[l,i]/ (180 / np.pi), fmean[i])  
                                NI, VAF_vec[l,i] = results.get('NI'), results.get('VAF')
                                print "- NI: " + str(NI) + ", VAF: " + str(VAF_vec[l,i]) 
                                
                            
                            if "peaks" in method_interpol[l]:
                                
                                # get peaks
                                spike_freq_interp = interp(t, freq_times, spike_freq, left=0, right=0) # interpolate to be eqivalent with input, set zero at beginning and end!
                                                        
                                freq_out_signal_interp = spike_freq_interp[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                                vamp, mag_vec[l,i], pha_vec[l,i], fm, _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_interp, t_input_signal, method = "peak", f = fu)
                                
                                results = est_quality(t_input_signal, fu, freq_out_signal_interp, self.amp*mag_vec[l,i], pha_vec[l,i]/ (180 / np.pi), fmean[i])  
                                NI, VAF_vec[l,i] = results.get('NI'), results.get('VAF')
                                
                      
                            if "quadr" in method_interpol[l]:   
                                
                                # quadratic interpolation
                                sfun = InterpolatedUnivariateSpline(freq_times, spike_freq, k=2)
                                spike_freq_interp = sfun(t)
                                                        
                                freq_out_signal_interp = spike_freq_interp[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                                vamp, mag_vec[l,i], pha_vec[l,i], fm, _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_interp, t_input_signal, method = "fft", f = fu)
                                
                                results = est_quality(t_input_signal, fu, freq_out_signal_interp, self.amp*mag_vec[l,i], pha_vec[l,i]/ (180 / np.pi), fm)  
                                NI, VAF_vec[l,i] = results.get('NI'), results.get('VAF')
                                
                            
                            if "shan" in method_interpol[l]:
                                
                                # shannon interpolation
                                spike_freq_interp = shannon_interp(freq_times, spike_freq, t)
                                
                                freq_out_signal_interp = spike_freq_interp[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                                vamp, mag_vec[l,i], pha_vec[l,i], fm, _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_interp, t_input_signal, method = "fft", f = fu)
                                
                                results = est_quality(t_input_signal, fu, freq_out_signal_interp, self.amp*mag_vec[l,i], pha_vec[l,i]/ (180 / np.pi), fm)  
                                NI, VAF_vec[l,i] = results.get('NI'), results.get('VAF')
                                
                            
                            if "syn" in method_interpol[l]:
                                
                                # synaptic integration                
                                freq_out_signal_interp = gsyn[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                                
                                #figure("gsyn")
                                #plot(t, gsyn)
                                #plt.show()
                                
                                vamp, mag_vec[l,i], pha_vec[l,i], fm, _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_interp, t_input_signal, method = "fft", f = fu, deb = 1)
                                
                                results = est_quality(t_input_signal, fu, freq_out_signal_interp, self.amp*mag_vec[l,i], pha_vec[l,i]/ (180 / np.pi), fm)  
                                NI, VAF_vec[l,i] = results.get('NI'), results.get('VAF')
                            
                            if i == 0:
                                freq_out_signal_interp_mat[l,:] = freq_out_signal_interp
                                  
                #print "rest: " + str(vrest) + " freq_used:" + str(freq_used) + " amp_vec:" + str(amp_vec) + " mag_vec:" + str(mag_vec) + " pha_vec:" + str(pha_vec)
        
                if is_impedance:
                    pha_vec = unwrap(pha_vec * (np.pi / 180)) * (180 / np.pi)  # unwrap for smooth phase
                else:    
                    for l in range(len(method_interpol)):  # unwrap 
                        pha_vec[l,:] = unwrap(pha_vec[l,:] * (np.pi / 180)) * (180 / np.pi)  # unwrap for smooth phase
                    
                    # normalize synaptic integration with with first magnitude, may by syn itself! 
                    bvec = ["syn" in st for st in method_interpol]
                    if np.any(bvec):
                        k = where(bvec)   
                        mag_vec[k,:]= mag_vec[0,0]*mag_vec[k,:]/mag_vec[k,0]     
                
                # only return fraction of actual signal, it is too long!!!     
                if time_plot[-1] > self.tmax:  
                    imax = where(time_plot > self.tmax)[0][0]  # for voltage, current and time
                    time_plot = time_plot[0:imax]; voltage_plot = voltage_plot[0:imax]; current_plot = current_plot[0:imax] 
                if freq_times_plot != []:   
                    if freq_times_plot[-1] > self.tmax:
                        imax2 = where(freq_times_plot > self.tmax)[0][0]  # for spike frequency   
                        freq_times_plot = freq_times_plot[0:imax2]; spike_freq_plot = spike_freq_plot[0:imax2] 
                
                VAF_vec = (freq_used, VAF_vec)
                results = {'freq_used':freq_used, 'amp':amp_vec,'mag':mag_vec,'pha':pha_vec,'ca':ca,'stimulus':stimulus_plot,'t2':time_plot2,'voltage':voltage_plot,
                    'current':current_plot,'t1':time_plot,'freq_times':freq_times_plot,'spike_freq':spike_freq_plot,'freq_out_signal_interp_mat':freq_out_signal_interp_mat,
                    'fmean':mean(fmean),'method_interpol':method_interpol,'VAF':VAF_vec}
                
                self.pickle_prefix
                pickle.dump( results, gzip.GzipFile( filepath, "wb" ) )
                
            else:
                
                results = pickle.load( gzip.GzipFile( filepath, "rb" ) )

                if self.minimal_dir: # save only info needed for plot
                
                    print {key:np.shape(value) for key,value in results.iteritems()}
                    
                    if "Fig1_cell_if_compare" in filename:
                        results['stimulus'] = [] 
                        results['t1'] = [] 
                        results['current'] = [] 
                        results['voltage'] = []     
                    else:
                        results['stimulus'] = [] 
                        results['t1'] = [] 
                        results['t2'] = [] 
                        results['current'] = [] 
                        results['voltage'] = [] 
                        results['freq_out_signal_interp_mat'] = []    
                        
#
#                    print {key:np.shape(value) for key,value in results.iteritems()}

                    pickle.dump( results, gzip.GzipFile( self.minimal_dir + "/" + filename, "wb" ) )
                
            
            # Remove outliners only with no interpolation                
            mag, pha, method_interpol, freq_used, freq_times = results.get('mag'), results.get('pha'), results.get('method_interpol'), results.get('freq_used'), results.get('freq_times')
            
            if ("none" in method_interpol[0]) and (len(freq_times) > 3 and len(freq_used) > 3):
                if (freq_used[-1]-freq_used[-1-1]<10):
        
                    lw = 20  # window length
                    md = 10  # margin
                    
                    outliner_vec = array([])
                    mdiff_vec = array([])
                    kdiff_vec = array([])
                    m = array(mag[0,:])
                    p = array(pha[0,:])
                                
                    for k in range(lw, len(m)):
                        #mm = mean(m_[k:k+lw-1])
                        mdiff = mean(abs(diff(m[k-lw:k])))
                        kdiff = abs(m[k]-m[k-1])
                        mdiff_vec = append(mdiff_vec, mdiff)   
                        kdiff_vec = append(kdiff_vec, kdiff)  
                        
                        if kdiff > md*mdiff:
                            m[k]=m[k-1]
                            p[k]=p[k-1]
                            outliner_vec = append(outliner_vec, freq_used[k]) 
                                          
                    print "Ssine fit outliner freq:" + str(outliner_vec)
                    #plt.plot(freq_used, mag[0,:], 'k', freq_used, m, 'r')
                    #plt.show()
         
                    results['mag'][0,:] = m
                    results['pha'][0,:] = p 
                    
                   
                    
            
            # remove frequnecies given 
            mag, pha, freq_used, VAF  = results.get('mag'), results.get('pha'), results.get('freq_used'), results.get('VAF')
            if len(self.del_freq) >= 1:
                for im in range(len(method_interpol)):
                    fi = zeros(len(self.del_freq))
                    for fr in range(len(self.del_freq)): 
                        fi[fr] = find(self.del_freq[fr] == freq_used)
                        #mag[im,fi] = mean(mag[im,fi-1:fi+1])
                        #pha[im,fi] = mean(pha[im,fi-1:fi+1])
                    
                results['mag'] = delete(mag, fi, 1)
                results['pha'] = delete(pha, fi, 1)
                results['freq_used'] = delete(freq_used, fi)
                results['VAF'] = (results['freq_used'], delete(VAF[1], fi, 1))            
                
                #plt.plot(results['freq_used'], results['mag'][0,:], 'k')
                #plt.show()
                
            return results
        
        
    def fun_ssine_Stim_pp(self, freq_used = array([1, 10, 100, 1000])*Hz, method_interpol = np.array(["none", "linear", "quadratic", "syn"])):
        """
        Compute impedance and/or transfer function using Single sine stimulation
        Only compute transfer function if there is a steady state (resting) firing rate!
        """
        
        self.run_pp = True
        filename = str(self.pickle_prefix) + "_results_sim_ssine.p"
        filepath = self.data_dir + "/" + filename
        
        if self.do_run or (os.path.isfile(filepath) is False):
            
            from scipy.interpolate import InterpolatedUnivariateSpline 
            import neuronpy.util.spiketrain
            
            for i, m in enumerate(method_interpol):
                    if "syn" in m: 
                        method_interpol[i] = "syn " + str(self.synout_tau1/ms) + "/" + str(self.synout_tau2/ms) + "ms"
                   
            fs = 1 / self.dt # sampling rate 
            fmax = fs / 2 # maximum frequency (nyquist)
            
            if self.syn_n == 0:
                
                ihold, self.amp, self.amp_noise = self.set_i() 
                if self.id == 0: print "- (CURRENT INPUT) starting single sine impedance/transfer function estimation! with amp = " + str(self.amp) + ", ihold = " + str(ihold) + ", dt = " + str(self.dt) + " => maximum frequency = " + str(fmax) + "\r"    

            else:
                
                ihold = self.ihold
                self.amp, self.amp_noise = self.set_amp(ihold, 0)  
                if self.id == 0: print "- (SYNAPTIC INPUT) starting single sine impedance/transfer function estimation! with amp = " + str(self.amp) + ", ihold = " + str(ihold) + ", dt = " + str(self.dt) + " => maximum frequency = " + str(fmax) + "\r"    

            freq_used_local = [] 
        
            for i in range(int(self.id), len(freq_used), int(self.nhost)): # loop over all freq_used
                freq_used_local.append(freq_used[i]) 
                
            if self.id == 0: print "nodeid: ", self.id, ", freq: ", freq_used_local  # check freq_used of node

            to_run = True
            if len(freq_used_local) == 0:
                freq_used_local.append(-1)
                to_run = False    
                 
            # construct vectors
            amp_vec = zeros(len(freq_used_local)) # amplitude vector
            fmean = zeros(len(freq_used_local)) # mean firing frequency
            ca = zeros(len(freq_used_local), dtype=complex)
            # create matrix to hold all different interpolation methods:
            mag_vec = zeros((len(method_interpol),len(freq_used_local)))  # transfer magnitude vector
            pha_vec = zeros((len(method_interpol),len(freq_used_local))) # transfer phase vector  
            VAF_vec = zeros((len(method_interpol),len(freq_used_local))) # transfer phase vector
                        
            if to_run:
                for i, fu in enumerate(freq_used_local):
                    
                    t, stimulus, i_startstop, t_startstop = create_singlesine(fu = fu, amp = self.amp, ihold = ihold, dt = self.dt, periods = 10, minlength = 6*s, t_prestim = 1*s)
                    tstop = t[-1]
                    
                    if self.syn_n == 0:
                        
                        self.set_IPlay(stimulus, t)
    
                    else:
                        
                        if self.id == 0: print "nodeid: ", self.id, "setting excitation"  
                        self.set_SynPlay(stimulus, t, n_syn=self.syn_n, g_s=self.syn_g, tau1=self.syn_tau1, tau2=self.syn_tau2, noise=self.syn_noise)
                                            
                        if self.inh_n > 0:
                            # Inhibition
                            t, stimulus_inh, _, _ = create_singlesine(fu = fu, amp = 0, ihold = self.inh_hold, dt = self.dt, periods = 10, minlength = 6*s, t_prestim = 1*s)
                            
                            if self.id == 0: print "nodeid: ", self.id, "setting inhibition"                    
                            self.set_InhPlay(stimulus_inh, t, n_inh=self.inh_n, g_i=self.inh_g, noise=self.inh_noise)
                              
                        
                    if self.id == 0: print "nodeid: ", self.id, "single sine processing frequency = " + str(fu)
                    
                    self.set_Ifluct(sigma = self.amp_noise, tau = self.tau_noise) 
                    self.run(1, tstop)
        
                    t1, voltage, current, g_in = self.get() # time in ms
                    freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = self.if_get(compute_mean = 0)  # time in ms
                    is_impedance = len(freq_times) <= 2  # get impedance if less than tree spikes!!
                    
                    if i == 0:   
                        
                        # select first sinusoidal to plot
                        stimulus_plot = stimulus[i_startstop[0]:i_startstop[1]]
                        time_plot2 = t1[i_startstop[0]:i_startstop[1]]-t1[i_startstop[0]]
                        voltage_plot = voltage
                        current_plot = current
                        time_plot = t1
                        freq_times_plot = freq_times
                        spike_freq_plot = spike_freq
                        freq_out_signal_interp_mat = zeros((len(method_interpol),len(time_plot2)))
                                                        
                            
                    response = voltage
                    # DON'T USE THIS, USE ORIGINAL SIGNAL
                    #stimulus_out = current  
                    #stimulus_out_interp = interp(t, t1[:-1],stimulus_out[:-1]) # interpolate (downsample) to be eqivalent with input
                    
                    if is_impedance:  # get impedance
                    
                        # Get impedance function
                        response_interp = interp(t, t1[:-1], response[:-1]) # interpolate (downsample) to be eqivalent with input
            
                        output_signal = response_interp[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                        stimulus_signal = stimulus[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                        t_input_signal = t[i_startstop[0]:i_startstop[1]] - t[i_startstop[0]]
                        
                        amp_vec[i], mag_vec[0,i], pha_vec[0,i], fm, _ = get_magphase(stimulus_signal, t_input_signal, output_signal, t_input_signal, method = "fft", f = fu)
                        ca[i] = 1 / (mag_vec[0,i] * exp(np.pi / 180 * 1j * pha_vec[0,i]))
                        
                    else:  # get transfer function
                    
                        for l in range(len(method_interpol)):
                            
                            stimulus_signal = stimulus[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                            t_input_signal = t[i_startstop[0]:i_startstop[1]] - t[i_startstop[0]]
                                
                            if "none" in method_interpol[l]:
                            
                                # no interpolation
                                                        
                                f_start = find(freq_times >= t_startstop[0])[0]
                                f_stop = find(freq_times <= t_startstop[1])[-1]
                                freq_out_signal = spike_freq[f_start:f_stop] # cut out relevant signal
                                freq_times_out = freq_times[f_start:f_stop] - t[i_startstop[0]] # cut out relevant signal      
                                
                                vamp, mag_vec[l,i], pha_vec[l,i], fm, freq_out_signal_interp  = get_magphase(stimulus_signal, t_input_signal, freq_out_signal, freq_times_out, method = "fit", f = fu)
                                
                                VAF_vec[l,i] = float("nan")
                                
                                if i == 0:   
                                    freq_times_plot = freq_times_out
                                    spike_freq_plot = freq_out_signal
                                                                                   
                            
                            if "linear" in method_interpol[l]:
                                
                                # linear interpolation
                                spike_freq_interp = interp(t, freq_times, spike_freq, left=0, right=0) # interpolate to be eqivalent with input, set zero at beginning and end!
                                                        
                                freq_out_signal_interp = spike_freq_interp[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                                vamp, mag_vec[l,i], pha_vec[l,i], fmean[i], _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_interp, t_input_signal, method = "fft", f = fu)
                            
                                results = est_quality(t_input_signal, fu, freq_out_signal_interp, self.amp*mag_vec[l,i], pha_vec[l,i]/ (180 / np.pi), fmean[i])  
                                NI, VAF_vec[l,i] = results.get('NI'), results.get('VAF')
                                print "- NI: " + str(NI) + ", VAF: " + str(VAF_vec[l,i]) 
                                                            
                            
                            if "dt" in method_interpol[l]:
                                # binary spike code using dt as bin size!
                                [response, _] = neuronpy.util.spiketrain.get_histogram(freq_times, bins = t)
                                spike_freq_interp = concatenate((zeros(1),response))
                                
                                freq_out_signal_interp = spike_freq_interp[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                                vamp, mag_vec[l,i], pha_vec[l,i], fm, _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_interp, t_input_signal, method = "fft", f = fu)
                            
                                results = est_quality(t_input_signal, fu, freq_out_signal_interp, self.amp*mag_vec[l,i], pha_vec[l,i]/ (180 / np.pi), fmean[i])  
                                NI, VAF_vec[l,i] = results.get('NI'), results.get('VAF')
                                print "- NI: " + str(NI) + ", VAF: " + str(VAF_vec[l,i]) 
                                
                            
                            if "peaks" in method_interpol[l]:
                                
                                # get peaks
                                spike_freq_interp = interp(t, freq_times, spike_freq, left=0, right=0) # interpolate to be eqivalent with input, set zero at beginning and end!
                                                        
                                freq_out_signal_interp = spike_freq_interp[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                                vamp, mag_vec[l,i], pha_vec[l,i], fm, _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_interp, t_input_signal, method = "peak", f = fu)
                                
                                results = est_quality(t_input_signal, fu, freq_out_signal_interp, self.amp*mag_vec[l,i], pha_vec[l,i]/ (180 / np.pi), fmean[i])  
                                NI, VAF_vec[l,i] = results.get('NI'), results.get('VAF')
                                
                      
                            if "quadr" in method_interpol[l]:   
                                
                                # quadratic interpolation
                                sfun = InterpolatedUnivariateSpline(freq_times, spike_freq, k=2)
                                spike_freq_interp = sfun(t)
                                                        
                                freq_out_signal_interp = spike_freq_interp[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                                vamp, mag_vec[l,i], pha_vec[l,i], fm, _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_interp, t_input_signal, method = "fft", f = fu)
                                
                                results = est_quality(t_input_signal, fu, freq_out_signal_interp, self.amp*mag_vec[l,i], pha_vec[l,i]/ (180 / np.pi), fm)  
                                NI, VAF_vec[l,i] = results.get('NI'), results.get('VAF')
                                
                            
                            if "shan" in method_interpol[l]:
                                
                                # shannon interpolation
                                spike_freq_interp = shannon_interp(freq_times, spike_freq, t)
                                
                                freq_out_signal_interp = spike_freq_interp[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                                vamp, mag_vec[l,i], pha_vec[l,i], fm, _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_interp, t_input_signal, method = "fft", f = fu)
                                
                                results = est_quality(t_input_signal, fu, freq_out_signal_interp, self.amp*mag_vec[l,i], pha_vec[l,i]/ (180 / np.pi), fm)  
                                NI, VAF_vec[l,i] = results.get('NI'), results.get('VAF')
                                
                            
                            if "syn" in method_interpol[l]:
                                
                                # synaptic integration                
                                freq_out_signal_interp = gsyn[i_startstop[0]:i_startstop[1]] # cut out relevant signal
                                
                                vamp, mag_vec[l,i], pha_vec[l,i], fm, _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_interp, t_input_signal, method = "fft", f = fu, deb = 1)
                                
                                results = est_quality(t_input_signal, fu, freq_out_signal_interp, self.amp*mag_vec[l,i], pha_vec[l,i]/ (180 / np.pi), fm)  
                                NI, VAF_vec[l,i] = results.get('NI'), results.get('VAF')
                            
                            if i == 0:
                                freq_out_signal_interp_mat[l,:] = freq_out_signal_interp
            
            vlen = []
            vlen = self.comm.allgather(len(freq_used_local), vlen) # get length first, then use proper Gather! 
            
            freq_used_all = -1 * ones( (self.comm.size, max(vlen)), dtype='d')
            amp_vec_all = zeros( (self.comm.size, max(vlen)), dtype='d') 
            fmean_all = zeros( (self.comm.size, max(vlen)), dtype='d')
            ca_all = zeros( (self.comm.size, max(vlen)), dtype=complex)
            mag_vec_all = zeros( (self.comm.size, len(method_interpol)*max(vlen)), dtype='d')
            pha_vec_all = zeros( (self.comm.size, len(method_interpol)*max(vlen)), dtype='d')
            VAF_vec_all = zeros( (self.comm.size, len(method_interpol)*max(vlen)), dtype='d')
            
            freq_used_local = np.array(freq_used_local, dtype='d')
            
            fillto = max(vlen)-len(freq_used_local)
            filler = np.zeros((len(method_interpol), fillto), dtype='d')
            mag_vec = np.concatenate(np.hstack((mag_vec, filler)))

            self.comm.Gatherv(sendbuf=[freq_used_local, MPI.DOUBLE], recvbuf=[freq_used_all, MPI.DOUBLE], root=0)
            self.comm.Gatherv(sendbuf=[amp_vec, MPI.DOUBLE], recvbuf=[amp_vec_all, MPI.DOUBLE], root=0)
            self.comm.Gatherv(sendbuf=[fmean, MPI.DOUBLE], recvbuf=[fmean_all, MPI.DOUBLE], root=0)
            
            self.comm.Gatherv(sendbuf=[ca, MPI.COMPLEX], recvbuf=[ca_all, MPI.COMPLEX], root=0)
            self.comm.Gatherv(sendbuf=[mag_vec, MPI.DOUBLE], recvbuf=[mag_vec_all, MPI.DOUBLE], root=0)
            self.comm.Gatherv(sendbuf=[pha_vec, MPI.DOUBLE], recvbuf=[pha_vec_all, MPI.DOUBLE], root=0)
            self.comm.Gatherv(sendbuf=[VAF_vec, MPI.DOUBLE], recvbuf=[VAF_vec_all, MPI.DOUBLE], root=0)
            
            
            if self.id == 0:
                
                mag_vec_all = mag_vec_all.reshape((self.comm.size,len(method_interpol),max(vlen)))
                pha_vec_all = pha_vec_all.reshape((self.comm.size,len(method_interpol),max(vlen)))
                VAF_vec_all = VAF_vec_all.reshape((self.comm.size,len(method_interpol),max(vlen)))

                freq_used_all = np.concatenate(freq_used_all)
                ca_all = np.concatenate(ca_all)
                amp_vec_all = np.concatenate(amp_vec_all)
                fmean_all = np.concatenate(fmean_all)
                
                i1 = argsort(freq_used_all) 
                i2 = i1[where(freq_used_all[i1]>0)]
                
                freq_used = freq_used_all[i2]
                ca = ca_all[i2]
                amp_vec = amp_vec_all[i2]
                fmean = fmean_all[i2]
                
                mag_vec = zeros((len(method_interpol),len(freq_used)))  # transfer magnitude vector
                pha_vec = zeros((len(method_interpol),len(freq_used))) # transfer phase vector  
                VAF_vec = zeros((len(method_interpol),len(freq_used))) # transfer phase vector 
                
                
                for il in range(len(method_interpol)):
                    
                    mag_vec0 = np.concatenate(mag_vec_all[:,il,:]) 
                    mag_vec[il,:] = mag_vec0[i2]
                    
                    pha_vec0 = np.concatenate(pha_vec_all[:,il,:]) 
                    pha_vec[il,:] = pha_vec0[i2]
                    
                    VAF_vec0 = np.concatenate(VAF_vec_all[:,il,:]) 
                    VAF_vec[il,:] = VAF_vec0[i2]
                    
 
                #print freq_used
                #print mag_vec  
             
            self.comm.Barrier()
    
            if self.id == 0: 
    
                if is_impedance:
                    pha_vec = unwrap(pha_vec * (np.pi / 180)) * (180 / np.pi)  # unwrap for smooth phase
                else:    
                    for l in range(len(method_interpol)):  # unwrap 
                        pha_vec[l,:] = unwrap(pha_vec[l,:] * (np.pi / 180)) * (180 / np.pi)  # unwrap for smooth phase
                    
                    # normalize synaptic integration with with first magnitude, may by syn itself! 
                    bvec = ["syn" in st for st in method_interpol]
                    if np.any(bvec):
                        k = where(bvec)   
                        mag_vec[k,:]= mag_vec[0,0]*mag_vec[k,:]/mag_vec[k,0]     
                
                # only return fraction of actual signal, it is too long!!!     
                if time_plot[-1] > self.tmax:  
                    imax = where(time_plot > self.tmax)[0][0]  # for voltage, current and time
                    time_plot = time_plot[0:imax]; voltage_plot = voltage_plot[0:imax]; current_plot = current_plot[0:imax] 
                if freq_times_plot != []:   
                    if freq_times_plot[-1] > self.tmax:
                        imax2 = where(freq_times_plot > self.tmax)[0][0]  # for spike frequency   
                        freq_times_plot = freq_times_plot[0:imax2]; spike_freq_plot = spike_freq_plot[0:imax2] 
                
                VAF_vec = (freq_used, VAF_vec)
                results = {'freq_used':freq_used, 'amp':amp_vec,'mag':mag_vec,'pha':pha_vec,'ca':ca,'stimulus':stimulus_plot,'t2':time_plot2,'voltage':voltage_plot,
                    'current':current_plot,'t1':time_plot,'freq_times':freq_times_plot,'spike_freq':spike_freq_plot,'freq_out_signal_interp_mat':freq_out_signal_interp_mat,
                    'fmean':mean(fmean),'method_interpol':method_interpol,'VAF':VAF_vec}
                
                self.pickle_prefix
                pickle.dump( results, gzip.GzipFile( filepath, "wb" ) )
            
        else:
            
            if self.id == 0:
                results = pickle.load( gzip.GzipFile( filepath, "rb" ) )
                
                if self.minimal_dir: # save only info needed for plot
                
                    print {key:np.shape(value) for key,value in results.iteritems()}
                    
                    results['stimulus'] = [] 
                    results['t1'] = [] 
                    results['t2'] = [] 
                    results['current'] = [] 
                    results['voltage'] = [] 
                    results['freq_out_signal_interp_mat'] = []    
#
#                    print {key:np.shape(value) for key,value in results.iteritems()}

                    pickle.dump( results, gzip.GzipFile( self.minimal_dir + "/" + filename, "wb" ) )
        
        if self.id == 0:
            # Remove outliners only with no interpolation                
            mag, method_interpol, freq_used, freq_times = results.get('mag'), results.get('method_interpol'), results.get('freq_used'), results.get('freq_times')
            
            if ("none" in method_interpol[0]) and (len(freq_times) > 3 and len(freq_used) > 3):
                if (freq_used[-1]-freq_used[-1-1]<10):
        
                    outliner_vec = array([])
                    mdiff_vec = array([])
                    kdiff_vec = array([])
                    m = array(mag[0,:])
                    lw = 20  # window length
                    md = 10  # margin
                                
                    for k in range(lw, len(m)):
                        #mm = mean(m_[k:k+lw-1])
                        mdiff = mean(abs(diff(m[k-lw:k])))
                        kdiff = abs(m[k]-m[k-1])
                        mdiff_vec = append(mdiff_vec, mdiff)   
                        kdiff_vec = append(kdiff_vec, kdiff)  
                        
                        if kdiff > md*mdiff:
                            m[k]=m[k-1]
                            outliner_vec = append(outliner_vec, freq_used[k]) 
                                          
                    print "Ssine fit outliner freq:" + str(outliner_vec)
                    #plt.plot(freq_used, mag[0,:], 'k', freq_used, m, 'r')
                    #plt.show()
         
                    results['mag'][0,:] = m    
            
            # remove frequencies given 
            mag, pha, freq_used, VAF  = results.get('mag'), results.get('pha'), results.get('freq_used'), results.get('VAF')
            if len(self.del_freq) >= 1:
                for im in range(len(method_interpol)):
                    fi = zeros(len(self.del_freq))
                    for fr in range(len(self.del_freq)):
                        fi[fr] = find(self.del_freq[fr] == freq_used)
                        #mag[im,fi] = mean(mag[im,fi-1:fi+1])
                        #pha[im,fi] = mean(pha[im,fi-1:fi+1])
                    
                results['mag'] = delete(mag, fi, 1)
                results['pha'] = delete(pha, fi, 1)
                results['freq_used'] = delete(freq_used, fi)
                results['VAF'] = (results['freq_used'], delete(VAF[1], fi, 1))            
                
                #plt.plot(results['freq_used'], results['mag'][0,:], 'k')
                #plt.show()
        
        else:
            results = {'freq_used':[], 'amp':[],'mag':[],'pha':[],'ca':[],'stimulus':[],'t2':[],'voltage':[],
                    'current':[],'t1':[],'freq_times':[],'spike_freq':[],'freq_out_signal_interp_mat':[],
                    'fmean':[],'method_interpol':[],'VAF':[]}    
                    
        return results


    def fun_msine_Stim(self, t_stim = 10*s, freq_used = None, do_csd = 0, method_interpol = np.array(["linear", "quadratic", "syn"])):
        """
        Stimulate cell with multisine noise having frequencies as given in freq_used
        do_csd = 1: use cross spectral density function for computation       
        """
        
        if self.id == 0: 
            
            if self.do_run:
                
                if freq_used == None:
                    fstart = 0.1*Hz; fstop = 1000*Hz; fsteps = 0.1*Hz # Hz
                    freq_used = arange(fstart,fstop,fsteps) # frequencies we want to examine
                        
                tstart = 0; 
                fs = 1 / self.dt # sampling rate 
                fmax = fs / 2 # maximum frequency (nyquist)
                
                ihold, self.amp, self.amp_noise = self.set_i()  
                print "- starting multi sine impedance/transfer function estimation! with amp = " + str(self.amp) + ", ihold = " + str(ihold) + ", t_stim = " + str(t_stim) + ", dt = " + str(self.dt) + " => maximum frequency = " + str(fmax) + "\r"    
        
                t_noise = arange(tstart, t_stim, self.dt) # create stimulus time vector, make sure stimulus is even!!!
                
                noise_data, freq, freq_wp, f_used_check = create_multisines(t_noise, freq_used)  # create multi sine signal
                noise_data_points = len(noise_data)  
                
                if np.any(abs(freq_used - f_used_check) > 0.00000001):  # bloody floating point precision       
                    raise ValueError('Requested and computed multisine frequencies are not the same')
         
                stimulus, t = construct_Stimulus(noise_data, fs, self.amp, ihold)
                tstop = t[-1]
                
                self.set_IPlay(stimulus, t)
                self.run(1, tstop)
                
                t1, voltage, current, g_in = self.get()
                freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = self.if_get(compute_mean = 0)  
                is_impedance = len(freq_times) <= 3  # get impedance if less than tree spikes!!
                
                if is_impedance:  # Get impedance function
                    mag, pha, ca, freq, freq_used = compute_Impedance(voltage, current, t1, stimulus, t, noise_data_points, freq_wp, do_csd = do_csd)
                    method_interpol = []; fmean = 0
                    
                else:  # Get transfer function
                    
                    for i, m in enumerate(method_interpol):
                        if "syn" in m: 
                            method_interpol[i] = "syn " + str(self.synout_tau1/ms) + "/" + str(self.synout_tau2/ms) + "ms"
                    
                    results = compute_Transfer(spike_freq = spike_freq, freq_times = freq_times, 
                        stimulus = stimulus, t = t, noise_data_points = noise_data_points, freq_wp = freq_wp, gsyn = gsyn, do_csd = do_csd,
                        method_interpol = method_interpol, w_length = 1)
                    mag, pha, ca, freq, freq_used, fmean = results.get('mag_mat'), results.get('pha_mat'), results.get('ca_mat'), results.get('freq'), results.get('freq_used')          
                  
                
                # only return fraction of actual signal, it is too long if not!!!             
                if t1[-1] > self.tmax:      
                    imax = where(t1 > self.tmax)[0][0]  # for voltage, current and time
                    t1 = t1[0:imax]; voltage = voltage[0:imax]; current = current[0:imax] 
                if freq_times != []:       
                    if freq_times[-1] > self.tmax:
                        imax2 = where(freq_times > self.tmax)[0][0]  # for spike frequency               
                        freq_times = freq_times[0:imax2]; spike_freq = spike_freq[0:imax2]
                        
                results = {'freq_used':freq_used,'mag':mag,'pha':pha,'ca':ca,'voltage':voltage,
                    'current':current,'t1':t1,'freq_times':freq_times,'spike_freq':spike_freq,
                    'fmean':fmean,'method_interpol':method_interpol}
    
                pickle.dump( results, gzip.GzipFile( self.data_dir + "/" + str(self.pickle_prefix) + "_results_sim_msine.p", "wb" ) )
                
            else:
                
                results = pickle.load( gzip.GzipFile( self.data_dir + "/" + str(self.pickle_prefix) + "_results_sim_msine.p", "rb" ) )
                
            return results
            
     
    def fun_cnoise_Stim(self, t_stim = 10, sexp = 4, cutf = None, do_csd = 1, method_interpol = np.array(["linear", "quadratic", "syn"])):
        """
        Stimulate cell with colored noise
        sexp = spectral exponent: Power ~ 1/freq^sexp
        cutf = frequency cutoff: Power flat (white) for freq <~ cutf  
        do_csd = 1: use cross spectral density function for computation
        """
        
        if self.id == 0: 
        
            if self.do_run:
                
                tstart = 0; 
                fs = 1 / self.dt # sampling rate 
                fmax = fs / 2 # maximum frequency (nyquist)
                
                t_noise = arange(tstart, t_stim, self.dt) # create stimulus time vector, make sure stimulus is even!!!
                noise_data = create_colnoise(t_noise, sexp, cutf, seed = self.seed)        
                noise_data_points = len(noise_data)      
    
                if self.syn_n == 0:
                    
                    ihold, self.amp, self.amp_noise = self.set_i() 
                    print "- (CURRENT INPUT) starting colored noise impedance/transfer function estimation! with amp = " + str(self.amp) + ", ihold = " + str(ihold) + ", t_stim = " + str(t_stim) + ", dt = " + str(self.dt) + " => maximum frequency = " + str(fmax) + "\r"    
                                    
                    stimulus, t, t_startstop = construct_Stimulus(noise_data, fs, self.amp, ihold)
                    tstop = t[-1]
                
                    self.set_IPlay(stimulus, t)
    
                else:
                    
                    self.give_freq = False # do not convert to current!
                    ihold = self.ihold
                    self.amp, self.amp_noise = self.set_amp(ihold, 0)  
                    
                    print "- (SYNAPTIC INPUT) starting colored noise impedance/transfer function estimation! with amp = " + str(self.amp) + ", ihold = " + str(ihold) + ", t_stim = " + str(t_stim) + ", dt = " + str(self.dt) + " => maximum frequency = " + str(fmax) + "\r"    
                    
                    stimulus, t, t_startstop = construct_Stimulus(noise_data, fs, self.amp, ihold)
                    tstop = t[-1]
                    
                    print "- setting excitation"      
                    self.set_SynPlay(stimulus, t, n_syn=self.syn_n, g_s=self.syn_g, tau1=self.syn_tau1, tau2=self.syn_tau2, noise=self.syn_noise)
                    print "- setting excitation done"      
                    
                    if self.inh_n > 0:
                        # Inhibition
                        stimulus_inh, t, _ = construct_Stimulus(ones(len(t_noise)), fs, 0, self.inh_hold)
                        
                        #plt.figure(32534)
                        #plt.plot(t, stimulus_inh)
                        #plt.show()
                        
                        print "- setting inhibition"                    
                        self.set_InhPlay(stimulus_inh, t, n_inh=self.inh_n, g_i=self.inh_g, noise=self.inh_noise)
                        print "- setting inhibition done"      
                        
                self.run(1, tstop)
                
                t1, voltage, current, g_in = self.get()
                freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = self.if_get(compute_mean = 0)  
                is_impedance = len(freq_times) <= 3  # get impedance if less than tree spikes!!
                
                #figure(1000)
                #plot(t1, voltage, 'k')  
                #show()            
                
                if is_impedance:  # Get impedance function
                    mag, pha, ca, freq, freq_used = compute_Impedance(voltage, current, t1, stimulus, t, noise_data_points, do_csd = do_csd)  # freq_wp not defined, use all frequencies
                    method_interpol = []; fmean = 0; SNR_mat = []; VAFf_mat = []
                    
                else:  # Get transfer function
                
                    for i, m in enumerate(method_interpol):
                        if "syn" in m: 
                            method_interpol[i] = "syn " + str(self.synout_tau1/ms) + "/" + str(self.synout_tau2/ms) + "ms"
    
                    results = compute_Transfer(spike_freq = spike_freq, freq_times = freq_times, stimulus = stimulus, t = t, 
                        noise_data_points = noise_data_points, gsyn = gsyn, do_csd = do_csd, method_interpol = method_interpol, w_length = 1*s, t_startstop = t_startstop)
                        
                    mag, pha, ca, freq, freq_used, fmean = results.get('mag_mat'), results.get('pha_mat'), results.get('ca_mat'), results.get('freq'), results.get('freq_used'), results.get('fmean')          
                    SNR_mat, VAFf_mat = results.get('SNR_mat'), results.get('VAFf_mat')
                    
                # only return fraction of actual signal, it is too long if not!!!     
                if t1[-1] > self.tmax:      
                    imax = where(t1 > self.tmax)[0][0]  # for voltage, current and time
                    t1 = t1[0:imax]; voltage = voltage[0:imax]; current = current[0:imax] 
                if freq_times != []:       
                    if freq_times[-1] > self.tmax:
                        imax2 = where(freq_times > self.tmax)[0][0]  # for spike frequency               
                        freq_times = freq_times[0:imax2]; spike_freq = spike_freq[0:imax2]
                        
                results = {'freq_used':freq_used,'mag':mag,'pha':pha,'ca':ca,'voltage':voltage,
                    'current':current,'t1':t1,'freq_times':freq_times,'spike_freq':spike_freq,
                    'fmean':fmean,'method_interpol':method_interpol, 'SNR':SNR_mat, 'VAF':VAFf_mat}
             
                pickle.dump( results, gzip.GzipFile( self.data_dir + "/" + str(self.pickle_prefix) + "_results_sim_cnoise.p", "wb" ) )
                
            else:
                
                results = pickle.load( gzip.GzipFile( self.data_dir + "/" + str(self.pickle_prefix) + "_results_sim_cnoise.p", "rb" ) )
                
                
            return results


    def stim_reconstruct(self, t_stim = 10, sexp = 4, cutf = 10, w_length = 1*s, t_kernel = 0, t_qual = 10*s, method_interpol = array(["linear", "quadratic", "syn"]), tau = 0*ms):
        """
        
        """
        
        tstart = 0; 
        fs = 1 / self.dt # sampling rate 
        fmax = fs / 2 # maximum frequency (nyquist)
        
        ihold, self.amp, self.amp_noise = self.set_i()
        print "- starting stim reconstruction test! with amp = " + str(self.amp) + ", ihold = " + str(ihold) + ", t_stim = " + str(t_stim) + ", dt = " + str(self.dt) + " => maximum frequency = " + str(fmax) + "\r"    

        t_noise = arange(tstart, t_stim, self.dt) # create stimulus time vector, make sure stimulus is even!!!
        noise_data = create_colnoise(t_noise, sexp, cutf, seed = self.seed)   
        
        noise_data_points = len(noise_data)        
        
        stimulus, t = construct_Stimulus(noise_data, fs, self.amp, ihold)
        tstop = t[-1]

        self.set_IPlay(stimulus, t)
        self.run(1, tstop)
        
        t1, voltage, current, g_in = self.get()
        freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = self.if_get(compute_mean = 0)  
        
        results = compute_Transfer(spike_freq = spike_freq, freq_times = freq_times, 
            stimulus = stimulus, t = t, noise_data_points = noise_data_points, gsyn = gsyn, do_csd = 1,
            method_interpol = method_interpol, w_length = w_length, t_kernel = t_kernel, t_qual = t_qual)  # freq_wp not defined, use all frequencies

        freq_used, tc, stim_re_mat, stim, resp_mat, noise_mat, CF_mat, VAF_mat, VAFf_mat, SNR_mat = results.get('freq_used'), results.get('tc'), results.get('stim_re_mat'), results.get('stim'), results.get('resp_mat'), results.get('noise_mat'), results.get('CF_mat'), results.get('VAF_mat'), results.get('VAFf_mat'), results.get('SNR_mat')
        tk, K_mat, mag_mat, P_ss = results.get('tk'), results.get('K_mat'), results.get('mag_mat'), results.get('P_ss')
        
        figure(1)
        for l in range(len(method_interpol)):
            semilogx(freq_used, 10*log10(real(SNR_mat[1][l,:]))) 
            
        figure(3)
        for l in range(len(method_interpol)):
            semilogx(freq_used, real(VAFf_mat[1][l,:])) 
            semilogx(freq_used, -1/real(SNR_mat[1][l,:]) + 1, 'r')

        figure(44)
        semilogx(freq_used, P_ss, 'k')  
        
        figure(22)
        semilogx(freq_used, mag_mat[0,:])
                
        #figure(9)        
        #SNRreal = real(SNR_mat[1][0,:])
        #semilogx(freq_used, (SNRreal - 1) / (max(SNRreal) - 1))      
        
        other_est = 0
        
        for l in range(len(method_interpol)):
            figure(2)              
            subplot(len(method_interpol),1,l+1)
            
            if other_est: # USED DIFFERENT SIGNAL NOT WORKING!!! WHY???
                
                sexp = 4 
                cutf = 10
                t_stim = 15
                t_noise = arange(tstart, t_stim, self.dt) # create stimulus time vector, make sure stimulus is even!!!
                noise_data = create_colnoise(t_noise, sexp, cutf)   
                noise_data_points = len(noise_data)        
                stimulus, t = construct_Stimulus(noise_data, fs, self.amp, ihold)
                tstop = t[-1]
        
                self.set_IPlay(stimulus, t)
                self.run(1, tstop)
                
                t1, voltage, current, g_in = self.get()
                freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = self.if_get(compute_mean = 0)  
                
                results = compute_Transfer(spike_freq = spike_freq, freq_times = freq_times, 
                    stimulus = stimulus, t = t, noise_data_points = noise_data_points, gsyn = gsyn, do_csd = 1,
                    method_interpol = method_interpol, w_length = w_length, t_kernel = t_kernel, t_qual = t_qual)  # freq_wp not defined, use all frequencies
        
                freq_used, tc, stim_re_mat, stim, resp_mat, noise_mat, CF_mat, SNR_mat = results.get('freq_used'), results.get('tc'), results.get('stim_re_mat'), results.get('stim'), results.get('resp_mat'), results.get('noise_mat'), results.get('CF_mat'), results.get('SNR_mat')
                # K_mat still form previous estimation
                
                tc, resp_cc, stim_cc, stim_re_cc, noise_cc, CF, VAF = reconstruct_Stimulus(K_mat[l,:], resp_mat[l,:], stim, t1)
                
                tk, K_mat = results.get('tk'), results.get('K_mat')
            
            else:
                
                resp_cc = resp_mat[l,:]
                stim_cc = stim
                stim_re_cc =  stim_re_mat[l,:] 
                noise_cc =  noise_mat[l,:]  
                CF = CF_mat[l]  
                VAF = VAF_mat[l] 
                K = K_mat[l,:]
            
            plt.plot(tc, stim_cc, 'k', label="stim")
            plt.plot(tc, stim_re_cc, 'b', label="stim_re, interp: " + str(method_interpol[l]) + ", CF=" + str(CF) + ", VAF=" + str(VAF)) 
            plt.plot(tc, noise_cc, 'r', label="noise")
            plt.axis(xmin=0, xmax=1) 
            lg = plt.legend()
            
            #plt.figure(33)
            #plt.plot(tc, resp_cc)
            #plt.axis(xmin=0, xmax=1) 
            
            plt.figure(10)
            plt.plot(tk, K)            
            
            #TEST: Estimate SNR from noise!        
            
            nfft = 2 ** int(ceil(log2(w_length * fs))) 
            
            nmean = mean(noise_cc) 
            smean = mean(stim_cc)
            
            figure(11) 
            P_nn, freq = csd(noise_cc-nmean, noise_cc-nmean, Fs = fs, NFFT = nfft, noverlap = nfft / 4, sides = 'twosided')
            P_ss, freq = csd(stim_cc-smean, stim_cc-smean, Fs = fs, NFFT = nfft, noverlap = nfft / 4, sides = 'twosided')
            SNR = ifftshift(P_ss / P_nn)[0:len(freq)/2] # 10*log10()
            freq =  ifftshift(freq)[0:len(freq)/2]  # only positive frequencies
            
            fstart = 0
            fend = max(find(freq < 500))  # only use frequencies smaller than 1kHz
            freq = freq[fstart:fend]   
            SNR = SNR[fstart:fend]   

            figure(1) 
            semilogx(freq, 10*log10(real(SNR)), 'r--') 
            
        return CF_mat
        


    def get_RCtau(self, amp = -0.001, delay = 1000*ms, dur = 500*ms, dtest = 100*ms, method = "simple", nruns = 3, C_old = None, do_print = 1, Vrest = None, dv=1*mV):
        """
        This function returnes the effective tau, resistance, and capacity at the resting potential
        """  
        
        if Vrest == None:
            self.vinit = 0
            self.run(0, dur)
            t1, voltage, current, g_in = self.get()
            Vrest = voltage[-1]
            self.stim = None 
            
            print Vrest
            #figure("rest")
            #plot(t1, voltage)
        
        if method == "vc":
            
            Vtest = Vrest+dv
            self.set_VClamp(amp0 = Vrest, amp1 = Vtest, dur0 = 1000*ms, dur1 = 1000*ms)
            self.run(0, 2000*ms)
            t1, voltage, current, g_in = self.get() 
            
            self.stim = None    
            amp = current[-1]-current[999*ms/self.dt]
            
            R = dv / amp  # MOhm 
            
            if do_print: print "R: " + str(R) + " MOhm"
            C = None
            tau = None
            
            figure("vc")
            plot(t1[900*ms/self.dt:-1], current[900*ms/self.dt:-1])
            plt.show()
            
        elif method == "simple":
            
            Vtest = Vrest-1*mV
            self.set_VClamp(amp0 = Vtest, dur0 = 500*ms)
            self.run(0, 500*ms)
            t0, voltage, current0, g_in = self.get()        
            amp = current0[-1]
            self.stim = None    
            
            print amp
            #figure("simple0")
            #plot(t0[100*ms/self.dt:-1], current0[100*ms/self.dt:-1])
            
            self.set_IClamp(amp = amp, delay = delay, dur = dur)
            self.run(0, delay + dur)
            t1, voltage, current, g_in = self.get()
            self.stim = None 
            
            #figure("simple")
            #plot(t1, voltage)
            #show()
        
            istart = delay / self.dt # min(find(t1 > 90))
            istop = -50*ms / self.dt
            t1 = t1[istart:istop]    
            voltage = voltage[istart:istop]   
            
            v_rest = voltage[0]
            v_min = voltage[-1] 
            dv = v_min - v_rest
                       
            R = dv / amp  # MOhm                
            
            exp_val = v_rest + ((1 - 1 / exp(1)) * dv)  # 0.6321 * dv
            itau = where(voltage < ( exp_val ))[0][0]
            
            tau = t1[itau] - t1[0]  # s
            
            C = tau / R  # uF
    
            if do_print: print "R: " + str(R) + " MOhm, C: " + str(C) + " uF, tau: " + str(tau) + " s"             
            
        elif method == "linfit":
            
            
            for i in range(nruns):
                self.run(0, delay + dur)
                t1, voltage, current, g_in = self.get()
                
                if i == 0:                    
                    voltagem = zeros(shape=(nruns,size(t1)))
        
                voltagem[i,:] = voltage
                
            voltage = mean(voltagem,0)
                
            istart = int((delay+dur-dtest) / self.dt) 
            istop = int((delay+dur) / self.dt)
            vmean = mean(voltage[istart:istop]) 
            
            istart = int((delay-dtest) / self.dt) 
            istop = int(delay / self.dt)   
            vrest = mean(voltage[istart:istop])
            
            istart = int(delay / self.dt) 
            istop = int(-50*ms / self.dt)
            t1 = t1[istart:istop]-delay    
            v_fit = voltage[istart:istop]-vrest  
            
            dv0 = vmean - vrest
            R_m = dv0 / amp  # MOhm 
            
            dv, tau, u = fit_exp(t1, v_fit, p0 = array([dv0, 1e-3]))

            R = dv / amp  # MOhm   
            C = tau / R
            
            if  C_old != None:  # when change of tau shoudl just be due to R change
                tau_m = R_m * C_old
            else:
                tau_m = 0
                
            if do_print: print "R_m: " + str(R_m) + " R(fit): " + str(R) + " MOhm, C: " + str(C) + " nF, C_old: " + str(C_old) + " nF, tau: " + str(tau) + " s" + " tau_m: " + str(tau_m) + " s"

            plt.plot(t1, v_fit, '.', t1, u)
            plt.show()
                               

        self.stim = None           
        
        return R, C, tau
        
        
    def get_onset_amp(self, stim_start = 0.1*s, stim_end = 0.1*s, len_pulse = 0.5*ms, input_vec = arange(0.01,1,0.01), stim_type = "i"):
        """
        This function returns the pulse current amplitude necessary to trigger a spike, should work for longer pulses too!
        """
        
        for amp_init in input_vec:  # find spike onset current
        
            if stim_type == "i":
                          
                t, ivec = construct_Pulsestim(dt = self.dt, pulses = 1, stim_start = stim_start, stim_end = stim_end, len_pulse = len_pulse, amp_init = amp_init)
                tstop = t[-1] 
            
                self.set_IPlay(ivec, t)                    
                
            if stim_type == "syn":

                syntimes = None
                syntimes = h.NetStim(0)
                syntimes.interval = 5/ms  # in ms
                syntimes.number = 1
                syntimes.start = stim_start/ms  # in ms
                tstop = stim_start + stim_end  # in ms

                n_syn = 1
                nclist = []
                
                if hasattr(self.cell, 'whatami'):
                    
                    self.cell.MF_L = []
                    
                    for i in range(n_syn):
                        self.cell.createsyn(nmf = 1, ngoc = 0, weight = amp_init) 
                                            
                    for i in range(len(self.cell.MF_L)):
                        nclist.append(h.NetCon(syntimes, self.cell.MF_L[i].input))  ## use internal function!!! 
                        nclist[i].delay = 0
                        nclist[i].weight[0] = 1                 
                    
                else:

                    self.cell.create_synapses()
                    
                    for i in range(len(self.cell.synlist)):
                        nclist.append(h.NetCon(syntimes, self.cell.synlist[i], sec = self.cell.soma))                        
                        nclist[i].delay = 0
                                            
                        if hasattr(self.cell, 'g_syn'):
                            nclist[i].weight[0] = self.cell.g_syn[i] * amp_init                            
                        else:
                            self.cell.synlist[i].gmax_factor = amp_init                  
            
            self.run(1, tstop)
        
            t1, voltage, current, g_in = self.get()
            
            #plt.plot(t1, voltage); draw() #draw() #show() 
            
            freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = self.if_get(compute_mean = 0)  
            
            num_spikes = len(spike_times)
            
            if num_spikes == 0:  # no spike found, do nothing
                pass    
            elif num_spikes >= 1:  # break, at least one spike was found
                break   
            else:
                
                print 'Error in spike threshold detection'
                plt.figure(23); plt.plot(t1, voltage, t1, current); plt.show()
                raise ValueError('Error in spike threshold detection, what happened?')
        
        if num_spikes == 0:  # check if loop was exited without triggering a spike
            raise ValueError('Supplied currents to weak, no spike triggered!')
                
        print "Threshold: " + str(amp_init) #+ " nA"
        #show()         
        
        return amp_init


    def get_refrac_pp(self, do_plot = 0, len_pulse = 0.5*ms, stim_start = 0.01, stim_end = 0.01, input_vec = arange(0.01,1,0.01), latency_vec = arange(0.5*ms, 50*ms, 0.5*ms), num_workers = 3):
        """
        Estimate relative (and absolute) refractory period using parallel worker program get_refrac-worker1.py.
        Slower since single NEURON runs are too short!!
        """
        
        from mpi4py import MPI
        import time
        
        if MPI.Query_thread() < MPI.THREAD_MULTIPLE:  # ONLY MPICH" SUPPORTS THREADING!!!
            sys.stderr.write("MPI does not provide enough thread support \r")
            sys.exit(0)

        try:
            import threading
        except ImportError:
            sys.stderr.write("threading module not available \r")
            sys.exit(0)

        
        start_time = time.time()        
        
        if do_plot: 
            ion()  # interactive plotting mode on
        
        dt = self.dt
         
        amp_init = self.get_onset_amp(stim_start, stim_end, len_pulse, input_vec)  # get amplitude for one pulse
        
        latency_vec = latency_vec[::-1] # reverse
        late_amp_vec = ones(len(latency_vec)) * float('NaN') # create NaN vector
        late_eff_lat_vec = ones(len(latency_vec)) * float('NaN') # create NaN vector
        
        # we have to tell the worker wich cell to run and which temperature!
        # change in ion channels not yet implemented!
        cell_path = os.path.dirname( os.path.realpath( __file__ ) )  # path to cell module
        cell_name = self.cell.__module__  # name of cell module
        worker_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'get_refrac-worker1.py'))
        
        worker = MPI.COMM_SELF.Spawn(sys.executable, args = [worker_path, str(self.temperature), cell_path, cell_name], maxprocs = num_workers)  # start workers to recieve signals
        print "use_workers:" + str(num_workers)
        
        # send the workers some basic information
        worker.bcast(dt, root = MPI.ROOT)
        worker.bcast(stim_start, root = MPI.ROOT)
        worker.bcast(stim_end, root = MPI.ROOT)
        worker.bcast(len_pulse, root = MPI.ROOT)
        worker.bcast(amp_init, root = MPI.ROOT)
        
        class worker_send(threading.Thread): 
            """Thread class with a stop() method. The thread itself has to check
            regularly for the stopped() condition."""
                
            def __init__(self, input_vec, latency):
                super(worker_send, self).__init__()  #or threading.Thread.__init__(self)
                self.input_vec = input_vec
                self.latency = latency
                self.is_free = empty(1, int)
                self.i_amp_next = -1 
                self.amp_next = []                                  
                    
                self._stop = threading.Event()                    
                    
            def get_i_amp_next(self):
                return self.i_amp_next
                    
            def stop(self):
                self._stop.set()

            def stopped(self):
                return self._stop.isSet()
 
            def run(self):
         
                for self.amp_next in self.input_vec:
                        
                    if self.stopped(): 
                        break
                    else:      
                        self.i_amp_next += 1
    
                    worker.Recv([self.is_free, MPI.INT] , source = MPI.ANY_SOURCE, tag = 1)  # wait for free worker we can give a job? 

                    end_work = array(0, int)        
                    worker.Ssend([end_work, MPI.INT] , dest = self.is_free, tag = 3)  # don't go on        
            
                    assert self.amp_next == self.input_vec[self.i_amp_next]
                                                
                    #print "FREE worker " + str(self.is_free) + " has to do latency: " + str(self.latency) + " and amp_next: " + str(self.amp_next) + "\r"
                    
                    worker.send(self.latency, dest = self.is_free)       
                    worker.send(self.i_amp_next, dest = self.is_free)  # also send index
                    worker.send(self.amp_next, dest = self.is_free)
                    
                    
        def worker_recv(has_result, i_amp_next, amp_next, spike_times, t1, current, voltage):
            """Small function to wait for new results from workers and to collect them"""
            
            # check if there are some results coming back
            worker.Recv([has_result, MPI.INT], source = MPI.ANY_SOURCE, tag = 2)  # search for free worker who has a result 

            #worker.send(has_result, dest = has_result)  
                    
            i_amp_next = worker.recv(i_amp_next, source = has_result)         
            amp_next = worker.recv(amp_next, source = has_result)         
            spike_times = worker.recv(spike_times, source = has_result)
                
            t1 = []
            current = []
            voltage = []
                
            # don't get any plots back, only for debuggung
            #t1 = worker.recv(t1, source = has_result)
            #current = worker.recv(current, source = has_result)
            #voltage = worker.recv(voltage, source = has_result)
                
            return i_amp_next, amp_next, spike_times, t1, current, voltage
                    
        
        for i_latency, latency in enumerate(latency_vec):  # cycle through all latencies we want to examine
        
            #print "latency: " + str(latency)
            
            no_amp_found = False
            t1 = array([1])
            current = array([1])
            voltage = array([1])
            
            send_thread = worker_send(input_vec, latency)  # create thread object
            send_thread.start()  # run sender thread
            
            has_result = empty(1, int)
            i_amp_next = []
            amp_next = []
            spike_times = []                               
                
            done_input_vec = ones(len(input_vec), int) * -1  # create vector of -1s
            temp_isi = ones(len(input_vec)) * float('NaN') # create NaN vector
                
            amp_found = True
            i_amp_collected = 0
            i_amp_sent = []
            

            for amp_next in input_vec: 
                
                i_amp_next, amp_next, spike_times, t1, current, voltage = worker_recv(has_result, i_amp_next, amp_next, spike_times, t1, current, voltage)
                    
                #print "RECV worker " + str(has_result) + " spike_times: " + str(spike_times) + " latency: " + str(latency) + " amp_next: " + str(amp_next) + "\r"
                i_amp_collected += 1  # count jobs that have been collected! 
                
                num_spikes = len(spike_times)
                    
                if num_spikes == 1:  # only first spike found, write down
                    
                    done_input_vec[i_amp_next] = 0  # no spike
                        
                elif num_spikes == 2:  # check if two spikes occur
                    
                    done_input_vec[i_amp_next] = 1  # spike
                    temp_isi[i_amp_next] = diff(spike_times)  # save temporary isis, wee need them later!
                                            
                else:  #  everything else has to be an error!!!      
                    raise ValueError('Error in spike threshold detection, maybe more than one spike occured')
                        
                # amp found is valid only:
                # if any element is bigger than 1, 
                # AND if the elements of 1s are continuous
                # AND if the elements of uncomputed -1s are continuous, 
                # e.g. if there are no elements that have not yet been computed before
                # -1 is added at the end to capture some "border effects"
                                         
                # Proof of concept:
                #done_input_vec = array([-1,-1,-1,-1])  # False    
                #done_input_vec = array([1,-1,-1,-1])  # True 
                #done_input_vec = array([-1,1,-1,-1])  # False
                #done_input_vec = array([1,-1,1,-1])  # False
                #done_input_vec = array([0,0,0,-1])  # False
                #done_input_vec = array([0,0,-1,1])  # False : -1 at the end is added for this
                
                #done_input_vec = array([0,1,-1,-1])  # True
                #done_input_vec = array([0,1,1,-1,-1])  # True
                #done_input_vec = array([0,0,0,0])  # False
                #done_input_vec = array([0,0,-1,1,-1,-1])  # False                    
                
                done_input_vec2 = concatenate((done_input_vec, array([-1])))
                amp_found = (not np.any(diff(where(done_input_vec2 < 0)) >= 2)) & (not np.any(diff(where(done_input_vec2 > 0)) >= 2)) & np.any(done_input_vec2 > 0) # & (done_input_vec[0] != -1)
                
                if amp_found:  # current was found
                
                    index = where(done_input_vec)[0][0]  # we know that done_input_vec is continuous check where 1 occurs first    
                    late_amp_vec[i_latency] = input_vec[index]  # write amplitude to vector and break
                    late_eff_lat_vec[i_latency] = temp_isi[index]  # save effective latency of spikes to vector
                    
                    send_thread.stop()  # don't send out jobs anymore!
                    i_amp_sent = send_thread.get_i_amp_next()+1  # how many jobs have been sent out?
                    #print "i_amp_sent: " + str(i_amp_sent) + " i_amp_collected: " + str(i_amp_collected) 
                    #print done_input_vec

                if amp_found & (i_amp_sent == i_amp_collected): # continue until all jobs have been collected! 
                    break                
                
                if (sum(abs(done_input_vec)) == 0):  # check if no current could be found
                    no_amp_found = True
                    break
                
            send_thread.join()
                   
            if no_amp_found:  # no current could be found for last latency
                break
            
            if do_plot:
                #plt.figure(10); clf(); xlabel("Time [s]"), ylabel("Voltage [mV], Current [nA]")
                #plt.plot(t1, current*50, 'b-', t1, voltage, 'g-'); draw()
                #plt.print "Amplitude of second pulse for a latency of " + str(latency * 1000) + " ms: " + str(late_amp_vec[i_latency]) + " nA"
                            
                plt.figure(11); plt.clf() 
                plt.xlabel("Latency till second spike [s]"); plt.ylabel("Current necessary for second spike [nA]")
                plt.plot(latency_vec, late_amp_vec, 'k+-', latency_vec, amp_init * (late_amp_vec / late_amp_vec) , 'k--')  # plot but also add unity line
                plt.draw()  # redraw the canvas            
                
            if  isnan(late_amp_vec[i_latency]):  # stop computation when no current in input_vec is found that triggers a spike
                break
            
        is_free = empty(1, int)
        end_work = array(1, int)
        for i in range(num_workers):  
            worker.Recv([is_free, MPI.INT] , source = MPI.ANY_SOURCE, tag = 1)  # wait for free worker we can give a job? 
            worker.Ssend([end_work, MPI.INT] , dest = is_free, tag = 3)  # don't go on                
                        
        worker.Disconnect()
        
        duration = time.time() - start_time  
        print "execution time: " + str(duration)
         
        return latency_vec, late_amp_vec, amp_init, late_eff_lat_vec  # latency vector, amplitude of second spike vector, initional amplitude and effective latency of two spikes


    def get_refrac(self, do_plot = 0, len_pulse = 0.5*ms, stim_start = 0.01, stim_end = 0.01, input_vec = arange(0.01,1,0.01), latency_vec = arange(0.5*ms, 50*ms, 0.5*ms)):
        """
        Estimate relative (and absolute) refractory period
        """

        import time
        
        start_time = time.time()        
        
        if do_plot: 
            ion()  # interactive plotting mode on
        
        amp_init = self.get_onset_amp(stim_start, stim_end, len_pulse, input_vec)  # get amplitude for one pulse
        
        latency_vec = latency_vec[::-1] # reverse
        late_amp_vec = ones(len(latency_vec)) * float('NaN') # create NaN vector
        late_eff_lat_vec = ones(len(latency_vec)) * float('NaN') # create NaN vector
        
        for i_latency, latency in enumerate(latency_vec):  # cycle through all latencies we want to examine

            for amp_next in input_vec:  # find spike onset current
            
                t, ivec = construct_Pulsestim(dt = self.dt, pulses = 2, latency = latency, stim_start = stim_start, stim_end = stim_end, len_pulse = len_pulse, amp_init = amp_init, amp_next = amp_next)
                tstop = t[-1]
            
                self.set_IPlay(ivec, t)
                self.run(1, tstop)
        
                t1, voltage, current, g_in = self.get()
                freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = self.if_get(compute_mean = 0) 
                                
                num_spikes = len(spike_times)
                
                is_first_spike = False
                is_second_spike = False
                
                if num_spikes == 1:  # check if one spike occurs
                    #if (stim_start + latency) >= spike_times[0] >= stim_start:  # this might not be necessary 
                        is_first_spike = True
                        
                if num_spikes == 2:  # check if two spikes occurs
                    #if (stim_start + latency) <= spike_times[1]:  # this might not be necessary 
                        is_second_spike = True        
                    
                if is_first_spike:  # only first spike found, do nothing
                    pass    
                
                elif is_second_spike:  # second spike was also found
                    late_amp_vec[i_latency] = amp_next  # write amplitude to vector and break
                    late_eff_lat_vec[i_latency] = diff(spike_times)  # save effective latency of spikes to vector
                    break   
                
                else:  #  everything else has to be an error!!!      
                    raise ValueError('Error in spike threshold detection, maybe more than one spike occured')
            
            if do_plot:
                plt.figure(10); plt.clf(); plt.xlabel("Time [s]"), plt.ylabel("Voltage [mV], Current [nA]")
                plt.plot(t1, current*50, 'b-', t1, voltage, 'g-'); plt.draw()
                            
                plt.figure(11); plt.clf() 
                plt.xlabel("Latency till second spike [s]"); plt.ylabel("Current necessary for second spike [nA]")
                plt.plot(latency_vec, late_amp_vec, 'k+-', latency_vec, amp_init * (late_amp_vec / late_amp_vec) , 'k--')  # plot but also add unity line
                plt.draw()  # redraw the canvas            
            
            print "Amplitude of second pulse for a latency of " + str(latency*1000) + " ms: " + str(late_amp_vec[i_latency]) + " nA"
            
            if  isnan(late_amp_vec[i_latency]):  # stop computation when no current in input_vec is found that triggers a spike
                break
            
        duration = time.time() - start_time  
        print "execution time: " + str(duration)        
        
        return latency_vec, late_amp_vec, amp_init, late_eff_lat_vec  # latency vector, amplitude of second spike vector, initional amplitude and effective latency of two spikes


    def get_refrac_train(self, do_plot = 0, len_pulse = 0.5*ms, stim_start = 0.05*s, stim_end = 0.05*s, input_vec = arange(0.01,1,0.01), latency_vec = arange(0.5*ms, 50*ms, 0.5*ms), pulses = 4, stim_type = "i", n_syn = 1, pure_inout = False, n_is_amp = False):
        """
        Estimate relative (and absolute) refractory period, using a train of spikes with number of pulses set by pulses (default: 4)
        """

        if do_plot: 
            ion()  # interactive plotting mode on
        
        if pure_inout:
            amp_init = 1
        else:
            amp_init = self.get_onset_amp(stim_start, stim_end, len_pulse, input_vec, stim_type = stim_type)  # get amplitude for one pulse    
        
        latency_vec = latency_vec[::-1]  # reverse
        amp_vec = ones(len(latency_vec)) * float('NaN')  # create NaN vector
        eff_lat_vec = ones(len(latency_vec)) * float('NaN')  # create NaN vector
        mean_lat_vec = ones(len(latency_vec)) * float('NaN')  # create NaN vector
        
        nan_found = 0
        n_syn_temp = n_syn
        
        for i_latency, latency in enumerate(latency_vec):  # cycle through all latencies we want to examine

            for amp in input_vec:  # find spike onset current
                print "amp: ", amp
                
                if n_is_amp:
                    amp = n_syn_temp
                    n_syn = 1                    
            
                if stim_type == "i":
                          
                    t, ivec = construct_Pulsestim(dt = self.dt, pulses = pulses, latency = latency, stim_start = stim_start, stim_end = stim_end, len_pulse = len_pulse, amp_init = amp)
                    tstop = t[-1] 
                
                    self.set_IPlay(ivec, t)
                        
                    
                if stim_type == "syn":

                    syntimes = None            
                    syntimes = h.NetStim(0)
                    syntimes.interval = latency/ms  # in ms
                    syntimes.number = pulses
                    syntimes.start = stim_start/ms  # in ms
                    tstop = stim_start + pulses * latency + stim_end  # in ms
                    
                    nclist = []
                    
                    
                    if hasattr(self.cell, 'MF_L'):
                        
                        self.cell.MF_L = []
                        
                        for i in range(n_syn):
                            self.cell.createsyn(nmf = 1, ngoc = 0, weight = amp) 
                                                
                        for i in range(len(self.cell.MF_L)):
                            nclist.append(h.NetCon(syntimes, self.cell.MF_L[i].input))  ## use internal function!!! 
                            nclist[i].delay = 0
                            nclist[i].weight[0] = 1     
                        
                    else:

                        self.cell.create_synapses()

                        for i in range(len(self.cell.synlist)):
                            nclist.append(h.NetCon(syntimes, self.cell.synlist[i], sec = self.cell.soma))  ## use internal function!!!
                            nclist[i].delay = 0
                            
                            if hasattr(self.cell, 'g_syn'):
                                nclist[i].weight[0] = self.cell.g_syn[i] * amp                            
                            else:
                                self.cell.synlist[i].gmax_factor = amp                       
                
                self.run(1, tstop)
        
                t1, voltage, current, g_in = self.get()
                                
                freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = self.if_get(compute_mean = 0) 
                
                num_spikes = len(spike_times)
                print "1/diff(spike_times):", 1/diff(spike_times)
                
                #if num_spikes > 1:
                #    if min(diff(spike_times)) <= latency+0.01*latency:
                #        amp_vec[i_latency] = amp  # write amplitude to vector and break
                #        eff_lat_vec[i_latency] = min(diff(spike_times))  # save max latency of spikes to vector
                #        break
                
                if pure_inout:
                    amp_vec[i_latency] = amp  # write amplitude to vector and break
                    ds = diff(spike_times)
                    if len(ds) != 0:
                        eff_lat_vec[i_latency] = ds[0]  # save max latency of spikes to vector
                        mean_lat_vec[i_latency] = mean(ds)
                    break 
                    
                else:
                    if (num_spikes >= pulses) and (mean(diff(spike_times[0:pulses])) <= latency+0.01*latency):  # all spikes occured + right frequency!!!
                        print "diff(spike_times[0:pulses]):", diff(spike_times[0:pulses])
                        amp_vec[i_latency] = amp  # write amplitude to vector and break
                        eff_lat_vec[i_latency] = min(diff(spike_times[0:pulses]))  # save max latency of spikes to vector
                        mean_lat_vec[i_latency] = mean(diff(spike_times[0:pulses]))
                        break 
                
                #elif num_spikes < pulses:
                #    pass
               
                #elif stim_type == "syn":  # don't care weither there are too many spikes if synaptic stimulation 
                #    pass
                
                #else:  #  everything else has to be an error!!!      
                #    raise ValueError('Error in spike threshold detection, too many spikes occured')
            
            if do_plot:
                plt.figure(10); plt.clf(); plt.xlabel("Time [s]"), plt.ylabel("Voltage [mV], Current [nA]")
                
                if stim_type == "syn":
                    plt.plot(t1, voltage, 'g-'); plt.draw()
                else:
                    plt.plot(t1,current*50, 'b-', t1, voltage, 'g-'); plt.draw()
                        
                plt.figure(11); plt.clf() 
                plt.xlabel("Latency between " + str(pulses) + " spikes [s]"); plt.ylabel("Current necessary for all spike [nA]")
                plt.plot(latency_vec, amp_vec, 'k+-', latency_vec, amp_init * (amp_vec / amp_vec) , 'k--')  # plot but also add unity line
                plt.draw()  # redraw the canvas   
                
                
            if stim_type == "i":
                print "Amplitude of " + str(pulses) + " pulses for with a latency of " + str(latency*1000) + " ms: " + str(amp_vec[i_latency]) + " nA"
                
            if stim_type == "syn":                
                print "Synaptic strength of " + str(pulses) + " pulses for with a latency of " + str(latency*1000) + " ms: " + str(amp_vec[i_latency])               
            
            if  isnan(amp_vec[i_latency]):  # stop computation when no current in input_vec is found that triggers a spike
                nan_found += 1            
            
            if  nan_found > 20: # if nan was found for more than 20 times, break cycle!
                break 
        
        return latency_vec, amp_vec, amp_init, eff_lat_vec, mean_lat_vec  # latency vector, amplitude of second spike vector, initional amplitude and effective latency of two spikes


    def plot(self, currlabel="control", do_freq=1):
        
        if self.id == 0: 
        
            t1, voltage, current, g_in = self.get()
    
            plt.subplot(3,1,1)
            plt.plot(t1, voltage, label=currlabel)
            plt.xlabel("Time [ms]")
            plt.ylabel("Voltage [mV]")
                    
            plt.axis(xmin = 0, xmax = self.tstop)
            plt.axis(ymin = -100, ymax = 50)
            
            lg = plt.legend()
            lg.get_frame().set_linewidth(0.5)        
    
            plt.subplot(3,1,2)
            plt.plot(t1, current)
            plt.xlabel("Time [ms]")
            plt.ylabel("Current [nA]")
    
            plt.axis(xmin=0, xmax=self.tstop)
    
            if do_freq:  
                freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = self.if_get()
                plt.subplot(3,1,3)
                plt.lot(freq_times, spike_freq)
                plt.xlabel("Time [ms]")
                plt.ylabel("Rate [Hz]")
    
                plt.axis(xmin=0, xmax=self.tstop)
            
            
    def get_if(self, istart = 0, istop = 1, di = 0.01, tstop = 1*s, change_factor = 0.3):

        current_vector = arange(istart,istop,di)
                
        current_vector_local = [] 
        
        spike_freq_vec = []
        freq_times_vec = []
        
        
        for i in range(int(self.id), len(current_vector), int(self.nhost)): # loop over all freq_used
            current_vector_local.append(current_vector[i]) 
                       
        to_run = True
        if len(current_vector_local) == 0:
            current_vector_local.append(-1)
            to_run = False 
         
        freq_vector = zeros(size(current_vector_local), dtype='d')
        freq_onset_vector = zeros(size(current_vector_local), dtype='d')
                    
        if to_run:
            for i, c in enumerate(current_vector_local): 
                print "- if current:", c 
                self.set_IClamp(c, delay = 0, dur = tstop+1)    
                print "- if clamped"
                self.run_steps(1, tstop = tstop, simstep = 1*s)
                print "- if ran"
                freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = self.if_get(compute_mean=tstop, change_factor = change_factor)
                freq_vector[i] = freq_mean
                print "- if freq_mean:", freq_mean
                spike_freq_vec.append(spike_freq)
                freq_times_vec.append(freq_times)
                print "- if spike_freq:", spike_freq
                freq_onset_vector[i] = freq_onset
                self.c = c    
                #self.stim = None
       
        vlen = []
        vlen = self.comm.allgather(len(current_vector_local), vlen) # get length first, then use proper Gather! 
        
        current_vector_all = -1 * ones( (self.comm.size, max(vlen)), dtype='d')
        freq_vector_all = zeros( (self.comm.size, max(vlen)), dtype='d') 
        freq_onset_vector_all = zeros( (self.comm.size, max(vlen)), dtype='d')
                    
        current_vector_local = np.array(current_vector_local, dtype='d')
        
        #if self.id == 0: print freq_vector_all, freq_vector
        
        self.comm.Gatherv(sendbuf=[current_vector_local, MPI.DOUBLE], recvbuf=[current_vector_all, MPI.DOUBLE], root=0)
        self.comm.Gatherv(sendbuf=[freq_vector, MPI.DOUBLE], recvbuf=[freq_vector_all, MPI.DOUBLE], root=0)
        self.comm.Gatherv(sendbuf=[freq_onset_vector, MPI.DOUBLE], recvbuf=[freq_onset_vector_all, MPI.DOUBLE], root=0)

        if self.id == 0:
                
            current_vector_all = np.concatenate(current_vector_all)
            freq_vector_all = np.concatenate(freq_vector_all)
            freq_onset_vector_all = np.concatenate(freq_onset_vector_all)
                        
            i1 = argsort(current_vector_all) 
            i2 = i1[where(current_vector_all[i1]>-1)]
            
            current_vector = current_vector_all[i2]
            freq_vector = freq_vector_all[i2]
            freq_onset_vector = freq_onset_vector_all[i2]

            plt.figure(95)
            l = len(freq_times_vec)
            for i in range(l):
                #plt.plot(freq_times_vec[i][1:], abs(diff(spike_freq_vec[i])), color=plt.cm.cool(1.*i/l) )
                plt.plot(freq_times_vec[i], spike_freq_vec[i], color=plt.cm.cool(1.*i/l) )
            plt.savefig("./figs/dump/latest_if_trains_" + self.celltype + ".pdf", dpi = 300)  # save it 
    
        self.comm.Barrier()    
        
        return current_vector, freq_vector, freq_onset_vector


    def if_plot(self, currlabel = "control", istart = 0, istop = 1, di = 0.01, ax = None):

        current_vector, freq_vector, freq_onset_vector = self.get_if(istart = istart, istop = istop, di = di)
        plot_if(currlabel, current_vector, freq_vector, freq_onset_vector, ax = ax, color_vec = self.color_vec)
        
    
    def iv_plot(self, currlabel = "control", istart = -0.01, istop = 0.01, di = 0.001, delay = 500*ms, dur = 1500*ms, dtest = 100*ms, amp = -0.002):

        current_vector = arange(istart,istop,di)

        zcheck = abs(current_vector)<=1e-12  # remove 0!
        if np.any(zcheck):
            ic = int(where(zcheck)[0]) 
            current_vector = delete(current_vector, ic)
    
        v_vector = zeros(size(current_vector))
        r_vector = zeros(size(current_vector))
        tau_vector = zeros(size(current_vector))

        for i, c in enumerate(current_vector):
            
            self.set_IClamp(amp = c, delay = delay, dur = dur, stims = 2)
            self.run_steps(0, delay + dur)
            
            t1, voltage, current, g_in = self.get()
            #plt.plot(t1, voltage)
            
            istart = int((delay+dur-dtest) / self.dt) 
            istop = int((delay+dur) / self.dt)
            vmean = mean(voltage[istart:istop]) 
            
            istart = int((delay-dtest) / self.dt) 
            istop = int(delay / self.dt)   
            vrest = mean(voltage[istart:istop])
            
            rmean = (vmean-vrest)/c
            
            delay2 = delay*2
            dur2 = dur - delay
            R, C, tau = self.get_RCtau(amp = amp, delay = delay2, dur = dur2, do_print = 0)
            
            #print R, tau
            #t1, voltage, current, g_in = self.get()
            #plt.plot(t1, voltage, 'r')
            #plt.show()
            
            v_vector[i] = vmean
            r_vector[i] = R #rmean
            tau_vector[i] = tau         
            

        self.stim = None
        plot_iv(currlabel, current_vector, v_vector, r_vector, tau_vector)
        
       
    def refrac_plot(self, do_plot = 1, method = "onset", currlabel = "control", input_vec = arange(0.01, 1, 0.01), latency_vec = arange(1*ms, 50*ms, 1*ms), len_pulse = 0.2*ms, stim_type = "i"):

        if method == "onset":  # check threshold for onset spikes
        
            latency_vec, late_amp_vec, amp_init, late_eff_lat_vec = self.get_refrac(do_plot = do_plot, input_vec = input_vec, latency_vec = latency_vec, len_pulse = len_pulse)
           
            plot_refrac_onset(currlabel, latency_vec, late_amp_vec, amp_init, late_eff_lat_vec) 
                
        if method == "train":  # check threshold for a train of spikes
                
            if stim_type == "syn":    
                pulses = 2
            else:    
                pulses = 3
            latency_vec, amp_vec, amp_init, eff_lat_vec = self.get_refrac_train(do_plot = do_plot, input_vec = input_vec, latency_vec = latency_vec, len_pulse = len_pulse, pulses = pulses, stim_type = stim_type)

            plot_refrac_train(currlabel, pulses, latency_vec, amp_vec, amp_init, eff_lat_vec, stim_type)


    def fun_plot(self, currlabel = "control", dowhat = "cnoise", freq_used = array([1,10,100,1000]), t_stim = 10*s, method_interpol = np.array(["linear"]), ymax = 0, ax = None, axP = None, SNR = None, VAF = None, sexp = 0, cutf = 0, method_interpol_plot = [], opt_plot = np.array([]), xmax = None):
            
        SNR_switch = SNR
        VAF_switch = VAF
        
        VAF = [] 
        SNR = []
            
        if dowhat == "msine":
            
            results = self.fun_msine_Stim(t_stim = t_stim, freq_used = freq_used, method_interpol = method_interpol)
            freq_used, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1') 
            freq_times, spike_freq, fmean, method_interpol = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol')                   
                            
        elif dowhat == "cnoise":
            
            results = self.fun_cnoise_Stim(t_stim = t_stim, sexp = sexp, cutf = cutf, method_interpol = method_interpol)
            freq_used, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1') 
            freq_times, spike_freq, fmean, method_interpol, SNR, VAF = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('SNR'), results.get('VAF')                      
                        
        elif dowhat == "ssine":
            
            results = self.fun_ssine_Stim_pp(freq_used = freq_used, method_interpol = method_interpol)
            freq_used, vamp, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1') 
            freq_times, spike_freq, fmean, method_interpol, VAF = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('VAF')   
                       
        
        ihold, amp, amp_noise = self.set_i()
        
        if self.id == 0:

            if "analytical" in opt_plot: # simplest case, only uses rm and tau, scaling necessary 
                rm, cm, taum = self.get_RCtau()
            else:
                rm = cm = taum = 0
                
            if hasattr(self.cell, 'tau_r') and analytical:  # does model have a resonance?, plot it!!! 
                gr = self.cell.gr 
                tau_r = self.cell.tau_r
                rm = self.cell.RM
                cm = self.cell.CM
            else:
                gr = tau_r = 0        

            if len(freq_times) <= 3: # Plot impedance function 
                
                if dowhat == "ssine":
                    mag = mag[0,:]  
                    pha = pha[0,:]    
                plot_impedance(currlabel, freq_used, mag, pha, ca, t1, current, voltage, rm, cm, gr,tau_r)   
    
            else:  # Plot transfer function        
                if "if" in opt_plot:
                    Vrest = self.cell.soma(0.5).pas.e*mV
                    Vth = self.cell.spkout.thresh*mV 
                    Vreset = self.cell.spkout.vrefrac*mV
                else:
                    Vreset = 0*mV; Vth = 1*mV; Vrest = 0*mV
                
                print "Mean rate: " + str(fmean)
                
                # Turn it off if set to zero
                if SNR_switch == 0: SNR = None
                if VAF_switch == 0: VAF = None  
                
                if xmax is not None:
                    iend = mlab.find(freq_used >= xmax)[0]
                    freq_used = freq_used[0:iend]
                    mag = mag[:,0:iend]
                    pha = pha[:,0:iend]
                
                plot_transfer(currlabel, freq_used, mag, pha, t1, current, voltage, freq_times, spike_freq, taum, fmean, ihold, rm, Vreset, Vth, Vrest, method_interpol, method_interpol_plot, ymax = ymax, SNR = SNR, VAF = VAF, ax = ax, axP = axP, color_vec = self.color_vec, opt_plot = opt_plot, linewidth = self.linewidth)            
        
        return VAF, SNR, mag, pha, freq_used   
        
            
    def save_file(self, currtitle, data = None, directory = "./figs/"):
        
        from datetime import datetime
        import re
        import h5py

        def make_name(filename):
            badchars1 = re.compile(r'[: ]+|^\.|\.$|^ | $|^$')
            badchars2 = re.compile(r'[^A-Za-z0-9_.]+|^\.|\.$|^ | $|^$')
            filename = badchars1.sub('', filename)
            filename = badchars2.sub('_', filename)
            
            return filename.lower()
            
        idate = datetime.now().strftime('%Y%m%d_%H%M')  # %S
        
        if not os.path.exists(directory):  # create directory if necessary
            os.makedirs(directory)

        filename = directory + idate + "-" + make_name(currtitle) + ".hdf5" 
  
        f = h5py.File(filename)
        
        for i in range(len(data.keys())):
            dset = f.create_dataset(data.keys()[i], data=data[data.keys()[i]], compression = 'gzip', compression_opts = 6)
        f.close()
        
  
# test code
if __name__ == '__main__': 
    
    # RUN:  mpdboot -n 5 -f /home/chris/mpd.hosts --chkup -v -d
    #       mpiexec -n 5 Stimulation.py
    #       mpdallexit
    #       pcmd 'mpdallexit; mpdcleanup'
    
    from Stimulation import *
    from Plotter import *
    from Stimhelp import *
    
    from cells.PassiveCell import *
    from cells.ActiveCell import *
    from cells.IfCell import *
    from cells.T1Cell import *
    
    do_run = 1
    pickle_prefix = "Stimulation.py"
    if results.norun:  # do not run again use pickled files!
        print "- Not running, using saved files"
        do_run = 0

    # SET DEFAULT VALUES FOR THIS PLOT
    fig_size =  [11.7, 8.3]
    params = {'backend': 'ps', 'axes.labelsize': 9, 'axes.linewidth' : 0.5, 'title.fontsize': 8, 'text.fontsize': 9,
        'legend.borderpad': 0.2, 'legend.fontsize': 8, 'legend.linewidth': 0.1, 'legend.loc': 'best', # 'lower right'    
        'legend.ncol': 4, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'text.usetex': False, 'figure.figsize': fig_size}
    rcParams.update(params)

    
    #do = array(["onset_refractory" , "train_refractory", "train_refractory_syn", "impedance", "iftransfer_ssine"])  # not parallel
    do = array(["impedance"])
    #do = array(["train_refractory_syn"])
    #do = array(["onset_refractory"])  # not parallel
    #do = array(["ifplot"])
    #do = array(["iftransfer"])
    #do = array(["stim_reconstruct_jitter"])
    #do = array(["stim_reconstruct"])
    #do = array(["CF"])
    #do = array(["tau_if"])
    #do = array(["iftransfer_syn"])
    #do = array(["grc_transfer_syn"])
    #do = array(["grc_transfer_fit"])
    #do = array(["grc_IF"])
    #do = array(["grc_cnoise_transfer_fit"])
    do = array(["ifpulse"])
    do = array(["prk_if"])
   
    if "ifplot" in do:
        
        ih = 0.01
        
        cell = ActiveCell(L_diam = 100)
        #cell = PassiveCell(GS = 1, CS = 10)
        
        sim = Stimulation(cell, temperature = 6.3)
        
        sim = None

        
        plt.figure(1)        
        sim = Stimulation(cell, temperature = 6.3)
        sim.get_RCtau(amp = -0.02, delay = 50*ms)
        sim.set_IClamp(ih)
        sim.run(0)
        t1, voltage, current, g_in = sim.get() 
        plt.figure(2)        
        plt.plot(t1, voltage, 'b')        

        #sim.if_plot(currlabel = "control", istart = 0, istop = 0.3, di = 0.01)        
        
        sim = None
        
        plt.figure(1)
        sim = Stimulation(cell, temperature = 6.3)
        sim.set_Gfluct(E_e = 0*mV, E_i = -75*mV, g_e0 = 0.00*uS, g_i0 = 0.0001*uS, std_e = 0.0002*uS, std_i = 0.001*uS, tau_e = 2*ms, tau_i = 10*ms)
        sim.get_RCtau(amp = -0.02, delay = 50*ms)
        sim.set_IClamp(ih)
        sim.run(0)
        t1, voltage, current, g_in = sim.get()
        plt.figure(2)
        plt.plot(t1, voltage, 'r') 
        
        #sim.if_plot(currlabel = "control", istart = 0, istop = 0.3, di = 0.01)
        
        sim = None
        
        #sim = Stimulation(cell, temperature = 6.3)
        #sim.get_RCtau()
        #sim = None
        
        plt.show()
        plt.clf()  # delete figure
        
        cell = None
        sim = None
        
        
    if "ifpulse" in do:
        
        from templates.granule.GRANULE_Cell import Grc
        
        cell = Grc(np.array([0.,0.,0.]))
           
        sim = Stimulation(cell, temperature = 37)
        sim.dt = 0.02*ms
    
        sim.icloc = "soma(0.5)" 
            
        sim.set_IClamp(amp = 0.016*nA, delay = 100*ms, dur = 2000*ms)  
        
        stim = h.IClamp(cell.soma(0.2)) 
        stim.amp = 0.1*nA
        stim.delay = 1000
        stim.dur = 10
        
        #stim2 = h.IClamp(cell.soma(0.2)) 
        #stim2.amp = 0.05
        #stim2.delay = 900
        #stim2.dur = 200
            
        tstop = 2200*ms
        sim.run(1, tstop)
        
        t1, voltage, current, g_in = sim.get()
        freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = sim.if_get(compute_mean = 0)
    
        print "freq_times: " + str(spike_freq)
        
        figure()
        plot(t1, voltage)
        
        figure()
        plot(freq_times, spike_freq)
        
        show()  
        

    if "train_refractory_syn" in do:
        
        di = 0.00001*uS; istart = di; istop = 0.01*uS;   # uS
        g_vec = arange(istart, istop, di)  # vector of synaptic strengths that will be analyzed
        
        lsteps = 1*ms; lstart = lsteps; lstop = 40*ms
        latency_vec = arange(lstart, lstop, lsteps)  # array of spike latencies we want to examine
        
        cell = ActiveCell()
        sim = Stimulation(cell, temperature = 6.3)
        #sim.get_RCtau()
        sim.set_IClamp(amp = 0.001*nA, delay = 10*ms, dur = 1000*ms)
        
        sim.refrac_plot(currlabel = "control", method = "train", do_plot = 1, input_vec = g_vec, latency_vec = latency_vec, stim_type = "syn")
        
        from datetime import datetime
        idate = datetime.now().strftime('%Y%m%d_%H%M')  # %S
        savefig("./figs/" + idate + "-refractory_syn_train_ActiveCell.pdf", dpi = 300) # save it 
       
        plt.show()
        plt.clf()  # delete figure   
        
        cell = None
        sim = None

   
    if "onset_refractory" in do:
        
        di = 0.01; istart = di; istop = 1;   # nA
        current_vec = arange(istart, istop, di)  # vector of currents that will be analyzed
        
        lsteps = 1*ms; lstart = lsteps; lstop = 30*ms
        latency_vec = arange(lstart, lstop, lsteps)  # array of spike latencies we want to examine
        
        cell = ActiveCell()
        sim = Stimulation(cell, temperature = 6.3)
        sim.get_RCtau()
        
        sim.refrac_plot(currlabel = "control", method = "onset", do_plot = 1, input_vec = current_vec, latency_vec = latency_vec)
        
        from datetime import datetime
        idate = datetime.now().strftime('%Y%m%d_%H%M')  # %S
        savefig("./figs/" + idate + "-refractory_onset_ActiveCell.pdf", dpi = 300) # save it 

        #plt.show()
        plt.clf()  # delete figure 
        
        
    if "train_refractory" in do:
        
        di = 0.01; istart = di; istop = 1;   # nA
        current_vec = arange(istart, istop, di)  # vector of currents that will be analyzed
        
        lsteps = 1*ms; lstart = lsteps; lstop = 30*ms
        latency_vec = arange(lstart, lstop, lsteps)  # array of spike latencies we want to examine
        
        cell = ActiveCell()
        sim = Stimulation(cell, temperature = 6.3)
        sim.get_RCtau()
        
        sim.refrac_plot(currlabel = "control", method = "train", do_plot = 1, input_vec = current_vec, latency_vec = latency_vec)
        
        from datetime import datetime
        idate = datetime.now().strftime('%Y%m%d_%H%M')  # %S
        savefig("./figs/" + idate + "-refractory_train_ActiveCell.pdf", dpi = 300) # save it 
       
        plt.show()
        plt.clf()  # delete figure   
        
        cell = None
        sim = None
        

    if "impedance" in do:
                
        freq_used = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        freq_used = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 30, 40, 50, 100])
                              
        plt.figure()
        cell = PassiveCell()
        
        from templates.granule.GRANULE_Cell import Grc
        cell = Grc(np.array([0.,0.,0.]))
        
        sim = Stimulation(cell, temperature = 0)
        sim.amp = 0.0005*nA
        sim.ihold = 0.002*nA 
        
        sim.fun_plot("Single Sine", "ssine", freq_used = freq_used) 
        
        from datetime import datetime
        idate = datetime.now().strftime('%Y%m%d_%H%M')  # %S
        savefig("./figs/" + idate + "-impedance_GrC.pdf", dpi = 300) # save it 
    
        #sim.fun_plot("Multi Sine", "msine", freq_used = freq_used, t_stim = 100) # rm [MOhm], cm [nF] TEST CELL
    
        sim.fun_plot("Colored Noise", "cnoise", t_stim = 100)
       
        cell = None
        sim = None
        
        from datetime import datetime
        idate = datetime.now().strftime('%Y%m%d_%H%M')  # %S
        savefig("./figs/" + idate + "-impedance_GrC.pdf", dpi = 300) # save it 
        
        plt.show()
        plt.clf()  # delete figure
        

    # construct frequencies
    freq_used = concatenate((arange(0.1, 1, 0.1), arange(1, 501, 1) ))
    freq_used = arange(1, 501, 1) #arange(10, 51, 1)
    f_rem = array([54,55,107,110,163,164,217,219,272,273,327,326,435,436,491,354,409,463,380,490,108,109,328,381])
    if_rem = zeros(len(f_rem)) 
    for i in range(len(f_rem)):
        if_rem[i] = int(where(freq_used==f_rem[i])[0])
    freq_used = delete(freq_used, if_rem.T)
    freq_used = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 50, 60, 70, 80, 90, 100, 1000])
    #print freq_used
    #freq_used = array([1,10,100,1000]) 

    
    if "iftransfer" in do:
        
        t_stim = 100*s
        
        celltype = "IfCell"   
        cell_exe = "cell = IfCell()"   
        exec cell_exe                 
        
        #cell.add_resonance(tau_r = 100*ms, gr = 0.0005*uS)
        #sim.set_Ifluct(sigma = amp*0.1, tau = 0.1*ms)
        
        sim = Stimulation(cell, celltype = celltype, cell_exe = cell_exe, temperature = 0, do_run = do_run, pickle_prefix = pickle_prefix)
        
        sim.ihold = 50
        sim.amod = 0.1       
        
        sim.give_freq = True
        
        opt_plot = np.array(["only_mag", "normalize", "analytical", "if", "loglog"]) 
        sexp = 0; cutf = 0
        
        from datetime import datetime
        idate = datetime.now().strftime('%Y%m%d_%H%M')  # %S
        suptitle("Transfer function of a " + celltype)    
        
        plt.figure(1)
        currtitle = "colored noise, " + celltype + ", amp: " + str(sim.amp) + ", ihold: " + str(sim.ihold) 
        
        method_interpol = np.array(["linear", "dt"])
        sim.fun_plot(currtitle, "cnoise", freq_used = freq_used, t_stim = t_stim, SNR = 0, method_interpol = method_interpol, sexp = sexp, cutf = cutf, opt_plot = opt_plot)
        savefig("./figs/" + idate + "-cnoise_transfer_" + celltype + "_ihold" + str(sim.ihold) + "_amp" + str(sim.amp) + ".pdf", dpi = 300)  # save it          
        
        plt.figure(2)        
        currtitle = "single sine, " + celltype + ", amp: " + str(sim.amp) + ", ihold: " + str(sim.ihold) 
        
        method_interpol = np.array(["none", "linear", "dt"])
        sim.fun_plot(currtitle, "ssine", freq_used = freq_used, VAF = 0, method_interpol = method_interpol, opt_plot = opt_plot)
        savefig("./figs/" + idate + "-ssine_transfer_" + celltype + "_ihold" + str(sim.ihold) + "_amp" + str(sim.amp) + ".pdf", dpi = 300)  # save it
        
        sim.jitter = 1*ms
                
        idate = datetime.now().strftime('%Y%m%d_%H%M')  # %S
        suptitle("Transfer function of a " + celltype)    
        
        plt.figure(3)
        currtitle = "colored noise, " + celltype + ", amp: " + str(sim.amp) + ", ihold: " + str(sim.ihold) 
        
        method_interpol = np.array(["linear", "dt"])
        sim.fun_plot(currtitle, "cnoise", freq_used = freq_used, t_stim = t_stim, SNR = 0, method_interpol=method_interpol, sexp = sexp, cutf = cutf, opt_plot = opt_plot)
        savefig("./figs/" + idate + "-cnoise_jitter__transfer_" + celltype + "_ihold" + str(sim.ihold) + "_amp" + str(sim.amp) + ".pdf", dpi = 300)  # save it          
        
        plt.figure(4) 
        
        currtitle = "single sine, " + celltype + ", amp: " + str(sim.amp) + ", ihold: " + str(sim.ihold) 
        
        method_interpol = np.array(["none", "linear", "dt"])
        sim.fun_plot(currtitle, "ssine", freq_used = freq_used, VAF = 0, method_interpol=method_interpol, opt_plot = opt_plot)
        savefig("./figs/" + idate + "-ssine_jitter_transfer_" + celltype + "_ihold" + str(sim.ihold) + "_amp" + str(sim.amp) + ".pdf", dpi = 300)  # save it
        

        print "Finished" 

        plt.show()
        plt.clf()  # delete figure 
        
    if "grc_IF" in do:
        
        sys.path.append('../NET/sheff/weasel/')

        fig = figure()
        
        gs = matplotlib.gridspec.GridSpec(1, 3,
                               width_ratios=[1,1,1],
                               height_ratios=[1]
                               )
        ax1 = subplot(gs[0,0])
        ax2 = subplot(gs[0,1])
        ax3 = subplot(gs[0,2])
                
        gs.update(bottom=0.63, wspace=0.2, hspace=0.3, top=0.97, left=0.1, right=0.90) 
        
        gs2 = matplotlib.gridspec.GridSpec(1, 1,
                               width_ratios=[1],
                               height_ratios=[1]
                               )
        gs2.update(bottom=0.17, wspace=0.1, hspace=0.1, top=0.45, left=0.1, right=0.90) 
        ax5 = subplot(gs2[0,0])
        
        cellimport = "from templates.granule.GRANULE_Cell import Grc"
        celltype = "Grc"
        cell_exe = "cell = Grc(np.array([0.,0.,0.]))" 
     
        prefix = "grc"   
                
        exec cellimport
        exec cell_exe  
        
        istart = 0.00568
        istop = 0.008
        di = 0.001
        
        sim = Stimulation(cell, temperature = 37)
        
        ie = 0.00567*nA
        sim.set_IClamp(ie, delay = 50*ms, dur = 800*ms)
        sim.run(1, tstop=1000*ms)
        t1, voltage, current, g_in = sim.get()
        ax1.plot(t1, voltage, linewidth=1)

        sim.stim = None
        
        # Step
        ie = 0.00568*nA
        sim.set_IClamp(ie, delay = 50*ms, dur = 800*ms)
        sim.run(1, tstop=1000*ms)
        t1, voltage, current, g_in = sim.get()
        ax2.plot(t1, voltage, linewidth=1)

        sim.stim = None
        
        # Step
        ie = 0.00569*nA
        sim.set_IClamp(ie, delay = 50*ms, dur = 800*ms)
        sim.run(1, tstop=1000*ms)
        t1, voltage, current, g_in = sim.get()
        ax3.plot(t1, voltage, linewidth=1)

        sim.stim = None
        
        # I/F Plot
                
        current_vector = arange(istart,istop,di)
        freq_vector = zeros(size(current_vector))
        freq_onset_vector = zeros(size(current_vector))
        num_spikes_vector = zeros(size(current_vector))
        
        for i, c in enumerate(current_vector): 
            sim.set_IClamp(c, delay = 100*ms, dur = 5000*ms)
            sim.run(1, tstop = 2100*ms)
            freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = sim.if_get()
            freq_vector[i] = mean(spike_freq)
            freq_onset_vector[i] = freq_onset
            num_spikes_vector[i] = len(spike_freq) 
            sim.c = c
        
        ax5.plot(current_vector, freq_vector)
        ax5.axis(xmin=0, xmax=current_vector[-1])
        
        plt.show()    

    
    if "grc_transfer_fit" in do:
        
        sys.path.append('../NET/sheff/weasel/')
        from scipy.optimize import fmin, leastsq
        from lmfit import minimize, Parameters, Parameter, report_errors

        ihold = 40
        xmax = 39
        fit_end = 20
        
        #xmax = 100
        #fit_end = 100
        
        amp = 0
        amod = 0.1
        
        istart = 0.004 
        istop = 0.02
        di = 0.001
        
        sexp = 0
        cutf = 0
        
        cellimport = "from templates.granule.GRANULE_Cell import Grc"
        celltype = "Grc"
        cell_exe = "cell = Grc(np.array([0.,0.,0.]))"    
        
        pickle_prefix = "cell_grc" 
        
        exec cellimport
        exec cell_exe   
        
        temperature = 37
        give_freq = True
        SNR = None 
        NI = None
        
        icloc = "soma(0.5)"

        pickle_prefix = pickle_prefix + "_transfer_fit"
                
        freq_used0 = concatenate(( arange(0.5, xmax+0.5, 0.5), array([]) )) #, arange(210, 1010, 10) ))
    
        sim = Stimulation(cell, celltype = celltype, cell_exe = cell_exe, temperature = temperature, do_run = do_run, pickle_prefix = pickle_prefix, give_freq = give_freq, istart = istart, istop = istop, di = di)
        if sim.id == 0: rm, cm, taum = sim.get_RCtau()
        
        sim.spikes_from_neuron = False
        
        sim.del_freq = array([20])
        sim.ihold = ihold
        sim.amp = amp
        sim.amod = amod

        method_interpol = np.array(["none"])  
                           
        results = sim.fun_ssine_Stim_pp(freq_used = freq_used0, method_interpol = method_interpol)
                
        if sim.id == 0:
            
            plt.figure(1)
            ax1 = plt.subplot(1,2,1)
            ax2 = plt.subplot(1,2,2)
            
            freq_used, vamp, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1') 
            freq_times, spike_freq, fmean, method_interpol, VAF = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('VAF')                   
    
            new_end = find(freq_used >= fit_end)[0]  # do not used all frequencies
            
            freq_used = freq_used[0:new_end]
            mag1 = mag[0,0:new_end] / mag[0,0]
            pha1 = pha[0,0:new_end]
            H_goal = (mag1 * exp(pi / 180 * 1j * pha1))
    
            #scale = 1/mag1[0]
            scale = 1
            ax1.semilogx(freq_used, 20*log10(mag1*scale), '-') # 20*log10
            ax2.semilogx(freq_used, pha1, '-')
                                  
            H, H0 = aiftransfer(freq_used, tau = 26.6*1e-3, f0 = ihold, i0 = 0)  
            magA = abs(H)
            phaA = angle(H, deg = True) # Transfer Phase in degree
            phaA = unwrap(angle(H)) * (180 / pi)
            scale = 1/magA[0]
            
            ax1.semilogx(freq_used, 20*log10(magA*scale), 'k--') # 20*log10
            ax2.semilogx(freq_used, phaA, 'k--')
            
            tau_fit, scale_fit, H_fit, delta_t_fit = fit_aiftransfer(freq_used, H_goal, f0 = ihold, i0 = 0)
            
            print "fit theor., tau=" + str(tau_fit/ms) + "ms, delta_t:", str(delta_t_fit/ms), "ms"

            magA = abs(H_fit)
            phaA = angle(H_fit, deg = True) # Transfer Phase in degree
            phaA = unwrap(angle(H_fit)) * (180 / pi)
            scale = 1/magA[0]
            
            ax1.semilogx(freq_used, 20*log10(magA*scale), 'r--') # 20*log10
            ax2.semilogx(freq_used, phaA, 'r--')
            
            plt.savefig("./figs/latest_grc_transfer.pdf", dpi = 300)  # save it  
            plt.clf()
            
        else:
            H_goal = None    
        
        H_goal = sim.comm.bcast(H_goal, root=0)
        sim.comm.Barrier()
            
        sim = None
        cell = None
        
        #CM = (9.76*1e-4)**2 * np.pi / cm = 2.9926057635859505e-06 uF = 3e-06 uF
        
        # OLD:
        # FIT TO MAG ONLY:
        # GrC resting: 737 MOhm, 2.985e-06 uF   tau: 0.0022 s, with GrC transfer fit: tau: 0.02129 s => with 3e-06 uF, R = 0.02129/3e-12 = 7097 MOhm    
        # e = -71.5*mV, thresh = -36*mV, vrefrac = -60*mV => thresh-e = 35.5*mV; i_reho_orig = 0.00568*nA
        
        # OLD, fit to MAG + PHASE: with GrC transfer fit: tau: 0.02658 s => with 3e-06 uF, R = 0.02658/3e-12 = 8860 MOhm
        
        # 1/R + gr = 1/737*MOhm + gr = 1/9045*MOhm => gr = 1/9045-1/737 = -0.0012462*uS
        # i_rheo = 1/9045*MOhm * 35.5*mV = 3.9248e-12 = 0.0039248*nA
        # i_reho_orig * 9045*MOhm = 51.37*mV = thresh-e => thresh = 51.37+-71.5 = -20.13
       
        gr = 5.56626611714e-05*uS 
        tau_r = 19.6497147661*ms
        R = 5226.95424101*MOhm
        
        delta_t = 4.85115500258*ms
        
        thresh = (0.00568*nA * R) - 71.5*mV #-31.18
        #tau_passive = 3e-06*5226.95424101 = 15.7ms
        
        p0 = array([tau_r, gr, R, delta_t])  # thresh, tau_r, gr 
          
   
        def peval(freq_used0, p):
            
            gr = p[1]
            tau_r = p[0]
            R = p[2] # 737*MOhm
            #R = 8860*MOhm
            #R = 7097*MOhm
            #gr2 = p[4]
            #tau_r2 = p[3] 
            delta_t = p[3]
     
            #gr = p['gr'].value
            #tau_r = p['tau_r'].value
            #R = p['R'].value
            #gr2 = p['gr2'].value
            #tau_r2 = p['tau_r2'].value
            
            #thresh = (0.00568*nA * (1/(1/(R) + gr))) - 71.5*mV # keep same rheobase!!!
            thresh = (0.00568*nA * R) - 71.5*mV
            
            if MPI.COMM_WORLD.rank == 0: print "thresh:", thresh, "gr:", gr, "tau_r:", tau_r, "R:", R
            
            celltype = "IfCell"
            #cell_exe = "cell = IfCell(C = 2.985e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -60*mV); cell.add_resonance_spike(tau_r =" + str(tau_r) + ", gr =" + str(gr) + ")" 
            cell_exe = "cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -71.5*mV, dgk =" + str(gr) + ", egk = -71.5*mV, ctau =" + str(tau_r) + ")"
            #cell_exe = "cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -71.5*mV, dgk =" + str(gr) + ", egk = -71.5*mV, ctau =" + str(tau_r) + ", dgk2 =" + str(gr2) + ", dtau =" + str(tau_r2) + ")"

            exec cell_exe
            
            istart = 0.004 
            istop = 0.02
            di = 0.0001
            give_freq = True
        
            sim = Stimulation(cell, celltype = celltype, cell_exe = cell_exe, temperature = 0, do_run = 1, give_freq = give_freq, istart = istart, istop = istop, di = di)
            sim.spikes_from_neuron = False
            sim.ihold = ihold
            sim.amp = amp
            sim.amod = amod
            sim.del_freq = array([20])
            sim.delta_t = delta_t

            method_interpol = np.array(["none"])  

            results = sim.fun_ssine_Stim_pp(freq_used = freq_used0, method_interpol = method_interpol)
                        
            sim = None
            cell = None

            return results
                        
        def residuals(p, H_goal, freq_used, fit_end):
            
            results = peval(freq_used, p)
            
            if MPI.COMM_WORLD.rank == 0: 

                freq_used, vamp, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1') 
                freq_times, spike_freq, fmean, method_interpol, VAF = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('VAF')                   
    
                new_end = find(freq_used >= fit_end)[0]  # do not used all frequencies
            
                freq_used = freq_used[0:new_end]
                mag1 = mag[0,0:new_end] / mag[0,0]
                pha1 = pha[0,0:new_end]
                H = (mag1 * exp(pi / 180 * 1j * pha1))

                #gr = p[1]
                #tau_r = p[0]
                #R = p[2] # 737*MOhm
                #R = 8860*MOhm              
                
                ## NEW:
                gr = p[1]
                tau_r = p[0]
                R = p[2] 
                #R = 7097*MOhm
                #gr2 = p[4]
                #tau_r2 = p[3]
                delta_t = p[3]
                
                #gr = p['gr'].value
                #tau_r = p['tau_r'].value
                #R = p['R'].value
                #gr2 = p['gr2'].value
                #tau_r2 = p['tau_r2'].value
                
                #thresh = (0.00568*nA * R) - 71.5*mV 
                #if thresh < -71.5*mV or thresh > 0*mV:
                #    H = H*0
                    
                thresh_goal = -36*mV 
                err_thresh = 0.1*abs(thresh_goal - thresh)
                    
                H_goal = H_goal[0:new_end]
                mag_goal = abs(H_goal)
                pha_goal = unwrap(angle(H_goal)) * (180 / pi)
                
                err_mag = (abs(H_goal) - abs(H))**2
                err_pha = ((unwrap(angle(H_goal)) - unwrap(angle(H))) * 180/pi)**2
                err =  sum( err_mag  +  err_pha  + err_thresh )
                err = sum( ( concatenate((real(H_goal), imag(H_goal), np.array([0.1*thresh_goal]))) - concatenate((real(H), imag(H), np.array([0.1*thresh]))) )**2 ) 
                err = sum( ( concatenate((real(H_goal), imag(H_goal) )) - concatenate((real(H), imag(H))) )**2 ) 
                #err = sum( err_mag )
                
                #err = ( concatenate((real(H_goal), imag(H_goal) )) - concatenate((real(H), imag(H))) )**2
            
                tau = 3e-06*uF * R
                HA, _ = aiftransfer(freq_used, tau = tau, f0 = 40, i0 = 0, delta_t = delta_t)
                
                print "TAU:",tau, p
                
                magA = abs(HA)/HA[0]
                phaA = unwrap(angle(HA)) * (180 / pi)

                plt.figure(2)
                ax1 = plt.subplot(1,3,1)
                ax2 = plt.subplot(1,3,2)
                ax3 = plt.subplot(1,3,3)
                
                ax1.semilogx(freq_used, 20*log10(mag1), 'g-') 
                ax2.semilogx(freq_used, pha1, 'g-')
                ax3.plot(concatenate((real(H), imag(H), np.array([0.1*thresh]))), 'g-')

                ax1.semilogx(freq_used, 20*log10(mag_goal), 'b-') 
                ax2.semilogx(freq_used, pha_goal, 'b-') 
                ax3.plot(concatenate((real(H_goal), imag(H_goal), np.array([0.1*thresh_goal]))), 'b-')
                
                ax1.semilogx(freq_used, 20*log10(magA), 'k--') 
                ax2.semilogx(freq_used, phaA, 'k--')
                ax3.plot(concatenate((real(HA), imag(HA))), 'k--')
                
                #ax1.semilogx(freq_used, 20*log10(err_mag), 'r-') 
                #ax2.semilogx(freq_used, err_pha, 'r-') 

                plt.savefig("./figs/latest_grc_transfer_fit.pdf", dpi = 300)  # save it  
                plt.clf()              
            
            else:
                err = None
            
            err = MPI.COMM_WORLD.bcast(err, root=0)
            MPI.COMM_WORLD.Barrier()

            return err
        
        # create a set of Parameters
        #p = Parameters()
        #p.add('tau_r',   value = tau_r,  min = 0, max = 50*ms)
        #p.add('gr', value = gr,  min = 0, max = 1*uS)
        #p.add('R', value = R,  min = 100, max = 10000*MOhm) 
        #p.add('tau_r2',   value = tau_r2,  min = 0, max = 50*ms)
        #p.add('gr2', value = gr2,  min = -1*uS, max = 0*uS)
    
        np.random.seed(seed=333)    
        #result = minimize(residuals, p, args=(H_goal, freq_used0, fit_end), method='leastsq') # 'nelder'
        
        plsq = fmin(residuals, p0, args=(H_goal, freq_used0, fit_end)) # fmin
         
        p = plsq
        
        #gr = p[1]
        #tau_r = p[0]
        #R = p[2] # 737*MOhm
        #R = 8860*MOhm
        #thresh = (0.00568*nA * (1/(1/(R) + gr))) - 71.5*mV # keep same rheobase!!!
        #thresh = (0.00568*nA * R) - 71.5*mV
        #thresh = -20.13*mV
        
        gr = p[1]
        tau_r = p[0]
        R = p[2] # 737*MOhm
        #R = 7097*MOhm
        #gr2 = p[4]
        #tau_r2 = p[3]
        delta_t = p[3]
        
        #thresh = (0.00568*nA * (1/(1/(R) + gr))) - 71.5*mV # keep same rheobase!!!
        thresh = (0.00568*nA * R) - 71.5*mV
        #thresh = -20.13*mV
        
        if MPI.COMM_WORLD.rank == 0: print "FINAL: thresh:", thresh, "gr:", gr, "tau_r:", tau_r, "R:", R, ",delta_t:", delta_t #, "gr2:", gr2, "tau_r2:", tau_r2
        
        #if MPI.COMM_WORLD.rank == 0: 
        #    print "FINAL: ", p
        #    report_errors(p)
        
        
    if "grc_cnoise_transfer_fit" in do:
        
        t_stim = 200*s
        
        sys.path.append('../NET/sheff/weasel/')
        from scipy.optimize import fmin, leastsq

        ihold = 40
        
        xmax = 39
        fit_end = 39
        
        xmax = 20
        fit_end = 20
        
        amp = 0
        amod = 0.1
        
        istart = 0.004 
        istop = 0.02
        di = 0.001
        
        sexp = 0
        cutf = 0
        
        cutf = 20
        sexp = -1
        
        cellimport = "from templates.granule.GRANULE_Cell import Grc"
        celltype = "Grc"
        cell_exe = "cell = Grc(np.array([0.,0.,0.]))"    
        
        pickle_prefix = "cell_grc" 
        
        exec cellimport
        exec cell_exe   
        
        temperature = 37
        give_freq = True
        SNR = None 
        NI = None
        
        icloc = "soma(0.5)"

        pickle_prefix = pickle_prefix + "_transfer_fit"

        sim = Stimulation(cell, celltype = celltype, cell_exe = cell_exe, temperature = temperature, do_run = do_run, pickle_prefix = pickle_prefix, give_freq = give_freq, istart = istart, istop = istop, di = di)
        sim.spikes_from_neuron = False
        sim.ihold = ihold
        sim.amp = amp
        sim.amod = amod

        method_interpol = np.array(["dt"])         
        results = sim.fun_cnoise_Stim(t_stim = t_stim, sexp = sexp, cutf = cutf, method_interpol = method_interpol)
                        
        if sim.id == 0:
            
            plt.figure(1)
            ax1 = plt.subplot(1,3,1)
            ax2 = plt.subplot(1,3,2)
            ax3 = plt.subplot(1,3,3)
            
            freq_used, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1') 
            freq_times, spike_freq, fmean, method_interpol, SNR, VAF = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('SNR'), results.get('VAF')                      
        
            new_end = find(freq_used >= fit_end)[0]  # do not used all frequencies
            
            freq_used = freq_used[0:new_end]
            mag1 = mag[0,0:new_end] / mag[0,0]
            pha1 = pha[0,0:new_end]
            H_goal = (mag1 * exp(pi / 180 * 1j * pha1))
    
            scale = 1
            #scale = 1/mag1[0]
            ax1.semilogx(freq_used, 20*log10(mag1*scale), '-') # 20*log10
            ax2.semilogx(freq_used, pha1, '-')
            ax3.plot(concatenate((real(H_goal), imag(H_goal))), '-')          
                        
            plt.savefig("./figs/latest_grc_cnoise_transfer.pdf", dpi = 300)  # save it  
            plt.clf()
            
        else:
            H_goal = None    
        
        H_goal = sim.comm.bcast(H_goal, root=0)
        sim.comm.Barrier()
            
        sim = None
        cell = None


        gr = 5.78815e-05*uS 
        tau_r = 0.018916
        R = 8860*MOhm
        thresh = (0.00568*nA * R) - 71.5*mV #-21.175
        #tau_passive = 3e-06*8860 = 26.58ms
        
        gr2 = -10e-05*uS 
        tau_r2 = 0.005
        
        p0 = array([tau_r, gr, R, tau_r2, gr2])  # thresh, tau_r, gr 
          
   
        def peval(p, cutf, sexp, t_stim):
            
            gr = p[1]
            tau_r = p[0]
            R = p[2] # 737*MOhm
            #R = 8860*MOhm
            gr2 = p[4]
            tau_r2 = p[3]
            
            #thresh = (0.00568*nA * (1/(1/(R) + gr))) - 71.5*mV # keep same rheobase!!!
            thresh = (0.00568*nA * R) - 71.5*mV
            
            if MPI.COMM_WORLD.rank == 0: print "thresh:", thresh, "gr:", gr, "tau_r:", tau_r, "R:", R, "gr2:", gr2, "tau_r2:", tau_r2
            
            celltype = "IfCell"
            #cell_exe = "cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -60*mV, dgk =" + str(gr) + ", egk = -71.5*mV, ctau =" + str(tau_r) + ")"
            cell_exe = "cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -71.5*mV, dgk =" + str(gr) + ", egk = -71.5*mV, ctau =" + str(tau_r) + ", dgk2 =" + str(gr2) + ", dtau =" + str(tau_r2) + ")"

            exec cell_exe
            
            istart = 0.003 
            istop = 0.02
            di = 0.001
            give_freq = True
           
            sim = Stimulation(cell, celltype = celltype, cell_exe = cell_exe, temperature = 0, do_run = 1, give_freq = give_freq, istart = istart, istop = istop, di = di)
            sim.spikes_from_neuron = False
            sim.ihold = ihold
            sim.amp = amp
            sim.amod = amod
            
            method_interpol = np.array(["dt"])         
            results = sim.fun_cnoise_Stim(t_stim = t_stim, sexp = sexp, cutf = cutf, method_interpol = method_interpol)
            
            sim = None
            cell = None

            return results
                        
        def residuals(p, H_goal, fit_end, cutf, sexp, t_stim):
            
            results = peval(p, cutf, sexp, t_stim)
            
            if MPI.COMM_WORLD.rank == 0: 

                freq_used, vamp, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1') 
                freq_times, spike_freq, fmean, method_interpol, VAF = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('VAF')                   
    
                new_end = find(freq_used >= fit_end)[0]  # do not used all frequencies
            
                freq_used = freq_used[0:new_end]
                mag1 = mag[0,0:new_end] / mag[0,0]
                pha1 = pha[0,0:new_end]
                H = (mag1 * exp(pi / 180 * 1j * pha1))

                ## NEW:
                gr = p[1]
                tau_r = p[0]
                R = p[2]
                gr2 = p[4]
                tau_r2 = p[3]
                
                thresh = (0.00568*nA * R) - 71.5*mV 
                
                #if thresh < -71.5*mV or thresh > 0*mV:
                #    H = H*0
                    
                H_goal = H_goal[0:new_end]
                mag_goal = abs(H_goal)
                pha_goal = unwrap(angle(H_goal)) * (180 / pi)

                #err = sum( abs(H_goal - H)**2 )
                err = sum( ( concatenate((real(H_goal), imag(H_goal))) - concatenate((real(H), imag(H))) )**2 )
                
                tau = 3e-06*uF * R
                H1, H0 = aiftransfer(freq_used, tau = tau, f0 = 40)            
                HA = H1/H0
                #HA = HA / H[0]

                print "TAU:",tau
                
                magA = abs(HA)/HA[0]
                phaA = unwrap(angle(HA)) * (180 / pi)


                plt.figure(2)
                ax1 = plt.subplot(1,3,1)
                ax2 = plt.subplot(1,3,2)
                ax3 = plt.subplot(1,3,3)
                
                ax1.semilogx(freq_used, 20*log10(mag1), 'g-') 
                ax2.semilogx(freq_used, pha1, 'g-')
                ax3.plot(concatenate((real(H), imag(H))), 'g-')

                ax1.semilogx(freq_used, 20*log10(mag_goal), 'b-') 
                ax2.semilogx(freq_used, pha_goal, 'b-') 
                ax3.plot(concatenate((real(H_goal), imag(H_goal))), 'b-')
                
                ax1.semilogx(freq_used, 20*log10(magA), 'k--') 
                ax2.semilogx(freq_used, phaA, 'k--')
                ax3.plot(concatenate((real(HA), imag(HA))), 'k--')
                
                plt.savefig("./figs/latest_grc_cnoise_transfer_fit.pdf", dpi = 300)  # save it  
                plt.clf()
            
            else:
                err = None
            
            err = MPI.COMM_WORLD.bcast(err, root=0)
            MPI.COMM_WORLD.Barrier()

            return err
        
        t_stim = 100*s
        
        plsq = fmin(residuals, p0, args=(H_goal, fit_end, cutf, sexp, t_stim)) # fmin
        p = plsq
        gr = p[1]
        tau_r = p[0]
        R = p[2] # 737*MOhm
        gr2 = p[4]
        tau_r2 = p[3]
        
        thresh = (0.00568*nA * R) - 71.5*mV

        if MPI.COMM_WORLD.rank == 0: print "FINAL: thresh:", thresh, "gr:", gr, "tau_r:", tau_r, "R:", R, "gr2:", gr2, "tau_r2:", tau_r2        
        

    if "iftransfer_syn" in do:
        
        t_stim = 100*s
        
        celltype = "IfCell"   
        cell_exe = "cell = IfCell()"   
        exec cell_exe                 
        
        sim = Stimulation(cell, celltype = celltype, cell_exe = cell_exe, temperature = 0, do_run = do_run, pickle_prefix = pickle_prefix)
        
        sim.ihold = 50
        sim.amod = 0.5       
        
        sim.give_freq = True
        
        sim.syn_n = 4
        sim.syn_g = 0.001
        sim.syn_noise = 0
        sim.syn_tau1 = 0*ms
        sim.syn_tau2 = 50*ms
        
        sexp = 0
        cutf = 0    

        from datetime import datetime
        idate = datetime.now().strftime('%Y%m%d_%H%M')  # %S
        suptitle("Transfer function of a " + celltype)    
        
        plt.figure(1)
        currtitle = "colored noise, " + celltype + ", amod: " + str(sim.amod) + ", ihold: " + str(sim.ihold) + ", syn_n: " + str(sim.syn_n) + ", syn_g: " + str(sim.syn_g) + ", syn_noise: " + str(sim.syn_noise) + ", syn_tau: " + str(sim.syn_tau1) + "/" + str(sim.syn_tau2)
        
        opt_plot = np.array(["only_mag", "normalize", "analytical", "if", "loglog"]) 
        #opt_plot = np.array(["analytical", "if"]) 
        
        sexp = 0; cutf = 0
        method_interpol = np.array(["linear", "dt"])
        sim.fun_plot(currtitle, "cnoise", freq_used = freq_used, t_stim = t_stim, SNR = 0, method_interpol = method_interpol, sexp = sexp, cutf = cutf, opt_plot = opt_plot)
        savefig("./figs/" + idate + "-cnoise_transfer_syn_" + celltype + "_ihold" + str(sim.ihold) + "_amp" + str(sim.amp) + ".pdf", dpi = 300)  # save it          
        
        plt.figure(2)        
        currtitle = "single sine, " + celltype + ", amod: " + str(sim.amod) + ", ihold: " + str(sim.ihold) 
        
        method_interpol = np.array(["none", "linear", "dt"])
        sim.fun_plot(currtitle, "ssine", freq_used = freq_used, VAF = 0, method_interpol = method_interpol, opt_plot = opt_plot)
        savefig("./figs/" + idate + "-ssine_transfer_syn_" + celltype + "_ihold" + str(sim.ihold) + "_amp" + str(sim.amp) + ".pdf", dpi = 300)  # save it
        
        
        print "Finished" 

        plt.show()
        plt.clf()  # delete figure  
        
        
    if "grc_transfer_syn" in do:
        
        sys.path.append('../NET/sheff/weasel/')
        
        freq_used = array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 50, 60, 70, 80, 90, 100, 1000])
        freq_used = array([1, 10, 100, 1000])
        
        t_stim = 100*s
        
        cellimport = "from templates.granule.GRANULE_Cell import Grc"
        celltype = "Grc"
        cell_exe = "cell = Grc(np.array([0.,0.,0.]))"   
                
        exec cellimport
        exec cell_exe   
        
        temperature = 37
        give_freq = True
        SNR = None 
        NI = None  

        sexp = 2
        cutf = 0  

        sexp = -1
        cutf = 20            
        
        sim = Stimulation(cell, celltype = celltype, cell_exe = cell_exe, temperature = temperature, do_run = do_run, pickle_prefix = pickle_prefix)
        
        sim.ihold = 50
        sim.amod = 1       
        

        # Excitation
        sim.syn_n = 4
        sim.syn_noise = 0
        
        # Inhibition
        sim.inh_n = 4
        sim.inh_hold = 5
        sim.inh_g = 1
        sim.inh_noise = 0
               

        from datetime import datetime
        idate = datetime.now().strftime('%Y%m%d_%H%M')  # %S
        suptitle("Transfer function of a " + celltype)    
        
        plt.figure(1)
        currtitle = "colored noise, " + celltype + ", amp: " + str(sim.amp) + ", ihold: " + str(sim.ihold) + ", syn_n: " + str(sim.syn_n) + ", syn_g: " + str(sim.syn_g) + ", syn_noise: " + str(sim.syn_noise) + ", syn_tau: " + str(sim.syn_tau1) + "/" + str(sim.syn_tau2)
        
        opt_plot = np.array(["only_mag", "normalize", "analytical", "loglog"]) 
        opt_plot = np.array(["normalize", "analytical", "loglog"]) 
        
        method_interpol = np.array(["linear", "dt"])
        sim.fun_plot(currtitle, "cnoise", freq_used = freq_used, t_stim = t_stim, SNR = 0, method_interpol = method_interpol, sexp = sexp, cutf = cutf, opt_plot = opt_plot)
        savefig("./figs/" + idate + "-cnoise_transfer_syn_" + celltype + "_ihold" + str(sim.ihold) + "_amp" + str(sim.amp) + ".pdf", dpi = 300)  # save it          
        
        plt.figure(2)        
        currtitle = "single sine, " + celltype + ", amp: " + str(sim.amp) + ", ihold: " + str(sim.ihold) 
        
        method_interpol = np.array(["none", "linear", "dt"])
        sim.fun_plot(currtitle, "ssine", freq_used = freq_used, VAF = 0, method_interpol = method_interpol, opt_plot = opt_plot)
        savefig("./figs/" + idate + "-ssine_transfer_syn_" + celltype + "_ihold" + str(sim.ihold) + "_amp" + str(sim.amp) + ".pdf", dpi = 300)  # save it
        
        
        print "Finished" 

        plt.show()
        plt.clf()  # delete figure  
        
        
    if "stim_reconstruct_jitter" in do:  
        
        celltype = "IfCell"                 
        from IfCell import *
        cell = IfCell()
                
        sim = Stimulation(cell, temperature = 6.3)
        sim.seed = 33
        
        method_interpol = array(["linear", "dt", "syn"])
        #method_interpol = array(["linear_ds", "binary"]) 
                
        sim.ihold = 0.14*nA 
        sim.amp = 0.005*nA
        
        t_stim = 10*s
        t_kernel = 1*s  # use full kernel length
        t_qual = 5*s
                
        sexp = -1
        cutf = 20
        
        currtitle = "colored noise, " + celltype + ", amp: " + str(sim.amp) + ", ihold: " + str(sim.ihold) 
        
        w_length = 1*s
        sim.jitter = 2*ms
        sim.stim_reconstruct(t_stim = t_stim, sexp = sexp, cutf = cutf, w_length = w_length, t_kernel = t_kernel, t_qual = t_qual, method_interpol = method_interpol)  
        
        w_length = 1*s
        sim.jitter = 0*ms
        sim.stim_reconstruct(t_stim = t_stim, sexp = sexp, cutf = cutf, w_length = w_length, t_kernel = t_kernel, t_qual = t_qual, method_interpol = method_interpol)  
        
        plt.show()
        plt.clf()  # delete figure  
        
        cell = None
        sim = None  
        
    
    if "stim_reconstruct" in do:
        
        celltype = "IfCell"                 
        from cells.IfCell import *
        cell = IfCell()
                
        sim = Stimulation(cell, temperature = 6.3)
        
        rm, cm, tau = sim.get_RCtau()
        
        method_interpol = array(["dslin"])
        method_interpol = array(["binary"]) 
        method_interpol = array(["linear", "dt"]) 
        #method_interpol = array(["dt"]) 
        
        sim.ihold = 0.14*nA 
        sim.amp = 0.005*nA
        
        t_stim = 400*s
        t_kernel = 1*s  # use full kernel length
        t_qual = 5*s
                
        currtitle = "colored noise, " + celltype + ", amp: " + str(sim.amp) + ", ihold: " + str(sim.ihold) 
        
        w_length = 1*s
        sim.jitter = 0*ms
        
        sexp = -1
        
        cutf = None
        sim.stim_reconstruct(t_stim = t_stim, sexp = sexp, cutf = cutf, w_length = w_length, t_kernel = t_kernel, t_qual = t_qual, method_interpol = method_interpol, tau = tau)  
        
        #cutf = 1000
        #sim.stim_reconstruct(t_stim = t_stim, sexp = sexp, cutf = cutf, w_length = w_length, t_kernel = t_kernel, t_qual = t_qual, method_interpol = method_interpol)  
        
        #cutf = 100
        #sim.stim_reconstruct(t_stim = t_stim, sexp = sexp, cutf = cutf, w_length = w_length, t_kernel = t_kernel, t_qual = t_qual, method_interpol = method_interpol)  
        
        #cutf = 50
        #sim.stim_reconstruct(t_stim = t_stim, sexp = sexp, cutf = cutf, w_length = w_length, t_kernel = t_kernel, t_qual = t_qual, method_interpol = method_interpol)  
        
        #cutf = 10
        #sim.stim_reconstruct(t_stim = t_stim, sexp = sexp, cutf = cutf, w_length = w_length, t_kernel = t_kernel, t_qual = t_qual, method_interpol = method_interpol)  
        
        plt.show()
        plt.clf()  # delete figure  
        
        cell = None
        sim = None  
        
        
    if "CF" in do:
        
        celltype = "IfCell"                 
        from IfCell import *
        cell = IfCell()
                
        sim = Stimulation(cell, temperature = 6.3)
        
        method_interpol = array(["linear_ds"])
        method_interpol = array(["binary"]) 
        method_interpol = array(["dt", "linear", "syn"]) 
        
        sim.ihold = 0.14*nA 
        sim.amp = 0.005*nA
        
        t_stim = 20*s
        t_kernel = 1*s  # use full kernel length
        t_qual = 5*s
                
        currtitle = "colored noise, " + celltype + ", amp: " + str(sim.amp) + ", ihold: " + str(sim.ihold) 
        
        w_length = 1*s
        sim.jitter = 0*ms
        sexp = -1

        cutf_vec = arange(1,500,1)
        CF_vec = zeros([len(method_interpol),len(cutf_vec)])  

        for i, cutf in enumerate(cutf_vec):
            CF_mat = sim.stim_reconstruct(t_stim = t_stim, sexp = sexp, cutf = cutf, w_length = w_length, t_kernel = t_kernel, t_qual = t_qual, method_interpol = method_interpol)  
            
            for j in range(len(method_interpol)):
                CF_vec[j,i] = CF_mat[j]
                print "interp: " + method_interpol[j] + ", cutf: " + str(cutf) + ", CF :" + str(CF_mat[j])  
                
            if all(CF_mat < 0.05):
                print "CF < 0.05"
                break
        
        plt.figure(5)
        plt.semilogx(cutf_vec, CF_vec[0,:], 'b')
        plt.semilogx(cutf_vec, CF_vec[1,:], 'r')
            
        plt.show()
                
        cell = None
        sim = None  
        
        
    if "tau_if" in do:  
    
        from NeuroTools import stgen
        import neuronpy.util.spiketrain
        
        celltype = "IfCell"                 
        from cells.IfCell import *
        cell = IfCell()
                
        sim = Stimulation(cell, temperature = 6.3)
         
        # has to be in ms!!        
        start_time = 100
        duration = 1500
        dt = 0.025
        steadyf = 100
        pulsef = 150
        pulse_start = 500
        pulse_len = 500            
        t_input = np.arange(0, duration, dt) # create stimulus time vector
        mod = np.concatenate(([np.zeros(start_time/dt), steadyf*np.ones((pulse_start-start_time)/dt), pulsef*np.ones(pulse_len/dt),steadyf*np.ones((duration-pulse_start-pulse_len)/dt)  ])) 
        
        modulation = (t_input, mod)
        
        train = spike_train(frequency=modulation, seed=1, noise=0, jitter=0)
        #train, u = if_spike_train(frequency=modulation, seed=1, noise=50)
        
        #gen = stgen.StGen(rng = np.random.mtrand.RandomState(), seed=1)
        #train = gen.inh_poisson_generator(rate=mod, t=t_input, t_stop=t_input[-1], array=True)
        
        cell.create_synapses(n_ex=1, tau1=0*ms, tau2=10*ms)
        
        vecstim = h.VecStim(.5)
        nc_vectim = h.NetCon(vecstim,None)
        spike_vec = h.Vector(train)
        vecstim.play(spike_vec) 
        
        nc = h.NetCon(vecstim, cell.synlist[0])
        
        nc.weight[0] = 0.002
        nc.delay = 0
    
        sim.run(do_freq = 1, tstop = 1500*ms)
        
        t1, voltage, current, g_in = sim.get()
        freq_times, spike_freq, freq_mean, freq_onset, spike_times, gsyn = sim.if_get(compute_mean = 0)
        
        plt.plot(freq_times, spike_freq)
        
        plt.show()
        plt.clf()  # delete figure  
        
        

            


    #print "Finished"
    #plt.show()