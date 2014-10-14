# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 20:06:39 2011

@author: chris
"""

from __future__ import division
import numpy as np
from neuron import h
from units import *

class Cell(object):
    """
    Generic cell template for NEURON cell objects.
    """
    
    def __init__(self):      
        
        self.synlist = []
        self.synlist_inh = []
        self.nc = []
        self.nc_inh = []
        #self.gid = []
        
        #self.spike = h.NetStim(0.5, sec= self.soma)
        #self.spike.start = -10
        #self.spike.number = 1
        #self.spike.interval = 1e9
        
        #self.nc_spike = h.NetCon(self.soma(1)._ref_v, self.spike, self.thresh, 0, 1, sec = self.soma)
        
        self.nc_spike = h.NetCon(self.soma(1)._ref_v, None, self.thresh, 1, 1, sec = self.soma)
        self.nc_syn = []
        
    def start_record(self, tau1 = 0, tau2 = 0, thresh = None):
    
        if thresh == None:   # if threshold is given use it, if not, use the cells internal variable
            threshold = self.thresh
        else:
            threshold = thresh
        
        self.record = {}
        
        # Synaptic integrated response
        if tau2 > 0:        
            self.record['gsyn'] = h.Vector()
            self.fake_cell = h.Section()
            self.syn = h.Exp2Syn(self.fake_cell(0.5))
            self.syn.tau1 = tau1
            self.syn.tau2 = tau2
            self.syn.e = 0
            self.nc_syn = h.NetCon(self.soma(1)._ref_v, self.syn,threshold,1,1, sec = self.soma)
            self.record['gsyn'].record(self.syn._ref_g) 
        
        
        self.record['w_inh'] = []
        self.record['w'] = []  
        
        if len(self.synlist) > 0:
            if hasattr(self.synlist[0], 'w'):
                for i in range(len(self.synlist)):
                    self.record['w'].append(h.Vector())
                    self.record['w'][-1].record(self.synlist[i]._ref_w)
        
        if len(self.synlist_inh) > 0:
            if hasattr(self.synlist_inh[0], 'w'):
                for i in range(len(self.synlist_inh)):
                    self.record['w_inh'].append(h.Vector())
                    self.record['w_inh'][-1].record(self.synlist_inh[i]._ref_w)
            
        
                 
        
    #Synapses    
    def create_synapses(self, n_ex=1, tau1=1*ms, tau2=2*ms, n_inh=0, tau1_inh=1*ms, tau2_inh=2*ms, e_inh=-90, w = 0, wmax = 0, taupre = 0, taupost = 0, apre = 0, apost = 0, tend = 1e9, e_ex=0): 
        """
        Crate a new synapse targeting this cells membrane
        """
        
        if n_inh == 0:
            # Excitation
            for i in range(n_ex):
                if tau1 == 0:
                    syn = h.ExpSyn(self.soma(0.5))
                    syn.tau = tau2/ms
                else: 
                    if wmax == 0:
                        syn = h.Exp2Syn(self.soma(0.5))
                        syn.tau1 = tau1/ms
                        syn.tau2 = tau2/ms

                    else: # STDP
                        syn = h.stdpE2S(self.soma(0.5))
                        syn.tau1 = tau1/ms
                        syn.tau2 = tau2/ms
                        
                        syn.on = 1
                        syn.thresh = self.thresh
                        
                        syn.wmax = wmax
                        syn.w = w
                        
                        syn.taupre  = taupre/ms	
                        syn.taupost  = taupost/ms
                        syn.apre    = apre
                        syn.apost   = apost
                        syn.tend = tend/ms 


                syn.e = e_ex/mV
                self.synlist.append(syn) # synlist is defined in Cell 
                return self.synlist[-1]
                
        else:	
            #Inibition
            for i in range(n_inh):
                if tau1_inh == 0:
                    syn = h.ExpSyn(self.soma(0.5))
                    syn.tau = tau2_inh/ms
                
                else:                            
                    if wmax == 0:
                        syn = h.Exp2Syn(self.soma(0.5))
                        syn.tau1 = tau1_inh/ms
                        syn.tau2 = tau2_inh/ms

                    else: # STDP
                        
                        syn = h.stdpE2S(self.soma(0.5))
                        syn.tau1 = tau1_inh/ms
                        syn.tau2 = tau2_inh/ms
                        
                        syn.on = 1
                        syn.thresh = self.thresh
                        
                        syn.wmax = wmax
                        syn.w = w
                        
                        syn.taupre  = taupre/ms	
                        syn.taupost  = taupost/ms
                        syn.apre    = apre
                        syn.apost   = apost
                        syn.tend = tend/ms 
                    
                syn.e = e_inh/mV
                self.synlist_inh.append(syn) # synlist is defined in Cell 
                return self.synlist_inh[-1]
                

    #Make Netcon  
    def connect_target(self, target, thresh=None, weight=1, delay=0):
        """
        Make a new NetCon with this cell's membrane
        potential at the soma as the source (i.e. the spike detector)
        onto the target passed in (i.e. a synapse on a cell).
        Subclasses may override with other spike detectors.
        """
        
        nc = h.NetCon(self.soma(0.5)._ref_v, target, sec = self.soma)
        nc.weight[0] = weight
        nc.delay = delay
        
        if thresh == None:   # if threshold is given use it, if not, use the cells internal variable
            nc.threshold = self.thresh
        else:
            nc.threshold = thresh
            
        return nc
        
        
    #Parallell        
    def pconnect_target(self, pc, source=0, target=0, syntype='ex', thresh=None, weight=1, delay=0):
        """
        Make a new NetCon with this cell's membrane
        Parallel version
        """
        
        if syntype == 'ex':
            source = int(source)
            self.nc.append(pc.gid_connect(source, self.synlist[target]))
            self.nc[-1].delay = delay
            self.nc[-1].weight[0] = weight
            
            if thresh != None:  # change threshold, if thresh == None threshold used in source before will be used (?)   
                self.nc[-1].threshold = self.thresh
                        
            return self.nc[-1]
            
        if syntype == 'inh':
            source = int(source)
            self.nc_inh.append(pc.gid_connect(source, self.synlist_inh[target]))
            self.nc_inh[-1].delay = delay
            self.nc_inh[-1].weight[0] = weight
            
            if thresh != None:  # change threshold, if thresh == None threshold used in source before will be used (?)   
                self.nc_inh[-1].threshold = self.thresh
                            
            return self.nc_inh[-1]
            

    def destroy(self):
        
        del self.nc_spike
        del self.nc_syn
        
        for m in self.synlist:
            del m
        for m in self.synlist_inh:
            del m
        for m in self.nc:
            del m
        for m in self.nc_inh:
            del m
        del self
    
    