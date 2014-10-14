# -*- coding: utf-8 -*-
from neuron import h
import random as rnd
from synapse.synapse import Synapse
import numpy as np

class Grc:
    def __init__(self,position, record_all = 0, sigma_L = 0, gid = 0, lkg2_noise=0, lkg2_gbar=6e-5):
        self.record_all = record_all
        if record_all:
            print "Recording all in Grc"
     
        self.soma = h.Section(cell=self)
        self.soma.nseg = 1 
 
        L = 9.76 #um
        #(9.76e-6)**2 * 3.14159265359 * 1e-6 / (1e-2)**2 = 3 pF  
      
        self.gid = gid

        if sigma_L == 0:
            self.soma.L = L
            self.soma.diam = L
        else:
            np.random.seed(self.gid*40)
            self.soma.L = np.random.normal(L, L*sigma_L, 1).clip(min=L*sigma_L) 
            self.soma.diam = self.soma.L
            print "soma.L:", self.soma.L
 
        self.soma.cm = 1

        self.soma.Ra = 100
        # h.celsius = 37

        self.lkg2_noise = lkg2_noise
        self.whatami = "grc"
                
        self.soma.push()
        h.pt3dclear()
        h.pt3dadd(position.item(0), position.item(1) - self.soma.L, position.item(2), self.soma.diam)
        h.pt3dadd(position.item(0), position.item(1) + self.soma.L, position.item(2), self.soma.diam)
        h.pop_section()
        
        self.record = {}
        self.record['L'] = h.Vector(np.array([self.soma.L]))
        self.record['position'] = h.Vector(position)

        self.soma.insert('GRANULE_LKG1')
        
        if lkg2_noise > 0:
            print "noise"
            self.noise = h.Random()  # provides NOISE with random stream            
            self.fluct = h.GRANULE_LKG2_noise(self.soma(0.5))
            self.fluct.gbar = lkg2_gbar
            self.fluct.std_i = lkg2_noise
            self.fluct.tau_i = 100
            self.fluct.noiseFromRandom(self.noise)  # connect random generator!
            
            self.noise.MCellRan4(1, gid+1)  # set lowindex to gid+1, set highindex to > 0 
            self.noise.normal(0,1)
        else:
            self.soma.insert('GRANULE_LKG2')
            #self.soma(0.5).GRANULE_LKG2.gbar = 6e-5
            
        self.soma.insert('GRANULE_Nmda_leak')
        self.soma.insert('GRANULE_NA')
        self.soma.insert('GRANULE_NAR')
        self.soma.insert('GRANULE_PNA')
        self.soma.insert('GRANULE_KV')
        self.soma.insert('GRANULE_KA')
        self.soma.insert('GRANULE_KIR')
        self.soma.insert('GRANULE_KCA')
        self.soma.insert('GRANULE_KM')
        self.soma.insert('GRANULE_CA')
        self.soma.insert('GRANULE_CALC')

        h.usetable_GRANULE_NA = 1
        h.usetable_GRANULE_NAR = 1
        h.usetable_GRANULE_PNA = 1
        h.usetable_GRANULE_KV  = 1
        h.usetable_GRANULE_KA = 1
        h.usetable_GRANULE_KIR = 1
        h.usetable_GRANULE_KCA = 0
        h.usetable_GRANULE_KM = 1
        h.usetable_GRANULE_CA = 1

        self.soma.ena = 87.39
        self.soma.ek = -84.69
        self.soma.eca = 129.33

        self.MF_L = []
        self.GOC_L = []	
        self.mfncpc = []
        self.gocncpc = []

        self.spike = h.NetStim(0.5, sec= self.soma)
        self.spike.start = -10
        self.spike.number = 1
        self.spike.interval = 1e9

        self.nc_spike = h.NetCon(self.soma(1)._ref_v, self.spike,-20,1,1, sec = self.soma)


        self.record['spk'] = h.Vector()
        self.nc_spike.record(self.record['spk'])

        if self.record_all:
            self.record['vm'] = h.Vector()
            self.record['vm'].record(self.soma(.5)._ref_v, sec = self.soma)
            self.record['time'] = h.Vector()
            self.record['time'].record(h._ref_t)
            
        self.nc_syn = []

    #Synapses
    def createsyn(self,nmf=0,nrel = 0,ngoc = -1, weight_var = 0, weight = 1, syntype = 'ANK', weight_gmax_var = 0, weight_gmax = 1, record_all = 0):
        # Use here the source target sting name
        # so the presynaptic link is not made
        # and it will have to be manged later
        # by the gid connect for parallel simulations
        #Mossy
        if ngoc <0 :
            ngoc = nmf
            
        w = weight
        g = weight_gmax
        

        for i in range(nmf):
            
            if weight_var > 0:
                np.random.seed(1*self.gid*(i+1))  
                w = np.random.normal(weight, weight_var, 1).clip(min=0) 
                
            if weight_gmax_var > 0:
                np.random.seed(1*self.gid*(i+1))  
                g = np.random.normal(weight_gmax, weight_gmax_var, 1).clip(min=0) 
                
            self.MF_L.append(Synapse('glom',self,self.soma,nrel,record_all=record_all,weight=w,weight_gmax=g,syntype=syntype))
            
            #Inibition
            # print "ngoc ", ngoc
        for i in range(ngoc):
            
            if weight_var > 0:
                np.random.seed(2*self.gid*(i+1))  
                w = np.random.normal(weight, weight_var, 1).clip(min=0) 
                
            if weight_gmax_var > 0:
                np.random.seed(2*self.gid*(i+1))  
                g = np.random.normal(weight_gmax, weight_gmax_var, 1).clip(min=0) 
                
            self.GOC_L.append(Synapse('goc',self,self.soma,nrel,record_all=record_all,weight=w,weight_gmax=g,syntype=syntype))

	
    def pconnect(self,pc,source,syn_idx,type_syn):
        if type_syn == 'mf':
            source = int(source)
            self.mfncpc.append(pc.gid_connect(source, self.MF_L[syn_idx].input))
            self.mfncpc[-1].delay = 1
            self.mfncpc[-1].weight[0] = 1
            return self.mfncpc[-1]
        if type_syn == 'goc':
            source = int(source)
            # print "syn_len ", len(self.GOC_L), syn_idx, self.whatami, source
            self.gocncpc.append(pc.gid_connect(source, self.GOC_L[syn_idx].input))
            self.gocncpc[-1].delay = 1
            self.gocncpc[-1].weight[0] = 1
            return self.gocncpc[-1]
            
            
    def start_record(self, tau1 = 0, tau2 = 0, thresh = -20):
    
        threshold = thresh
        
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


    def destroy(self):
        del self.nc_spike
        for m in self.MF_L:
            m.destroy()
            del m
        for m in self.GOC_L:
            m.destroy()
            del m
        for m in self.mfncpc:
            del m
        for m in self.gocncpc:
            del m
        for r in self.record:
            del r
        del self
