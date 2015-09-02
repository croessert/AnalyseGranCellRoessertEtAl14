# -*- coding: utf-8 -*-
"""
@author: chris

Modified from THOMAS MCTAVISH (2010-11-04).

mpiexec -f ~/machinefile -enable-x -n 96 python Population.py --noplot
"""

from __future__ import with_statement
from __future__ import division

import sys
sys.path.append('NEURON/')
#sys.path.append('../NET/sheffprk/template/')

import os

#use_pc = True

import sys
argv = sys.argv

if "-python" in argv:
    use_pc = True
else:
    use_pc = False

if use_pc == True:

    from neuron import h
    pc = h.ParallelContext()
    rank = int(pc.id())
    nhost = pc.nhost()

else:

    from mpi4py import MPI
    from neuron import h
    rank = MPI.COMM_WORLD.rank

#print sys.version

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', action='store', dest='opt')
    parser.add_argument('--noplot', action='store_true')
    parser.add_argument('--norun', action='store_true')
    parser.add_argument('--noconst', action='store_true')
    parser.add_argument('--noqual', action='store_true')
    pars, unknown = parser.parse_known_args(['-o','--noplot','--norun','--noconst','--noqual'])

if __name__ == '__main__':

    import matplotlib
    if rank == 0:
        matplotlib.use('Tkagg', warn=True)
    else:
        matplotlib.use('Agg', warn=True)

if __name__ == '__main__':

    do_plot = 1
    if results.noplot:  # do not plot to windows
        matplotlib.use('Agg', warn=True)
        if rank == 0: print "- No plotting"
        do_plot = 0

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import random as rnd
import neuronpy.util.spiketrain

#set_printoptions(threshold='nan')

from Stimulation import *
from Stimhelp import *
from units import *

from cells.PassiveCell import *

from itertools import izip

try:
    import cPickle as pickle
except:
    import pickle

import gzip
import h5py

from synapse.synapse import Synapse
#from synapsepfpurk import Synapse as Synapse2
if use_pc is False: import mdp

import time as ttime
from scipy.optimize import fmin, leastsq

from NeuroTools import stgen, signals

import md5

#from guppy import hpy
#hpy = hpy()


class Population:
    """
    A population of N cells
    """

    def __init__(self, cellimport = [], celltype = None, N = [10], temperature = 6.3, cell_exe = 0, ihold = [0*nA], ihold_sigma = [0*nA], amp = [0*nA], amod = [None], anoise = [None], give_freq = False, do_run = 1, pickle_prefix = "default", istart = 0, istop = 0.07, di = 0.001, dt = 0.025*ms, use_mpi = True, use_pc = False):
        """
        :param N: Number of cells.
        :param fluct_m:
        :param fluct_s:
        :param fluct_tau:
        """

        p = "figs/Pub/"
        if not os.path.isdir(p): os.makedirs(p)
        p = "figs/txt/"
        if not os.path.isdir(p): os.makedirs(p)
        p = "figs/dump/"
        if not os.path.isdir(p): os.makedirs(p)
        p = "data/"
        if not os.path.isdir(p): os.makedirs(p)
        p = "log/"
        if not os.path.isdir(p): os.makedirs(p)

        self.use_pc = use_pc

        if type(celltype) is not list: celltype = [celltype] #convert to list if it is not given as one
        self.celltype = celltype

        if type(cell_exe) is not list: cell_exe = [cell_exe] #convert to list if it is not given as one
        self.cell_exe = cell_exe

        if cellimport is not None:
            if cellimport == []:
                for n in range(len(celltype)):
                    cellimport.append("from cells." + self.celltype[n] + " import *")
        self.cellimport = cellimport

        if type(N) is not list: N = [N]
        self.N = N # Total number of cells in the net

        self.n_celltypes = len(self.N)
        self.a_celltype = [0]     # celltype to analyse

        self.factor_celltype = [1]*self.n_celltypes

        self.set_init(ihold, ihold_sigma, amp, amod)

        self.CF_var = False

        self.inh_hold_sigma = [0]
        self.intr_hold_sigma = [0]

        #self.sigma_inh_hold = 0
        #self.sigma_ihold = 0


        if type(anoise) is not list: anoise = [anoise]*self.n_celltypes
        if len(anoise) < self.n_celltypes: anoise = [anoise[0]]*self.n_celltypes
        self.anoise = anoise # RUN self.set_i()

        self.give_freq = give_freq # RUN self.set_i()

        self.temperature = temperature

        self.gid_count = 0
        self.gidlist = []           # List of global identifiers on this host
        self.global_gidlist = []    # List of global identifiers
        self.cells = []             # Cells on this host

        self.t_vec = []
        self.id_vec = []
        self.rec_v = []

        for n in range(self.n_celltypes):
            if use_mpi:
                self.t_vec.append(h.Vector()) # np.array([0])
                self.id_vec.append(h.Vector()) # np.array([-1], dtype=int)
            else:
                self.t_vec.append([])

            self.rec_v.append(h.Vector())

        #self.t_vec = h.Vector(np.array([0]))     # Spike time of all cells on this host
        #self.id_vec = h.Vector(np.array([-1]))    # Ids of spike times on this host

        self.flucts = []            # Fluctuating inputs on this host
        self.fluct_m = 0            # [nA]
        self.fluct_s = [0]          # [nA]
        self.fluct_tau = 0*ms          # [ms]

        self.noises = []            # Random number generators on this host
        self.plays = []             # Play inputs on this host
        self.rec_is = []

        self.trains = []
        self.vecstim = []
        self.nc_vecstim = []
        self.spike_vec = []

        self.syn_tau1 = 5*ms        # Synapse of virtual target neuron
        self.syn_tau2 = 5*ms        # Synapse of virtual target neuron
        self.tmax = 10*sec          # maximum length of plot that should be plotted!!

        self.nc_delay = 0 #500*ms  # only important if syn_output is used, not used currently
        self.dt = dt
        self.bin_width = dt
        self.jitter = 0*ms
        self.delta_t = 0*ms

        self.istart = istart
        self.istop = istop
        self.di = di

        self.ic_holds = []
        self.i_holdrs = []
        self.i_holds = []
        self.ic_starts = []
        self.vc_starts = []
        self.ic_steps = []

        self.rec_step = []

        self.tvecs = []
        self.ivecs = []

        self.noises = []

        self.record_syn = []
        self.id_all_vec_input = []
        self.t_all_vec_input = []

        if len(self.N) == len(self.cell_exe) == len(self.celltype):
            pass
        else:
            raise ValueError('N, cell_exe, celltype do NOT have equal length!')

        self.use_mpi = use_mpi
        self.use_pc = use_pc

        if self.use_mpi:

            #### Make a new ParallelContext object
            self.pc = h.ParallelContext()
            self.id = self.pc.id()
            self.nhost = int(self.pc.nhost())

            if self.use_pc == False:

                s = "mpi4py thinks I am %d of %d on %s, NEURON thinks I am %d of %d\n"
                processorname = MPI.Get_processor_name()
                self.comm = MPI.COMM_WORLD

                if self.id == 0:
                    print s % (self.comm.rank, self.comm.size, processorname, self.id, self.nhost)

            else:

                s = "NEURON thinks I am %d of %d\n"
                if self.id == 0:
                    print s % (self.id, self.nhost)

            self.barrier()

        else:
            self.id = 0
            self.nhost = 1

        self.do_run = do_run

        self.first_run = True

        self.set_numcells()  # Build the portion of cells on this host.

        self.pickle_prefix = pickle_prefix

        # plot options
        self.ymax = 0
        self.ax = None
        self.linewidth = 1.5
        self.color_vec = None
        self.alpha = 0.8
        self.method_interpol = np.array(['bin','syn'])
        self.dumpsave = 1
        self.called_syn_out_all = False
        self.no_fmean=False

        self.tau1_ex=[0*ms]*self.n_celltypes
        self.tau2_ex=[10*ms]*self.n_celltypes
        self.tau1_inh=[0*ms]*self.n_celltypes
        self.tau2_inh=[100*ms]*self.n_celltypes

        self.n_syn_ex = [0]*self.n_celltypes
        self.g_syn_ex = [1]*self.n_celltypes
        self.g_syn_ex_s = [0]*self.n_celltypes
        self.mglufac_ex = [1,0]

        self.noise_syn = [0]*self.n_celltypes
        self.noise_syn_tau = [0*ms]*self.n_celltypes
        self.noise_syn_inh = [0]*self.n_celltypes
        self.noise_syn_tau_inh = [0*ms]*self.n_celltypes

        self.noise_a = [1e9]*self.n_celltypes
        self.noise_a_inh = [1e9]*self.n_celltypes

        self.inh_hold = [0]*self.n_celltypes
        self.n_syn_inh = [0]*self.n_celltypes
        self.g_syn_inh = [1]*self.n_celltypes
        self.g_syn_inh_s = [0]*self.n_celltypes
        self.intr_hold = [0]*self.n_celltypes
        self.n_syn_intr = [0]*self.n_celltypes
        self.g_syn_intr = [0]*self.n_celltypes
        self.syn_max_mf = [1]*self.n_celltypes # possible mossy fibres per synapse
        self.syn_max_inh = [1]*self.n_celltypes # possible Golgi cells per synapse
        self.syn_max_intr = [1]*self.n_celltypes # possible Intruding cells per synapse


        self.seed = 50

        self.force_run = False
        self.give_psd = False
        self.do_if = True

        self.fluct_g_e0 = []
        self.fluct_g_i0 = []
        self.fluct_std_e = []
        self.fluct_std_i = []
        self.fluct_tau_e = []
        self.fluct_tau_i = []

        self.adjinh = True # adjust inhibition to get CFo instead of g_ex
        self.adjfinh = True # adjust frequnecy of inhibition to get CFo instead of g_ex

        self.syn_ex_dist = []
        self.syn_inh_dist = []

        self.stdp_used = False
        self.xmax = 20
        self.use_multisplit = False
        self.use_local_dt = False
        self.simstep = 0
        self.plot_train = True
        self.inh_delay = 0 # in ms
        self.plot_input = True
        self.delay_baseline = 8

        self.tstop_if = 1
        self.gsyn_in_fac = []

        self.netcons = [] # keeping track of!
        self.nclist = []

        self.ST_stims = []
        self.PF_stims = []

        self.data_dir = "./data"
        self.minimal_dir = False


    def set_init(self, ihold,  ihold_sigma, amp, amod):
        # important for all methods:
        if type(ihold) is not list: ihold = [ihold] #convert to list if it is not given as one
        self.ihold = ihold
        self.ihold_orig = ihold

        if type(amp) is not list: amp = [amp]
        if len(amp) < self.n_celltypes: amp = [amp[0]]*self.n_celltypes
        self.amp = amp

        if type(amod) is not list: amod = [amod]*self.n_celltypes
        self.amod = amod # RUN self.set_i()

        self.ihold_sigma = ihold_sigma

    def barrier(self):
        if self.use_mpi:
            if self.use_pc == True:
                self.pc.barrier()
            else:
                self.comm.Barrier()

    def broadcast(self, vec, root = 0, fast = False):
        if self.use_mpi:
            if self.use_pc:

                if fast:
                    hvec = h.Vector(vec)
                    v = self.pc.broadcast(hvec,root)
                    vec = np.array(hvec)
                else:

                    sendlist = [None]*self.nhost
                    if self.id == root:
                        for i in range(self.nhost):
                            sendlist[i] = vec
                    getlist = self.pc.py_alltoall(sendlist)
                    vec = getlist[root]

            else:
                #vec = np.array(vec, dtype=np.float64)
                #self.comm.Bcast([vec, MPI.DOUBLE])
                vec = self.comm.bcast(vec, root=0)

        return vec

    def set_numcells(self, N = []):
        """
        Create, layout, and connect N cells.
        """
        self.set_gids(N)
        self.create_cells()

        #self.syn_output() # generate synaptic "output" in neuron
        #self.connect_cells()


    def set_gids(self, N = []):
        """Set the gidlist on this host.
        Round-robin counting. Each host as an id from 0 to pc.nhost()-1.
        Example:
        if N = 5 cells and nhost() = 3
        node id() = 0 will get cells [0, 3]
        node id() = 1 will get cells [1, 4]
        node id() = 2 will get cells [2]
        """

        self.gidlist = []

        if N == []:
            N = self.N

        # borders where another celltype begins
        self.global_gidlist = []
        self.n_borders = [0]
        for l in range(1,self.n_celltypes+1):
            self.n_borders.append(sum(N[0:l]))
            self.global_gidlist.append(range(self.n_borders[-2], self.n_borders[-1]))

        for n in range(self.n_celltypes): # create list in list
            self.gidlist.append([])

        for i in range(int(self.id), sum(N), int(self.nhost)): # loop over all cells

            n = np.where((np.array(self.n_borders)-i)>0)[0][0]-1 # find out what cell type this is
            self.gidlist[n].append(i) # put in specific gidlist for that celltype

        self.gid_count = self.gid_count + sum(N)

        if self.id == 0: print "nodeid:" , self.id , ", gidlist:" , self.gidlist , ", total gids:" , len(self.global_gidlist) , ", sum(N):" , sum(N)    # check gids of node


    def del_cells(self):
        if self.cells != []:
            for n in range(self.n_celltypes):
                for m in self.cells[n]:
                    print "deleting cell", m
                    del m
            del self.cells
            self.cells = []
        if self.use_mpi: self.pc.gid_clear()


    def create_cells(self):
        """
        Create cell objects on this host.
        """
        if self.do_run:

            self.del_cells()

            if self.id == 0: print "creating cells"

            for n in range(self.n_celltypes):
                self.cells.append([]) # create list in list

                #print self.cellimport[n]
                exec self.cellimport[n]

                #print self.gidlist
                for i in self.gidlist[n]:

                    #if "sigma" not in self.cell_exe[n]:
                    #    exec self.cell_exe[n]
                    #    cell.gid = i # tell cell it's gid!
                    #    print i
                    #else:

                    if (self.celltype[n] == "IfCell") or (self.celltype[n] == "Grc"):

                        # add gid to cell and execute!
                        if self.cell_exe[n][-2] == "(":
                            exec self.cell_exe[n][0:-1] + "gid=" + str(i) + ")"
                        else:
                            exec self.cell_exe[n][0:-1] + ", gid=" + str(i) + ")"

                    else:
                        exec self.cell_exe[n]
                        cell.gid = i

                    self.cells[n].append(cell)  # add to (local) list

                    if self.use_mpi:
                        #### Tell this host it has this gid
                        #### gids can be any integer, they just need to be unique.
                        #### In this simple case, we set the gid to i.
                        self.pc.set_gid2node(i, int(self.id))
                        self.pc.cell(i, cell.nc_spike) # Associate the cell with this host and gid

                        ## NOT NECESSARY ANYMORE ##
                        #### Means to tell the ParallelContext that this cell is a source.
                        #nc = cell.connect_target(None)
                        #self.ncs[n].append(nc)

                        #### Record spikes of this cell
                        self.pc.spike_record(i, self.t_vec[n], self.id_vec[n])

                        #print n, self.cells[n][-1].nc_spike.thresh
                    else:

                        self.t_vec[n].append(h.Vector())
                        cell.nc_spike.record(self.t_vec[n][-1])



    def connect_cells(self, conntype=[], stdp=[], tend=1e9):
        """
        Connect cells as specified.
        """

        if self.do_run:

            stdp = stdp[:]
            conntype = conntype[:]

            if len(stdp) == 0:
                for i in conntype:
                    stdp.append({'wmax':0, 'taupre':0, 'taupost':0, 'apre':0, 'apost':0})
            else:
                self.stdp_used = True

            for i, conn in enumerate(conntype):

                typ = conn['type']
                conv = conn['conv']
                src = conn['src']
                tgt = conn['tgt']
                w0 = conn['w']
                var = conn['var']
                tau1 = conn['tau1']
                tau2 = conn['tau2']

                if 'mgr2' in conn.keys():
                    mgr2 = conn['mgr2']
                    mgr2_var = conn['mgr2_var']
                else:
                    mgr2 = 0
                    mgr2_var = 0

                if 'e_inh' in conn.keys():
                    e_inh = conn['e_inh']
                else:
                    e_inh = -65

                if 'e_ex' in conn.keys():
                    e_ex = conn['e_ex']
                else:
                    e_ex = 0

                wmax = stdp[i]['wmax']
                taupre = stdp[i]['taupre']
                taupost = stdp[i]['taupost']
                apre = stdp[i]['apre']
                apost = stdp[i]['apost']

                # Connect conv cells of celltype src to every cell of celltype tgt
                for ni, i in enumerate(self.cells[tgt]):

                    rnd.seed(i.gid*10*self.seed)

                    if conv >= len(self.global_gidlist[src]):
                        gids = self.global_gidlist[src]
                        if self.id == 0: print "more or equal conv to len(self.global_gidlist[src])"
                    else:
                        gids = rnd.sample(self.global_gidlist[src],conv)

                    if self.id == 0: print conn['type'], ":", ni, ":", gids[0], "\n"

                    for ng, g in enumerate(gids):

                        np.random.seed(g*12)
                        #np.random.seed(int(g%10+1)*12)

                        if len(shape(w0))>0: # array is given
                            print "w array is given"

                            if len(w0[ng]) == self.N[0]:
                                w = w0[ng][ni]

                        elif (var > 0) and (w0>0):
                            w = np.random.normal(w0, w0*var, 1).clip(min=0)
                        else:
                            w = w0

                        if (mgr2_var > 0) and (mgr2>0):
                            mg = np.random.normal(mgr2, mgr2*mgr2_var, 1).clip(min=0)
                        else:
                            mg = mgr2


                        #print conn['type'], ":", i.gid, ":", g, ", w:", w, "\n"

                        if self.celltype[tgt] == 'IfCell':

                            if typ == 'gogr':

                                i.whatami = "grc"
                                i.synlist_inh.append(Synapse('goc', i, i.soma, nrel=0, record_all=0, weight_gmax=w))
                                i0 = int(len(i.synlist_inh)-1)

                                i.nc_inh.append(self.pc.gid_connect(g, i.synlist_inh[i0].input))
                                i.nc_inh[-1].delay = 1
                                i.nc_inh[-1].weight[0] = 1

                            if typ == 'grgo':

                                i.whatami = "goc"
                                i.synlist.append(Synapse('grc', i, i.soma, syntype = 'D', nrel=0, record_all=0, weight_gmax=w))
                                e0 = int(len(i.synlist)-1)

                                i.nc.append(self.pc.gid_connect(g, i.synlist[e0].input))
                                i.nc[-1].delay = 1
                                i.nc[-1].weight[0] = 1

                            if typ == 'grgom':

                                i.whatami = "goc"
                                i.synlist.append(Synapse('grc', i, i.soma, syntype = 'DM', nrel=0, record_all=0, weight_gmax=w, mglufac = mg))
                                e0 = int(len(i.synlist)-1)

                                i.nc.append(self.pc.gid_connect(g, i.synlist[e0].input))
                                i.nc[-1].delay = 1
                                i.nc[-1].weight[0] = 1


                            if typ == 'e2inh':

                                i.create_synapses(n_inh=1, tau1_inh=tau1, tau2_inh=tau2, e_inh=e_inh, w = w, wmax = wmax, taupre = taupre, taupost = taupost, apre = apre, apost = apost, tend=tend)
                                i0 = len(i.synlist_inh)-1

                                if self.use_mpi:
                                    if wmax == 0:
                                        i.pconnect_target(self.pc, source=g, target=i0, syntype='inh', weight=w, delay=1)
                                    else:
                                        i.pconnect_target(self.pc, source=g, target=i0, syntype='inh', weight=1, delay=1)

                                else:
                                    if wmax == 0:
                                        i.nc_inh.append(self.cells[1][g-self.N[0]].connect_target(target=i.synlist_inh[i0], weight=w, delay=1))
                                    else:
                                        i.nc_inh.append(self.cells[1][g-self.N[0]].connect_target(target=i.synlist_inh[i0], weight=1, delay=1))

                            if typ == 'e2ex':

                                i.create_synapses(n_ex = 1, tau1 = tau1, tau2 = tau2, e_ex=e_ex, w = w, wmax = wmax, taupre = taupre, taupost = taupost, apre = apre, apost = apost, tend=tend)
                                e0 = len(i.synlist)-1

                                if self.use_mpi:
                                    if wmax == 0:
                                        i.pconnect_target(self.pc, source=g, target=e0, syntype='ex', weight=w, delay=1)
                                    else:
                                        i.pconnect_target(self.pc, source=g, target=e0, syntype='ex', weight=1, delay=1)

                                else:
                                    if wmax == 0:
                                        i.nc.append(self.cells[0][g].connect_target(target=i.synlist[e0], weight=w, delay=1))
                                    else:
                                        i.nc.append(self.cells[0][g].connect_target(target=i.synlist[e0], weight=1, delay=1))

                        else: # No IfCell

                            if typ == 'gogr':
                                i.createsyn(ngoc = 1, weight_gmax=w) # multiplication factor
                                i0 = len(i.GOC_L)-1 # get number of current synapse!
                                i.pconnect(self.pc,g,i0,'goc')

                            if typ == 'grgo':
                                i.createsyn(ngrc = 1, weight_gmax=w) # multiplication factor
                                i0 = len(i.GRC_L)-1 # get number of current synapse!
                                i.pconnect(self.pc,g,i0,'grc',conduction_speed=0,grc_positions=[1])

                            if typ == 'grgom':
                                #print w, mg
                                i.createsyn(ngrcm = 1, weight_gmax=w, mglufac = mg) # multiplication factor
                                i0 = len(i.GRC_L)-1 # get number of current synapse!
                                i.pconnect(self.pc,g,i0,'grc',conduction_speed=0,grc_positions=[1])

                            if typ == 'grstl':
                                i.createsyn(ngrc = 1, weight_gmax=w) # multiplication factor
                                i0 = len(i.GRC_L)-1 # get number of current synapse!
                                i.pconnect(self.pc,g,i0,'grc',conduction_speed=0,grc_positions=[1])


                            if 'e2' in typ:

                                if 'inh' in typ:
                                    Erev = -65
                                elif 'ex' in typ:
                                    Erev = 0

                                if tau1 == 0:
                                    syn = h.ExpSyn(i.soma(0.5))
                                    syn.tau = tau2/ms
                                else:
                                    if wmax == 0:
                                        syn = h.Exp2Syn(i.soma(0.5))
                                        syn.tau1 = tau1/ms
                                        syn.tau2 = tau2/ms

                                    else: # STDP
                                        syn = h.stdpE2S(i.soma(0.5))
                                        syn.tau1 = tau1/ms
                                        syn.tau2 = tau2/ms

                                        syn.on = 1
                                        syn.thresh = -20

                                        syn.wmax = wmax
                                        syn.w = w

                                        syn.taupre  = taupre/ms
                                        syn.taupost  = taupost/ms
                                        syn.apre    = apre
                                        syn.apost   = apost

                                syn.e = Erev/mV

                                if self.celltype[tgt] == 'Grc':

                                    i.GOC_L.append(syn)
                                    i0 = int(len(i.GOC_L)-1) # get number of current synapse!

                                    i.gocncpc.append(self.pc.gid_connect(g, i.GOC_L[i0]))
                                    i.gocncpc[-1].delay = 1

                                    if wmax == 0:
                                        i.gocncpc[-1].weight[0] = w
                                    else:
                                        i.gocncpc[-1].weight[0] = 1

                                elif self.celltype[tgt] == 'Goc':

                                    i.GRC_L.append(syn)
                                    e0 = int(len(i.GRC_L)-1) # get number of current synapse!

                                    i.pfncpc.append(self.pc.gid_connect(g, i.GRC_L[e0]))
                                    i.pfncpc[-1].delay = 1
                                    i.pfncpc[-1].weight[0] = w

                                    if wmax == 0:
                                        i.pfncpc[-1].weight[0] = w
                                    else:
                                        i.pfncpc[-1].weight[0] = 1

            #self.rec_s1 = h.Vector()
            #self.rec_s1.record(self.cells[0][0].synlist_inh[0]._ref_g)
            #self.rec_s2 = h.Vector()
            #self.rec_s2.record(self.cells[1][0].synlist_inh[0]._ref_g)


    def syn_output(self):
        """
        Connect cell n to target cell sum(self.N) + 100.
        """

        if self.id == 0:  # create target cell

            tgt_gid = self.gid_count
            self.gid_count = self.gid_count + 1

            # Synaptic integrated response
            self.rec_g = h.Vector()
            self.passive_target = PassiveCell()
            if self.use_mpi: self.pc.set_gid2node(tgt_gid, 0)  # Tell this host it has this gid

            syn = self.passive_target.create_synapses(tau1 = self.syn_tau1, tau2 = self.syn_tau2)  # if tau1=tau2: alpha synapse!

            for i in range(self.n_borders[self.a_celltype[0]],self.n_borders[self.a_celltype[0]+1]): # take all cells, corresponding to self.a_celltype, not just the ones in self.gidlist:

                src_gid = i

                if self.use_mpi:
                    nc = self.pc.gid_connect(src_gid, syn)
                    nc.weight[0] = 1
                    nc.delay = self.nc_delay/ms #0.05  # MUST be larger than dt!!!

                else:
                    nc = self.cells[self.a_celltype[0]][src_gid].connect_target(target=syn, weight=1, delay=self.nc_delay/ms)

                self.nclist.append(nc)

            self.rec_g.record(syn._ref_g)


    def syn_out_all(self, tau1 = 1*ms, tau2 = 30*ms):

        if self.do_run:

            for n in range(self.n_celltypes):
                for i, gid in enumerate(self.gidlist[n]):

                    self.cells[n][i].start_record(tau1 = tau1/ms, tau2 = tau2/ms)

            self.called_syn_out_all = True


    def get_i(self, a, n, do_plot = True):

        import md5
        m = md5.new()

        if ", sigma" in self.cell_exe[n]:
            cell_exe_new = self.cell_exe[n].split(", sigma")[0] + ")"
        else:
            cell_exe_new = self.cell_exe[n]

        m.update(cell_exe_new)
        filename = self.data_dir + '/if_' + self.celltype[n] + '_' + m.hexdigest() + '.p'

        #print filename

        if self.id == 0:
            is_there = os.path.isfile(filename)
        else:
            is_there = None

        is_there = self.broadcast(is_there)

        if (is_there is not True) or (self.force_run is True): # run i/f estimation

            if self.id == 0: print '- running i/f estimation for ', self.celltype[n], ' id: ' , m.hexdigest()
            exec self.cellimport[n]
            exec cell_exe_new
            sim = Stimulation(cell, temperature = self.temperature, use_multisplit = self.use_multisplit)
            sim.spikes_from_neuron = False
            sim.celltype =  self.celltype[n]
            current_vector, freq_vector, freq_onset_vector = sim.get_if(istart = self.istart, istop = self.istop, di = self.di, tstop = self.tstop_if)

            sim = None
            cell = None

            if self.id == 0:
                if do_plot:
                    plt.figure(99)
                    plt.plot(current_vector, freq_vector, 'r*-')
                    plt.plot(current_vector, freq_onset_vector, 'b*-')
                    plt.savefig("./figs/dump/latest_if_" + self.celltype[n]  + ".pdf", dpi = 300)  # save it
                    plt.clf()
                    #plt.show()

                ifv = {'i':current_vector,'f':freq_vector}
                print ifv

                pickle.dump(ifv, gzip.GzipFile(filename, "wb" ))

            self.barrier()

        else:

            if self.id == 0:
                ifv = pickle.load(gzip.GzipFile(filename, "rb" ))
                #print ifv

        self.barrier()

        if self.id == 0:

            f =  ifv.get('f')
            i =  ifv.get('i')

            i = i[~isnan(f)]
            f = f[~isnan(f)]

            iin = if_extrap(a, f, i)

        else:

            iin = [0]

        iin = self.broadcast(iin, root=0, fast = True)
        self.barrier()

        return iin


    def set_i(self, ihold = [0]):

        ihold = list(ihold)
        self.ihold_orig = list(ihold)

        self.barrier()   # wait for other nodes

        # Ihold given as frequency, convert to current

        if ((self.give_freq)):

            ihold0 = [[] for _ in range(self.n_celltypes)]

            for n in range(self.n_celltypes):
                a = np.array([ihold[n]])
                #print "a:", a
                iin = self.get_i(a, n)
                #print "iin:", iin
                ihold0[n] = iin[0]

            if self.id == 0: print '- ihold: ', ihold, 'Hz, => ihold: ', ihold0, 'nA'

        # Modulation depth given, not always applied to current!
        for n in range(self.n_celltypes):

            if self.amod[n] is not None:

                if self.give_freq:

                    # Apply to amplitude:
                    a = np.array([ihold[n]]) + self.amod[n]*np.array([ihold[n]])
                    self.amp[n] = self.get_i(a, n) - ihold0[n]

                    if self.id == 0:
                        print '- amp: ihold: ', ihold[n], 'Hz , amod: ', self.amod[n], ', => amp: ', self.amp[n], 'nA (' #, self.get_i(a, n), ')'

                elif self.n_syn_ex[n] > 0:

                    if self.id == 0:
                        print '- amp: ihold: ', ihold[n], 'Hz , amod: ', self.amod[n], ', => amp will be set for each spike generator'

                else:

                    self.amp[n] = self.amod[n] * ihold[n]

                    if self.id == 0:
                        print '- amp: ihold: ', ihold[n], 'nA , amod: ', self.amod[n], ', => amp: ', self.amp[n], 'nA'

            # Noise depth given, not always applied to current!
            if self.anoise[n] is not None:

                if (self.give_freq is True) or (self.n_syn_ex[n] > 0):

                    # Apply to amplitude:
                    a = np.array([ihold[n]]) + self.anoise[n]*np.array([ihold[n]])
                    self.fluct_s[n] = ((self.get_i(a, n) - ihold0[n]))/2. # adjust with /2 so that noise = +-2*std

                    if self.id == 0:
                        print '- noise: ihold: ', ihold[n], 'Hz , anoise: ', self.anoise[n], ', => fluct_s: ', self.fluct_s[n], 'nA'

                else:

                    self.fluct_s[n] = self.anoise[n] * ihold[n]

                    if self.id == 0:
                        print '- noise: ihold: ', ihold[n], 'nA , anoise: ', self.anoise[n], ', => fluct_s: ', self.fluct_s[n], 'nA'


        if self.give_freq is True:
            ihold = ihold0

        return ihold


    def calc_fmean(self, t_vec, t_startstop):

        #t_startstop[0] = 1
        #t_startstop[1] = 5

        f_cells_mean = 0
        f_cells_cv = np.nan
        f_cells_std = np.nan

        if len(t_vec) > 0:

            f_start_in = mlab.find(t_vec >= t_startstop[0]) # 1
            f_stop_in = mlab.find(t_vec <= t_startstop[1]) # 5

            if (len(f_start_in) > 0) & (len(f_stop_in) > 0):

                f_start = f_start_in[0]
                f_stop = f_stop_in[-1]+1
                use_spikes = t_vec[f_start:f_stop]*1e3

                if len(use_spikes) > 1:
                    s1 = signals.SpikeTrain(use_spikes)
                    isi = s1.isi()
                    f_cells_mean = s1.mean_rate() # use mean of single cells
                    f_cells_cv = np.std(isi)/np.mean(isi)
                    f_cells_std = np.std(isi)

            #f_start_in = mlab.find(t_vec >= 1)
            #f_stop_in = mlab.find(t_vec <= 2)

            #if (len(f_start_in) > 0) & (len(f_stop_in) > 0):

            #    f_start = f_start_in[0]
            #    f_stop = f_stop_in[-1]+1
            #    use_spikes = t_vec[f_start:f_stop]*1e3

            #    if len(use_spikes) > 1:
            #        s1 = signals.SpikeTrain(use_spikes)
            #        isi = s1.isi()
            #        f_cells_cv = np.std(isi)/np.mean(isi)

        return f_cells_mean, f_cells_cv, f_cells_std


    def get_fmean(self, t_all_vec_vecn, id_all_vec_vecn, t_startstop, gidlist, facborder = 3): # 1e9

        f_cells_mean = zeros(len(gidlist))
        f_cells_base = zeros(len(gidlist))
        f_cells_std = nans(len(gidlist))
        f_cells_cv = nans(len(gidlist))
        f_cells_gid = nans(len(gidlist))

        fbase = np.nan
        fmean = np.nan
        fmax = np.nan
        fmstd = np.nan
        fcvm = np.nan
        fstdm = np.nan

        f_cells_mean_all = []
        f_cells_base_all = []
        f_cells_cv_all = []
        f_cells_std_all = []

        gid_del = np.array([])

        if self.no_fmean == False:

            if self.id == 0: print "- sorting for fmean"

            for i, l in enumerate(gidlist):

                t_0_vec = t_all_vec_vecn[where(id_all_vec_vecn==l)]
                f_cells_mean[i], f_cells_cv[i], f_cells_std[i] = self.calc_fmean(t_0_vec, t_startstop)
                f_cells_base[i], _, _ = self.calc_fmean(t_0_vec, [self.delay_baseline-4,self.delay_baseline])
                f_cells_gid[i] = l

            if self.id == 0:  print "- gather fmean"
            f_cells_mean_all = self.do_gather(f_cells_mean)
            f_cells_base_all = self.do_gather(f_cells_base)
            f_cells_std_all = self.do_gather(f_cells_std)
            f_cells_cv_all = self.do_gather(f_cells_cv)
            f_cells_gid_all = self.do_gather(f_cells_gid)

            if self.id == 0:

                #print f_cells_mean_all

                f_cells_mean_all = np.nan_to_num(f_cells_mean_all)
                fmean = mean(f_cells_mean_all)  # compute mean of mean rate for all cells
                fmstd = std(f_cells_mean_all)
                fmax = max(f_cells_mean_all)

                f_cells_base_all = np.nan_to_num(f_cells_base_all)
                fbase = mean(f_cells_base_all)  # compute mean of mean rate for all cells

                f_cells_cv_all = f_cells_cv_all[~np.isnan(f_cells_cv_all)]
                f_cells_std_all = f_cells_std_all[~np.isnan(f_cells_std_all)]
                fcvm = mean(f_cells_cv_all)
                fstdm = mean(f_cells_std_all)

                print "- get_fmean, fmean: ",fmean, "fmax: ",fmax, "Hz", "fmstd: ",fmstd, "Hz", "fcvm: ",fcvm, "fstdm: ",fstdm, "Hz" ,"fbase: ", fbase, "Hz"

                if facborder < 1e9:

                    fborder = fmean + facborder*fmstd
                    i = mlab.find(f_cells_mean_all > fborder)
                    gid_del = f_cells_gid_all[i]

                #    f_cells_mean_all[i] = 0
                #    f_cells_cv_all[i] = np.nan
                #    f_cells_std_all[i] = np.nan

                #    fmean2 = mean(np.nan_to_num(f_cells_mean_all))  # compute mean of mean rate for all cells
                #    fmstd2 = std(np.nan_to_num(f_cells_mean_all))
                #    fmax2 = max(np.nan_to_num(f_cells_mean_all))

                #    fcvm2 = mean(f_cells_cv_all[~np.isnan(f_cells_cv_all)])
                #    fstdm2 = mean(f_cells_std_all[~np.isnan(f_cells_std_all)])

                #    print "- after facborder: get_fmean, fmean: ",fmean2, "fmax: ",fmax2, "Hz", "fmstd: ",fmstd2, "Hz", "fcvm: ",fcvm2, "fstdm: ",fstdm2, "Hz, gid_del: ", gid_del


        return fmean, fmax, fmstd, fcvm, fstdm, gid_del, f_cells_mean_all, f_cells_cv_all, f_cells_std_all, fbase, f_cells_base_all


    def connect_fluct(self):
        """
        Create fluctuating input onto every cell.
        """

        if self.do_run:

            for m in self.flucts:
                del m
            del self.flucts

            for m in self.noises:
                del m
            del self.noises

            self.flucts = []
            self.noises = []

            for n in range(self.n_celltypes):

                for i, gid in enumerate(self.gidlist[n]):  # for every cell in the gidlist

                    #h.mcell_ran4_init(gid)

                    noiseRandObj = h.Random()  # provides NOISE with random stream
                    self.noises.append(noiseRandObj)  # has to be set here not inside the nmodl function!!

                    # print str(gid) + ": " + str(noiseRandObj.normal(0,1))

                    fluct = h.Ifluct2(self.cells[n][i].soma(0.5))
                    fluct.m = self.fluct_m/nA      # [nA]
                    fluct.s = self.fluct_s[n]/nA      # [nA]
                    fluct.tau = self.fluct_tau/ms    # [ms]
                    self.flucts.append(fluct)   # add to list
                    self.flucts[-1].noiseFromRandom(self.noises[-1])  # connect random generator!

                    self.noises[-1].MCellRan4(1, gid+1)  # set lowindex to gid+1, set highindex to > 0
                    self.noises[-1].normal(0,1)


    def connect_gfluct(self, E_e=0, E_i=-65):
        """
        Create fluctuating conductance input onto every cell.
        """
        if self.do_run:

            for m in self.flucts:
                del m
            del self.flucts

            for m in self.noises:
                del m
            del self.noises

            self.flucts = []
            self.noises = []

            for n in range(self.n_celltypes):

                fluct_g_i0_n = self.fluct_g_i0[n]

                if type(fluct_g_i0_n) is not ndarray: fluct_g_i0_n = np.array([fluct_g_i0_n])

                if len(fluct_g_i0_n) == len(self.global_gidlist[n]):
                    pass
                else:
                    fluct_g_i0_n = np.ones(int(len(self.global_gidlist[n])))*fluct_g_i0_n[0]
                    if self.id == 0: print "- single value in fluct_g_i0_n"

                #print fluct_g_i0_n

                for i, gid in enumerate(self.gidlist[n]):  # for every cell in the gidlist

                    #h.mcell_ran4_init(gid)

                    noiseRandObj = h.Random()  # provides NOISE with random stream
                    self.noises.append(noiseRandObj)  # has to be set here not inside the nmodl function!!

                    # print str(gid) + ": " + str(noiseRandObj.normal(0,1))

                    fluct = h.Gfluct3(self.cells[n][i].soma(0.5))
                    fluct.E_e = E_e/mV  # [mV]
                    fluct.E_i = E_i/mV  # [mV]
                    fluct.g_e0 = self.fluct_g_e0[n]/uS  # [uS]
                    fluct.g_i0 = fluct_g_i0_n[i]/uS  # [uS]
                    fluct.std_e = self.fluct_std_e[n]/uS  # [uS]
                    fluct.std_i = self.fluct_std_i[n]/uS  # [uS]
                    fluct.tau_e = self.fluct_tau_e/ms #tau_e/ms  # [ms]
                    fluct.tau_i = self.fluct_tau_i/ms #tau_i/ms  # [ms]

                    self.flucts.append(fluct)   # add to list
                    self.flucts[-1].noiseFromRandom(self.noises[-1])  # connect random generator!

                    self.noises[-1].MCellRan4(1, gid+1)  # set lowindex to gid+1, set highindex to > 0
                    self.noises[-1].normal(0,1)


    def connect_synfluct(self, PF_BG_rate=6, PF_BG_cv=1, STL_BG_rate=20, STL_BG_cv=1):
        """
        Create fluctuating synaptic input onto every cell.
        """

        if self.do_run:

            for m in self.ST_stims:
                del m
            del self.ST_stims

            for m in self.PF_stims:
                del m
            del self.PF_stims

            self.ST_stims = []
            self.PF_stims = []


            for n in range(self.n_celltypes):

                for i, gid in enumerate(self.gidlist[n]):  # for every cell in the gidlist

                    PF_syn_list = self.cells[n][i].createsyn_PF()

                    for d in PF_syn_list:
                        d.input.newnetstim.number = 1e9
                        d.input.newnetstim.noise = PF_BG_cv
                        d.input.newnetstim.interval = 1000.0 / PF_BG_rate
                        d.input.newnetstim.start = 0

                    self.PF_stims.append(PF_syn_list)

                    ST_stim_list = self.cells[n][i].createsyn_ST(record_all=0)

                    for d in ST_stim_list:
                        d.newnetstim.number = 1e9
                        d.newnetstim.noise = STL_BG_cv
                        d.newnetstim.interval =  1000.0 / STL_BG_rate
                        d.newnetstim.start = 0

                    self.ST_stims.append(ST_stim_list)

            if self.id == 0: print "- PF and ST stimulation added."



    def set_IStim(self, ihold = None, ihold_sigma = None, random_start = True, tstart_offset = 0):
        """
        Add (random) ihold for each cell and offset!
        """
        if self.do_run:

            # if not given, use the one in self
            if ihold == None:
                ihold = self.ihold
            if ihold_sigma == None:
                ihold_sigma = self.ihold_sigma

            if ihold[self.a_celltype[0]] != 0:
                ihold = self.set_i(ihold)

            for m in self.ic_holds:
                #m.destroy()
                del m
            del self.ic_holds

            for m in self.ic_starts:
                #m.destroy()
                del m
            del self.ic_starts

            for m in self.vc_starts:
                #m.destroy()
                del m
            del self.vc_starts

            self.ic_holds = []
            self.ic_starts = []
            self.vc_starts = []
            self.i_holdrs = []
            self.i_holds = ihold

            for n in range(self.n_celltypes):
                self.i_holdrs.append([])

                for i, gid in enumerate(self.gidlist[n]):  # for every cell in the gidlist

                    np.random.seed(gid*20)

                    tis = 1

                    if random_start == True:

                        # random start time
                        tstart = np.random.uniform(tstart_offset+0, tstart_offset+0.5)
                        #if self.id == 0: print "tstart:", tstart
                        vc_start = h.SEClamp(self.cells[n][i].soma(0.5))
                        vc_start.dur1 = tstart/ms
                        vc_start.amp1 = -80
                        self.vc_starts.append(vc_start)
                        tis = 0

                    else:

                        tis = 0


                    if ihold_sigma[n] != 0:
                        #print ihold_sigma[n], ihold[n]
                        ihold_r = np.random.normal(ihold[n], ihold[n]*ihold_sigma[n], 1).clip(min=0)
                        #ihold_r = np.random.uniform(ihold[n]*ihold_sigma[n], ihold[n])

                    elif self.CF_var is not False:  # CF gets not adapted to current but final frequnecy!

                        r_ok = False
                        while r_ok == False:
                            r_temp = np.random.normal(self.ihold_orig[n], self.CF_var[n][1], 1)
                            if (r_temp <= self.CF_var[n][2]) and (r_temp >= self.CF_var[n][0]): # check borders!
                                r_ok = True

                        #print r_temp
                        ihold_r = self.get_i(r_temp, n)
                        #print ihold_r
                        #if self.id == 0:
                        print "set self.CF_var", r_temp, ihold_r

                    else:  # same ihold for all cells!
                        ihold_r = ihold[n]

                    self.i_holdrs[n].append(ihold_r)

                    if ihold_r != 0:

                        if hasattr(self.cells[n][i], 'input_vec'):

                            ic_hold = []
                            for vec in self.cells[n][i].input_vec:
                                for inv in vec:
                                    #print ihold_r
                                    ic_hold.append(h.IClamp(inv(0.5)))
                                    ic_hold[-1].amp = self.cells[n][i].ifac * ihold_r / self.cells[n][i].n_input_spiny / nA
                                    ic_hold[-1].delay = tis/ms
                                    ic_hold[-1].dur = 1e9

                        else:

                            # holding current
                            ic_hold = h.IClamp(self.cells[n][i].soma(0.5))
                            ic_hold.delay = tis/ms
                            ic_hold.dur = 1e9
                            ic_hold.amp = ihold_r/nA

                        self.ic_holds.append(ic_hold)

            if self.id == 0: print "set_IStim finished. ihold: ", ihold, ", ihold_sigma: ", ihold_sigma


    def set_IStep(self, istep = [0], istep_sigma = [0], tstep = 5, tdur = 1e6, give_freq = True):
        """
        Add istep for each cell and offset!
        """
        if self.do_run:
            #for m in self.ic_steps:
            #    m.destroy()
            #    del m
            #del self.ic_steps

            #self.ic_steps = []

            istep = list(istep)
            neg = False

            for n in range(self.n_celltypes):

                if istep[n] < 0:
                    neg = True
                    istep[n] = abs(istep[n]) # make positive again

                if istep[n] != 0:
                    if give_freq is True:
                        a = np.array([istep[n]])
                        iin = self.get_i(a, n)[0]
                        if self.id == 0: print "celltype: ", n, " istep: ", istep[n], "Hz => ", iin, " nA"
                        istep[n] = iin

            for n in range(self.n_celltypes):

                for i, gid in enumerate(self.gidlist[n]):  # for every cell in the gidlist

                    np.random.seed(gid*30)

                    if self.i_holdrs == []:

                        if istep_sigma[n] != 0:
                            istep_r = np.random.normal(istep[n], istep[n]*istep_sigma[n], 1).clip(min=0)
                        else:  # same ihold for all cells!
                            istep_r = istep[n]

                    else: # ihold has been set!

                        if istep_sigma[n] != 0:
                            istep_r = np.random.normal(istep[n]-self.i_holds[n], (istep[n]-self.i_holds[n])*istep_sigma[n], 1).clip(min=0) # delta now! put on top of hold!
                        else:  # same ihold for all cells!
                            istep_r = istep[n]-self.i_holds[n] # delta now! put on top of hold!

                    if neg:
                        istep_r = -1*istep_r

                    if istep[n] == 0:
                        istep_r = -1*self.i_holdrs[n][i]

                    #print 'is:' + str(istep_r) + 'was:' + str(self.i_holdrs[n][i])

                    if istep_r != 0:
                        # step current
                        ic_step = h.IClamp(self.cells[n][i].soma(0.5))
                        ic_step.delay = tstep/ms
                        ic_step.dur = tdur/ms
                        ic_step.amp = istep_r/nA
                        self.ic_steps.append(ic_step)


            if self.id == 0: print "set_IStep finished. istep: ", istep, ", istep_sigma: ", istep_sigma


    def set_IPlay(self, stimulus, t):
        """
        Initializes values for current clamp to play a signal.
        """

        if self.do_run:

            for m in self.tvecs:
                #m.destroy()
                del m
            del self.tvecs

            for m in self.ivecs:
                #m.destroy()
                del m
            del self.ivecs

            for m in self.plays:
                #m.destroy()
                del m
            del self.plays

            self.tvecs = []
            self.ivecs = []
            self.plays = []

            for i, gid in enumerate(self.gidlist[self.a_celltype[0]]):  # for every cell in the gidlist

                tvec = h.Vector(t/ms)
                ivec = h.Vector(stimulus/nA)

                play = h.IClamp(self.cells[self.a_celltype[0]][i].soma(0.5))
                play.delay = 0
                play.dur = 1e9

                ivec.play(play._ref_amp, tvec, 1)

                self.plays.append(play)   # add to list
                self.tvecs.append(tvec)   # add to list
                self.ivecs.append(ivec)   # add to list

            if self.id == 0: print "set_IPlay finished."


    def set_IPlay2(self, stimulus, t):
        """
        Initializes values for current clamp to play a signal.
        """

        if self.do_run:

            for m in self.tvecs:
                #m.destroy()
                del m
            del self.tvecs

            for m in self.ivecs:
                #m.destroy()
                del m
            del self.ivecs

            for m in self.plays:
                #m.destroy()
                del m
            del self.plays

            self.tvecs = []
            self.ivecs = []
            self.plays = []

            for j in self.a_celltype:

                tvec = h.Vector(t/ms)
                ivec = []
                for s in stimulus:
                    if hasattr(self.cells[j][0], 'input_vec'):
                        ivec.append(h.Vector(self.factor_celltype[j] * self.cells[j][0].ifac * s / self.cells[j][0].n_input_spiny / nA))
                    else:
                        ivec.append(h.Vector(self.factor_celltype[j]*s/nA))

                self.tvecs.append(tvec)   # add to list
                self.ivecs.append(ivec)   # add to list

                for i, gid in enumerate(self.gidlist[j]):  # for every cell in the gidlist

                    if hasattr(self.cells[j][i], 'input_vec'):

                        play = []
                        for iloc, vec in enumerate(self.cells[j][i].input_vec):
                            isig = self.syn_ex_dist[j][iloc]-1
                            #print isig
                            for inv in vec:
                                play.append(h.IClamp(inv(0.5)))
                                play[-1].delay = 0
                                play[-1].dur = 1e9
                                ivec[isig].play(play[-1]._ref_amp, tvec, 1)

                    else:
                        #fluctuating current
                        play = h.IClamp(self.cells[j][i].soma(0.5))
                        play.delay = 0
                        play.dur = 1e9
                        ivec[0].play(play._ref_amp, tvec, 1)

                    self.plays.append(play)   # add to list


            if self.id == 0: print "set_IPlay2 finished."


    def set_IPlay3(self, stimulus, t, amp = None):
        """
        Initializes values for current clamp to play a signal.
        """

        if self.do_run:

            for m in self.tvecs:
                #m.destroy()
                del m
            del self.tvecs

            for m in self.ivecs:
                #m.destroy()
                del m
            del self.ivecs

            for m in self.plays:
                #m.destroy()
                del m
            del self.plays

            self.tvecs = []
            self.ivecs = []
            self.plays = []

            for j in self.a_celltype:

                if amp is None:
                    amp0 = 0
                else:
                    amp0 = amp[j]

                tvec = h.Vector(t/ms)
                self.tvecs.append(tvec)   # add to list

                for i, gid in enumerate(self.gidlist[j]):  # for every cell in the gidlist

                    if isinstance(self.factor_celltype[j], ( int, long ) ):
                        ivec = h.Vector(self.factor_celltype[j]*(stimulus*amp0)/nA)
                    else:
                        np.random.seed(gid*40)
                        rnd.seed(gid*40)
                        if self.factor_celltype[j][1] > 0:
                            f = np.random.normal(self.factor_celltype[j][0], self.factor_celltype[j][1], 1).clip(min=0)
                        else:
                            f = self.factor_celltype[j][0]
                        if self.factor_celltype[j][2] > 0: # add inverted input with 50% probability, in future versions this will indicate the propability for -1 and 1
                            f = rnd.sample([-1,1],1)[0] * f
                            if self.id == 0: print "- inverted input with 50% probability:", f
                        if self.id == 0: print "- randomize play stimulus height"
                        ivec = h.Vector(f*(stimulus*amp0)/nA)

                    self.ivecs.append(ivec)   # add to list

                    #fluctuating current
                    play = h.IClamp(self.cells[j][i].soma(0.5))
                    play.delay = 0
                    play.dur = 1e9
                    ivec.play(play._ref_amp, tvec, 1)

                    self.plays.append(play)   # add to list

            if self.id == 0: print "set_IPlay3 finished."


    def set_PulseStim(self, start_time=[100*ms], dur=[1500*ms], steadyf=[100*Hz], pulsef=[150*Hz], pulse_start=[500*ms], pulse_len=[500*ms], weight0=1, tau01=[1*ms], tau02=[20*ms], weight1=1, tau11=[0*ms], tau12=[1*ms], noise = 1):

        if self.do_run:

            modulation_vec = []

            for n in range(self.n_celltypes):

                t_input = np.arange(0, dur[n], self.dt) # create stimulus time vector has to be in ms!!
                mod = np.concatenate(([np.zeros(round(start_time[n]/self.dt)), steadyf[n]*np.ones(round((pulse_start[n]-start_time[n])/self.dt)), pulsef[n]*np.ones(round(pulse_len[n]/self.dt)),steadyf[n]*np.ones(round((dur[n]-pulse_start[n]-pulse_len[n])/self.dt)) ]))
                modulation = (t_input, mod)

                #print shape(t_input), shape(mod), shape(modulation)

                for i, gid in enumerate(self.gidlist[n]):  # for every cell in the gidlist

                    if dur[n] > 0:

                        if self.celltype[n] == 'Grc':

                            nmf = 4

                            for j in range(nmf):

                                self.cells[n][i].createsyn(nmf = 1, ngoc = 0, weight = weight0)
                                e0 = len(self.cells[n][i].MF_L)-1 # get number of current synapse!

                                pulse_gid = int(self.gid_count + gid*1000 + j)

                                train = mod_spike_train(modulation, noise = noise, seed = pulse_gid)

                                self.setup_Play_train(train = train, input_gid = pulse_gid)

                                self.cells[n][i].pconnect(self.pc,pulse_gid,int(e0),'mf')

                        elif self.celltype[n] == 'Goc':

                            nmf = 53

                            for j in range(nmf):

                                self.cells[n][i].createsyn(nmf = 1, weight = weight1)
                                e0 = len(self.cells[n][i].MF_L)-1 # get number of current synapse!

                                pulse_gid = int(self.gid_count + gid*1000 + j)

                                train = mod_spike_train(modulation, noise = noise, seed = pulse_gid)

                                self.setup_Play_train(train = train, input_gid = pulse_gid)

                                self.cells[n][i].pconnect(self.pc,pulse_gid,int(e0),'mf')


                        elif self.celltype[n] == 'Goc_noloop':

                            ngrc = 100

                            for j in range(ngrc):

                                self.cells[n][i].createsyn(ngrc = 1, weight = weight0)
                                e0 = len(self.cells[n][i].GRC_L)-1 # get number of current synapse!

                                pulse_gid = int(self.gid_count + gid*1000 + j)

                                train = mod_spike_train(modulation, noise = noise, seed=pulse_gid)

                                self.setup_Play_train(train = train, input_gid = pulse_gid)

                                self.cells[n][i].pconnect(self.pc,pulse_gid,int(e0),'grc')

                        else:

                            pulse_gid = int(self.gid_count + gid*1000 + 100)

                            train = mod_spike_train(modulation, noise = noise, seed = pulse_gid)
                            self.trains.append(train)

                            setup_Play_train(train = train, input_gid = pulse_gid)

                            # NMDA
                            self.cells[n][i].create_synapses(n_ex=1, tau1=tau01[n], tau2=tau02[n])
                            e0 = len(self.cells[n][i].synlist)-1

                            weight=weight0[n]
                            np.random.seed(gid*60)
                            #weight = np.random.normal(weight, weight*0.5, 1).clip(min=0)
                            self.cells[n][i].pconnect_target(self.pc, source=pulse_gid, target=e0, syntype='ex', weight=weight, delay=1)

                            # AMPA
                            self.cells[n][i].create_synapses(n_ex=1, tau1=tau11[n], tau2=tau12[n])
                            e0 = len(self.cells[n][i].synlist)-1

                            weight=weight1[n]
                            np.random.seed(gid*60)
                            #weight = np.random.normal(weight, weight*0.5, 1).clip(min=0)
                            self.cells[n][i].pconnect_target(self.pc, source=pulse_gid, target=e0, syntype='ex', weight=weight, delay=1)


                modulation = (t_input, mod) # mack to s!
                modulation_vec.append(modulation)

            return modulation_vec


    def connect_Synapse(self, pulse_gid, nt, i, n, gid, j, syntype = "ex", nsyn=0):

        if self.do_run:

            if 'gsyn_in' in self.method_interpol:
                if isinstance(self.factor_celltype[nt], ( int, long ) ):
                    f = self.factor_celltype[nt]
                else:
                    f = self.factor_celltype[nt][0]

            if syntype == "ex":

                # each cell can receive different g_syn_ex !
                if type(self.g_syn_ex[nt]) is ndarray:
                    if len(self.g_syn_ex[nt]) == len(self.global_gidlist[nt]):
                        w = self.g_syn_ex[nt][n]
                    else:
                        w = self.g_syn_ex[nt]
                else:
                    w = self.g_syn_ex[nt]

                seed = int(10000 + 10*gid + j)
                np.random.seed(seed*41)

                if self.g_syn_ex_s[nt] > 0:
                    w = np.random.normal(w, w*self.g_syn_ex_s[nt], 1).clip(min=0)  #  self.g_syn_ex_s[nt]

                if self.celltype[nt] == 'Grc':

                    # delete old
                    if j == 0:
                        self.cells[nt][i].MF_L = []
                        self.cells[nt][i].mfncpc = []

                    if "gr" not in str(self.tau1_ex[nt]):

                        if "amfit" in str(self.tau1_ex[nt]):
                            syn = h.ExpZSyn(self.cells[nt][i].soma(0.5))

                            syn.tau1_ampa = 0.254
                            syn.tau2_ampa = 0.254
                            syn.tau3_ampa = 0.363
                            syn.tau4_ampa = 6.523
                            syn.f1_ampa = 8.8376e-05
                            syn.f2_ampa = 5.5257e-05

                            syn.f1_nmda = 0

                        elif "nmfit" in str(self.tau1_ex[nt]):
                            syn = h.ExpYSyn(self.cells[nt][i].soma(0.5))

                            syn.f1_ampa = 0
                            syn.f2_ampa = 0

                            syn.tau1_nmda = 1.902
                            syn.tau2_nmda = 82.032
                            syn.f1_nmda = 7.853857483005277e-05

                        elif "fit" in str(self.tau1_ex[nt]):
                            syn = h.ExpGrcSyn(self.cells[nt][i].soma(0.5))

                            syn.tau1_ampa = 0.254
                            syn.tau2_ampa = 0.254
                            syn.tau3_ampa = 0.363
                            syn.tau4_ampa = 6.523
                            syn.f1_ampa = 8.8376e-05
                            syn.f2_ampa = 5.5257e-05

                            syn.tau1_nmda = 1.902
                            syn.tau2_nmda = 82.032
                            syn.f1_nmda = 7.853857483005277e-05

                        else:
                            tau1 = self.tau1_ex[nt]
                            tau2 = self.tau2_ex[nt]

                            if tau1 == 0:
                                syn = h.ExpSyn(self.cells[nt][i].soma(0.5))
                                syn.tau = tau2/ms

                            else:
                                syn = h.Exp2Syn(self.cells[nt][i].soma(0.5))
                                syn.tau1 = tau1/ms
                                syn.tau2 = tau2/ms

                        syn.e = 0/mV

                        self.cells[nt][i].MF_L.append(syn)

                        e0 = len(self.cells[nt][i].MF_L)-1 # get number of current synapse!

                        syn_idx = int(e0)

                        source = int(pulse_gid)
                        self.cells[nt][i].mfncpc.append(self.pc.gid_connect(source, self.cells[nt][i].MF_L[syn_idx]))
                        self.cells[nt][i].mfncpc[-1].delay = 1
                        self.cells[nt][i].mfncpc[-1].weight[0] = w

                        if 'gsyn_in' in self.method_interpol:
                            self.record_syn.append(h.Vector())
                            self.record_syn[-1].record(self.cells[nt][i].MF_L[-1]._ref_g)
                            self.gsyn_in_fac.append(f)

                    else:

                        nrel = 0

                        if "stoch" in str(self.tau1_ex[nt]):
                            nrel = 4

                        self.cells[nt][i].createsyn(nmf = 1, ngoc = 0, weight_gmax = w, nrel=nrel)

                        if "ampa" in str(self.tau1_ex[nt]):
                            self.cells[nt][i].MF_L[-1].postsyns['NMDA'][0].gmax_factor = 0
                            if "nopre" in str(self.tau1_ex[nt]):
                                print "- no pre"
                                self.cells[nt][i].MF_L[-1].postsyns['AMPA'][0].tau_rec = 1e-9
                                self.cells[nt][i].MF_L[-1].postsyns['AMPA'][0].tau_facil  = 1e-9
                                self.cells[nt][i].MF_L[-1].postsyns['AMPA'][0].tau_1 = 0

                        if "nostdampa" in str(self.tau1_ex[nt]):
                            self.cells[nt][i].MF_L[-1].postsyns['NMDA'][0].gmax_factor = 0
                            self.cells[nt][i].MF_L[-1].postsyns['AMPA'][0].tau_rec = 1e-9
                            self.cells[nt][i].MF_L[-1].postsyns['AMPA'][0].tau_facil  = 1e-9
                            self.cells[nt][i].MF_L[-1].postsyns['AMPA'][0].tau_1 = 0
                            self.cells[nt][i].MF_L[-1].postsyns['AMPA'][0].r6FIX = 0

                        if "nostdnmda" in str(self.tau1_ex[nt]):
                            self.cells[nt][i].MF_L[-1].postsyns['AMPA'][0].gmax_factor = 0
                            self.cells[nt][i].MF_L[-1].postsyns['NMDA'][0].tau_rec = 1e-9
                            self.cells[nt][i].MF_L[-1].postsyns['NMDA'][0].tau_facil  = 1e-9
                            self.cells[nt][i].MF_L[-1].postsyns['NMDA'][0].tau_1 = 0
                            self.cells[nt][i].MF_L[-1].postsyns['NMDA'][0].RdRate = 0

                        if "nmda" in str(self.tau1_ex[nt]):
                            self.cells[nt][i].MF_L[-1].postsyns['AMPA'][0].gmax_factor = 0
                            if "nopre" in str(self.tau1_ex[nt]):
                                self.cells[nt][i].MF_L[-1].postsyns['NMDA'][0].tau_rec = 1e-9
                                self.cells[nt][i].MF_L[-1].postsyns['NMDA'][0].tau_facil  = 1e-9
                                self.cells[nt][i].MF_L[-1].postsyns['NMDA'][0].tau_1 = 0

                        if "nostdgr" in str(self.tau1_ex[nt]):
                            self.cells[nt][i].MF_L[-1].postsyns['AMPA'][0].r6FIX	= 0 #1.12
                            self.cells[nt][i].MF_L[-1].postsyns['NMDA'][0].RdRate = 0 #12e-3
                            print "- no std"

                        if "nomggr" in str(self.tau1_ex[nt]):
                            self.cells[nt][i].MF_L[-1].postsyns['NMDA'][0].v0_block = -1e9
                            print "- no mg block"

                        e0 = len(self.cells[nt][i].MF_L)-1 # get number of current synapse!

                        self.cells[nt][i].pconnect(self.pc,pulse_gid,int(e0),'mf')

                        if 'gsyn_in' in self.method_interpol:
                            self.record_syn.append(h.Vector())
                            self.record_syn[-1].record(self.cells[nt][i].MF_L[-1].postsyns['AMPA'][0]._ref_g)
                            self.record_syn.append(h.Vector())
                            self.record_syn[-1].record(self.cells[nt][i].MF_L[-1].postsyns['NMDA'][0]._ref_g)
                            self.gsyn_in_fac.append(f)
                            self.gsyn_in_fac.append(f)


                elif self.celltype[nt] == 'Goc':

                    # delete old
                    if j == 0:
                        self.cells[nt][i].MF_L = []
                        self.cells[nt][i].mfncpc = []

                    if "go" not in str(self.tau1_ex[nt]):

                        tau1 = self.tau1_ex[nt]
                        tau2 = self.tau2_ex[nt]

                        if tau1 == 0:
                            syn = h.ExpSyn(self.cells[nt][i].soma(0.5))
                            syn.tau = tau2/ms

                        else:
                            syn = h.Exp2Syn(self.cells[nt][i].soma(0.5))
                            syn.tau1 = tau1/ms
                            syn.tau2 = tau2/ms

                        syn.e = 0/mV

                        self.cells[nt][i].MF_L.append(syn)

                        e0 = len(self.cells[nt][i].MF_L)-1 # get number of current synapse!

                        syn_idx = int(e0)

                        source = int(pulse_gid)
                        self.cells[nt][i].mfncpc.append(self.pc.gid_connect(source, self.cells[nt][i].MF_L[syn_idx]))
                        self.cells[nt][i].mfncpc[-1].delay = 1
                        self.cells[nt][i].mfncpc[-1].weight[0] = w

                        if 'gsyn_in' in self.method_interpol:
                            self.record_syn.append(h.Vector())
                            self.record_syn[-1].record(self.cells[nt][i].MF_L[-1]._ref_g)
                            self.gsyn_in_fac.append(f)
                    else:

                        nrel = 0

                        mg = self.mglufac_ex[0]
                        if self.mglufac_ex[1] > 0:
                            mg = np.random.normal(self.mglufac_ex[0], self.mglufac_ex[1]*self.mglufac_ex[0], 1).clip(min=0)  #  self.g_syn_ex_s[nt]

                        if "stoch" in str(self.tau1_ex[nt]):
                            nrel = 4

                        self.cells[nt][i].createsyn(nmf = 1, weight_gmax = w, nrel=nrel, mglufac = mg)

                        e0 = len(self.cells[nt][i].MF_L)-1 # get number of current synapse!

                        self.cells[nt][i].pconnect(self.pc,pulse_gid,int(e0),'mf')

                        if 'gsyn_in' in self.method_interpol:
                            self.record_syn.append(h.Vector())
                            self.record_syn[-1].record(self.cells[nt][i].MF_L[-1].postsyns['AMPA'][0]._ref_g)
                            self.record_syn.append(h.Vector())
                            self.record_syn[-1].record(self.cells[nt][i].MF_L[-1].postsyns['NMDA'][0]._ref_g)
                            self.gsyn_in_fac.append(f)
                            self.gsyn_in_fac.append(f)

                elif self.celltype[nt] == 'IfCell':

                    # delete old
                    if j == 0:
                        self.cells[nt][i].synlist = []
                        self.cells[nt][i].nc = []

                    if "gr" in str(self.tau1_ex[nt]):

                        self.cells[nt][i].whatami = "grc"

                        nrel = 0
                        if "stoch" in str(self.tau1_ex[nt]):
                            nrel = 4

                        self.cells[nt][i].MF_L = self.cells[nt][i].synlist
                        self.cells[nt][i].synlist.append(Synapse('glom', self.cells[nt][i], self.cells[nt][i].soma, nrel=nrel, record_all=0, weight_gmax = w))

                        if "ampa" in str(self.tau1_ex[nt]):
                            self.cells[nt][i].synlist[-1].postsyns['NMDA'][0].gmax_factor = 0
                            if "nopre" in str(self.tau1_ex[nt]):
                                print "- no pre"
                                self.cells[nt][i].synlist[-1].postsyns['AMPA'][0].tau_rec = 1e-9
                                self.cells[nt][i].synlist[-1].postsyns['AMPA'][0].tau_facil  = 1e-9
                                self.cells[nt][i].synlist[-1].postsyns['AMPA'][0].tau_1 = 0

                        if "nmda" in str(self.tau1_ex[nt]):
                            self.cells[nt][i].synlist[-1].postsyns['AMPA'][0].gmax_factor = 0
                            if "nopre" in str(self.tau1_ex[nt]):
                                self.cells[nt][i].synlist[-1].postsyns['NMDA'][0].tau_rec = 1e-9
                                self.cells[nt][i].synlist[-1].postsyns['NMDA'][0].tau_facil  = 1e-9
                                self.cells[nt][i].synlist[-1].postsyns['NMDA'][0].tau_1 = 0

                        if "nostdampa" in str(self.tau1_ex[nt]):
                            self.cells[nt][i].synlist[-1].postsyns['AMPA'][0].tau_rec = 1e-9
                            self.cells[nt][i].synlist[-1].postsyns['AMPA'][0].tau_facil  = 1e-9
                            self.cells[nt][i].synlist[-1].postsyns['AMPA'][0].tau_1 = 0
                            self.cells[nt][i].synlist[-1].postsyns['AMPA'][0].r6FIX	= 0 #1.12

                        if "nostdnmda" in str(self.tau1_ex[nt]):
                            self.cells[nt][i].synlist[-1].postsyns['NMDA'][0].tau_rec = 1e-9
                            self.cells[nt][i].synlist[-1].postsyns['NMDA'][0].tau_facil  = 1e-9
                            self.cells[nt][i].synlist[-1].postsyns['NMDA'][0].tau_1 = 0
                            self.cells[nt][i].synlist[-1].postsyns['NMDA'][0].RdRate = 0

                        if "nostdgr" in str(self.tau1_ex[nt]):
                            self.cells[nt][i].synlist[-1].postsyns['AMPA'][0].r6FIX	= 0 #1.12
                            self.cells[nt][i].synlist[-1].postsyns['NMDA'][0].RdRate = 0 #12e-3
                            print "- no std"

                        if "nomggr" in str(self.tau1_ex[nt]):
                            self.cells[nt][i].synlist[-1].postsyns['NMDA'][0].v0_block = -1e9 #.k_block  = 1e-9
                            print "- no mg block"

                        e0 = len(self.cells[nt][i].synlist)-1
                        syn_idx = int(e0)

                        source = int(pulse_gid)
                        self.cells[nt][i].nc.append(self.pc.gid_connect(source, self.cells[nt][i].synlist[syn_idx].input))
                        self.cells[nt][i].nc[-1].delay = 1
                        self.cells[nt][i].nc[-1].weight[0] = 1

                        if 'gsyn_in' in self.method_interpol:
                            self.record_syn.append(h.Vector())
                            self.record_syn[-1].record(self.cells[nt][i].synlist[syn_idx].postsyns['AMPA'][0]._ref_g)
                            self.record_syn.append(h.Vector())
                            self.record_syn[-1].record(self.cells[nt][i].synlist[syn_idx].postsyns['NMDA'][0]._ref_g)
                            self.gsyn_in_fac.append(f)
                            self.gsyn_in_fac.append(f)
                    else:

                        if "amfit" in str(self.tau1_ex):

                            syn = h.ExpGrcSyn(self.cells[nt][i].soma(0.5))

                            syn.tau1_ampa = 0.254
                            syn.tau2_ampa = 0.254
                            syn.tau3_ampa = 0.363
                            syn.tau4_ampa = 6.523
                            syn.f1_ampa = 8.8376e-05
                            syn.f2_ampa = 5.5257e-05

                            syn.f1_nmda = 0

                            self.cells[nt][i].synlist.append(syn) # synlist is defined in Cell

                        elif "nmfit" in str(self.tau1_ex):

                            syn = h.ExpGrcSyn(self.cells[nt][i].soma(0.5))

                            syn.f1_ampa = 0
                            syn.f2_ampa = 0

                            syn.tau1_nmda = 1.902
                            syn.tau2_nmda = 82.032
                            syn.f1_nmda = 7.853857483005277e-05

                            self.cells[nt][i].synlist.append(syn) # synlist is defined in Cell

                        elif "fit" in str(self.tau1_ex):

                            syn = h.ExpGrcSyn(self.cells[nt][i].soma(0.5))

                            syn.tau1_ampa = 0.254
                            syn.tau2_ampa = 0.254
                            syn.tau3_ampa = 0.363
                            syn.tau4_ampa = 6.523
                            syn.f1_ampa = 8.8376e-05
                            syn.f2_ampa = 5.5257e-05

                            syn.tau1_nmda = 1.902
                            syn.tau2_nmda = 82.032
                            syn.f1_nmda = 7.853857483005277e-05

                            self.cells[nt][i].synlist.append(syn) # synlist is defined in Cell

                        else:

                            self.cells[nt][i].create_synapses(n_ex=1, tau1=self.tau1_ex[nt], tau2=self.tau2_ex[nt])


                        e0 = len(self.cells[nt][i].synlist)-1
                        syn_idx = int(e0)

                        self.cells[nt][i].pconnect_target(self.pc, source=pulse_gid, target=int(e0), syntype='ex', weight=w, delay=1)

                        if 'gsyn_in' in self.method_interpol:
                            self.record_syn.append(h.Vector())
                            self.record_syn[-1].record(self.cells[nt][i].synlist[syn_idx]._ref_g)
                            self.gsyn_in_fac.append(f)

                elif self.celltype[nt] == 'Prk':

                    # delete old
                    if j == 0:
                        self.cells[nt][i].PF_Lsync = []
                        self.cells[nt][i].spk_nc_pfsync = []
                        self.cells[nt][i].pfrand = []

                        m = len(self.cells[nt][i].dendrange)

                        seed = int(4*gid)
                        np.random.seed(seed)

                        for k in xrange(nsyn):
                            m -= 1
                	    mi = np.random.randint(0, m)
                	    self.cells[nt][i].dendrange[mi], self.cells[nt][i].dendrange[m] = self.cells[nt][i].dendrange[m], self.cells[nt][i].dendrange[mi]
                	    self.cells[nt][i].pfrand.append(self.cells[nt][i].dendrange[m])

                        #print self.cells[nt][i].pfrand

                    if "prk" not in str(self.tau1_ex[nt]):
                        pass
                    else:
                        self.cells[nt][i].PF_Lsync.append(Synapse2('pf',self.cells[nt][i],self.cells[nt][i].pfrand[j],record_all=0))

                        e0 = len(self.cells[nt][i].PF_Lsync)-1 # get number of current synapse!
                        syn_idx = int(e0)

                        self.cells[nt][i].spk_nc_pfsync.append(self.pc.gid_connect(pulse_gid, self.cells[nt][i].PF_Lsync[syn_idx].input.newnetstim))
                        self.cells[nt][i].spk_nc_pfsync[-1].delay = 1
                        self.cells[nt][i].spk_nc_pfsync[-1].weight[0] = 1

                        if 'gsyn_in' in self.method_interpol:
                            self.record_syn.append(h.Vector())
                            self.record_syn[-1].record(self.cells[nt][i].PF_Lsync[-1].postsyns['AMPA'][0]._ref_g)
                            self.gsyn_in_fac.append(f)

            elif syntype == "inh":

                w = self.g_syn_inh[nt]

                seed = int(10000 + 10*gid + j)
                np.random.seed(seed*42)

                if self.g_syn_inh_s[nt] > 0:
                    w = np.random.normal(w, w*self.g_syn_inh_s[nt], 1).clip(min=w*0.1)  # self.g_syn_inh_s[nt]

                if self.celltype[nt] == 'Grc':

                    if j == 0:
                        self.cells[nt][i].GOC_L = []
                        self.cells[nt][i].gocncpc = []

                    if "gr" not in str(self.tau1_inh[nt]):

                        tau1 = self.tau1_inh[nt]
                        tau2 = self.tau2_inh[nt]

                        if tau1 == 0:
                            syn = h.ExpSyn(self.cells[nt][i].soma(0.5))
                            syn.tau = tau2/ms

                        else:
                            syn = h.Exp2Syn(self.cells[nt][i].soma(0.5))
                            syn.tau1 = tau1/ms
                            syn.tau2 = tau2/ms

                        syn.e = -65

                        self.cells[nt][i].GOC_L.append(syn)

                        i0 = len(self.cells[nt][i].GOC_L)-1 # get number of current synapse!

                        syn_idx = int(i0)
                        source = int(pulse_gid)
                        self.cells[nt][i].gocncpc.append(self.pc.gid_connect(source, self.cells[nt][i].GOC_L[syn_idx]))
                        self.cells[nt][i].gocncpc[-1].delay = 1
                        self.cells[nt][i].gocncpc[-1].weight[0] = w

                    else:

                        self.cells[nt][i].createsyn(nmf = 0, ngoc = 1, weight_gmax = w)
                        i0 = len(self.cells[nt][i].GOC_L)-1 # get number of current synapse!
                        self.cells[nt][i].pconnect(self.pc,pulse_gid,int(i0),'goc')


                if self.celltype[nt] == 'IfCell':

                    if j == 0:
                        self.cells[nt][i].synlist_inh = []
                        self.cells[nt][i].nc_inh = []

                    if "gr" in str(self.tau1_inh[nt]):

                        nrel = 0
                        if "stoch" in str(self.tau1_ex[nt]):
                            nrel = 4

                        self.cells[nt][i].GOC_L = self.cells[nt][i].synlist
                        self.cells[nt][i].whatami = "grc"
                        self.cells[nt][i].synlist_inh.append(Synapse('goc', self.cells[nt][i], self.cells[nt][i].soma, nrel=nrel, record_all=0, weight_gmax = w))

                        i0 = len(self.cells[nt][i].synlist_inh)-1
                        syn_idx = int(i0)

                        source = int(pulse_gid)
                        self.cells[nt][i].nc_inh.append(self.pc.gid_connect(source, self.cells[nt][i].synlist_inh[syn_idx].input))
                        self.cells[nt][i].nc_inh[-1].delay = 1
                        self.cells[nt][i].nc_inh[-1].weight[0] = 1

                        if "gaba" in str(self.tau1_ex[nt]):

                            if 'gsyn_in' in self.method_interpol:

                                if "nostdgaba" in str(self.tau1_ex[nt]):

                                    self.cells[nt][i].synlist_inh[syn_idx].postsyns['GABA'][0].tau_rec = 1e-9
                                    self.cells[nt][i].synlist_inh[syn_idx].postsyns['GABA'][0].tau_facil = 1e-9
                                    self.cells[nt][i].synlist_inh[syn_idx].postsyns['GABA'][0].tau_1 = 0
                                    self.cells[nt][i].synlist_inh[syn_idx].postsyns['GABA'][0].d3 = 0
                                    self.cells[nt][i].synlist_inh[syn_idx].postsyns['GABA'][0].d1d2 = 0
                                    self.cells[nt][i].synlist_inh[syn_idx].postsyns['GABA'][0].d1 = 0
                                    self.cells[nt][i].synlist_inh[syn_idx].postsyns['GABA'][0].d2 = 0
                                    self.cells[nt][i].synlist_inh[syn_idx].postsyns['GABA'][0].d3_a6 = 0
                                    self.cells[nt][i].synlist_inh[syn_idx].postsyns['GABA'][0].d1d2_a6 = 0
                                    self.cells[nt][i].synlist_inh[syn_idx].postsyns['GABA'][0].d1_a6 = 0
                                    self.cells[nt][i].synlist_inh[syn_idx].postsyns['GABA'][0].d2_a6 = 0

                                self.record_syn.append(h.Vector())
                                self.record_syn[-1].record(self.cells[nt][i].synlist_inh[syn_idx].postsyns['GABA'][0]._ref_g)
                                self.gsyn_in_fac.append(f)

                    else:

                        self.cells[nt][i].create_synapses(n_inh=1, tau1_inh=self.tau1_inh[nt], tau2_inh=self.tau2_inh[nt], e_inh=-65)
                        i0 = len(self.cells[nt][i].synlist_inh)-1
                        syn_idx = int(i0)
                        self.cells[nt][i].pconnect_target(self.pc, source=pulse_gid, target=int(i0), syntype='inh', weight=w, delay=1)


            elif syntype == "intr":

                if self.celltype[nt] == 'Prk':

                    pass


    def set_SynPlay(self, farray, tarray, N = [], t_startstop = [], amode = 1):

        if self.do_run:

            delay = 1
            if (self.use_pc is False):
                delay = 0.1

            if N == []:
                N = self.N

            self.pulse_list = []
            self.global_pulse_list = []
            self.global_pulse_list_inh = []
            self.global_pulse_list_intr = []

            f_cells_mean_local = []
            f_cells_cv_local = []
            f_cells_std_local = []

            for nt in range(self.n_celltypes): # loop over all cells

                if (self.n_syn_ex[nt] > 0) or (self.n_syn_inh[nt] > 0) or (self.n_syn_intr[nt] > 0):

                    local_gid_count = 0
                    local_gid_count_type = []


                    # EXCITATION
                    if str(type(self.g_syn_ex[nt] )) is not ndarray: self.g_syn_ex[nt]  = np.array([self.g_syn_ex[nt] ]) # each cell can receive different g_syn_ex !

                    if len(self.g_syn_ex[nt]) == len(self.global_gidlist[nt]):
                        pass
                    else:
                        self.g_syn_ex[nt] = np.ones(len(self.global_gidlist[nt]))*self.g_syn_ex[nt][0]
                        #print "- single value in g_syn_ex, cells:", len(self.global_gidlist[nt])

                    self.global_pulse_list.append([])
                    for ns in range(self.n_syn_ex[nt]): # loop over all excitatory synapses!
                        self.global_pulse_list[-1].append([])
                        for n in range(self.syn_max_mf[nt]): # number of cells of this celltype
                            self.global_pulse_list[-1][-1].append(local_gid_count+self.gid_count)
                            local_gid_count += 1
                            local_gid_count_type.append([])
                            local_gid_count_type[-1].append('ex')
                            local_gid_count_type[-1].append(n) # number of cell within their population 0..N[nt]
                            local_gid_count_type[-1].append(ns) # number of synapse


                    # INHIBITION
                    if np.array(self.inh_hold[nt]).size <= 1:
                        self.inh_hold[nt] = np.ones(len(self.global_gidlist[nt]))*self.inh_hold[nt]
                        #print "- single value in inh_hold", self.inh_hold[nt]


                    self.global_pulse_list_inh.append([])
                    for ns in range(self.n_syn_inh[nt]): # loop over all inhibitory synapses!
                        self.global_pulse_list_inh[-1].append([])
                        for n in range(self.syn_max_inh[nt]): # number of cells of this celltype
                            self.global_pulse_list_inh[-1][-1].append(local_gid_count+self.gid_count)
                            local_gid_count += 1
                            local_gid_count_type.append([])
                            local_gid_count_type[-1].append('inh')
                            local_gid_count_type[-1].append(n) # number of cell within their population 0..N[nt]
                            local_gid_count_type[-1].append(ns) # number of synapse


                    # INTRUDER SYNAPSE
                    if str(type(self.g_syn_intr[nt] )) is not ndarray: self.g_syn_intr[nt]  = np.array([self.g_syn_intr[nt] ]) # each cell can receive different g_syn_intr !

                    if len(self.g_syn_intr[nt]) == len(self.global_gidlist[nt]):
                        pass
                    else:
                        self.g_syn_intr[nt] = np.ones(len(self.global_gidlist[nt]))*self.g_syn_intr[nt][0]
                        #print "- single value in g_syn_intr, cells:", len(self.global_gidlist[nt])

                    self.global_pulse_list_intr.append([])
                    for ns in range(self.n_syn_intr[nt]): # loop over all intruding synapses!
                        self.global_pulse_list_intr[-1].append([])
                        for n in range(self.syn_max_intr[nt]): # number of generators for this celltype
                            self.global_pulse_list_intr[-1][-1].append(local_gid_count+self.gid_count)
                            local_gid_count += 1
                            local_gid_count_type.append([])
                            local_gid_count_type[-1].append('intr')
                            local_gid_count_type[-1].append(n) # number of cell within their population 0..N[nt]
                            local_gid_count_type[-1].append(ns) # number of synapse


                    t_vec_input = np.array([]) # input trains
                    id_vec_input = np.array([]) # input trains id
                    fs = 1 / self.dt
                    ih_use_v = []

                    for i in range(int(self.id), local_gid_count, int(self.nhost)): # loop over all train generators and generate them

                        self.pulse_list.append(i+self.gid_count)
                        pulse_gid = self.pulse_list[-1]
                        gid = local_gid_count_type[i][1] # should correspond to this gid when multiple values inserted

                        if local_gid_count_type[i][0] == 'ex':

                            seed = int(10001 + pulse_gid) #  unique gid for generators!
                            np.random.seed(seed*423)

                            if self.ihold_sigma[nt] > 0:
                                ih_use = np.random.normal(self.ihold[nt], self.ihold[nt]*self.ihold_sigma[nt], 1).clip(min=0) # self.ihold[nt]*self.ihold_sigma[nt]

                            elif self.ihold_sigma[nt] < 0:
                                ih_use = np.random.uniform(0.1, self.ihold[nt])

                            else:
                                ih_use = self.ihold[nt]

                            ih_use_v.append(ih_use)

                            if ih_use > 0:
                                # train has to be contructed here, to insert different train into each "dendrite"
                                ## different ihold has to be implemented here!!
                                iholdvec = concatenate((zeros(round(fs)), ones(round(len(tarray) - 1 * fs)) * ih_use))

                                if isinstance(self.syn_ex_dist[nt], ( tuple ) ): # distribution of amplitude, only one noise source!

                                    np.random.seed(pulse_gid*40)
                                    if self.syn_ex_dist[nt][1] > 0:
                                        f = np.random.normal(self.syn_ex_dist[nt][0], self.syn_ex_dist[nt][1], 1).clip(min=0)
                                    else:
                                        f = self.syn_ex_dist[nt][0]

                                    f2 = f
                                    rnd.seed(pulse_gid*40) # use gid so type 1, 2 is identical for each cell
                                    #rnd.seed(gid*40) # use gid so type 1, 2 is identical for each cell
                                    if self.syn_ex_dist[nt][2] > 0: # add inverted input with 50% probability, in future versions this will indicate the propability for -1 and 1
                                        f2 = rnd.sample([-1,1],1)[0] * f
                                        #f2 = f

                                    if amode == 1:
                                        inamp = (f2 * self.amod[nt] * ih_use)
                                    elif amode == 2:
                                        inamp = (f2 * self.amod[nt] * self.ihold[nt])

                                    modulation = (tarray, inamp * farray[0] + iholdvec)

                                    #if self.id == 0: print "- randomize play stimulus height, pulse_gid=", pulse_gid, " gid=", gid ," f=", f
                                    if (gid==0): print "- randomize play stimulus height, pulse_gid=", pulse_gid, " gid=", gid ," f2=", f2,"inamp=",inamp

                                    #rnd.seed(local_gid_count_type[i][1]*300) # pick seed based on number of cell
                                    #nj = rnd.sample(range(len(farray)),1)[0]
                                    nj = 1

                                else: # different noise sources can be used at different synapses, linear combination test in openloop

                                    nj = self.syn_ex_dist[nt][local_gid_count_type[i][2]]

                                    if nj == 0:
                                        modulation = (tarray, iholdvec)
                                    else:
                                        if amode == 1:
                                            inamp = (self.factor_celltype[nt] * self.amod[nt] * ih_use)
                                        elif amode == 2:
                                            inamp = (self.factor_celltype[nt] * self.amod[nt] * self.ihold[nt])

                                        modulation = (tarray, inamp * farray[nj-1] + iholdvec)
                                        if self.id == 0: print "ex farray number:", nj-1, "ih_use:", ih_use, "self.amod[nt]:", self.amod[nt], "inamp: ", inamp


                                # will be done n_syn_ex * number of cells!
                                if self.noise_syn_tau[nt] < 0: # variable threshold
                                    no = self.noise_syn[nt]
                                else:
                                    no = self.noise_syn[nt]*ih_use

                                train, self.n_train_ex = mod_spike_train(modulation, noise = no, seed = seed, noise_tau = self.noise_syn_tau[nt], noise_a = self.noise_a[nt])

                                #plt.figure("input")
                                #plt.plot(train, train*0, '|')
                                #plt.show()

                                t_vec_input = np.append(t_vec_input, train*ms).flatten() # use ms to save!!
                                id_vec_input = np.append(id_vec_input, np.ones(len(train))*pulse_gid).flatten()

                                f_cells_mean_local0, f_cells_cv_local0, f_cells_std_local0 = self.calc_fmean(train*ms, t_startstop)
                                f_cells_mean_local.append(f_cells_mean_local0); f_cells_cv_local.append(f_cells_cv_local0); f_cells_std_local.append(f_cells_std_local0)

                                if self.id == 0: print "TRAIN: requ. mean:", ih_use ,"eff. mean:", f_cells_mean_local0, "cv: " , f_cells_cv_local0, "std:" , f_cells_std_local0

                            else:
                                train = []
                                self.n_train_ex = []



                        elif local_gid_count_type[i][0] == 'intr':

                            # train has to be contructed here, to insert different train into each "dendrite"
                            nj = 0

                            seed = int(10001 + pulse_gid)
                            np.random.seed(seed*4411)

                            if self.intr_hold_sigma[nt] > 0:
                                ih_use = np.random.normal(self.intr_hold[nt], self.intr_hold[nt]*self.intr_hold_sigma[nt], 1).clip(min=0)
                            else:
                                ih_use = self.intr_hold[nt]

                            ih_use_v.append(ih_use)

                            if ih_use > 0:

                                iholdvec = concatenate((zeros(round(fs)), ones(round(len(tarray) - 1 * fs)) * ih_use))
                                modulation = (tarray, iholdvec)

                                # will be done n_syn_in * number of cells!
                                if self.noise_syn_tau_intr[nt] < 0: # variable threshold
                                    no = self.noise_syn_intr[nt]
                                else:
                                    no = self.noise_syn_intr[nt]*ih_use

                                if self.noise_syn_tau_intr[nt] >= -1:
                                    train, _ = mod_spike_train(modulation, noise = no, seed = seed, noise_tau = self.noise_syn_tau_intr[nt], noise_a = self.noise_a_intr[nt]) # train in ms
                                else:
                                    train = oscill_spike_train(sor = 4, spike_prob = 1/4, noise_fraction = 4, end_time = tarray[-1]/ms, seed = seed)


                        elif local_gid_count_type[i][0] == 'inh':

                            # train has to be contructed here, to insert different train into each "dendrite"

                            seed = int(10001 + pulse_gid)

                            np.random.seed(seed*44)

                            if self.inh_hold_sigma[nt] > 0:
                                ih_use = np.random.normal(self.inh_hold[nt][gid], self.inh_hold[nt][gid]*self.inh_hold_sigma[nt], 1).clip(min=0)
                            else:
                                ih_use = self.inh_hold[nt][gid]


                            iholdvec = concatenate((zeros(round(fs)), ones(round(len(tarray) - 1 * fs)) * ih_use))

                            nj = self.syn_inh_dist[nt][local_gid_count_type[i][2]]
                            if nj == 0:
                                modulation = (tarray, iholdvec)
                            else:
                                inamp = (self.amod[nt] * ih_use)
                                modulation = (tarray, inamp * farray[nj-1] + iholdvec)
                                #print "inh farray number:", nj-1, "ih_use:", ih_use, "amp: ", inamp #old: nj-1+nemax

                            # will be done n_syn_in * number of cells!
                            if self.noise_syn_tau_inh[nt] < 0: # variable threshold
                                no = self.noise_syn_inh[nt]
                            else:
                                no = self.noise_syn_inh[nt]*ih_use

                            train, _ = mod_spike_train(modulation, noise = no, seed = seed, noise_tau = self.noise_syn_tau_inh[nt], noise_a = self.noise_a_inh[nt]) # train in ms
                            #print train

                        #print train
                        if len(train) > 0:
                            if self.id == 0:
                                print "-", pulse_gid, local_gid_count_type[i], "seed: ", seed, "ih_use:", ih_use, no, nj #, "first spike: ", train[0]
                            self.setup_Play_train(train = train+self.inh_delay, input_gid = pulse_gid, delay = delay) # train in ms


                    self.gid_count += local_gid_count # increase gid count

                    self.barrier()

                    for i, gid in enumerate(self.gidlist[nt]):  # for all input cells

                        rnd.seed(gid*200)
                        n = self.global_gidlist[nt].index(gid) # index of cell within their population 0..N[nt]
                        # i is index on this node only!

                        self.record_syn = []
                        for j in range(self.n_syn_ex[nt]):
                            if N[nt] == len(self.global_pulse_list[nt][j]):
                                pulse_gid = self.global_pulse_list[nt][j][n] #every cell of this type receives one pulse gid
                                if self.id == 0: print "- gid:", gid ," n:", n ," one ex train for each synapse:", pulse_gid, "self.g_syn_ex[nt][n]:", self.g_syn_ex[nt][n]
                            else:
                                pulse_gid = rnd.sample(self.global_pulse_list[nt][j],1)[0] # not enough, just pick one at random, for inh/f search only one synapse available!
                                if self.id == 0: print "- gid:", gid ," n:", n ," one ex train from", len(self.global_pulse_list[nt][j]), ":", pulse_gid, "self.g_syn_ex[nt][n]:", self.g_syn_ex[nt][n]

                            if "gaba" in str(self.tau1_ex[nt]):
                                self.connect_Synapse(pulse_gid, nt, i, n, gid, j, syntype = "inh")
                            else:
                                self.connect_Synapse(pulse_gid, nt, i, n, gid, j, syntype = "ex", nsyn = self.n_syn_ex[nt])


                        if self.n_syn_inh[nt] > 0:
                            for j in range(self.n_syn_inh[nt]):

                                if N[nt] == len(self.global_pulse_list_inh[nt][j]):
                                    pulse_gid = self.global_pulse_list_inh[nt][j][n] #every cell of this type receives one pulse gid
                                    if self.id == 0: print "- one inh train for each synapse:", pulse_gid
                                else:
                                    pulse_gid = rnd.sample(self.global_pulse_list_inh[nt][j],1)[0] # not enough, just pick one at random
                                    if self.id == 0: print "- one inh train from", len(self.global_pulse_list_inh[nt][j]), ":", pulse_gid

                                self.connect_Synapse(pulse_gid, nt, i, n, gid, j, syntype = "inh")


                        if self.n_syn_intr[nt] > 0:
                            for j in range(self.n_syn_intr[nt]):

                                if N[nt] == len(self.global_pulse_list_intr[nt][j]):
                                    pulse_gid = self.global_pulse_list_intr[nt][j][n] #every cell of this type receives one pulse gid
                                    if self.id == 0: print "- one intruding train for each synapse:", pulse_gid
                                else:
                                    pulse_gid = rnd.sample(self.global_pulse_list_intr[nt][j],1)[0] # not enough, just pick one at random
                                    if self.id == 0: print "- one intruding train from", len(self.global_pulse_list_intr[nt][j]), ":", pulse_gid

                                if (self.use_pc is False):

                                    if self.celltype[nt] == 'Prk': self.cells[nt][i].delrerun()

                                    (msg,CF_input) = self.cells[nt][i].createsyn_CF(record_all=0,factor=self.g_syn_intr[nt][0],cf_setup_select='old')
                                    CF_input.number = 3 # three bursts
                                    CF_input.start = -0.3 # See synapsepfpurk.py
                                    CF_input.interval = 3 # 3 ms interval between bursts

                                    self.cells[nt][i].input_to_CF_nc.append(h.NetCon(self.vecstim[j], CF_input, 0, 0.1, 1))
                                    self.netcons.append(self.cells[nt][i].input_to_CF_nc[-1])

                                else:
                                    print "NOT IMPLEMENTED"


                    if self.id == 0: print "trains connected"

                    if local_gid_count_type[i][0] == 'intr':
                        pass
                    else:
                        self.id_all_vec_input.append(self.do_gather(id_vec_input, dtype = 'i'))
                        self.t_all_vec_input.append(self.do_gather(t_vec_input))

                        f_cells_mean = self.do_gather(f_cells_mean_local)
                        f_cells_cv = self.do_gather(f_cells_cv_local)
                        f_cells_std = self.do_gather(f_cells_std_local)

                    self.fmean_input = np.nan
                    self.fmax_input = np.nan
                    self.fmstd_input = np.nan
                    self.fcvm_input = np.nan
                    self.fstdm_input = np.nan

                    ih_use_v_all = self.do_gather(ih_use_v)

                    if self.id == 0 and local_gid_count_type[i][0] != 'intr':

                        self.fmean_input = mean(np.nan_to_num(f_cells_mean))  # compute mean of mean rate for all cells
                        self.fmstd_input = std(np.nan_to_num(f_cells_mean))
                        self.fmax_input = max(np.nan_to_num(f_cells_mean))

                        self.fcvm_input = mean(f_cells_cv[~np.isnan(f_cells_cv)])
                        self.fstdm_input = mean(f_cells_std[~np.isnan(f_cells_std)])

                        self.ih_use_max = max(ih_use_v_all)

                        print "- trains, fmean: ",self.fmean_input, "fmax: ",self.fmax_input, "Hz", "fmstd: ",self.fmstd_input, "Hz", "fcvm: ",self.fcvm_input, "fstdm: ",self.fstdm_input, "Hz, ih_use_max:", self.ih_use_max

                else:
                    self.global_pulse_list.append([])
                    self.global_pulse_list_inh.append([])



    def do_gather(self, v_local, dtype = 'd'):

        if self.use_mpi:

            self.barrier()

            #v_local = v_local.astype(dtype).flatten()
            v_local = np.array(v_local, dtype=dtype).flatten()

            if self.use_pc == False:

                v_global = None
                counts_local = np.array(len(v_local), dtype='i')

                counts = 0
                if self.id == 0:
                    counts = np.empty(self.nhost, dtype='i')

                self.comm.Gather(sendbuf=[counts_local, MPI.INT], recvbuf=[counts, MPI.INT], root=0)

                if self.id == 0:
                    v_global = np.empty(sum(counts), dtype=dtype)


                if dtype == 'd':
                    self.comm.Gatherv(sendbuf=[v_local, MPI.DOUBLE], recvbuf=[v_global, (counts, None), MPI.DOUBLE], root=0)
                elif dtype == 'i':
                    self.comm.Gatherv(sendbuf=[v_local, MPI.INT], recvbuf=[v_global, (counts, None), MPI.INT], root=0)

                #v_global = np.hstack(v_global)

            else:
                sendlist = [None]*self.nhost
                sendlist[0] =  v_local
                getlist = self.pc.py_alltoall(sendlist)

                v_global = np.hstack(getlist)

        else:

            v_global = np.hstack(v_local)

        return v_global


    def setup_Play_train(self, train = [], input_gid = 0, delay = 1):

        self.trains.append(train)

        # possibility to play spikes into the cells!
        self.vecstim.append(h.VecStim(.5))
        self.nc_vecstim.append(h.NetCon(self.vecstim[-1],None))
        self.nc_vecstim[-1].delay = delay

        self.spike_vec.append(h.Vector(self.trains[-1]))
        self.vecstim[-1].play(self.spike_vec[-1])

        if (self.use_mpi):
            self.pc.set_gid2node(input_gid, self.id)  # associate gid with this host
            self.pc.cell(input_gid,self.nc_vecstim[-1])  # associate gid with spike detector


    def record(self):
        """
        Initializes recording vectors. Internal function
        """

        if self.n_celltypes > 1:
            #print "self.n_borders:",self.n_borders
            for n in range(self.n_celltypes):
                if self.n_borders[n] in self.gidlist[n]:
                    #print "np.shape(self.rec_v):",np.shape(self.rec_v)
                    #print "np.shape(self.cells):",np.shape(self.cells)
                    self.rec_v[n].record(self.cells[n][0].soma(0.5)._ref_v)


        if self.id == 0:  # only for first node and first cell

            # Voltage
            self.rec_v[0].record(self.cells[self.a_celltype[0]][0].soma(0.5)._ref_v)

            # Stimuli
            self.rec_i = h.Vector()

            if (self.plays != []):
                if (isinstance(self.plays[0], list) is False):
                    self.rec_i.record(self.plays[0]._ref_i)
                else:
                    self.rec_i.record(self.plays[0][0]._ref_i)

            self.rec_ich = h.Vector()
            if self.ic_holds != [] and (isinstance(self.ic_holds[0], list) is False):
                self.rec_ich.record(self.ic_holds[0]._ref_i)

            self.rec_ics = h.Vector()
            if self.ic_starts != []:
                self.rec_ics.record(self.ic_starts[0]._ref_i)

            self.rec_n = h.Vector()

            if self.fluct_s[0] > 0:
                # Fluctuating input
                self.rec_n.record(self.flucts[0]._ref_i)
                print "recording noise"
            elif (len(self.flucts) > 0) and (len(self.fluct_g_i0)>0):
                self.rec_n.record(self.flucts[0]._ref_g_i)
                print "recording g noise"
            else:
                print "nonoise"

            if hasattr(self.cells[self.a_celltype[0]][0], 'lkg2_noise'):
                if self.cells[self.a_celltype[0]][0].lkg2_noise > 0:
                    self.rec_n.record(self.cells[self.a_celltype[0]][0].fluct._ref_il)
                    print "recording tonic gaba noise"

            self.rec_step = h.Vector()
            if self.ic_steps != []:
                self.rec_step.record(self.ic_steps[0]._ref_i)

            # Time
            self.rec_t = h.Vector()
            self.rec_t.record(h._ref_t)


    def run(self, tstop = 10*s, do_loadstate = True):
        """
        Starts the stimulation.
        """
        self.record()

        if self.first_run:

            if self.use_mpi: self.pc.set_maxstep(100)
            #self.pc.spike_compress(1) #test

            if self.use_multisplit:
                import multiprocessing

                Hines = h.CVode()
                Hines.active(0)

                h.load_file("parcom.hoc")
                p = h.ParallelComputeTool()

                if self.use_mpi:
                    cpus = multiprocessing.cpu_count() #32 #self.pc.nhost()
                else:
                    cpus = multiprocessing.cpu_count() #32

                p.change_nthread(cpus,1)
                p.multisplit(1)
                print "Using multisplit, cpus:", cpus

            else:

                h.load_file("stdrun.hoc")

            if self.use_local_dt:
                h.cvode.active(1)
                h.cvode.use_local_dt(1)

            h.celsius = self.temperature
            h.dt = self.dt/ms  # Fixed dt
            h.steps_per_ms = 1 / (self.dt/ms)

            if self.cells[self.a_celltype[0]] != []:
                if hasattr(self.cells[self.a_celltype[0]][0], 'v_init'):
                    h.v_init = self.cells[self.a_celltype[0]][0].v_init  # v_init is supplied by cell itself!
                else:
                    h.v_init = -60

            h.stdinit()

            h.finitialize()

            if hasattr(self.cells[self.a_celltype[0]][0], 'load_states') and do_loadstate:
                m = md5.new()
                cell_exe_new = self.cell_exe[0]
                m.update(cell_exe_new)
                filename = './states_' + self.celltype[0] + '_' + m.hexdigest() + '_Population.b'
                self.cells[self.a_celltype[0]][0].load_states(filename)

        else:

            pass


        if self.id == 0:
            import time
            t0 = time.time()

        if self.simstep == 0:
            if self.id == 0: print "Running without steps",

            if self.use_mpi:
                self.pc.psolve(tstop/ms)
            else:
                h.init()
                h.tstop = tstop/ms
                h.run()

        else:

            h.finitialize()
            cnt = 1

            #if self.id == 50:
            #    print len(self.cells[1][0].nc), self.cells[1][0].nc[0].weight[0]
            #    print len(self.cells[0][0].nc_inh), self.cells[0][0].nc_inh[0].weight[0]

            h.t = 0
            while h.t < tstop/ms:

                if self.id == 0:
                    print "Running...",
                    if self.use_mpi:
                        past_time = self.pc.time()

                h.continuerun(cnt*self.simstep/ms)
                if self.use_mpi: self.pc.barrier()

                if self.id == 0:
                    if self.use_mpi:
                        print "Simulated time =",h.t*ms, "s, Real time = ", (self.pc.time()-past_time), 's'
                    else:
                        print "Simulated time =",h.t*ms, "s"

                #if self.id == 0:
                #    print hpy.heap().byrcs
                cnt += 1

        if self.id == 0: print "psolve took ", time.time() - t0, "seconds"

        self.first_run = False

        self.barrier()   # wait for other nodes

        self.tstop = tstop


    def get(self, t_startstop=[], i_startstop=[], N = []):
        """
        Gets the recordings.
        """

        if N == []:
            N = self.N

        if t_startstop == []:
            t_startstop = np.array([2, self.tstop])

        t_all_vec = []
        id_all_vec = []

        fmean = []
        fbase = []
        fmax = []
        fmstd = []
        fcvm = []
        fstdm = []
        gid_del = []
        f_cells_mean_all = []
        f_cells_base_all = []
        f_cells_cv_all = []
        f_cells_std_all = []

        fmeanA = []
        fmstdA = []
        fmaxA = []
        fcvmA = []
        fstdmA = []
        fbaseA = []
        fbstdA = []

        if self.id == 0: print "start gathering spikes"

        for n in range(self.n_celltypes):

            if self.use_mpi:

                self.barrier()   # wait for other node
                t_vec = np.array(self.t_vec[n]).flatten()*ms - 1*ms # shift time because of output delay
                id_vec = np.array(self.id_vec[n]).flatten()

            else:

                t_vec = np.array([])
                id_vec = np.array([])
                print np.shape(self.t_vec)
                for i in self.gidlist[n]:
                    t_vec0 = np.array(self.t_vec[n][i]).flatten()*ms
                    t_vec = np.append(t_vec, t_vec0).flatten()
                    id_vec = np.append(id_vec, np.ones(len(t_vec0))*i).flatten()

            fmean0, fmax0, fmstd0, fcvm0, fstdm0, gid_del0, f_cells_mean_all0, f_cells_cv_all0, f_cells_std_all0, fbase0, f_cells_base_all0 = self.get_fmean(t_vec, id_vec, t_startstop = t_startstop, gidlist = self.gidlist[n])
            fmean.append(fmean0); fmax.append(fmax0), fmstd.append(fmstd0), fcvm.append(fcvm0), fstdm.append(fstdm0), gid_del.append(gid_del0), f_cells_mean_all.append(f_cells_mean_all0), f_cells_cv_all.append(f_cells_cv_all0), f_cells_std_all.append(f_cells_std_all0)
            fbase.append(fbase0); f_cells_base_all.append(f_cells_base_all0)

            t_all_vec.append(self.do_gather(t_vec))
            id_all_vec.append(self.do_gather(id_vec))

        if (self.id == 0) and (self.no_fmean == False):
            f_cells_mean_all = np.array(f_cells_mean_all).flatten()
            fmeanA = mean(f_cells_mean_all)  # compute mean of mean rate for all cells
            fmstdA = std(f_cells_mean_all)
            fmaxA = max(f_cells_mean_all)

            f_cells_base_all = np.array(f_cells_base_all).flatten()
            fbaseA = mean(f_cells_base_all)  # compute mean of mean rate for all cells
            fbstdA = std(f_cells_base_all)

            f_cells_cv_all = np.concatenate((np.array(f_cells_cv_all)))
            f_cells_std_all = np.concatenate((np.array(f_cells_std_all)))

            fcvmA = mean(f_cells_cv_all)
            fstdmA = mean(f_cells_std_all)

            print "- ALL, fmean: ",fmeanA, "fmax: ",fmaxA, "Hz", "fmstd: ",fmstdA, "Hz", "fcvm: ",fcvmA, "fstdm: ",fstdmA, "Hz", "fbase: ",fbaseA, "Hz", "fbstd: ", fbstdA, "Hz"

        if self.id == 0: print "all spikes have been gathered"

        self.barrier()

        # do this here to have something to return
        voltage = []
        current = []
        time = []

        freq_times = []
        spike_freq = []
        gsyn = []

        if self.id == 0:  # only for first node

            time = np.array(self.rec_t)*ms

            # use self.bin_width as bin width!
            freq_times = arange(0, time[-1], self.bin_width)

            voltage.append(np.array(self.rec_v[0])*mV)
            current = np.zeros(len(time))

            if len(np.array(self.rec_ics)) > 0:
                current = current + np.array(self.rec_ics)

            if len(np.array(self.rec_ich)) > 0:
                current = current + np.array(self.rec_ich)

            if len(np.array(self.rec_i)) > 0:
                current = current + np.array(self.rec_i)

            if len(np.array(self.rec_n)) > 0:
                current = current + np.array(self.rec_n)
                print np.array(self.rec_n)

            if len(np.array(self.rec_step)) > 0:
                current = current + np.array(self.rec_step)

        else:
            time = [0]

        self.barrier()
        time = self.broadcast(time, fast = True)

        gsyn_in = []
        gsyn_in0 = []

        if 'gsyn_in' in self.method_interpol:

            gsyn_in = None
            if self.id == 0: print "- collecting gsyn_in"
            gsyn_in0 = np.zeros(len(time), dtype='d')
            if self.record_syn is not []:
                for i, j in enumerate(self.record_syn):
                    gsyn_in0 = gsyn_in0 + self.gsyn_in_fac[i] * np.array(j, dtype='d')

            if self.use_mpi:
                count = len(time)

                #if self.id == 0: gsyn_in = np.empty(count*self.nhost, dtype='d')
                #self.comm.Gatherv(sendbuf=[gsyn_in0, MPI.DOUBLE], recvbuf=[gsyn_in, MPI.DOUBLE], root=0)

                gsyn_in = self.do_gather(gsyn_in0)

                if self.id == 0:
                    gsyn_in = np.reshape(gsyn_in, (self.nhost,count))
                    gsyn_in = sum(gsyn_in,0)

            else:
                gsyn_in = gsyn_in0

        self.barrier()   # wait for other nodes

        if self.n_celltypes > 1:
            if self.id == 0: print "more than one celltype send voltage of first other cell to root"

            for n in range(1, self.n_celltypes):

                if self.use_pc == True:

                    srclist = [None]*self.nhost

                    if (self.n_borders[n] in self.gidlist[n]):
                        srclist[0] = np.array(self.rec_v[n])*mV

                    destlist = self.pc.py_alltoall(srclist)

                    if self.id == 0:
                        idx = [i for i, x in enumerate(destlist) if x is not None]
                        if len(idx) > 1: raise ValueError('Error, too many vectors sent, should be one at a time!')
                        voltage.append(np.array(destlist[idx[0]]))

                else:

                    if self.id == 0:
                        if (self.n_borders[n] in self.gidlist[n]):  # first node has it, do not wait to receive it!
                            v_temp = np.array(self.rec_v[n])*mV
                        else:
                            v_temp = np.zeros(len(voltage[0]))
                            self.comm.Recv([v_temp, MPI.DOUBLE], source = MPI.ANY_SOURCE, tag=int(sum(N)+33))

                        voltage.append(v_temp)
                    else:
                        if self.n_borders[n] in self.gidlist[n]:
                            voltage = np.array(self.rec_v[n])*mV
                            self.comm.Ssend([voltage, MPI.DOUBLE], dest=0, tag=int(sum(N)+33))

        self.barrier()   # wait for other nodes

        times = arange(0, time[-1], 1*ms)
        gsyns = []
        if self.called_syn_out_all == True:

            for n in range(self.n_celltypes):
                gsyns.append([])

                if self.use_pc == True:

                    for i, gid in enumerate(self.global_gidlist[n]):

                        srclist = [None]*self.nhost

                        if gid in self.gidlist[n]: #only one node does this
                            a = np.array(self.cells[n][self.gidlist[n].index(gid)].record['gsyn'])
                            c = np.zeros(int((1*ms)/self.dt))
                            temp = np.append(a, c).flatten()
                            temp = temp[int((1*ms)/self.dt):len(temp)+1]
                            gtemp = interp(times,time,temp)

                            srclist[0] = gtemp # send to root only

                        destlist = self.pc.py_alltoall(srclist)

                        if self.id == 0:
                            idx = [i for i, x in enumerate(destlist) if x is not None]
                            if len(idx) > 1: raise ValueError('Error, too many vectors sent, should be one at a time!')
                            gsyns[n].append(np.array(destlist[idx[0]]))

                else:

                    for i, gid in enumerate(self.global_gidlist[n]):

                        if self.id == 0:
                            if gid in self.gidlist[n]:
                                a = np.array(self.cells[n][self.gidlist[n].index(gid)].record['gsyn'])
                                c = np.zeros(int((1*ms)/self.dt))
                                temp = np.append(a, c).flatten()
                                temp = temp[int((1*ms)/self.dt):len(temp)+1]
                                gtemp = interp(times,time,temp)

                            else:
                                gtemp = np.zeros(len(times))
                                self.comm.Recv([gtemp, MPI.DOUBLE], source = MPI.ANY_SOURCE, tag=int(gid))

                            gsyns[n].append(np.array(gtemp))

                        else:
                            if gid in self.gidlist[n]:
                                a = np.array(self.cells[n][self.gidlist[n].index(gid)].record['gsyn'])
                                c = np.zeros(int((1*ms)/self.dt))
                                temp = np.append(a, c).flatten()
                                temp = temp[int((1*ms)/self.dt):len(temp)+1]
                                gtemp = interp(times,time,temp)
                                #np.array(self.cells[n][self.gidlist[n].index(gid)].record['gsyn'])
                                self.comm.Ssend([gtemp, MPI.DOUBLE], dest=0, tag=int(gid))

            if self.id == 0: print "root gathered synaptic output conductance"


        self.barrier()   # wait for other nodes

        times = arange(0, time[-1], 10*ms)

        w_mat = []
        winh_mat = []

        if self.stdp_used == True:

            for n in range(self.n_celltypes):
                w_mat.append([])

                for i, gid in enumerate(self.global_gidlist[n]):

                    if self.id == 0:

                        wall = []

                        if gid in self.gidlist[n]:

                            walltemp = self.cells[n][self.gidlist[n].index(gid)].record['w']
                            if len(walltemp) > 0:
                                for l in range(len(walltemp)):
                                    wtemp = np.array(walltemp[l])
                                    wtemp = interp(times,time,wtemp)
                                    wall.append(wtemp)

                        else:

                            while 1:
                                wtemp = np.zeros(len(times))
                                self.comm.Recv([wtemp, MPI.DOUBLE], source = MPI.ANY_SOURCE, tag=int(gid))

                                if wtemp[0] == -1:
                                    break
                                else:
                                    wall.append(wtemp)

                        w_mat[n].append(wall)

                    else:
                        if gid in self.gidlist[n]:
                            walltemp = self.cells[n][self.gidlist[n].index(gid)].record['w']

                            if len(walltemp) > 0:
                                for l in range(len(walltemp)):
                                    wtemp = np.array(walltemp[l])
                                    wtemp = interp(times,time,wtemp)
                                    self.comm.Ssend([wtemp, MPI.DOUBLE], dest=0, tag=int(gid))

                            wtemp = np.ones(len(times))*-1
                            self.comm.Ssend([wtemp, MPI.DOUBLE], dest=0, tag=int(gid))

            if self.id == 0:
                print "root gathered synaptic input conductance"


            self.barrier()   # wait for other nodes


            for n in range(self.n_celltypes):
                    winh_mat.append([])

                    for i, gid in enumerate(self.global_gidlist[n]):

                        if self.id == 0:

                            wall = []

                            if gid in self.gidlist[n]:

                                walltemp = self.cells[n][self.gidlist[n].index(gid)].record['w_inh']
                                if len(walltemp) > 0:
                                    for l in range(len(walltemp)):
                                        wtemp = np.array(walltemp[l])
                                        wtemp = interp(times,time,wtemp)
                                        wall.append(wtemp)

                            else:

                                while 1:
                                    wtemp = np.zeros(len(times))
                                    self.comm.Recv([wtemp, MPI.DOUBLE], source = MPI.ANY_SOURCE, tag=int(gid))

                                    if wtemp[0] == -1:
                                        break
                                    else:
                                        wall.append(wtemp)

                            winh_mat[n].append(wall)

                        else:
                            if gid in self.gidlist[n]:
                                walltemp = self.cells[n][self.gidlist[n].index(gid)].record['w_inh']

                                if len(walltemp) > 0:
                                    for l in range(len(walltemp)):
                                        wtemp = np.array(walltemp[l])
                                        wtemp = interp(times,time,wtemp)
                                        self.comm.Ssend([wtemp, MPI.DOUBLE], dest=0, tag=int(gid))

                                wtemp = np.ones(len(times))*-1
                                self.comm.Ssend([wtemp, MPI.DOUBLE], dest=0, tag=int(gid))


            if self.id == 0:
                print "root gathered synaptic input conductance"


            self.barrier()   # wait for other nodes


        t_all_vec_vec = []
        id_all_vec_vec = []
        f_cells_mean = []

        if self.id == 0:  # only for first node

            for n in range(self.n_celltypes):

                ie = argsort(t_all_vec[n])
                t_all_vec_vec.append( t_all_vec[n][ie] )
                id_all_vec_vec.append( id_all_vec[n][ie].astype(int) ) #

            print "all spikes have been sorted"

            if self.jitter > 0:  # add jitter!
                np.random.seed(40)
                x = np.random.normal(0, self.jitter, len(t_all_vec_vec[self.a_celltype[0]]))
                t_all_vec_vec[self.a_celltype[0]] = t_all_vec_vec[self.a_celltype[0]] + x

            if self.delta_t > 0:
                t_all_vec_vec[self.a_celltype[0]] = t_all_vec_vec[self.a_celltype[0]] + self.delta_t

            gsyn = zeros(len(freq_times))

            if 'gsyn_in' in self.method_interpol:
                pass
            else:
                bvec = ["syn" in st for st in self.method_interpol]
                if np.any(bvec):

                    if (not hasattr(self, 'passive_target')) | (self.jitter > 0):  # if not already done in neuron via artificial cell

                        [resp, _] = neuronpy.util.spiketrain.get_histogram(t_all_vec_vec[self.a_celltype[0]], bins = freq_times)
                        resp = np.concatenate((zeros(1),resp))

                        Ksyn = syn_kernel(arange(0,10*self.syn_tau2,self.bin_width), self.syn_tau1, self.syn_tau2)
                        Ksyn = np.concatenate((zeros(len(Ksyn)-1),Ksyn))
                        gsyn = np.convolve(Ksyn, resp, mode='same')
                        print "Generated gsyn by convolution with Ksyn"
                        self.nc_delay = 0

                    else:
                        gsyn = interp(freq_times,time,np.array(self.rec_g))

                spike_freq = np.zeros(len(freq_times))

                for j in self.a_celltype:

                    #plt.figure('results_voltage')
                    #ax99 = plt.subplot(2,1,1)
                    #ax99.plot(time,voltage[j])

                    #plt.text(0.5, 1.1, r'CF=' + str(round(fmean,1)) + ',fmax=' + str(round(fmax,1)) + ',fmstd=' + str(round(fmstd,1)), transform=ax99.transAxes, fontsize=10, va='center', ha='center')
                    #plt.savefig("./figs/Pub/Voltage_" + str(self.pickle_prefix) + "_cell" + str(j) + "_N" + str(self.N[j]) + ".pdf", dpi = 300, transparent=True) # save it
                    #plt.show()
                    #plt.clf()

                    [num_spikes, _] = neuronpy.util.spiketrain.get_histogram(t_all_vec_vec[j], bins = freq_times)

                    if isinstance(self.factor_celltype[j], ( int, long ) ):
                        f = self.factor_celltype[j]
                    else:
                        f = self.factor_celltype[j][0]

                    spike_freq = spike_freq + f * np.concatenate((zeros(1),num_spikes)) / self.bin_width

        self.barrier()   # wait for other nodes

        #figure('1')
        #plot(time,np.array(self.rec_s1),'b', time,np.array(self.rec_s2),'r')
        #plt.show()

        return {'time':time, 'voltage':voltage, 'current':current, 'fmean':fmean, 'f_cells_mean':f_cells_mean,
        'gsyn':gsyn, 'freq_times':freq_times, 'spike_freq':spike_freq, 'gsyn_in':gsyn_in, 'fmeanA':fmeanA, 'fmaxA':fmaxA, 'fmstdA':fmstdA, 'fcvmA':fcvmA, 'fstdmA':fstdmA, 'fbstdA':fbstdA,
        't_all_vec_vec':t_all_vec_vec, 'id_all_vec_vec':id_all_vec_vec, 'gsyns':gsyns, 'w_mat':w_mat, 'winh_mat':winh_mat, 'fmax':fmax, 'fmstd':fmstd, 'fcvm':fcvm, 'fbaseA':fbaseA, 'fbase':fbase}


    def clean(self):

        self.pc.runworker()
        self.pc.done()


    def compute_Transfer(self, stimulus, spike_freq, freq_times, t, noise_data_points, gsyn, gsyn_in, do_csd, t_qual, K_mat_old, t_startstop, inh_factor=[1]):

        stimulus0 = np.zeros(len(stimulus[0]))

        for a in self.a_celltype:
            # sum input to produce linear input that should be reconstructed!

            if (any(self.syn_inh_dist) > 0) and (any(self.syn_ex_dist) > 0):
                if  max(self.syn_inh_dist) == max(self.syn_ex_dist): # same signal through ex and inh
                    print "inh_factor = [0,1]"
                    inh_factor = [0,1]

            for ni in self.syn_ex_dist[a]:
                if ni != 0:
                    stimulus0 += inh_factor[ni-1] * stimulus[ni-1]
                    print "+ex:", ni-1

            for ni in self.syn_inh_dist[a]:
                if ni != 0:
                    stimulus0 -= inh_factor[ni-1] * stimulus[ni-1] #old: +nemax
                    print "-inh:", ni-1 #old: +nemax

        if (max(self.n_syn_ex) == 0) and (max(self.n_syn_inh) == 0):
            stimulus0 += stimulus[0]
            print "current"

        #if self.n_syn_ex[self.celltype_syn[0]] == 0:
        #    stimulus0 += stimulus[0]

        # amplitude should not matter since filter amplitude is simply adjusted
        #stimulus = stimulus0 #/len(self.syn_ex_dist)

        stimulus0 = stimulus0 / std(stimulus0) / 2

        # linear interpolation inside compute_Transfer !!!
        print "max(stimulus0):",max(stimulus0)
        results = compute_Transfer(spike_freq = spike_freq, freq_times = freq_times,
            stimulus = stimulus0, t = t, noise_data_points = noise_data_points, gsyn = gsyn, gsyn_in = gsyn_in, do_csd = do_csd, t_kernel = 1*s,
            method_interpol = self.method_interpol, nc_delay = self.nc_delay, w_length = 3, t_qual = t_qual, K_mat_old = K_mat_old, t_startstop = t_startstop, give_psd = self.give_psd) # freq_wp not defined, use all frequencies

        # TEST:
        #VAF = results.get('VAFf_mat')
        #freq_used = results.get('freq_used')

        #iend = mlab.find(freq_used >= self.xmax)[0]
        #err = 1-mean(VAF[1][0,1:iend-1])
        #print "err: ", err

        return results


    def residuals_compute_Transfer(self, p, stimulus, spike_freq, freq_times, t, noise_data_points, gsyn, gsyn_in, do_csd, t_qual, K_mat_old, t_startstop, inh_factor):

        inh_factor_in = inh_factor[:]
        ip = 0
        for i, inhf in enumerate(inh_factor_in):
            if inhf < 0:
                inh_factor_in[i] = p[ip]
                ip += 1

        results = self.compute_Transfer(stimulus = stimulus, spike_freq = spike_freq, freq_times = freq_times,
            t = t, noise_data_points = noise_data_points, gsyn = gsyn, gsyn_in = gsyn_in,
            do_csd = do_csd, t_qual = t_qual, K_mat_old = K_mat_old, t_startstop = t_startstop, inh_factor = inh_factor_in)

        VAF = results.get('VAFf_mat')
        freq_used = results.get('freq_used')

        iend = mlab.find(freq_used >= self.xmax)[0]
        err = 1-mean(VAF[1][0,0:iend])
        print "inh_factor:", inh_factor_in, "err: ", err

        return err

    #@profile
    def fun_cnoise_Stim(self, t_stim = 10*s, sexp = 0, cutf = 0, do_csd = 1, t_qual = 0, freq_used = np.array([]), K_mat_old = np.array([]), inh_factor = [1], onf = None, equi = 0):
        """
        Stimulate cell with colored noise
        sexp = spectral exponent: Power ~ 1/freq^sexp
        cutf = frequency cutoff: Power flat (white) for freq <~ cutf
        do_csd = 1: use cross spectral density function for computation
        """
        self.barrier()   # wait for other nodes

        filename = str(self.pickle_prefix) + "_results_pop_cnoise.p"
        filepath = self.data_dir + "/" + filename

        if self.id == 0: print "- filepath:", filepath

        if self.do_run or (os.path.isfile(filepath) is False):

            tstart = 0;
            fs = 1 / self.dt # sampling rate
            fmax = fs / 2 # maximum frequency (nyquist)

            t_noise = arange(tstart, t_stim, self.dt) # create stimulus time vector, make sure stimulus is even!!!

            #print self.syn_ex_dist
            #print self.syn_inh_dist
            #exit()

            if (self.syn_ex_dist == []):
                for nt in range(self.n_celltypes): # loop over all cells
                    #print "nt", nt
                    if hasattr(self.cells[nt][0], 'input_vec'):
                        self.syn_ex_dist.append([1] * len(self.cells[nt][0].input_vec)) # default ex for all by default!!!
                    else:
                        self.syn_ex_dist.append([1] * self.n_syn_ex[nt]) # default ex for all by default!!!

            #print self.syn_ex_dist

            if (self.syn_ex_dist[0] == []):
                nemax = 1
            else:
                nemax = max([item for sublist in self.syn_ex_dist for item in sublist])

            if (self.syn_inh_dist == []): # and (any(self.n_syn_inh) > 0)
                for nt in range(self.n_celltypes): # loop over all cells
                        self.syn_inh_dist.append([0] * self.n_syn_inh[nt]) # default no inh for all by default!!!

            #print self.syn_inh_dist
            #exit()

            if (self.syn_inh_dist[0] == []):
                nimax = 0
            else:
                nimax = max([item for sublist in self.syn_inh_dist for item in sublist])

            #print "self.syn_inh_dist, self.syn_ex_dist", self.syn_inh_dist, self.syn_ex_dist

            n_noise = max([nemax,nimax]) # number of noise sources
            #print n_noise,nemax,nimax
            # create reproduceable input
            noise_data = []

            for nj in range(n_noise):

                if self.id == 0:  # make sure all have the same signal !!!
                    if len(freq_used) == 0:
                        noise_data0 = create_colnoise(t_noise, sexp, cutf, self.seed+nj, onf = onf)
                    else:
                        noise_data0, _, _, _ = create_multisines(t_noise, freq_used)  # create multi sine signal
                else:
                    noise_data0 = np.empty(len(t_noise), dtype=np.float64)

                noise_data0 = self.broadcast(noise_data0, fast = True)

                noise_data.append(noise_data0)
                noise_data0 = []

            noise_data_points = len(noise_data[0])

            # Create signal weight vector inh_factor if it is not fully given
            if len(noise_data) > len(inh_factor):
                inh_factor = [inh_factor[0]] * len(noise_data)
                print "inh_factor:", inh_factor

            #if equi:
                #pass
            #    tstop = t_stim

            if max(self.n_syn_ex) == 0: # this means current input

                self.set_IStim() # sets amp

                if self.fluct_s != []:
                    if self.fluct_s[self.a_celltype[0]] > 0:
                        if self.id == 0: print "- adding i fluct"
                        self.connect_fluct()

                for i, m in enumerate(self.method_interpol):
                            if "syn" in m: self.method_interpol[i] = "syn " + str(self.syn_tau1/ms) + "/" + str(self.syn_tau2/ms) + "ms"
                            if "bin" in m: self.method_interpol[i] = "bin " + str(self.bin_width/ms) + "ms"

                stimulus = []
                for nj in range(len(noise_data)):
                    stimulus0, t, t_startstop = construct_Stimulus(noise_data[nj], fs, self.amp[self.a_celltype[0]], ihold = 0, delay_baseline = self.delay_baseline) # , tail_points = 0
                    stimulus.append(stimulus0)
                tstop = t[-1]

                self.set_IPlay2(stimulus, t)
                if self.id == 0: print "- starting colored noise transfer function estimation! with amp = " + str(np.round(self.amp[self.a_celltype[0]],4)) + ", ihold = " + str(np.round(self.ihold[self.a_celltype[0]],4)) + ", ihold_sigma = " + str(np.round(self.ihold_sigma,4)) + ", dt = " + str(self.dt) + " => maximum frequency = " + str(fmax) + "\r"

            else:

                self.give_freq = False
                ihold = self.set_i(self.ihold) # just sets amp, ihold should not change!

                if 'gsyn_in' not in self.method_interpol:
                    pass
                else:
                    self.g_syn_ex = [1]*len(self.N)


                if ((self.fluct_g_e0 != []) or (self.fluct_g_i0 != [])):
                    if ((self.fluct_g_e0[self.a_celltype[0]] > 0) or (self.fluct_g_i0[self.a_celltype[0]] > 0)):
                        if self.id == 0: print "- adding g fluct"
                        self.connect_gfluct(E_i=-65)

                stimulus = []
                for nj in range(len(noise_data)):
                    stimulus0, t, t_startstop = construct_Stimulus(noise_data[nj], fs, amp=1, ihold = 0, tail_points = 0, delay_baseline = self.delay_baseline) # self.amp
                    stimulus.append(stimulus0)

                noise_data = []
                tstop = t[-1]

                if self.N[self.a_celltype[0]] > 1:
                    self.set_IStim(ihold = [0]*self.n_celltypes, ihold_sigma = [0]*self.n_celltypes, random_start = True, tstart_offset = 1)
                    if self.id == 0: print "- add random start"

                #print "Enter Synplay()"
                self.set_SynPlay(stimulus, t, t_startstop = t_startstop)
                #print "Exit Synplay()"

                if self.id == 0: print "- starting colored noise transfer function estimation with synaptic input! with amp = " + str(np.round(self.amp,4)) + ", ihold = " + str(np.round(self.ihold,4)) + ", ihold_sigma = " + str(np.round(self.ihold_sigma,4)) + ", dt = " + str(self.dt) + " => maximum frequency = " + str(fmax) + "\r"

            amp_vec = []
            mag_vec = []
            pha_vec = []
            freq_used = []
            ca = []
            SNR_mat = []
            VAFf_mat = []
            Qual_mat = []
            CF_mat = []
            VAF_mat = []
            stim = []
            stim_re_mat = []
            resp_mat = []
            current_re = []
            ihold1 = []
            tk = []
            K_mat = []
            gsyn_in = []
            fmean = []
            fmax = []
            fmstd = []
            fcvm = []
            fmeanA = []
            fmaxA = []
            fmstdA = []
            fcvmA = []
            t_all_vec_input_sorted = []
            id_all_vec_input_sorted = []

            if (self.id == 0) and (max(self.n_syn_ex) > 0):
                print range(self.n_celltypes), np.shape(self.t_all_vec_input)
                for l in range(self.n_celltypes):
                    ie = argsort(self.t_all_vec_input[l])
                    t_all_vec_input_sorted.append( self.t_all_vec_input[l][ie] )
                    id_all_vec_input_sorted.append( self.id_all_vec_input[l][ie].astype(int) )

            #if (self.id == 0):
            #    print self.g_syn_ex
            #    print np.array(self.g_syn_ex)>= 0

            #print "g_syn_ex:",self.g_syn_ex
            if np.array(np.array(self.g_syn_ex)>= 0).any():

                if hasattr(self.cells[self.a_celltype[0]][0], 'get_states') and equi:
                    print "- Equilibrate!"
                    self.run(tstop, do_loadstate = False)
                    m = md5.new()
                    cell_exe_new = self.cell_exe[0]
                    m.update(cell_exe_new)
                    filename = './states_' + self.celltype[0] + '_' + m.hexdigest() + '_Population.b'
                    self.cells[self.a_celltype[0]][0].get_states(filename)
                else:
                    self.run(tstop, do_loadstate = False)

                i_startstop = []

                results = self.get(t_startstop, i_startstop)
                time = results.get('time')
                current = results.get('current')
                voltage = results.get('voltage')
                fmean = results.get('fmean')
                gsyn = results.get('gsyn')
                freq_times = results.get('freq_times')
                spike_freq = results.get('spike_freq')
                t_all_vec_vec = results.get('t_all_vec_vec')
                id_all_vec_vec = results.get('id_all_vec_vec')
                gsyns = results.get('gsyns')
                gsyn_in = results.get('gsyn_in')

                fmax = results.get('fmax')
                fmstd = results.get('fmstd')
                fcvm = results.get('fcvm')

                fmeanA = results.get('fmeanA')
                fmaxA = results.get('fmaxA')
                fmstdA = results.get('fmstdA')
                fcvmA = results.get('fcvmA')

                fbaseA = results.get('fbaseA')
                fbase = results.get('fbase')
                fbstdA = results.get('fbstdA')


            else: # do not run, analyse input!!!

                time = t
                voltage = []
                for l in range(self.n_celltypes):
                    voltage.append(np.zeros(len(t)))
                current = []

                freq_times = []
                spike_freq = []
                gsyn = []
                gsyn_in = []

                t_all_vec_vec = []
                id_all_vec_vec = []

                fmean = []
                fmax = []
                fmstd = []
                fcvm = []
                fstdm = []

                fmeanA = []
                fmaxA = []
                fmstdA = []
                fcvmA = []
                fbaseA = []
                fbase = []
                fbstdA = []

                if self.id == 0:

                    current = self.n_train_ex

                    #t_all_vec = self.t_all_vec_input
                    #id_all_vec = self.id_all_vec_input

                    #ie = argsort(t_all_vec)
                    #t_all_vec_vec.append( t_all_vec[ie] )
                    #id_all_vec_vec.append( id_all_vec[ie].astype(int) )

                    t_all_vec_vec = t_all_vec_input_sorted
                    id_all_vec_vec = id_all_vec_input_sorted

                    freq_times = arange(0, tstop, self.bin_width)
                    spike_freq = np.zeros(len(freq_times))

                    for j in self.a_celltype:

                        [num_spikes, _] = neuronpy.util.spiketrain.get_histogram(t_all_vec_vec[j], bins = freq_times)

                        if self.tau2_ex[0] > 0:
                            spike_freq = np.concatenate((zeros(1),num_spikes))
                            print "NOSYN TEST: start convolution with Ksyn"
                            Ksyn = syn_kernel(arange(0,10*self.tau2_ex[0],self.bin_width), self.tau1_ex[0], self.tau2_ex[0])
                            Ksyn = np.concatenate((zeros(len(Ksyn)-1),Ksyn))
                            spike_freq = np.convolve(Ksyn, spike_freq, mode='same')
                            print "NOSYN TEST: convolution finished"
                        else:

                            if isinstance(self.factor_celltype[j], ( int, long ) ):
                                f = self.factor_celltype[j]
                            else:
                                f = self.factor_celltype[j][0]

                            spike_freq = spike_freq + f * np.concatenate((zeros(1),num_spikes)) / self.bin_width

                    fmean.append(self.fmean_input)
                    fmax.append(self.fmax_input)
                    fmstd.append(self.fmstd_input)
                    fcvm.append(self.fcvm_input)
                    fstdm.append(self.fstdm_input)

                    if self.no_fmean == True:
                        fmean.append(ihold)

                    #plt.figure('spike_freq')
                    #plt.plot(freq_times, spike_freq)
                    #plt.savefig("./figs/Pub/Spike_freq_" + str(self.pickle_prefix) + ".pdf", dpi = 300, transparent=True) # save it
                    #plt.clf()

                    fmeanA = fmean[0]
                    fmaxA = fmax[0]
                    fmstdA = fmstd [0]
                    fcvmA = fcvm[0]
                    fstdmA = fstdm[0]


            if self.id == 0:

                if any([i<0 for i in inh_factor]):

                    p0 = []
                    inhf_idx = []
                    for i, inhf in enumerate(inh_factor):
                        if inhf < 0:
                            p0.append(0)
                            inhf_idx.append(i)

                    plsq = fmin(self.residuals_compute_Transfer, p0, args=(stimulus, spike_freq, freq_times, t, noise_data_points, gsyn, gsyn_in, do_csd, t_qual, K_mat_old, t_startstop, inh_factor))
                    p = plsq

                    ip = 0
                    for i in inhf_idx:
                        inh_factor[i] = p[ip]
                        ip += 1


                    print "Final inh_factor: ", inh_factor


                results = self.compute_Transfer(stimulus, spike_freq = spike_freq, freq_times = freq_times,
                    t = t, noise_data_points = noise_data_points, gsyn = gsyn, gsyn_in = gsyn_in,
                    do_csd = do_csd, t_qual = t_qual, K_mat_old = K_mat_old, t_startstop = t_startstop, inh_factor=inh_factor)

                mag_vec, pha_vec, ca, freq, freq_used, fmean_all = results.get('mag_mat'), results.get('pha_mat'), results.get('ca_mat'), results.get('freq'), results.get('freq_used'), results.get('fmean')
                SNR_mat, VAFf_mat, Qual_mat, CF_mat, VAF_mat = results.get('SNR_mat'), results.get('VAFf_mat'), results.get('Qual_mat'), results.get('CF_mat'), results.get('VAF_mat')
                stim, resp_mat, stim_re_mat, tk, K_mat = results.get('stim'), results.get('resp_mat'), results.get('stim_re_mat'), results.get('tk'), results.get('K_mat')


            self.barrier()   # wait for other nodes


            if self.id == 0:

                if t_qual > 0:
                    #print t_startstop[0], t_startstop[0]/self.dt, (t_startstop[0]+t_qual)/self.dt
                    current_re = current[int(t_startstop[0]/self.dt):int((t_startstop[0]+t_qual)/self.dt)]
                    current_re = current_re[int(len(K_mat[self.a_celltype[0]])):int(len(current_re))-int(len(K_mat[self.a_celltype[0]]))]

                if len(self.i_holdrs) > 0:
                    ihold1 = self.i_holdrs[self.a_celltype[0]][0]
                else:
                    ihold1 = []

                for l in range(len(self.method_interpol)):  # unwrap
                    pha_vec[l,:] = unwrap(pha_vec[l,:] * (pi / 180)) * (180 / pi)  # unwrap for smooth phase

                # only return fraction of actual signal, it is too long!!!
                if time[-1] > self.tmax:
                    imax = -1*int(self.tmax/self.dt)
                    time = time[imax:]; current = current[imax:]; gsyn = gsyn[imax:]; gsyn_in = gsyn_in[imax:]
                    for n in range(self.n_celltypes):
                        voltage[n] = voltage[n][imax:]

                if freq_times != []:
                    if freq_times[-1] > self.tmax:
                        imax2 = where(freq_times > self.tmax)[0][0]  # for spike frequency
                        freq_times = freq_times[0:imax2]; spike_freq = spike_freq[0:imax2]

                bvec = ["_syn" in st for st in self.method_interpol]
                if np.any(bvec):
                    # normalize synaptic integration with others
                    mag_vec[1,:]= mag_vec[0,0]*mag_vec[1,:]/mag_vec[1,0]

            if self.id == 0: print "start pickle"

            results = {'freq_used':freq_used, 'amp':amp_vec,'mag':mag_vec,'pha':pha_vec,'ca':ca,'voltage':voltage,'tk':tk,'K_mat':K_mat, 'ihold1': ihold1, 't_startstop':t_startstop, #'stimulus':stimulus,
                        'current':current,'t1':time,'freq_times':freq_times,'spike_freq':spike_freq, 'stim':stim, 'stim_re_mat':stim_re_mat, 'resp_mat':resp_mat, 'current_re':current_re, 'gsyn_in':gsyn_in, 'fmeanA':fmeanA, 'fmaxA':fmaxA, 'fmstdA':fmstdA, 'fcvmA':fcvmA, 'fbaseA':fbaseA, 'fbase':fbase, 'fbstdA':fbstdA,
                        'fmean':fmean,'method_interpol':self.method_interpol, 'SNR':SNR_mat, 'VAF':VAFf_mat, 'Qual':Qual_mat, 'CF':CF_mat, 'VAFs':VAF_mat, 'fmax':fmax, 'fmstd':fmstd, 'fcvm':fcvm, 'inh_factor':inh_factor, 't_all_vec_vec':t_all_vec_vec, 'id_all_vec_vec':id_all_vec_vec}

            if self.id == 0:
                if self.dumpsave == 1:
                    pickle.dump( results, gzip.GzipFile( filepath, "wb" ) )
                    print "pickle done"


                if self.plot_train:

                    for a in self.a_celltype:

                        #i_start = mlab.find(t_all_vec_vec[a] >= 0)[0]
                        #i_stop = mlab.find(t_all_vec_vec[a] >= 5)[0]

                        #t_all_cut = t_all_vec_vec[a][i_start:i_stop]
                        #id_all_cut = id_all_vec_vec[a][i_start:i_stop]

                        t_all_cut = t_all_vec_vec[a]
                        id_all_cut = id_all_vec_vec[a]

                        f_start_in = mlab.find(t_all_cut >= 0)
                        f_stop_in = mlab.find(t_all_cut <= 10)

                        f_start = f_start_in[0]
                        f_stop = f_stop_in[-1]+1
                        use_spikes = t_all_cut[f_start:f_stop]
                        use_id = id_all_cut[f_start:f_stop]

                        plt.figure('results_train')
                        ax99 = plt.subplot(1,1,1)
                        ax99.plot(use_spikes,use_id,'|', ms=2)
                        plt.text(0.5, 1.1, r'CF=' + str(round(fmean,1)) + ',fmax=' + str(round(fmax,1)) + ',fmstd=' + str(round(fmstd,1)), transform=ax99.transAxes, fontsize=10, va='center', ha='center')
                        plt.savefig("./figs/Pub/Train_" + str(self.pickle_prefix) + "_cell" + str(a) + "_N" + str(self.N[a]) + ".pdf", dpi = 300, transparent=True) # save it

                        plt.clf()

                        if len(t_all_cut) > 0:

                            tbin = 100*ms
                            tb = np.arange(0,t[-1],tbin)
                            [all_rate, _] = neuronpy.util.spiketrain.get_histogram(t_all_cut, bins = tb)
                            all_rate = np.concatenate((np.zeros(1),all_rate)) / self.N[a] / tbin

                            plt.figure('results_train2')
                            plt.plot(tb,all_rate)
                            plt.savefig("./figs/Pub/PSTH_" + str(self.pickle_prefix) + "_cell" + str(a) + "_N" + str(self.N[a]) + ".pdf", dpi = 300, transparent=True) # save it
                            plt.clf()

                        plt.figure('results_noise')
                        plt.plot(time,current)
                        plt.savefig("./figs/Pub/Noise_" + str(self.pickle_prefix) + "_cell" + str(a) + "_N" + str(self.N[a]) + ".pdf", dpi = 300, transparent=True) # save it
                        plt.clf()


                if self.plot_input:

                    if len(t_all_vec_input_sorted[0]) > 0:

                        i_start = mlab.find(t_all_vec_input_sorted[0] >= 0)[0]
                        i_stop = mlab.find(t_all_vec_input_sorted[0] >= 5)[0]

                        t_all_cut = t_all_vec_input_sorted[0][i_start:i_stop]
                        id_all_cut = id_all_vec_input_sorted[0][i_start:i_stop]

                        plt.figure('results_input')
                        ax99 = plt.subplot(1,1,1)
                        ax99.plot(t_all_cut,id_all_cut,'|', ms=2)
                        plt.text(0.5, 1.1, r'fmean=' + str(round(self.fmean_input,1)) + ',fmax=' + str(round(self.fmax_input,1)) + ',fmstd=' + str(round(self.fmstd_input,1)) + ',fcvm=' + str(round(self.fcvm_input,1)) + ',fstdm=' + str(round(self.fstdm_input,1)), transform=ax99.transAxes, fontsize=10, va='center', ha='center')
                        plt.savefig("./figs/Pub/Input_" + str(self.pickle_prefix) + "_N" + str(self.N[self.a_celltype[0]]) + ".pdf", dpi = 300, transparent=True) # save it
                        plt.clf()


        else:

            if self.id == 0:
                results = pickle.load( gzip.GzipFile( filepath, "rb" ) )

                #print results
                #print {key:np.shape(value) for key,value in results.iteritems()}

                if self.minimal_dir: # save only info needed for plot

                    print {key:np.shape(value) for key,value in results.iteritems()}

                    if "Fig6_pop_transfer_grc_syngr_nsyn4_cn_a1_noisesynlow_inhlow_adjfinh_varih_N100_CFo6.0_results_pop_cnoise.p" in filename:
                        results['ca'] = []
                        results['resp_mat'] = []
                        results['stim'] = []
                        results['current'] = []
                        results['tk'] = []
                        results['K_mat'] = []
                        results['freq_times'] = []
                        results['spike_freq'] = []
                        results['stim_re_mat'] = []
                        results['current_re'] = []
                        results['t_all_vec_vec'] = []
                        results['id_all_vec_vec'] = []
                        results['gsyn_in'] = []

                    elif ("Fig8_pop_transfer_none_synno_cn_cutf30_a1_noisesynlow_ih20_varih_N100_CFo-1_results_pop_cnoise.p" in filename) \
                         or ("Fig8_pop_transfer_none_synno_cn_cutf30_a10_noisesynlow_ih20_varih_N100_CFo-1_results_pop_cnoise.p" in filename) \
                         or ("Fig8_pop_transfer_grc_syngr_nsyn4_cn_cutf30_a1_noisesynlow_inhlow_adjfinh_varih_varinhn_N100_CFo9.0_results_pop_cnoise.p" in filename) \
                         or ("Fig8_pop_transfer_grc_syngr_nsyn4_cn_cutf30_a10_noisesynlow_inhlow_adjfinh_varih_varinhn_N100_is0.14_CFo9.0_results_pop_cnoise.p" in filename) \
                         :

                        results['ca'] = []
                        results['resp_mat'] = []
                        results['current'] = []
                        results['tk'] = []
                        results['K_mat'] = []
                        results['voltage'] = []
                        results['current_re'] = []
                        results['t_all_vec_vec'] = []
                        results['id_all_vec_vec'] = []
                        results['t1'] = []
                        results['gsyn_in'] = []

                    elif ("Fig8_pop_transfer_none_synno_cn_cutf30_a1_noisesynlow_ih20_varih_N50_twopop_CFo-1_results_pop_cnoise.p" in filename) \
                         or ("Fig8_pop_transfer_none_synno_cn_cutf30_a10_noisesynlow_ih20_varih_N50_twopop_CFo-1_results_pop_cnoise.p" in filename) \
                         or ("Fig8_pop_transfer_grc_syngr_nsyn4_cn_cutf30_a1_noisesynlow_inhlow_adjfinh_varih_varinhn_N50_twopop_CFo9.0_results_pop_cnoise.p" in filename) \
                         or ("Fig8_pop_transfer_grc_syngr_nsyn4_cn_cutf30_a10_noisesynlow_inhlow_adjfinh_varih_varinhn_N50_is0.14_twopop_CFo9.0_results_pop_cnoise.p" in filename) \
                         or ("Fig8_pop_transfer_grc_syngr_nsyn4_cn_cutf5_a1_noisesynlow_inhlow_adjfinh_varih_varinhn_N100_CFo14.0_results_pop_cnoise.p" in filename) \
                         or ("Fig8_pop_transfer_grc_syngr_nsyn4_cn_cutf5_a1_noisesynlow_inhlow_adjfinh_varih_varinhn_N50_twopop_CFo14.0_results_pop_cnoise.p" in filename) \
                         :

                        results['ca'] = []
                        results['resp_mat'] = []
                        results['current'] = []
                        results['tk'] = []
                        results['K_mat'] = []
                        results['voltage'] = []
                        results['current_re'] = []
                        results['t_all_vec_vec'] = []
                        results['id_all_vec_vec'] = []
                        results['t1'] = []
                        results['gsyn_in'] = []
                        results['freq_times'] = []
                        results['spike_freq'] = []

                    elif ("Fig4_pop_transfer_grc_cn_addn100_N[100]_CF[40]_amod[1]_results_pop_cnoise.p" in filename) \
                         or ("Fig4_pop_transfer_grc_cn_addn1_N[100]_CF[40]_amod[1]_results_pop_cnoise.p" in filename) \
                         or ("Fig4b_pop_transfer_grc_lowcf_cn_twopop_N[50, 50]_CF[0.0055, 0.0055]_amod[None, None]_results_pop_cnoise.p" in filename) \
                         or ("Fig4b_pop_transfer_grc_lowcf_cn_N[100]_CF[0.0055]_amod[None]_results_pop_cnoise.p" in filename) \
                         or ("Fig4b_pop_transfer_grc_lowcf_slownoise_cn_twopop_N[50, 50]_CF[0.0051, 0.0051]_amod[None, None]_results_pop_cnoise.p" in filename) \
                         or ("Fig4b_pop_transfer_grc_lowcf_slownoise_cn_N[100]_CF[0.0051]_amod[None]_results_pop_cnoise.p" in filename) \
                         :

                        results['ca'] = []
                        results['resp_mat'] = []
                        results['current'] = []
                        results['tk'] = []
                        results['K_mat'] = []
                        results['voltage'] = []
                        results['t_all_vec_vec'] = []
                        results['id_all_vec_vec'] = []
                        results['t1'] = []
                        results['gsyn_in'] = []
                        results['freq_times'] = []
                        results['spike_freq'] = []

                    elif ("Fig2_pop_transfer_" in filename) \
                         :

                        results['ca'] = []
                        results['resp_mat'] = []
                        results['current'] = []
                        results['t1'] = []
                        results['voltage'] = []
                        results['freq_times'] = []
                        results['spike_freq'] = []
                        results['current_re'] = []
                        results['t_all_vec_vec'] = []
                        results['id_all_vec_vec'] = []
                        results['gsyn_in'] = []

                    else:
                        results['ca'] = []
                        results['resp_mat'] = []
                        results['stim'] = []
                        results['current'] = []
                        results['tk'] = []
                        results['K_mat'] = []
                        results['t1'] = []
                        results['voltage'] = []
                        results['freq_times'] = []
                        results['spike_freq'] = []
                        results['stim_re_mat'] = []
                        results['current_re'] = []
                        results['t_all_vec_vec'] = []
                        results['id_all_vec_vec'] = []
                        results['gsyn_in'] = []

                    print {key:np.shape(value) for key,value in results.iteritems()}

                    pickle.dump( results, gzip.GzipFile( self.minimal_dir + "/" + filename, "wb" ) )

            else:
                results = {'freq_used':[], 'amp':[],'mag':[],'pha':[],'ca':[],'voltage':[], 'tk':[],'K_mat':[], 'ihold1':[], 't_startstop':[], #'stimulus':[],
                    'current':[],'t1':[],'freq_times':[],'spike_freq':[], 'stim':[], 'stim_re_mat':[], 'current_re':[], 'gsyn_in':[], 'fmeanA':[], 'fmaxA':[], 'fmstdA':[], 'fcvmA':[], 'fbaseA':[], 'fbase':[], 'fbstdA':[],
                     'fmean':[],'method_interpol':self.method_interpol, 'SNR':[], 'VAF':[], 'Qual':[], 'CF':[], 'VAFs':[], 'fmax':[], 'fmstd':[], 'fcvm':[], 'inh_factor':[], 't_all_vec_vec':[], 'id_all_vec_vec':[]}

        if self.id == 0:

            if self.plot_train:

                for a in self.a_celltype:

                    t1 = results.get('t1')
                    voltage = results.get('voltage')
                    fmean = results.get('fmean')
                    fmax = results.get('fmax')
                    fmstd = results.get('fmstd')


                    if results.has_key('t_all_vec_vec'):

                        if len(results['t_all_vec_vec']) > 0:
                            t_all_vec_vec = results.get('t_all_vec_vec')
                            id_all_vec_vec = results.get('id_all_vec_vec')

                            t_all_cut = t_all_vec_vec[a]
                            id_all_cut = id_all_vec_vec[a]

                            f_start_in = mlab.find(t_all_cut >= 0)
                            f_stop_in = mlab.find(t_all_cut <= 10)

                            f_start = f_start_in[0]
                            f_stop = f_stop_in[-1]+1
                            use_spikes = t_all_cut[f_start:f_stop]
                            use_id = id_all_cut[f_start:f_stop]

                            plt.figure('results_train')
                            ax97 = plt.subplot(1,1,1)
                            ax97.plot(use_spikes,use_id,'|', ms=6)
                            plt.text(0.5, 1.1, r'CF=' + str(round(fmean,1)) + ',fmax=' + str(round(fmax,1)) + ',fmstd=' + str(round(fmstd,1)), transform=ax97.transAxes, fontsize=10, va='center', ha='center')
                            plt.savefig("./figs/Pub/Train_" + str(self.pickle_prefix) + "_cell" + str(a) + "_N" + str(self.N[a]) + ".pdf", dpi = 300, transparent=True) # save it


                    plt.figure('results_voltage')
                    ax99 = plt.subplot(2,1,1)
                    ax99.plot(t1,voltage[a])

                    t_noise = arange(0, t_stim, self.dt)
                    noise_data = create_colnoise(t_noise, sexp, cutf, 50, onf = onf)
                    stimulus, t, t_startstop = construct_Stimulus(noise_data, 1/self.dt, amp=1, ihold = 0, tail_points = 0, delay_baseline = self.delay_baseline)
                    ax98 = plt.subplot(2,1,2)
                    ax98.plot(t[0:10/self.dt],stimulus[0:10/self.dt],color='k')

                    plt.text(0.5, 1.1, r'CF=' + str(round(fmean,1)) + ',fmax=' + str(round(fmax,1)) + ',fmstd=' + str(round(fmstd,1)), transform=ax99.transAxes, fontsize=10, va='center', ha='center')
                    plt.savefig("./figs/Pub/Voltage_" + str(self.pickle_prefix) + "_cell" + str(a) + "_N" + str(self.N[a]) + ".pdf", dpi = 300, transparent=True) # save it
                    plt.show()
                    plt.clf()

        if (self.id == 0) and (do_csd == 1):
            Qual = results.get('Qual')
            for i, ii in enumerate(self.method_interpol):
                        print "\n[QUAL:] Interpol:", ii, "SNR0:", Qual[i,0,0], "SNR_cutff:", Qual[i,0,1], "SNR_mean:", Qual[i,0,2], "\n VAF0:", Qual[i,1,0], "VAF_cutff:", Qual[i,1,1], "VAF_mean:", Qual[i,1,2],  "\n CF(subtracted):", Qual[i,2,0], "VAF(subtracted):", Qual[i,2,1]

            VAF = results.get('VAF')
            freq_used = results.get('freq_used')
            iend = mlab.find(freq_used >= self.xmax)[0]
            print 'm(VAF)=' + str(np.mean(VAF[1][0,0:iend]))

        self.barrier()   # wait for other nodes

        return results


#    def fun_ssine_Stim(self, freq_used = np.array([1, 10, 100, 1000])*Hz):
#        """
#        Compute impedance and/or transfer function using Single sine stimulation
#        Only compute transfer function if there is a steady state (resting) firing rate!
#        """
#        self.barrier()   # wait for other nodes
#
#        filepath = "./data/" + str(self.pickle_prefix) + "_results_pop_ssine.p"
#
#        if self.do_run or (os.path.isfile(filepath) is False):
#
#            fs = 1 / self.dt # sampling rate
#            fmax = fs / 2 # maximum frequency (nyquist)
#
#            if self.id == 0: print "- starting single sine transfer function estimation! with amp = " + str(np.round(self.amp[a_celltype[0]],4)) + ", ihold = " + str(np.round(self.ihold[self.a_celltype[0]],4)) + ", dt = " + str(self.dt) + " => maximum frequency = " + str(fmax) + "\r"
#
#            if max(self.n_syn_ex) == 0:
#                self.set_IStim()
#
#                if self.fluct_s != []:
#                    if self.fluct_s[self.a_celltype[0]] > 0:
#                        if self.id == 0: print "- adding i fluct"
#                        self.connect_fluct()
#
#                for i, m in enumerate(self.method_interpol):
#                    if "syn" in m: self.method_interpol[i] = "syn " + str(self.syn_tau1/ms) + "/" + str(self.syn_tau2/ms) + "ms"
#                    if "bin" in m: self.method_interpol[i] = "bin " + str(self.bin_width/ms) + "ms"
#
#            else:
#                self.give_freq = False
#                ihold = self.set_i(self.ihold) # just sets amp, ihold should not change!
#
#                if ((self.fluct_g_e0 != []) or (self.fluct_g_i0 != [])):
#                    if ((self.fluct_g_e0[self.a_celltype[0]] > 0) or (self.fluct_g_i0[self.a_celltype[0]] > 0)):
#                        if self.id == 0: print "- adding g fluct"
#                        self.connect_gfluct(E_i=-65)
#
#                #if ((self.fluct_std_e[self.a_celltype[0]] != []) or (self.fluct_std_i[self.a_celltype[0]] != [])):
#                #    if ((self.fluct_std_e[self.a_celltype[0]] > 0) or (self.fluct_std_i[self.a_celltype[0]] > 0)):
#                #        if self.id == 0: print "- adding g fluct"
#                #        self.connect_gfluct(E_i=-65)
#
#                if 'gsyn_in' not in self.method_interpol:
#                    pass
#                else:
#                    self.g_syn_ex = 1
#
#
#            for i, fu in enumerate(freq_used):
#
#                if self.id == 0: print "- single sine processing frequency = " + str(fu)
#
#                t, stimulus, i_startstop, t_startstop = create_singlesine(fu = fu, amp = self.amp[a_celltype[0]], ihold = 0, dt = self.dt, periods = 20, minlength = 2*s, t_prestim = 1*s)
#                tstop = t[-1]
#
#                if i == 0: t_startstop_plot = t_startstop
#
#                if max(self.n_syn_ex) == 0:
#                    self.set_IPlay(stimulus, t)
#                else:
#                    self.set_SynPlay(stimulus, t)
#
#                if self.g_syn_ex >= 0: # should also be true for current input!!!
#
#                    self.run(tstop)
#
#                    if i == 0:   # do this here to have something to return
#
#                        # select first sinusoidal to plot, later
#                        voltage_plot = []
#                        current_plot = []
#                        time_plot = []
#                        freq_times_plot = []
#                        spike_freq_plot = []
#                        gsyn_plot = []
#
#                        # construct vectors
#                        amp_vec = zeros(len(freq_used)) # amplitude vector
#                        fmean_all = zeros(len(freq_used)) # mean firing frequency (all cells combined)
#                        fmean = zeros(len(freq_used)) # mean firing frequency (one cell)
#                        ca = zeros(len(freq_used), dtype=complex)
#
#                        # create matrix to hold all different interpolation methods:
#                        mag_vec = zeros((len(self.method_interpol),len(freq_used)))  # magnitude vector
#                        pha_vec = zeros((len(self.method_interpol),len(freq_used))) # phase vector
#                        NI_vec = zeros((len(self.method_interpol),len(freq_used)))  # NI vector
#                        VAF_vec = zeros((len(self.method_interpol),len(freq_used)))  # VAF vector
#
#                    results = self.get(t_startstop, i_startstop) # t1 should be equal to t!!!
#                    time, voltage, current, fmean0, gsyn = results.get('time'), results.get('voltage'), results.get('current'), results.get('fmean'), results.get('gsyn')
#                    freq_times, spike_freq, t_all_vec_vec, id_all_vec_vec, gsyns = results.get('freq_times'), results.get('spike_freq'), results.get('t_all_vec_vec'), results.get('id_all_vec_vec'), results.get('gsyns')
#
#                else:
#
#                    time = t
#                    voltage = []
#                    voltage.append(np.zeros(len(t)))
#                    current = stimulus
#
#                    freq_times = []
#                    spike_freq = []
#                    fmean0 = ihold
#                    gsyn = []
#                    gsyn_in = []
#
#                    t_all_vec_vec = []
#                    id_all_vec_vec = []
#
#
#                    if self.id == 0:
#
#                        t_all_vec = []
#                        t_all_vec.append([])
#                        t_all_vec[0] = np.concatenate(self.t_all_vec_input)
#
#                        id_all_vec = []
#                        id_all_vec.append([])
#                        id_all_vec[0] = np.concatenate(self.id_all_vec_input)
#
#                        ie = argsort(t_all_vec[0])
#                        t_all_vec_vec.append( t_all_vec[0][ie] )
#                        id_all_vec_vec.append( id_all_vec[0][ie].astype(int) ) #
#
#
#                        freq_times = arange(0, tstop, self.bin_width)
#                        [num_spikes, _] = neuronpy.util.spiketrain.get_histogram(t_all_vec_vec[0], bins = freq_times)
#                        spike_freq = np.concatenate((zeros(1),num_spikes)) / self.bin_width
#
#
#                if self.id == 0:
#
#                    fmean[i] = fmean0[0]
#
#                    if i == 0:
#
#                        # select first sinusoidal to plot
#                        voltage_plot = voltage
#                        current_plot = current
#                        time_plot = time
#                        freq_times_plot = freq_times
#                        spike_freq_plot = spike_freq
#                        gsyn_plot = gsyn
#
#
#                    for l in range(len(self.method_interpol)):
#
#                        if "bin" in self.method_interpol[l]:
#
#                            # binning and linear interpolation
#                            stimulus_signal = stimulus[i_startstop[0]:i_startstop[1]] # cut out relevant signal
#                            t_input_signal = t[i_startstop[0]:i_startstop[1]] - t[i_startstop[0]]
#
#                            spike_freq_interp = interp(t, freq_times, spike_freq, left=0, right=0) # interpolate to be eqivalent with input, set zero at beginning and end!
#                            freq_out_signal_interp = spike_freq_interp[i_startstop[0]:i_startstop[1]] # cut out relevant signal
#                            vamp, mag_vec[l,i], pha_vec[l,i], fmean_all[i], _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_interp, t_input_signal, method = "fft", f = fu)
#
#                            results = est_quality(t_input_signal, fu, freq_out_signal_interp, self.amp[a_celltype[0]]*mag_vec[l,i], pha_vec[l,i]/ (180 / pi), fmean_all[i])
#                            NI_vec[l,i], VAF_vec[l,i] = results.get('NI'), results.get('VAF')
#                            print "-[bin] NI: " + str(NI_vec[l,i]) + ", VAF: " + str(VAF_vec[l,i])
#
#                        if "syn" in self.method_interpol[l]:
#
#                            # synaptic integration
#                            dt_out = t_input_signal[2] - t_input_signal[1]
#                            shift = self.nc_delay/dt_out  # shift response by the nc delay to remove offset
#                            freq_out_signal_syn = gsyn[i_startstop[0]+shift:i_startstop[1]+shift] # cut out relevant signal
#
#                            vamp, mag_vec[l,i], pha_vec[l,i], fm, _ = get_magphase(stimulus_signal, t_input_signal, freq_out_signal_syn, t_input_signal, method = "fft", f = fu)
#
#                            results = est_quality(t_input_signal, fu, freq_out_signal_syn, self.amp[a_celltype[0]]*mag_vec[l,i], pha_vec[l,i]/ (180 / pi), fm)
#                            NI_vec[l,i], VAF_vec[l,i] = results.get('NI'), results.get('VAF')
#                            print "-[syn] NI: " + str(NI_vec[l,i]) + ", VAF: " + str(VAF_vec[l,i])
#
#
#                self.barrier()   # wait for other nodes
#
#            #print "rest: " + str(vrest) + " freq_used:" + str(freq_used) + " amp_vec:" + str(amp_vec) + " mag_vec:" + str(mag_vec) + " pha_vec:" + str(pha_vec)
#
#            if self.id == 0:
#
#                for l in range(len(self.method_interpol)):  # unwrap
#                    pha_vec[l,:] = unwrap(pha_vec[l,:] * (pi / 180)) * (180 / pi)  # unwrap for smooth phase
#
#                # only return fraction of actual signal, it is too long!!!
#                if time_plot[-1] > self.tmax:
#                    imax = where(time_plot > self.tmax)[0][0]  # for voltage, current and time
#                    time_plot = time_plot[0:imax];  current_plot = current_plot[0:imax]; gsyn_plot = gsyn_plot[0:imax]
#                    for n in range(self.n_celltypes):
#                        voltage_plot[n] = voltage_plot[n][0:imax]
#
#                if freq_times_plot != []:
#                    if freq_times_plot[-1] > self.tmax:
#                        imax2 = where(freq_times_plot > self.tmax)[0][0]  # for spike frequency
#                        freq_times_plot = freq_times_plot[0:imax2]; spike_freq_plot = spike_freq_plot[0:imax2]
#
#                # normalize synaptic integration with with first magnitude, may by syn itself!
#                bvec = ["syn" in st for st in self.method_interpol]
#                if np.any(bvec):
#                    k = where(bvec)
#                    mag_vec[k,:]= mag_vec[0,0]*mag_vec[k,:]/mag_vec[k,0]
#
#            NI_vec = (freq_used, NI_vec)
#            VAF_vec = (freq_used, VAF_vec)
#            results = {'freq_used':freq_used, 'amp':amp_vec,'mag':mag_vec,'pha':pha_vec,'ca':ca,'voltage':voltage_plot, 't_startstop':t_startstop_plot,
#                'current':current_plot,'t1':time_plot,'freq_times':freq_times_plot,'spike_freq':spike_freq_plot,
#                'fmean':mean(fmean),'method_interpol':self.method_interpol, 'NI':NI_vec, 'VAF':VAF_vec}
#
#            if self.id == 0:
#                pickle.dump( results, gzip.GzipFile( filepath, "wb" ) )
#
#        else:
#
#            if self.id == 0:
#                results = pickle.load( gzip.GzipFile( filepath, "rb" ) )
#            else:
#                results = {'freq_used':[], 'amp':[],'mag':[],'pha':[],'ca':[],'voltage':[], 't_startstop':[],
#                    'current':[],'t1':[],'freq_times':[],'spike_freq':[],
#                     'fmean':[],'method_interpol':self.method_interpol,'NI':[],'VAF':[]}
#
#        return results

    def get_RC(self, opt_plot):

        if self.id == 0:
            if "analytical" in opt_plot: # simplest case, only uses rm and tau, scaling necessary
                exec self.cell_exe[self.a_celltype[0]]
                sim = Stimulation(cell, temperature = self.temperature)
                rm, cm, taum = sim.get_RCtau()
            else:
                rm = cm = taum = 0

            if "if" in opt_plot:
                Vrest = cell.soma(0.5).pas.e*mV
                Vth = cell.spkout.thresh*mV
                Vreset = cell.spkout.vrefrac*mV
            else:
                Vreset = 0*mV; Vth = 1*mV; Vrest = 0*mV

            sim = None
            cell = None
        else:
            rm = cm = taum = 0
            Vreset = 0*mV; Vth = 1*mV; Vrest = 0*mV

        return rm, cm, taum, Vreset, Vth, Vrest


    def fun_plot(self, currlabel="control", dowhat="cnoise", freq_used=np.array([]), cutf=10, sexp=0, t_stim=100*s, ymax=0, ax=None, SNR=None, VAF=None, t_qual=0, opt_plot=np.array([]), method_interpol_plot=[], do_csd = 1):

        SNR_switch = SNR
        VAF_switch = VAF

        rm, cm, taum, Vreset, Vth, Vrest = self.get_RC(opt_plot)

        if dowhat == "cnoise":

            if do_csd == 0:
                t_qual = 0; SNR_switch = 0; VAF_switch = 0

            results = self.fun_cnoise_Stim(t_stim = t_stim, cutf = cutf, sexp = sexp, t_qual = t_qual, freq_used = freq_used, do_csd = do_csd)

            freq_used, amp_vec, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1')
            freq_times, spike_freq, fmean, method_interpol, SNR, VAF, Qual = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('SNR'), results.get('VAF'), results.get('Qual')
            stim, stim_re_mat, current_re, tk, K_mat_old = results.get('stim'), results.get('stim_re_mat'), results.get('current_re'), results.get('tk'), results.get('K_mat')

        elif dowhat == "ssine":

            results = self.fun_ssine_Stim(freq_used = freq_used0)

            freq_used, amp_vec, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1')
            freq_times, spike_freq, fmean, method_interpol, VAF = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('VAF')
            tk = []
            K_mat_old = []

        # analyse
        if self.id == 0:

            print "Mean rate: " + str(fmean)

            # Turn it off if set to zero
            if SNR_switch == 0: SNR = None
            if VAF_switch == 0: VAF = None


            if t_qual > 0:

                plt.figure("Reconstruct")

                ax1 = subplot(2,1,1)

                ax1.plot(np.arange(len(stim))*dt-1, current_re*1e3, 'b', linewidth=1)
                ax1.plot(np.arange(len(stim))*dt-1, (stim)*1e3, 'k-', linewidth=1)
                ax1.plot(np.arange(len(stim))*dt-1, (stim_re_mat[0,:])*1e3, 'r', linewidth=1, alpha=1)

                #adjust_spines(ax1, ['left','bottom'], d_out = 10)
                #ax1.axis(xmin=0, xmax=1)

                #ax1.axis(ymin=8.3, ymax=10.7)
                #ax1.yaxis.set_ticks(array([8.5,9,9.5,10,10.5]))
                #ax1.set_title("Reconstruction")

                #ax1.set_xlabel("s")
                #ax1.set_ylabel("pA")

                #ax1.text(0.15, 10.7, "Input current", color=color3, fontsize = 8)
                #ax1.text(0.8, 10.7, "Signal", color="#000000", fontsize = 8)
                #ax1.text(0.0, 8.2, "Reconstruction", color=color2, fontsize = 8)

                ax2 = subplot(2,1,2)
                ax2.plot(tk, K_mat_old[0], 'k', linewidth=1)


                self.save_plot(directory = "./figs/dump/", prefix = "reconstruct")

            plt.figure("Transfer")

            currtitle = currlabel + " pop " + dowhat + ", " + self.celltype[self.a_celltype[0]]

            ax = plot_transfer(currtitle, freq_used, mag, pha, t1, current, voltage[self.a_celltype[0]], freq_times, spike_freq, taum, fmean, self.ihold, rm, Vreset, Vth, Vrest, method_interpol, method_interpol_plot, SNR = SNR, VAF = VAF, ymax = self.ymax, ax = self.ax, linewidth = self.linewidth, color_vec = self.color_vec, alpha = self.alpha, opt_plot = opt_plot)

            suptitle("Population transfer function of " + str(self.N[self.a_celltype[0]]) + " " + self.celltype[self.a_celltype[0]] + ", amp: " + str(np.round(self.amp[self.a_celltype[0]],4)) + ", amod: " + str(self.amod) + ", ih: " + str(np.round(self.ihold,4)) + ", ih_s: " + str(np.round(self.ihold_sigma,4)) + ", fm: " + str(np.round(fmean,2)) + ", fl_s: " + str(self.fluct_s))

        return VAF, SNR, ax, tk, K_mat_old


    def save_plot(self, directory = "./figs/dump/", prefix = " "):

        if pop.id == 0:

            from datetime import datetime
            idate = datetime.now().strftime('%Y%m%d_%H%M')  # %S
            savefig(directory + idate + "-pop_transfer_" + prefix + "_" + self.celltype[self.a_celltype[0]] + "_N" + str(self.N[self.a_celltype[0]]) + "_ihold" + str(np.round(self.ihold,4)) + "_amp" + str(np.round(self.amp[self.a_celltype[0]],4)) + ".pdf", dpi = 300)  # save it


    def do_pca_ica(self, t_analysis_delay=0, t_analysis_stop=1, time=0, signals=0, output_dim=10, n_processes=32, n_chunks=32, do_ica=1, n_celltype = 0):

        if self.use_mpi:

            filepath = self.data_dir + "/" + str(self.pickle_prefix) + "_results_pop_pca_ica.p"

            if self.do_run or (os.path.isfile(filepath) is False):

                # PCA

                # remove beginning
                dt = time[2]-time[1]
                t = time[int(t_analysis_delay/dt):int(t_analysis_stop/dt)]
                pca_mat = np.array(signals[n_celltype]).T[int(t_analysis_delay/dt):int(t_analysis_stop/dt),:]

                node = mdp.nodes.PCANode(output_dim=output_dim, svd=True)

                # pad with zeros to be able to split into chunks!
                n_add = n_chunks-np.remainder(np.shape(pca_mat)[0],n_chunks)
                mat_add = np.zeros((n_add, np.shape(pca_mat)[1]))
                pca_mat_add = np.concatenate((pca_mat, mat_add))
                pca_mat_iter = np.split(pca_mat_add, n_chunks)

                flow = mdp.parallel.ParallelFlow([node])

                start_time = ttime.time()

                with mdp.parallel.ProcessScheduler(n_processes=n_processes, verbose=True) as scheduler:
                    flow.train([pca_mat_iter], scheduler=scheduler) # input has to be list, why??

                process_time = ttime.time() - start_time

                s = np.array(flow.execute(pca_mat_iter))
                s = s[0:len(t),:]  # resize to length of t!

                #print "node.d: ",node.d
                var_vec = node.d/sum(node.d)
                print 'Explained variance (', 0, ') : ',  round(node.explained_variance,4)
                print 'Variance (' , 0, ') : ', var_vec
                print 'Time to run (' , 0, ') : ', process_time

                s2 = []
                if do_ica:
                # ICA
                    #s2 = mdp.fastica(s)
                    ica = mdp.nodes.FastICANode() #CuBICANode()
                    ica.train(s)
                    s2 = ica(s)

                results = {'t':t, 'pca':s,'pca_var':var_vec,'pca_var_expl':round(node.explained_variance,4), 'ica':s2}

                if self.id == 0:
                    if self.dumpsave == 1:
                        pickle.dump( results, gzip.GzipFile( filepath, "wb" ) )

            else:

                if self.id == 0:
                    results = pickle.load( gzip.GzipFile( filepath, "rb" ) )

        else:

            # remove beginning
            dt = time[2]-time[1]
            t = time[int(t_analysis_delay/dt):int(t_analysis_stop/dt)]
            pca_mat = np.array(signals[n_celltype]).T[int(t_analysis_delay/dt):int(t_analysis_stop/dt),:]

            node = mdp.nodes.PCANode(output_dim=output_dim, svd=True)

            start_time = ttime.time()

            node.train(pca_mat)
            s = node(pca_mat)

            process_time = ttime.time() - start_time
            #print "node.d: ",node.d
            var_vec = node.d/sum(node.d)
            print 'Explained variance (', 0, ') : ',  round(node.explained_variance,4)
            print 'Variance (' , 0, ') : ', var_vec
            print 'Time to run (' , 0, ') : ', process_time

            s2 = []
            if do_ica:
            # ICA
                #s2 = mdp.fastica(s)
                ica = mdp.nodes.FastICANode() #CuBICANode()
                ica.train(s)
                s2 = ica(s)

            results = {'t':t, 'pca':s,'pca_var':var_vec,'pca_var_expl':round(node.explained_variance,4), 'ica':s2}

        return results


    def net_run(self, tstop, simprop = "default", t_analysis_delay=0, t_analysis_stop=1, stim_start=0):

        freq_times = []
        t_all_vec_vec = []
        id_all_vec_vec = []
        gsyns = []
        w_mat = []
        winh_mat = []
        time = []
        voltage = []
        current = []

        filepath = self.data_dir + "/" + str(self.pickle_prefix) + "_results_pop_randomnet.hdf5"

        if self.do_run or (os.path.isfile(filepath) is False):

            self.run(tstop)

            self.no_fmean = True
            results = self.get()

            time, voltage, current, fmean, gsyn = results.get('time'), results.get('voltage'), results.get('current'), results.get('fmean'), results.get('gsyn')
            freq_times, spike_freq, t_all_vec_vec, id_all_vec_vec, gsyns, w_mat, winh_mat = results.get('freq_times'), results.get('spike_freq'), results.get('t_all_vec_vec'), results.get('id_all_vec_vec'), results.get('gsyns'), results.get('w_mat'), results.get('winh_mat')

            if self.id == 0:
                if self.dumpsave == 1:
                    #pickle.dump( results, open( filepath, "wb" ) ) # gzip.GzipFile

                    print "- Saving", filepath

                    f = h5py.File(filepath, 'w')
                    f.create_dataset('time', data=time, compression='gzip', shuffle=True)
                    f.create_dataset('voltage', data=np.array(voltage), compression='gzip', shuffle=True)
                    f.create_dataset('current', data=current, compression='gzip', shuffle=True)
                    f.create_dataset('freq_times', data=freq_times, compression='gzip', shuffle=True)

                    #f.create_dataset('t_all_vec_vec', data=np.array(t_all_vec_vec), compression='lzf', shuffle=True)
                    #f.create_dataset('id_all_vec_vec', data=np.array(id_all_vec_vec), compression='lzf', shuffle=True)
                    #f.create_dataset('gsyns', data=np.array(gsyns), compression='lzf', shuffle=True)

                    for i in range(len(self.N)):
                        subgroup = f.create_group("cell" + str(i))
                        subgroup.create_dataset('t_all_vec_vec', data=t_all_vec_vec[i], compression='gzip', shuffle=True)
                        subgroup.create_dataset('id_all_vec_vec', data=id_all_vec_vec[i], compression='gzip', shuffle=True)
                        subgroup.create_dataset('g', data=gsyns[i], compression='gzip', shuffle=True)

                        #for j in range(len(gsyns[i])):
                        #    subsubgroup = subgroup.create_group("gsyn" + str(j))
                        #    subsubgroup.create_dataset('g', data=gsyns[i][j], compression='lzf', shuffle=True)

                    f.close()
                    print "- Save finished"

                    #filename = slugify(simprop)

                    #syn_grc = np.array(gsyns[0])

                    #import scipy
                    #from scipy import io

                    #print "Saving .mat"
                    #data = {}
                    #data['syn_grc'] = syn_grc[:,int(t_analysis_delay/self.bin_width):int(t_analysis_stop/self.bin_width)]
                    #data['time'] = freq_times[int(t_analysis_delay/self.bin_width):int(t_analysis_stop/self.bin_width)]-stim_start
                    #scipy.io.savemat('./figs/' + filename + '.mat',data)

        else:

            if self.id == 0:
                #results = pickle.load( open( filepath, "rb" ) ) #gzip.GzipFile
                f = h5py.File(filepath, 'r')

                time = np.array(f['time'])
                voltage = np.array(f['voltage'])
                current = np.array(f['current'])
                freq_times = np.array(f['freq_times'])


                for i in range(len(self.N)):
                    t_all_vec_vec.append(np.array(f['/cell' + str(i) + '/t_all_vec_vec']))
                    id_all_vec_vec.append(np.array(f['/cell' + str(i) + '/id_all_vec_vec']))
                    gsyns.append(np.array(f['/cell' + str(i) + '/g']))

                    #gsyns.append([])
                    #for j in range(self.N[i]):
                    #    gsyns[i].append(np.array(f['/cell' + str(i) + '/gsyn' + str(j) + '/g' ]))

                f.close()

        return time, voltage, current, t_all_vec_vec, id_all_vec_vec, gsyns, freq_times, w_mat, winh_mat


    def delall(self):

        if self.use_mpi:
            self.pc.gid_clear()
            print "- clearing gids"
        else:
            pass
            #h.topology()
            #for sec in h.allsec():
            #    print "- deleting section:", sec.name()
            #    #h("%s{delete_section()}"%sec.name())
            #    sec.push()
            #    h.delete_section()
            #h.topology()

        for n in range(self.n_celltypes):
            for m in self.cells[n]:
                m.destroy()
                del m
        del self.cells
        del self.nc_vecstim
        del self.netcons
        del self.nclist
        print h.topology()


    def delrerun(self):

        del self.nc_vecstim
        del self.netcons
        del self.nclist
        del self.vecstim
        del self.spike_vec
        del self.ST_stims
        del self.PF_stims

        self.netcons = []
        self.nclist = []
        self.nc_vecstim = []
        self.vecstim = []
        self.spike_vec = []
        self.ST_stims = []
        self.PF_stims = []

        self.t_vec = []
        self.id_vec = []
        self.rec_v = []

        for n in range(self.n_celltypes):
            if self.use_mpi:
                self.t_vec.append(h.Vector()) # np.array([0])
                self.id_vec.append(h.Vector()) # np.array([-1], dtype=int)
            else:
                self.t_vec.append([])

            self.rec_v.append(h.Vector())

            for cell in self.cells[n]:
                self.t_vec[n].append(h.Vector())
                cell.nc_spike.record(self.t_vec[n][-1])

        self.flucts = []            # Fluctuating inputs on this host
        self.noises = []            # Random number generators on this host
        self.plays = []             # Play inputs on this host
        self.rec_is = []
        self.trains = []

        self.ic_holds = []
        self.i_holdrs = []
        self.i_holds = []
        self.ic_starts = []
        self.vc_starts = []
        self.ic_steps = []
        self.tvecs = []
        self.ivecs = []
        self.noises = []
        self.record_syn = []
        self.id_all_vec_input = []
        self.t_all_vec_input = []
        self.syn_ex_dist = []
        self.syn_inh_dist = []


# test code
if __name__ == '__main__':

    # mpiexec -f ~/machinefile -enable-x -n 96 python Population.py --noplot

    from Stimulation import *
    from Plotter import *
    from Stimhelp import *

    from cells.IfCell import *
    import scipy
    from scipy import io

    dt = 0.1*ms
    dt = 0.025*ms

    do_run = 1
    if results.norun:  # do not run again use pickled files!
        print "- Not running, using saved files"
        do_run = 0


    do = np.array(["transfer"])
    opts = np.array(["if_cnoise", "grc_cnoise"]) #ssine
    #opts = np.array(["if_cnoise"]) #ssine
    #opts = np.array(["if_recon"]) #ssine
    opts = np.array(["if_syn_CFvec"])
    #opts = np.array(["prk_cnoise"])
    opts = np.array(["if_cnoise", "if_ssine"]) #ssine
    opts = np.array(["if_ssine"]) #ssine
    opts = np.array(["grc_cnoise_addn_cn_", "grc_cnoise_cn_", "grc_cnoise_addn_cn_a01"])
    opts = np.array(["grc_cnoise_addn100_cn_", "grc_cnoise_addn_cn_", "grc_cnoise_cn_"])
    opts = np.array(["grc_cnoise_addn100_cn_"])
    opts = np.array(["grc_cnoise_addn100_"])
    opts = np.array(["grc_cnoise_addn_cn_"])
    #opts = np.array(["grc_cnoise"])
    #opts = np.array(["grc_cnoise_cn", "grc_cnoise_addn_cn"])
    #opts = np.array(["if_cnoise_addn", "if_cnoise"])

    do = np.array(["timeconst"])

    #do = np.array(["transfer"])
    #opts = np.array(["grc_cnoise_syn"])
    #opts = np.array(["grc_recon_syn"])

    #do = np.array(["prk_test"])


    if "prk_test" in do:

        import multiprocessing
        from Purkinje import Purkinje
        cell = Purkinje()

        # set up recording
        # Time
        rec_t = h.Vector()
        rec_t.record(h._ref_t)

        # Voltage
        rec_v = h.Vector()
        rec_v.record(cell.soma(0.5)._ref_v)

        tstop = 500
        v_init = -60

        stim = h.IClamp(cell.soma(0.5))
        stim.amp = 0.0/nA
        stim.delay = 1
        stim.dur = 1000

        cpu = multiprocessing.cpu_count()
        h.load_file("parcom.hoc")
        p = h.ParallelComputeTool()
        p.change_nthread(cpu,1)
        p.multisplit(1)
        print 'cpus:', cpu

        h.load_file("stdrun.hoc")
        h.celsius = 37
        h.init()
        h.tstop = tstop
        dt = 0.025 # ms
        h.dt = dt
        h.steps_per_ms = 1 / dt
        h.v_init = v_init

        h.finitialize()
        h.run()

        t1 = np.array(rec_t)
        voltage = np.array(rec_v)
        s, spike_times = get_spikes(voltage, -20, t1)

        print 1000/diff( spike_times)

        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(t1, voltage)

        plt.show()


    if "transfer" in do:

        # SET DEFAULT VALUES FOR THIS PLOT
        fig_size =  [11.7, 8.3]
        params = {'backend': 'ps', 'axes.labelsize': 9, 'axes.linewidth' : 0.5, 'title.fontsize': 8, 'text.fontsize': 9,
            'legend.borderpad': 0.2, 'legend.fontsize': 8, 'legend.linewidth': 0.1, 'legend.loc': 'best', # 'lower right'
            'legend.ncol': 4, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'text.usetex': False, 'figure.figsize': fig_size}
        rcParams.update(params)


        freq_used0 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 100, 1000])*Hz
        #freq_used0 = np.concatenate((arange(0.1, 1, 0.1), arange(1, 501, 1) ))
        freq_used0 = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 200, 400, 600, 800, 1000])

        SNR = None
        NI = None
        VAF = None

        t_stim = 1000*s  # only for cnoise

        opt_plot = np.array(["only_mag","normalize", "dB"]) #
        #opt_plot = np.array(["normalize", "dB"]) #

        color_vec = (np.array(["Red", "Blue", "HotPink", "Indigo"]), np.array(["Blue", "Orange", "HotPink", "Indigo"]))
        #color=cm.jet(1.*i/x)

        method_interpol = np.array(['bin','syn'])
        method_interpol = np.array(['bin'])

        for i, o in enumerate(opts):

            dt = 0.025*ms
            bin_width = 5*ms
            bin_width = dt
            jitter = 0*ms

            n_syn_ex = [0]
            g_syn_ex = [1]
            noise_syn = 0
            inh_hold = 0
            n_syn_inh = [0]
            g_syn_inh = [1]
            tau1_ex = 0
            tau2_ex = 10*ms
            tau1_inh = 0
            tau2_inh = 100*ms

            cutf = 20
            sexp = -1

            cutf = 0
            sexp = 0

            ihold = [10]
            amod = 0.1 # relative value
            give_freq = True

            anoise = [0]
            fluct_tau = 0*ms

            N = [100]

            amp = 0 # absolute value
            fluct_s = [0] # absolute value 0.0008
            ihold_sigma = [0] # 0.01 absolute value

            CF_var = [[5,10,20]]
            CF_var = False

            syn_tau1 = 5*ms
            syn_tau2 = 5*ms

            do_csd = 1

            if "if" in o:

                do_csd = 1

                color_vec = (np.array(["Blue"]), np.array(["Blue"]))
                #color_vec = (np.array(["Red"]), np.array(["Red"]))

                cellimport = []
                celltype = ["IfCell"]
                #cell_exe = ["cell = IfCell()"]
                #cell_exe = ["cell = IfCell(e = -70*mV, thresh = -69*mV, vrefrac = -70*mV)"]
                #cell_exe = ["cell = IfCell(e = 0*mV, thresh = 1*mV, vrefrac = 0*mV)"]

                # Brunel
                #cell_exe = ["cell = IfCell(C = 0.0005 *uF, R = 40*MOhm, e = -70*mV, thresh = -50*mV, vrefrac = -56*mV); cell.add_resonance(tau_r = 100*ms, gr = 0.025*uS)"]

                #cell_exe = ["cell = IfCell(C = 0.0001*uF, R = 40*MOhm, sigma_C = 0.2, sigma_R = 0.2)"]
                #cell_exe = ["cell = IfCell(C = 0.0001*uF, R = 40*MOhm)"] # tau = 4 ms
                #cell_exe = ["cell = IfCell(C = 0.0001*uF, R = 40*MOhm, s_reset_noise = 0*mV)"] # tau = 4 ms

                #GrC resting: 737 MOhm, 2.985e-06 uF   tau: 0.0022 s
                #GrC transfer fit: tau: 0.027 s => with 2.985e-06 uF, R = 0.027/2.985e-12 = 9045 MOhm

                #cell_exe = ["cell = IfCell(C = 2.985e-06*uF, R = 9045*MOhm)"]

                thresh = -41.8
                R = 5227*MOhm
                #tau_passive = 3e-06*5227 = 15.7ms

                cell_exe = ["cell = IfCell(C = 3.0e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -71.5*mV)"]

                prefix = "if_tf"

                istart = 0
                istop = 0.01
                di = 0.00001


                syn_tau1 = 10*ms
                syn_tau2 = 10*ms

                # Indirect
                give_freq = True
                ihold = [40]
                amod = 1 # relative value

                anoise = [0]
                fluct_tau = 0*ms

                #anoise = 0.1
                #fluct_tau = 100*ms

#                # Direct
#                give_freq = False
#                ihold = [0.00569223341176]
#                amod = None
#                amp = 7.31353725e-06
#
#                anoise = None
#                fluct_s = [3.65676863e-06]
#                fluct_tau = 0*ms
#
#                # Low CF, No low noise
#                N = [10000]
#                give_freq = False
#                ihold = [0.004]
#                ihold_sigma = [0.1/2] # 0.1/2 0.01 realtive value
#                amod = None
#                amp = 0.0021
#
#                anoise = None
#                fluct_s = [0.00]  # .005
#                fluct_tau = 0*ms


#                # Low CF, With low noise
#                N = [10000]
#                give_freq = False
#                ihold = [0.002]
#                ihold_sigma = [0.1/2] # 0.1/2 0.01 realtive value
#                amod = None
#                amp = 0.001
#
#                anoise = None
#                fluct_s = [0.002]  # .005
#                fluct_tau = 100*ms

            if "resif" in o:

                do_csd = 1

                color_vec = (np.array(["Blue"]), np.array(["Blue"]))
                #color_vec = (np.array(["Red"]), np.array(["Red"]))

                cellimport = []
                celltype = ["IfCell"]

                gr = 5.56e-05*uS
                tau_r = 19.6*ms
                R = 5227*MOhm
                delta_t = 4.85*ms
                thresh = (0.00568*nA * R) - 71.5*mV #
                thresh = -41.8

                cellimport = []
                celltype = "IfCell"
                cell_exe = "cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -71.5*mV, dgk =" + str(gr) + ", egk = -71.5*mV, ctau =" + str(tau_r) + ")"

                prefix = "resif_tf"

                istart = 0
                istop = 0.01
                di = 0.00001


                syn_tau1 = 10*ms
                syn_tau2 = 10*ms

                # Indirect
                give_freq = True
                ihold = [40]
                amod = 1 # relative value

                anoise = [0]
                fluct_tau = 0*ms
                dt = 0.1*ms



            if "if_syn" in o:

                N = [1]
                ihold = [40]
                amod = 1 # relative value

                prefix = "if_syntf"

                n_syn_ex = 1

                g_syn_ex = 0

                noise_syn = 0

                fluct_tau = 0*ms

                freq_used = np.array([])

                tau1_ex=0*ms
                tau2_ex=10*ms

                anoise = [0]


            if "grc" in o:

                color_vec = (np.array(["Blue"]), np.array(["Blue"]))

                cellimport = ["from GRANULE_Cell import Grc"]
                celltype = ["Grc"]
                cell_exe = ["cell = Grc(np.array([0.,0.,0.]))"]

                prefix = "grc_tf"

                istart = 0
                istop = 0.1
                di = 0.01

                syn_tau1 = 10*ms
                syn_tau2 = 10*ms

                # Indirect
                give_freq = True
                ihold = [40]
                amod = 1 # relative value

                anoise = [0]
                fluct_tau = 0*ms

                #anoise = 0.1
                #fluct_tau = 100*ms

#                # Direct
#                give_freq = False
#                ihold = [0.0058021085712642992]
#                amod = None
#                amp = 7.31353725e-06
#
#                anoise = None
#                fluct_s = [3.65676863e-06]
#                fluct_tau = 0*ms
#
#                # Low CF, No low noise
#                N = [50]
#                give_freq = False
#                ihold = [0.0049]
#                ihold_sigma = [0.1/2] # 0.1/2 0.01 realtive value
#                amod = None
#                amp = 0.0021
#
#                anoise = None
#                fluct_s = [0.00]  # .005
#                fluct_tau = 0*ms
#
#
#                # Low CF, With low noise
#                N = [10000]
#                give_freq = False
#                ihold = [0.003]
#                ihold_sigma = [0.1/2] # 0.1/2 0.01 realtive value
#                amod = None
#                amp = 0.001
#
#                anoise = None
#                fluct_s = [0.002]  # .005
#                fluct_tau = 100*ms


            use_multisplit = False
            use_mpi = True
            simstep = 1*s

            if "prk" in o:

                N = [1]
                ihold = [60]

                color_vec = (np.array(["Blue"]), np.array(["Blue"]))

                cellimport = ["from Purkinje import Purkinje"]
                celltype = ["Prk"]
                cell_exe = ["cell = Purkinje()"]

                prefix = "prk_tf"

                temperature = 37

                istart = 0
                istop = 0.1
                di = 0.005

                use_multisplit = True
                use_mpi = False

                t_stim = 5*s  # only for cnoise
                simstep = 1*s


            if "grc_syn" in o:

                N = [1]
                ihold = [125]
                amod = 1 # relative value

                prefix = "grc_syntf"

                cutf = 20
                sexp = -1

                cutf = 0
                sexp = 0

                n_syn_ex = 1
                g_syn_ex = -1
                noise_syn = 1

                n_syn_inh = -1
                inh_hold = 0
                g_syn_inh = 0

                fluct_tau = 0*ms

                freq_used = np.array([])

                anoise = 0


            if "_addn" in o:

                anoise = [6] # RESPONSIBLE FOR FILTERING EFFECT!!!
                fluct_tau = 1*ms
                prefix = prefix + "_addn"
                color_vec = (np.array(["Red"]), np.array(["Red"]))

            if "_addn100" in o:

                anoise = [2] # RESPONSIBLE FOR FILTERING EFFECT!!!
                fluct_tau = 100*ms
                prefix = prefix + "100"
                color_vec = (np.array(["Green"]), np.array(["Green"]))

            if "_cn_" in o:

                cutf = 20
                sexp = -1
                prefix = prefix + "_cn"

            if "_a01" in o:

                amod=0.1
                prefix = prefix + "_a01"



            plt.figure(i)
            pickle_prefix = "Population.py_" + prefix

            #comm = MPI.COMM_WORLD
            #comm.Barrier()   # wait for other nodes

            pop = Population(cellimport = cellimport,  celltype = celltype, cell_exe = cell_exe, N = N, temperature = 37, ihold = ihold, ihold_sigma = ihold_sigma, amp = amp, amod = amod, give_freq = give_freq, do_run = do_run, pickle_prefix = pickle_prefix, istart = istart, istop = istop, di = di, dt = dt)
            pop.bin_width = bin_width
            pop.jitter = jitter
            pop.anoise = anoise
            pop.fluct_s = fluct_s
            pop.fluct_tau = fluct_tau
            pop.method_interpol = method_interpol
            pop.no_fmean = False
            pop.CF_var = CF_var

            pop.tau1_ex=tau1_ex
            pop.tau2_ex=tau2_ex
            pop.tau1_inh=tau1_inh
            pop.tau2_inh=tau2_inh

            pop.n_syn_ex = n_syn_ex
            pop.g_syn_ex = g_syn_ex

            pop.noise_syn = noise_syn
            pop.inh_hold = inh_hold
            pop.n_syn_inh = n_syn_inh
            pop.g_syn_inh = g_syn_inh

            pop.force_run = False
            pop.use_multisplit = use_multisplit
            pop.use_mpi = use_mpi
            pop.simstep = simstep
            pop.use_local_dt = False
            pop.syn_tau1 = syn_tau1
            pop.syn_tau2 = syn_tau2
            pop.plot_input = False


            if n_syn_inh == -1:
                pop.connect_gfluct(g_i0=g_syn_inh)

            #pop.test_mod(n_syn_ex = n_syn_ex, g_syn_ex = g_syn_ex, noise_syn = noise_syn, inh_hold = inh_hold, n_syn_inh = n_syn_inh, g_syn_inh = g_syn_inh, do_plot = True)

            if "ssine" in o:
                pop.color_vec = color_vec
                #pop.color_vec = (np.array(["Red", "Orange", "HotPink", "Indigo"]), np.array(["Red", "Orange", "HotPink", "Indigo"]))
                pop.fun_plot(currlabel = "control", dowhat = "ssine", freq_used = freq_used0, opt_plot = opt_plot)

                pop.save_plot(directory = "./figs/dump/")

            if "cnoise" in o:

                freq_used = np.array([])
                pop.color_vec = color_vec
                #pop.color_vec = (np.array(["Blue", "Green", "DimGray", "DarkGoldenRod"]), np.array(["Blue", "Green", "DimGray", "DarkGoldenRod"]))
                pop.fun_plot(currlabel = "control", dowhat = "cnoise", t_stim = t_stim, cutf = cutf, sexp = sexp, t_qual = 0, opt_plot = opt_plot, freq_used = freq_used, do_csd = do_csd)

                pop.save_plot(directory = "./figs/dump/")


            if "recon" in o:

                pop.color_vec = color_vec
                #VAF, SNR, ax, tk, K_mat_old = pop.fun_plot(currlabel = "control", dowhat = "cnoise", t_stim = t_stim, cutf = cutf, sexp = sexp, t_qual = 0, opt_plot = opt_plot, n_syn_ex = n_syn_ex, g_syn_ex = g_syn_ex, noise_syn = noise_syn, inh_hold = inh_hold, n_syn_inh = n_syn_inh, g_syn_inh = g_syn_inh, SNR=0, freq_used = freq_used)

                # RECONSTRUCT!
                freq_used = np.array([9, 47, 111, 1000])*Hz
                t_stim = 10*s

                tk = arange(0,0.8192*2,pop.dt)
                K_mat_old = zeros((len(method_interpol),len(tk)), dtype=complex)

                if pop.id == 0:

                    sigma = 0.1e-3
                    a=0.1
                    t0 = tk[floor(len(tk)/2)]
                    K_mat_old[0] = gauss_func(tk, a, t0, sigma)

                K_mat_old = np.array([])

                results = pop.fun_cnoise_Stim(t_stim = t_stim, cutf = cutf, sexp = sexp, t_qual = 5, n_syn_ex = n_syn_ex, g_syn_ex = g_syn_ex, noise_syn = noise_syn, inh_hold = inh_hold, n_syn_inh = n_syn_inh, g_syn_inh = g_syn_inh, freq_used = freq_used, K_mat_old = K_mat_old, seed = 311)

                freq_used, amp_vec, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1')
                freq_times, spike_freq, fmean, method_interpol, SNR, VAF, Qual = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('SNR'), results.get('VAF'), results.get('Qual')
                stim, resp_mat, stim_re_mat = results.get('stim'), results.get('resp_mat'), results.get('stim_re_mat')

                if pop.id == 0:

                    plt.figure('Reconstruct')
                    axR0 = plt.subplot(4,1,1)
                    axR1 = plt.subplot(4,1,2)
                    axR2 = plt.subplot(4,1,3)
                    axR3 = plt.subplot(4,1,4)

                    axR0.plot(np.arange(len(stim))*pop.dt, resp_mat[0,:])
                    axR0.axis(xmin=0.9, xmax=1)
                    #axR0.plot(t1, voltage[0])
                    axR1.plot(np.arange(len(stim))*pop.dt, stim, 'b')
                    axR1.axis(xmin=0.9, xmax=1)
                    axR2.plot(np.arange(len(stim))*pop.dt, stim_re_mat[0,:], 'r')
                    axR2.axis(xmin=0.9, xmax=1)
                    axR3.plot(tk, K_mat_old[0])
                    plt.savefig("./figs/dump/Reconstruct.pdf", dpi = 300, transparent=True) # save it

            pop = None


    plt.show()


    if "timeconst" in do:

        from lmfit import minimize, Parameters

        # SET DEFAULT VALUES FOR THIS PLOT
        fig_size =  [11.7, 8.3]
        params = {'backend': 'ps', 'axes.labelsize': 9, 'axes.linewidth' : 0.5, 'title.fontsize': 8, 'text.fontsize': 9,
            'legend.borderpad': 0.2, 'legend.fontsize': 8, 'legend.linewidth': 0.1, 'legend.loc': 'best', # 'lower right'
            'legend.ncol': 4, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'text.usetex': False, 'figure.figsize': fig_size}
        rcParams.update(params)

        dt = 0.025*ms

        prefix = "timeconst"
        pickle_prefix = "Population.py_" + prefix

        stimtype = "inh_50ms_20ms"

        if stimtype == "ex_20ms":

            trun = 2.9
            tstart = 1.8
            tstop = 2.7

            celltype = ["IfCell"]
            cell_exe = ["cell = IfCell(C = 0.0001*uF, R = 200*MOhm)"]
            N = [5000]

            pop = Population(celltype = celltype, cell_exe = cell_exe, N = N, temperature = 0, do_run = do_run, pickle_prefix = pickle_prefix, dt = dt)

            pop.method_interpol = np.array(["bin", "syn"])
            pop.method_interpol = np.array(["bin"])

            modulation_vec = pop.set_PulseStim(start_time=[100*ms], dur=[3000*ms], steadyf=[100*Hz], pulsef=[150*Hz], pulse_start=[2000*ms], pulse_len=[500*ms], weight0=[1*nS], tau01=[0*ms], tau02=[20*ms], weight1=[0*nS], tau11=[0*ms], tau12=[1*ms])

            params = Parameters()
            params.add('amp', value=0.1)
            params.add('shift', value=10)
            params.add('tau1', value=1, vary=False) # alpha!
            params.add('tau2', value=20*ms)


        if stimtype == "ex_gr":

            trun = 6.9
            tstart = 4.8
            tstop = 6.5

            cellimport = ["from GRANULE_Cell import Grc"]
            celltype = ["Grc"]
            cell_exe = ["cell = Grc(np.array([0.,0.,0.]))"]
            N = [4096*10]

            pop = Population(cellimport = cellimport, celltype = celltype, cell_exe = cell_exe, N = N, temperature = 37, do_run = do_run, pickle_prefix = pickle_prefix, dt = dt)

            pop.method_interpol = np.array(["bin", "syn"])
            pop.method_interpol = np.array(["bin"])

            modulation_vec = pop.set_PulseStim(start_time=[100*ms], dur=[7000*ms], steadyf=[20*Hz], pulsef=[30*Hz], pulse_start=[5000*ms], pulse_len=[500*ms])

            params = Parameters()
            params.add('amp', value=0.1)
            params.add('shift', value=10)
            params.add('tau1', value=1, vary=False) # alpha!
            params.add('tau2', value=20*ms)


        if stimtype == "inh_50ms_20ms":

            trun = 2.9
            tstart = 1.8
            tstop = 2.7

            celltype = ["IfCell", "IfCell"]
            cell_exe = ["cell = IfCell()", "cell = IfCell()"]

            N = [10000,10000]

            pop = Population(celltype = celltype, cell_exe = cell_exe, N = N, temperature = 0, do_run = do_run, pickle_prefix = pickle_prefix, dt = dt)

            pop.method_interpol = np.array(["bin", "syn"])
            pop.method_interpol = np.array(["bin"])

            modulation_vec = pop.set_PulseStim(start_time=[100*ms,100*ms], dur=[3000*ms,3000*ms], steadyf=[100*Hz,50*Hz], pulsef=[100*Hz,80*Hz], pulse_start=[2000*ms,2000*ms], pulse_len=[500*ms,500*ms], weight0=[1*nS,1*nS], tau01=[1*ms,1*ms], tau02=[20*ms,20*ms], weight1=[0,0], tau11=[0*ms,0*ms], tau12=[1*ms,1*ms])

            pop.connect_cells(conntype='inh', weight=0.001, tau=50)

            params = Parameters()
            params.add('amp', value=-0.1)
            params.add('shift', value=10)
            params.add('tau1', value=1, vary=False) # alpha!
            params.add('tau2', value=20*ms)


        if stimtype == "inh_gr":

            trun = 9.9
            tstart = 4.8
            tstop = 8

            cellimport = ["from GRANULE_Cell import Grc", "from templates.golgi.Golgi_template import Goc"]
            celltype = ["Grc","Goc_noloop"]
            cell_exe = ["cell = Grc(np.array([0.,0.,0.]))","cell = Goc(np.array([0.,0.,0.]))"]
            N = [100,4]
            #N = [4096, 27]
            #N = [4096*5, 27*5]

            pop = Population(cellimport = cellimport, celltype = celltype, cell_exe = cell_exe, N = N, temperature = 37, do_run = do_run, pickle_prefix = pickle_prefix, dt = dt)

            pop.method_interpol = np.array(["bin", "syn"])
            pop.method_interpol = np.array(["bin"])

            modulation_vec = pop.set_PulseStim(start_time=[100*ms,100*ms], dur=[9800*ms,9800*ms], steadyf=[60*Hz,10*Hz], pulsef=[60*Hz,22*Hz], pulse_start=[5000*ms,5000*ms], pulse_len=[1500*ms,1500*ms])

            pop.connect_cells(conntype='inh_gr', weight = 0.3)

            params = Parameters()
            params.add('amp', value=-0.1)
            params.add('shift', value=10)
            params.add('tau1', value=1, vary=False) # alpha!
            params.add('tau2', value=20*ms)


        if stimtype == "inh_50ms_curr":

            trun = 2.9
            tstart = 1.8
            tstop = 2.8

            celltype = ["IfCell", "IfCell"]
            cell_exe = ["cell = IfCell()", "cell = IfCell()"]

            N = [1000,1000]

            give_freq = True

            istart = 0
            istop = 0.2
            di = 0.01

            ihold = [100, 50]
            ihold_sigma = [0.01, 0.01] # relative sigma

            pop = Population(celltype = celltype, cell_exe = cell_exe, N = N, temperature = 0, ihold = ihold, ihold_sigma = ihold_sigma, give_freq = give_freq, do_run = do_run, pickle_prefix = pickle_prefix, istart = istart, istop = istop, di = di, dt = dt)

            pop.method_interpol = np.array(["bin", "syn"])
            pop.method_interpol = np.array(["bin"])

            tstep = 2
            tdur = 0.5

            istep = [100,100]
            current1 = np.concatenate(([ihold[1]*np.ones(round((tstep)/pop.dt)), istep[1]*np.ones(round(tdur/pop.dt)),ihold[1]*np.ones(round((trun-tstep-tdur)/pop.dt)) ]))

            pop.set_IStim()
            pop.set_IStep(istep = istep, istep_sigma = [0.01,0.01], tstep = tstep, tdur = tdur)

            pop.connect_cells(conntype='inh', weight=0.0003, tau=50)

            pop.fluct_s = [0.02,0.05]
            pop.connect_fluct()

            params = Parameters()
            params.add('amp', value=-0.1)
            params.add('shift', value=10)
            params.add('tau1', value=1, vary=False) # alpha!
            params.add('tau2', value=20*ms)


        if stimtype == "inh_gr_curr":

            trun = 9.9
            tstart = 4.8
            tstop = 8

            cellimport = ["from GRANULE_Cell import Grc", "from templates.golgi.Golgi_template import Goc"]
            celltype = ["Grc","Goc_noloop"]
            cell_exe = ["cell = Grc(np.array([0.,0.,0.]))","cell = Goc(np.array([0.,0.,0.]))"]
            N = [100,4]
            N = [4096, 27]
            N = [4096*10, 27*10]

            give_freq = True

            # GRC
            #istart = 0
            #istop = 0.1
            #di = 0.01

            #GOC
            istart = 0
            istop = 0.5
            di = 0.02

            ihold = [100, 10]
            ihold_sigma = [0, 0] # relative sigma

            pop = Population(cellimport = cellimport, celltype = celltype, cell_exe = cell_exe, N = N, temperature = 37, ihold = ihold, ihold_sigma = ihold_sigma, give_freq = give_freq, do_run = do_run, pickle_prefix = pickle_prefix, istart = istart, istop = istop, di = di, dt = dt)

            pop.method_interpol = np.array(["bin", "syn"])
            pop.method_interpol = np.array(["bin"])

            tstep = 5
            tdur = 2

            istep = [100,50]
            current1 = np.concatenate(([ihold[1]*np.ones(round((tstep)/pop.dt)), istep[1]*np.ones(round(tdur/pop.dt)),ihold[1]*np.ones(round((trun-tstep-tdur)/pop.dt)) ]))

            pop.set_IStim()
            pop.set_IStep(istep = istep, istep_sigma = [0,0], tstep = tstep, tdur = tdur)

            pop.connect_cells(conntype='inh_gr', weight = 0.4)

            pop.fluct_s = [0.05,2]
            pop.connect_fluct()

            params = Parameters()
            params.add('amp', value=-0.1)
            params.add('shift', value=10)
            params.add('tau1', value=1, vary=False) # alpha!
            params.add('tau2', value=20*ms)


        pop.run_steps(trun)

        self.no_fmean = True
        results = pop.get()
        time, voltage, current, fmean, gsyn = results.get('time'), results.get('voltage'), results.get('current'), results.get('fmean'), results.get('gsyn')
        freq_times, spike_freq, t_all_vec_vec, id_all_vec_vec, gsyns = results.get('freq_times'), results.get('spike_freq'), results.get('t_all_vec_vec'), results.get('id_all_vec_vec'), results.get('gsyns')

        if pop.id == 0:

            bin_width = 1*ms
            freq_times = arange(0, time[-1], bin_width)
            [num_spikes, _] = neuronpy.util.spiketrain.get_histogram(t_all_vec_vec[0], bins = freq_times)
            spike_freq = np.concatenate((zeros(1),num_spikes)) / bin_width / N[0]


            if "inh" in stimtype: # generate input current, to complicated to get it out

                if "curr" in stimtype:
                    time1 = np.arange(0, trun, pop.dt)

                    r_mod = interp(freq_times, time1, current1, left=0, right=0)

                    [num_spikes, _] = neuronpy.util.spiketrain.get_histogram(t_all_vec_vec[1], bins = freq_times)
                    spike_freq1 = np.concatenate((zeros(1),num_spikes)) / bin_width / N[1]
                else:
                    r_mod = interp(freq_times, modulation_vec[1][0], modulation_vec[1][1], left=0, right=0)

                    [num_spikes, _] = neuronpy.util.spiketrain.get_histogram(t_all_vec_vec[1], bins = freq_times)
                    spike_freq1 = np.concatenate((zeros(1),num_spikes)) / bin_width / N[1]

            elif "ex" in stimtype:
                r_mod = interp(freq_times, modulation_vec[0][0], modulation_vec[0][1], left=0, right=0)


            def modelfun(amp, shift, tau1, tau2, bin_width, r_mod):

                tau1 = tau1
                tau2 = tau2

                t1 = np.arange(0,10*tau2,bin_width)
                K = amp*syn_kernel(t1, tau1, tau2)
                K = np.concatenate((np.zeros(len(K)-1),K))
                t2 = np.arange(0,len(K)*bin_width,bin_width)

                model = np.convolve(K, r_mod, mode='same') + shift

                return model


            def residual(params, r_mod, data=None, bin_width=1*ms, tstart=0, tstop=3):

                amp = params['amp'].value
                shift = params['shift'].value
                tau1 = params['tau1'].value
                tau2 = params['tau2'].value

                model = modelfun(amp, shift, tau1, tau2, bin_width, r_mod)

                return (data[int(tstart/bin_width):int(tstop/bin_width)]-model[int(tstart/bin_width):int(tstop/bin_width)])


            result = minimize(residual, params, args=(r_mod, spike_freq, bin_width, tstart, tstop))

            print "chisqr: ", result.chisqr
            print 'Best-Fit Values:'
            for name, par in params.items():
                print '  %s = %.4f +/- %.4f ' % (name, par.value, par.stderr)

            amp = params['amp'].value
            shift = params['shift'].value
            tau1 = params['tau1'].value
            tau2 = params['tau2'].value

            model = modelfun(amp, shift, tau1, tau2, bin_width = bin_width, r_mod = r_mod)


            if "ex" in stimtype:
                plt.figure(0)
                plt.plot(freq_times[int(0.5/bin_width):int(trun/bin_width)], spike_freq[int(0.5/bin_width):int(trun/bin_width)], freq_times[int(0.5/bin_width):int(trun/bin_width)], model[int(0.5/bin_width):int(trun/bin_width)])
                plt.figure(1)
                plt.plot(time, voltage[0]), freq_times, r_mod, time, current
                #plt.figure(100)
                #plt.plot(t_all_vec_vec[0],id_all_vec_vec[0],'k|')
                #plt.savefig("./figs/dump/taufit_" + str(stimtype) + "_spikes.pdf", dpi = 300)  # save it

            else:
                plt.figure(0)
                plt.plot(freq_times[int(0.5/bin_width):int(trun/bin_width)], spike_freq1[int(0.5/bin_width):int(trun/bin_width)], freq_times[int(0.5/bin_width):int(trun/bin_width)], spike_freq[int(0.5/bin_width):int(trun/bin_width)], freq_times[int(0.5/bin_width):int(trun/bin_width)], model[int(0.5/bin_width):int(trun/bin_width)])
                plt.figure(1)
                plt.plot(time, voltage[0], time, voltage[1], freq_times, r_mod, time, current)
                plt.figure(100)
                #plt.plot(t_all_vec_vec[0],id_all_vec_vec[0],'k|')
                #plt.plot(t_all_vec_vec[1],id_all_vec_vec[1],'b|')
                #plt.savefig("./figs/dump/taufit_" + str(stimtype) + "_spikes.pdf", dpi = 300)  # save it


            plt.figure(0)
            plt.title('Fit: ' + str(stimtype) + ', tau1=' + str(tau1) +  ' tau2=' + str(tau2))
            plt.savefig("./figs/dump/taufit_" + str(stimtype) + "_rate.png", dpi = 300)  # save it

            plt.figure(1)
            plt.savefig("./figs/dump/taufit_" + str(stimtype) + "_voltage.png", dpi = 300)  # save it


    plt.show()
