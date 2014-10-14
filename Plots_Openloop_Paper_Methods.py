# -*- coding: utf-8 -*-
"""
@author: chris

mpiexec -f ~/machinefile -enable-x -n 96 python Plots_Openloop_Paper_Methods.py -o fig1 --noplot 2>&1 | tee log/log1.txt
qsub -v J=Plots_Openloop_Paper_Results_Methods.py,O=fig1 -pe ompigige 96 PBSinsigneo.sh

mpiexec -f ~/machinefile -enable-x -n 1 python Plots_Openloop_Paper_Methods.py -o fig2 --noplot 2>&1 | tee log/log2.txt
qsub -v J=Plots_Openloop_Paper_Results_Methods.py,O=fig2 -pe ompigige 1 -l rmem=32G -l mem=32G PBSinsigneo.sh


"""

from __future__ import with_statement
from __future__ import division

import sys
sys.path.append('NEURON/')

import os
from mpi4py import MPI

#print sys.version

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-o', action='store', dest='opt')
parser.add_argument('--noplot', action='store_true')
parser.add_argument('--norun', action='store_true')
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
    
do_run = 1
if results.norun:  # do not run again use pickled files!
    print "- Not running, using saved files"
    do_run = 0

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
font0 = FontProperties()

font = font0.copy()
font.set_family('sans-serif')
font.set_weight('semibold')  

from neuron import h

from units import *
from Population import *        
from Stimulation import *
from Plotter import *
from Stimhelp import *

# PLOS: Fonts: 8 - 12, Multi-panel figures labels: 12


fig_size =  [28*0.3937, 10*0.3937]
params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 10, 
          'text.fontsize': 8,
          'font.size':8,
          'axes.titlesize':8,
          'legend.fontsize': 10,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'text.usetex': False,
          'figure.figsize': fig_size}
rcParams.update(params)

#matplotlib.rc('font', **{'sans-serif' : 'Arial'}) #, 'family' : 'sans-serif'})
#matplotlib.rc('font', family='sans-serif') 

b1 = '#1F78B4'
b2 = '#A6CEE3'
g1 = '#33A02C'
g2 = '#B2DF8A'
r1 = '#E31A1C'
r2 = '#FB9A99'
o1 = '#FF7f00'
o2 = '#FDBF6F'
p1 = '#6A3D9A'
p2 = '#CAB2D6'

color0 = 'black' # BLACK
color1 = 'blue' # BLUE
color2 = 'red' # RED
color3 = 'gray' # GRAY 
color4 = 'purple' # PURPLE
color5 = 'orange' # ORANGE
color6 = 'green' # GREEN

linewidth = 1

d_out = 5
d_down = 5

xmax = 130

dt = 0.025*ms

# Used for conversion, ignore
data_dir = "./publish/openloop/fulldata"
minimal_dir = "./publish/openloop/minimal"
export = False

data_dir = "./publish/openloop/minimal"
minimal_dir = False


t_stim = 1000*s # only for cnoise

# FIGURE 1
if results.opt == "fig1":
    export = False
    do_vec = np.array(["cell_methods_transfer_if_fig1_"])   
    
if results.opt == "fig1test":
    export = False
    do_vec = np.array(["cell_methods_transfer_if_fig1_"])   

# FIGURE 2
if results.opt == "fig2":
    export = True
    combine = "yes"
    do_vec = np.array(["pop_transfer_if_wn_fig2", "pop_transfer_if_cn_fig2"]) 


# TEST
#do_vec = np.array(["cell_methods_transfer_grc_fig1_"])
#do_vec = np.array(["pop_transfer_if_wn_fig2_0noise"]) 
#do_vec = np.array(["pop_transfer_if_wn_fig2_poster", "pop_transfer_if_cn_fig2_poster"]) 
#do_vec = np.array(["pop_transfer_if_wn_fig2_SNR", "pop_transfer_if_cn_fig2_SNR"]) 
#do_vec = np.array(["pop_transfer_grc_wn_talk"])
#do_vec = np.array(["pop_transfer_grc_cn_talk","pop_transfer_if_cn_talk","pop_transfer_resif_cn_talk"]) 



for d, do in enumerate(do_vec):
    
    pickle_prefix = ""
    if "_fig" in do:
        fig_num = str(do).split("_fig")[1].split("_")[0]
        pickle_prefix = pickle_prefix + "Fig" + str(fig_num) + "_"
    
    if ("poster" in do) or ("talk" in do):
        color0 = 'black' # BLACK
        color1 = '#00A0E3' # Cyan 
        color2 = '#E5097F' # Magenta
        color3 = '#FFED00' # Yellow
        color4 = '#393476' # Uni Blue
        color5 = '#E42A24' # Red
        color6 = '#009A47' # Dark Green
        color7 = '#78317B' # Lila 
        color8 = '#BFB5B1' # Gray
        color9 = '#EC671F' # Orange
        
        linewidth = 1.5
    
    if "cell" in do:
            
        anoise = 0
        tau_noise = 0*ms
        
        
        if "if" in do:
            
            ihold = 40
            
            amp = 0
            amod = 0.1
            
            istart = 0.002 
            istop = 0.05
            di = 0.0001
            
            sexp = 0
            cutf = 0
            
            #thresh = -21.175*mV 
            #R = 8860*MOhm
            #tau_passive = 3e-06*8860 = 26.6ms
            
            thresh = -41.8    
            R = 5227*MOhm
            #tau_passive = 3e-06*5227 = 15.7ms
            
            celltype = "IfCell"
            cell_exe = "cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -71.5*mV)"
            cellimport = "from cells." + celltype + " import *"
            
            pickle_prefix = pickle_prefix + "cell_if" 
            
            exec cellimport
            exec cell_exe   
            
            temperature = 0
            give_freq = True
            SNR = None 
            NI = None
            
            synout_tau1 = 100*ms
            synout_tau2 = 100*ms
            spikes_from_neuron = True
            
            icloc = "soma(0.5)"
            
            #give_freq = False
            #ihold = 0.003
            #amod = 1

            
        if "grc" in do:
            
            ihold = 40
            
            amp = 0
            amod = 0.1
            
            istart = 0 
            istop = 0.1
            di = 0.005
            
            sexp = 0
            cutf = 0
                        
            cellimport = "from GRANULE_Cell import Grc"
            celltype = "Grc"
            cell_exe = "cell = Grc(np.array([0.,0.,0.]))"  
                     
            
            pickle_prefix = pickle_prefix + "cell_grc" 
            
            exec cellimport
            exec cell_exe   
            
            temperature = 37
            give_freq = True
            SNR = None 
            NI = None
            
            synout_tau1 = 5*ms
            synout_tau2 = 5*ms
            spikes_from_neuron = False
            
            icloc = "soma(0.5)"
 
            
        if "methods_transfer" in do:
            
            d_out = 5
            d_down = 5
            
            fig_size =  [85*0.03937, 150*0.03937] # 1-Column
            params['figure.figsize'] = fig_size
            rcParams.update(params)
        
            fig1 = plt.figure('methods_transfer')  
            
            # METHODS
            gs = matplotlib.gridspec.GridSpec(3, 2,
                           width_ratios=[1,1],
                           height_ratios=[1,1,1]
                           )
   
            ax1 = plt.subplot(gs[0,0])
            ax2 = plt.subplot(gs[1,0])
            ax3 = plt.subplot(gs[2,0])
            ax7 = plt.subplot(gs[0,1]) #, sharex=ax2
            ax8 = plt.subplot(gs[1,1]) #, sharex=ax2
            ax9 = plt.subplot(gs[2,1]) #, sharex=ax2
            
            gs.update(left=0.25, right=0.91, bottom=0.44, top=0.93, wspace=0.4, hspace=0.6)  

            sim = Stimulation(cell, celltype = celltype, cell_exe = cell_exe, temperature = temperature, do_run = do_run, give_freq = give_freq, istart = istart, istop = istop, di = di)
            #if MPI.COMM_WORLD.rank == 0: rm, cm, taum = sim.get_RCtau()
            
            sim.spikes_from_neuron = spikes_from_neuron
            sim.ihold = ihold
            sim.amp = amp
            sim.amod = amod
            sim.anoise = anoise
            sim.tau_noise = tau_noise
            
            sim.synout_tau1 = synout_tau1
            sim.synout_tau2 = synout_tau2
            sim.icloc = icloc
            sim.data_dir = data_dir
            sim.minimal_dir = minimal_dir
            
            method_interpol = np.array(["none", "linear", "shan", "syn", "dt"]) 
            
            sim.pickle_prefix = pickle_prefix + "_compare1"
            freq_used0 = array([5]) 
            
            if MPI.COMM_WORLD.rank == 0:

                results = sim.fun_ssine_Stim(freq_used = freq_used0, method_interpol = method_interpol)
                freq_used, vamp, mag, pha, ca, input_signal, t2, voltage, current, t1, freq_out_signal_interp_mat = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('stimulus'), results.get('t2'), results.get('voltage'), results.get('current'), results.get('t1'), results.get('freq_out_signal_interp_mat') 
                freq_times, spike_freq, fmean, method_interpol, VAF = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('VAF')                   
    
                ax1.plot(freq_times, spike_freq, 'ko', markersize=3) 
                ax1.plot(t2, freq_out_signal_interp_mat[0], color = g1, linewidth=linewidth)
                ax1.axis(xmin=-0.01, xmax=0.31)  
                adjust_spines(ax1, ['left'], d_out = d_out, d_down = d_down)
                ax1.axis(ymin=35, ymax=45) 
                ax1.yaxis.set_ticks(array([35,40,45]))
                ax1.set_yticklabels(('35', '40', '45'))
                ax1.set_ylabel("spikes/s", labelpad=0)
                ax1.text(-0.27,46,'Input\n(Hz)', fontsize=10)
                #ax1.text(-0.17,45,'Input\n'+r'($f_{in}$)', fontsize=16, horizontalalignment='center')
                ax1.text(-0.27,40,r'5$\,\,\blacktriangleright$', fontsize=10)
                ax1.set_title('Sinusoidal fit', fontsize=10, y=1.25)
                
                ax1.text(-0.27, 1.30, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
                
    
                #ax4.plot(freq_times, spike_freq, 'ko', t2, freq_out_signal_interp_mat[1], color = p1, linewidth=linewidth)
                #ax4.plot(t2, freq_out_signal_interp_mat[2], color = o1, linewidth=linewidth)
                #ax4.axis(xmin=0, xmax=0.21)
                #ax4.axis(ymin=35, ymax=45) 
                #adjust_spines(ax4, [], d_out = d_out, d_down = d_down)
                #ax4.set_title('Interpolation', fontsize=16)
                #ax4.text(0.03, 38, "linear", color=p1, fontsize = params['text.fontsize'])
                #ax4.text(0.12, 42, "shannon", color=o1, fontsize = params['text.fontsize'])
                
                #ax7.plot(freq_times, spike_freq, 'ko')
                #ax7.axis(xmin=0, xmax=0.21)
                #ax7.axis(ymin=38.5, ymax=41.5) 
                #adjust_spines(ax7, [], d_out = d_out, d_down = d_down)
                
                #ax7b = ax7.twinx()
                #ax7b.plot(t2, freq_out_signal_interp_mat[3], 'm', t2, freq_out_signal_interp_mat[4], 'c') 
                #adjust_spines(ax7b, ['right'], color='gray', d_out = d_out, d_down = d_down)
                #ax7b.axis(ymin=0, ymax=1.2) 
                #ax7b.yaxis.set_ticks(array([0,1]))
                #ax7b.set_yticklabels(('0', '1'))
                #ax7b.axis(xmin=0, xmax=0.21)
                #ax7.set_title('Filter', fontsize=15)
                
                ax7b = ax7.twinx()        
                ax7.plot(t2, freq_out_signal_interp_mat[3], color = r1, linewidth=linewidth)
                adjust_spines(ax7, ['left'], color=r1, d_out = d_out, d_down = d_down)                
                ax7.axis(xmin=-0.01, xmax=0.31)
                ax7.axis(ymin=10.7, ymax=11.0) 
                ax7.yaxis.set_ticks(array([10.7,11.0]))
                ax7.set_yticklabels(('10.7', '11.0'))
                ax7.set_ylabel("a.u.", labelpad=-10)
                ax7.set_title('Filter', fontsize=10, y=1.25)
                ax7.text(0.01, 10.64, "synaptic", color=r1, fontsize = params['text.fontsize'])
                
                
                
                ax7b.plot(t2, freq_out_signal_interp_mat[4], color=b1, linewidth=linewidth)
                adjust_spines(ax7b, ['right'], color=b1, d_out = d_out, d_down = d_down)
                ax7b.axis(xmin=-0.01, xmax=0.31)
                ax7b.axis(ymin=-0.01, ymax=1.01) 
                ax7b.yaxis.set_ticks(array([0,1]))
                ax7b.set_yticklabels(('0', '1'))
                ax7b.set_ylabel("a.u.", labelpad=0)
                ax7b.text(0.11, -0.45, "sampling-rate", color=b1, fontsize = params['text.fontsize'])
                
                
                
                ax7.text(-0.27, 1.3, 'B1', transform=ax7.transAxes, fontsize=12, va='top', fontproperties=font)
                
                
                sim.pickle_prefix = pickle_prefix + "_compare2"
                freq_used0 = array([15]) 
                results = sim.fun_ssine_Stim(freq_used = freq_used0, method_interpol = method_interpol)
                freq_used, vamp, mag, pha, ca, input_signal, t2, voltage, current, t1, freq_out_signal_interp_mat = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('stimulus'), results.get('t2'), results.get('voltage'), results.get('current'), results.get('t1'), results.get('freq_out_signal_interp_mat') 
                freq_times, spike_freq, fmean, method_interpol, VAF = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('VAF')                   
                                
                ax2.plot(freq_times, spike_freq, 'ko', markersize=3)
                ax2.plot(t2, freq_out_signal_interp_mat[0], color=g1, linewidth=linewidth)
                ax2.axis(xmin=-0.01, xmax=0.31) 
                adjust_spines(ax2, ['left'], d_out = d_out, d_down = d_down)
                ax2.axis(ymin=35, ymax=45) 
                ax2.yaxis.set_ticks(array([35,40,45]))
                ax2.set_yticklabels(('35', '40', '45'))
                ax2.set_ylabel("spikes/s", labelpad=0)
                ax2.text(-0.27,40,r'15$\,\blacktriangleright$', fontsize=10)
                
                ax2.text(-0.27, 1.3, 'A2', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
    
                #ax5.plot(freq_times, spike_freq, 'ko', t2, freq_out_signal_interp_mat[1], color = p1, linewidth=linewidth)
                #ax5.plot(t2, freq_out_signal_interp_mat[2], color = o1, linewidth=linewidth)
                #ax5.axis(xmin=0, xmax=0.21) 
                #ax5.axis(ymin=35, ymax=45) 
                #adjust_spines(ax5, [], d_out = d_out, d_down = d_down)
                
                #ax8.plot(freq_times, spike_freq, 'ko')
                #ax8.axis(xmin=0, xmax=0.21)
                #ax8.axis(ymin=39, ymax=41) 
                #adjust_spines(ax8, [], d_out = d_out, d_down = d_down)
                
                #ax8b = ax8.twinx()
                #ax8b.plot(t2, freq_out_signal_interp_mat[3], 'm', t2, freq_out_signal_interp_mat[4], 'c') 
                #adjust_spines(ax8b, ['right'], color='gray', d_out = d_out, d_down = d_down)
                #ax8b.axis(ymin=0, ymax=1.2) 
                #ax8b.yaxis.set_ticks(array([0,1]))
                #ax8b.set_yticklabels(('0', '1'))
                #ax8b.axis(xmin=0, xmax=0.21)
                
                ax8b = ax8.twinx()                
                ax8.plot(t2, freq_out_signal_interp_mat[3], color = r1, linewidth=linewidth)
                adjust_spines(ax8, ['left'], color=r1, d_out = d_out, d_down = d_down)                
                ax8.axis(xmin=-0.01, xmax=0.31)
                ax8.axis(ymin=10.7, ymax=11.0) 
                ax8.yaxis.set_ticks(array([10.7,11.0]))
                ax8.set_yticklabels(('10.7', '11.0'))
                ax8.set_ylabel("a.u.", labelpad=-10)
                
                ax8b.plot(t2, freq_out_signal_interp_mat[4], color=b1, linewidth=linewidth)
                adjust_spines(ax8b, ['right'], color=b1, d_out = d_out, d_down = d_down)
                ax8b.axis(xmin=-0.03, xmax=0.31)
                ax8b.axis(ymin=-0.01, ymax=1.01) 
                ax8b.yaxis.set_ticks(array([0,1]))
                ax8b.set_yticklabels(('0', '1'))
                ax8b.set_ylabel("a.u.", labelpad=0)
                
                ax8.text(-0.27, 1.3, 'B2', transform=ax8.transAxes, fontsize=12, va='top', fontproperties=font)
                                
                
                sim.pickle_prefix = pickle_prefix + "_compare3"
                freq_used0 = array([35]) 
                results = sim.fun_ssine_Stim(freq_used = freq_used0, method_interpol = method_interpol)
                freq_used, vamp, mag, pha, ca, input_signal, t2, voltage, current, t1, freq_out_signal_interp_mat = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('stimulus'), results.get('t2'), results.get('voltage'), results.get('current'), results.get('t1'), results.get('freq_out_signal_interp_mat') 
                freq_times, spike_freq, fmean, method_interpol, VAF = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('VAF')                   
    
                ax3.plot(freq_times, spike_freq, 'ko', markersize=3)
                ax3.plot(t2, freq_out_signal_interp_mat[0], color=g1, linewidth=linewidth)
                adjust_spines(ax3, ['left','bottom'], d_out = d_out, d_down = d_down)
                ax3.axis(ymin=35, ymax=45) 
                ax3.yaxis.set_ticks(array([35,40,45]))
                ax3.set_yticklabels(('35', '40', '45'))
                ax3.axis(xmin=-0.01, xmax=0.31) 
                ax3.xaxis.set_ticks(array([0,0.1,0.2,0.3]))
                ax3.set_xticklabels(('0', '0.1', '0.2', '0.3'))
                ax3.set_ylabel("spikes/s", labelpad=0)
                ax3.set_xlabel("s", labelpad=0)
                ax3.text(-0.27,40,r'35$\,\blacktriangleright$', fontsize=10)
                #ax3.text(0.1,25,r'$\blacktriangledown$'+'\n $Mag(sin.\,fit)$ ', fontsize=16, horizontalalignment='center')
                
                ax3.text(-0.27, 1.3, 'A3', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)
                
                #ax6.plot(freq_times, spike_freq, 'ko', t2, freq_out_signal_interp_mat[1], color=p1, linewidth=linewidth) 
                #ax6.plot(t2, freq_out_signal_interp_mat[2], color=o1, linewidth=linewidth)
                #ax6.axis(ymin=35, ymax=45) 
                #ax6.yaxis.set_ticks(array([35,40,45]))
                #ax6.set_yticklabels(('35', '40', '45'))
                #adjust_spines(ax6, ['bottom'], d_out = d_out, d_down = d_down)
                #ax6.axis(xmin=0, xmax=0.21)
                #ax6.xaxis.set_ticks(array([0,0.1,0.2]))
                #ax6.set_xticklabels(('0', '0.1', '0.2'))
                #ax6.set_xlabel("s", labelpad=0)
                #ax6.text(0.1,25,r'$\blacktriangledown$'+'\n'+'$Mag(FFT(y)(f_{in}))$', fontsize=16, horizontalalignment='center')
                
                #ax9.plot(freq_times, spike_freq, 'ko')
                #ax9.axis(ymin=39.65, ymax=40.35) 
                #adjust_spines(ax9, ['bottom'], d_out = d_out, d_down = d_down)
                #ax9.axis(xmin=0, xmax=0.21)
                #ax9.set_xlabel("s", labelpad=0)
                
                #ax9b = ax9.twinx()
                #ax9b.plot(t2, freq_out_signal_interp_mat[3], 'm', t2, freq_out_signal_interp_mat[4], 'c') 
                #adjust_spines(ax9b, ['bottom','right'], color='gray', d_out = d_out, d_down = d_down)
                #ax9b.axis(ymin=0, ymax=1.2) 
                #ax9b.yaxis.set_ticks(array([0,1]))
                #ax9b.set_yticklabels(('0', '1'))
                #ax9b.axis(xmin=0, xmax=0.21)
                #ax9b.xaxis.set_ticks(array([0,0.1,0.2]))
                #ax9b.set_xticklabels(('0', '0.1', '0.2'))
                #ax9b.set_xlabel("s", labelpad=0)
                #ax9.text(0.1,38.9,r'$\blacktriangledown$'+'\n'+'$Mag(FFT(y)(f_{in}))$', fontsize=16, horizontalalignment='center')
                
                #ax9.text(0.15,10.40,r'$\blacktriangledown$'+'\n'+'$Mag(FFT(y)(f_{in}))$', fontsize=16, horizontalalignment='center')
                
                ax9b = ax9.twinx()        
                ax9b.plot(t2, freq_out_signal_interp_mat[4], color=b1, linewidth=linewidth)
                adjust_spines(ax9b, ['right'], color=b1, d_out = d_out, d_down = d_down)
                ax9b.axis(xmin=-0.01, xmax=0.31)
                #ax9b.xaxis.set_ticks(array([0,0.1,0.2,0.3]))
                #ax9b.set_xticklabels(('0', '0.1', '0.2', '0.3'))
                ax9b.axis(ymin=-0.01, ymax=1.01) 
                ax9b.yaxis.set_ticks(array([0,1]))
                ax9b.set_yticklabels(('0', '1'))
                ax9b.set_ylabel("a.u.", labelpad=0)
                
                ax9.plot(t2, freq_out_signal_interp_mat[3], color=r1, linewidth=linewidth)
                adjust_spines(ax9, ['left','bottom'], color=r1, d_out = d_out, d_down = d_down)                
                ax9.axis(xmin=-0.01, xmax=0.31)
                ax9.xaxis.set_ticks(array([0,0.1,0.2,0.3]))
                ax9.set_xticklabels(('0', '0.1', '0.2', '0.3'))
                ax9.axis(ymin=10.7, ymax=11.0) 
                ax9.yaxis.set_ticks(array([10.7,11.0]))
                ax9.set_yticklabels(('10.7', '11.0'))
                ax9.set_ylabel("a.u.", labelpad=-10)
                ax9.set_xlabel("s", labelpad=0)    
     
                ax9.text(-0.27, 1.3, 'B3', transform=ax9.transAxes, fontsize=12, va='top', fontproperties=font)
                
                filename="./figs/Pub/Fig1_Methods_" + str(pickle_prefix) 
                savefig(filename + ".png", dpi = 300) # save it
                savefig(filename + ".pdf", dpi = 300) # save it
                #os.system('rsvg-convert -f pdf -o ' + filename +'.pdf ' + filename + '.svg')
                
            MPI.COMM_WORLD.Barrier()
                
            #######################
            
            # TRANSFER
            sim = None
            
            gs2 = matplotlib.gridspec.GridSpec(1, 1,
                           width_ratios=[1],
                           height_ratios=[1]
                           )
            
            gs2.update(bottom=0.06, top=0.38, left=0.16, right=0.97, wspace=0.35, hspace=0.1) 
            ax10 = plt.subplot(gs2[0,0])
            #ax11 = plt.subplot(gs2[0,1])
                
                            
            pickle_prefix = pickle_prefix + "_transfer"
            
            freq_used0 = concatenate(( arange(0.5, xmax+1, 1), array([]) )) #, arange(210, 1010, 10) ))
        
            sim = Stimulation(cell, celltype = celltype, cell_exe = cell_exe, temperature = temperature, do_run = do_run, pickle_prefix = pickle_prefix, give_freq = give_freq, istart = istart, istop = istop, di = di)
            #if MPI.COMM_WORLD.rank == 0: rm, cm, taum = sim.get_RCtau()
            
            # plot theoretical 
            tau = 15.7*ms
            H1, H0 = aiftransfer(freq_used0, tau = tau, f0 = ihold)            
            H = H1/H0
            H = H/H[2]
            #ax10.semilogx(freq_used0, 20*log10(H), 'k--', linewidth=linewidth) 
            
            sim.spikes_from_neuron = False
            
            #sim.del_freq = array([20, 60, 100])
            sim.ihold = ihold
            sim.amp = amp
            sim.amod = amod
            sim.anoise = anoise
            sim.tau_noise = tau_noise
            
            sim.synout_tau1 = synout_tau1
            sim.synout_tau2 = synout_tau2
            
            method_interpol = np.array(["none", "linear", "shan", "syn", "dt"]) 
            method_interpol_plot = np.array(["none", "syn", "dt"])  
              
            currtitle = "single sinusoid, " + celltype + ", amp: " + str(sim.amp) + ", ihold: " + str(sim.ihold) 
            sim.color_vec = (array([g1,p1,o1,r1,b1]),array([g1,p1,o1,r1,b1]))
            
            sim.data_dir = data_dir
            sim.minimal_dir = minimal_dir
            
            if "if" in do:
                opt_plot = np.array(["only_mag", "normalize", "analytical", "if", "dB"]) #, "dB"
            elif "grc" in do:
                opt_plot = np.array(["only_mag", "normalize", "dB"]) 
            else:        
                opt_plot = np.array(["only_mag", "normalize", "do_fit", "dB"]) 
            
            sim.linewidth = linewidth
            
            sim.fun_plot(currtitle, "ssine", freq_used = freq_used0, method_interpol = method_interpol, method_interpol_plot = method_interpol_plot, ax = ax10, axP = None, SNR = 0, VAF = 0, opt_plot = opt_plot)       

            if MPI.COMM_WORLD.rank == 0:
                
                adjust_spines(ax10, ['left','bottom'], d_out = d_out, d_down = d_down) 
                #adjust_spines(ax11, ['left','bottom'], d_out = d_out, d_down = d_down) 
                                
                if "grc" in do:
                    ax10.set_title(r'Granule cell model (Solinas et al. 2010)', fontsize=10)
                    tfit = r"best fit analytical function ($\tau$ = 15.7 ms)"
                    
                elif "if" in do:
                    #ax10.set_title(r'Integrate & Fire neuron ($\tau$ = 20 ms)', fontsize=10)
                    #tfit = r"analytical function"
                    pass
                    
                elif "grD" in do:
                    ax10.set_title(r'Granule cell model (Diwakar et al. 2009)', fontsize=10)
                    tfit = r"best fit analytical function ($\tau$ = 4.8 ms)"
                    
                elif "grVSCS" in do:
                    ax10.set_title(r'Granule cell model (Steuber)', fontsize=10)
                    tfit = r"best fit analytical function ($\tau$ =  ms)"
    
                #handles, labels = ax10.get_legend_handles_labels()
                # reverse the order
                #lg = ax10.legend([handles[1], handles[2], handles[3], handles[4], handles[5], handles[0]], ('sinusoid fit', 'linear interpolation', 'shannon interpolation', r'synaptic filter $\tau_{\alpha}$ = 100 ms', 'sampling-rate filter', tfit,))
                #lg = ax10.legend([handles[0], handles[1], handles[2], handles[3], handles[4]], ('sinusoid fit', 'linear interpolation', 'shannon interpolation', r'synaptic filter $\tau_{\alpha}$ = 100 ms', 'sampling-rate filter',))
                #lg.get_frame().set_linewidth(0.5)
                
                ax10.text(2, 10, "sampling-rate filter", color=b1, fontsize = params['text.fontsize'])
                ax10.text(45, -38, "sinusoidal\n fit", color=g1, fontsize = params['text.fontsize'])
                ax10.text(10, -14, "analytical", color="#000000", fontsize = params['text.fontsize'])
                #ax10.text(5, -11, "linear interpolation", color=p1, fontsize = params['text.fontsize'])
                ax10.text(2, -45, 'synaptic filter \n' + r'$\tau_{\alpha}$ = 100 ms', color=r1, fontsize = params['text.fontsize'])
                #ax10.text(9, -70, "shannon interpolation", color=o1, fontsize = params['text.fontsize'])
                
                ax10.set_ylabel("Gain (dB)", labelpad=0) 
                plt.xscale('log', subsx=[2, 3, 4, 5, 6, 7, 8, 9])
                ax10.set_xscale('log')
                ax10.xaxis.set_ticks(array([1,10,40,100]))
                ax10.set_xticklabels(('1', '', '40', '100'))
                ax10.axis(xmin=0.5, xmax=xmax)
                ax10.axis(ymin=-71, ymax=30)
                ax10.yaxis.set_ticks(array([-60, -40, -20, 0, 20]))
                ax10.set_xlabel("Hz", labelpad=2)
                
                ax10.text(-0.01, 1.0, 'C', transform=ax10.transAxes, fontsize=12, va='top', fontproperties=font)
                
                
                #ax11.set_ylabel("Phase ($^\circ$)") 
                #plt.xscale('log', subsx=[2, 3, 4, 5, 6, 7, 8, 9])
                #ax11.set_xscale('log')
                #ax11.xaxis.set_ticks(array([1,10,40,100]))
                #ax11.set_xticklabels(('1', '10', '40', '100'))
                #ax11.axis(xmin=1, xmax=xmax)
                #ax11.axis(ymin=-280, ymax=180)
                #ax11.yaxis.set_ticks(array([-200, -100, 0, 100]))
                #ax11.set_xlabel("Hz", labelpad=0)
                
                #ax11.text(-0.02, 1.1, 'D', transform=ax11.transAxes, fontsize=12, va='top', fontproperties=font)
    
                filename="./figs/Pub/" + str(pickle_prefix) 
                savefig(filename + ".png", dpi = 300) # save it
                savefig(filename + ".pdf", dpi = 300) # save it
                #os.system('rsvg-convert -f pdf -o ' + filename +'.pdf ' + filename + '.svg')           
  
        sim = None
        cell = None
            
   
    if "pop_transfer" in do:
        
        ih = 40
        amod = 0.1
        anoise = [0] 
        tau_noise = 0*ms
        N = [1]
        
        if "talk" in do:
            
            fig_size =  [11.7*0.3937,7.4*0.3937]
            params = {'backend': 'ps',
                      'axes.labelsize': 6,
                      'axes.linewidth' : 0.5,
                      'title.fontsize': 10, 
                      'text.fontsize': 6,
                      'font.size':6,
                      'axes.titlesize':6,
                      'legend.fontsize': 10,
                      'xtick.labelsize': 6,
                      'ytick.labelsize': 6,
                      'legend.borderpad': 0.2,
                      'legend.linewidth': 0.1,
                      'legend.loc': 'best',
                      'legend.ncol': 4,
                      'text.usetex': False,
                      'figure.figsize': fig_size}
            rcParams.update(params)
            
            linewidth = 1
            d_out = 8
            d_down = 3

            
        else:
            fig_size =  [85*0.03937, 140*0.03937] # 2-Column
            params['figure.figsize'] = fig_size
            rcParams.update(params)
            
        dt = 0.025*ms
       
        if "_if" in do:
            
            prefix = pickle_prefix + "pop_transfer" + "_if" 
            
            thresh = -21.175*mV 
            R = 8860*MOhm
            #tau_passive = 3e-06*8860 = 26.6ms
            
            thresh = -41.8    
            R = 5227*MOhm
            #tau_passive = 3e-06*5227 = 15.7ms
            
            cellimport = []
            celltype = "IfCell"
            cell_exe = "cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -71.5*mV)"

            color = r1
            linestyle = '-'
            if "_talk" in do: 
                color = 'k'  
                linestyle = '--'
            temperature = 0
            
            
        if "_resif" in do:
            
            prefix = pickle_prefix + "pop_transfer" + "_resif" 
            
                        
            gr = 5.56e-05*uS 
            tau_r = 19.6*ms
            R = 5227*MOhm
            delta_t = 4.85*ms
            
            thresh = (0.00568*nA * R) - 71.5*mV # 
            thresh = -41.8          

            cellimport = []
            celltype = ["IfCell"]
            cell_exe = ["cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -71.5*mV, dgk =" + str(gr) + ", egk = -71.5*mV, ctau =" + str(tau_r) + ")"]
          
            color = r1
            if "_talk" in do: color = 'k'  
            linestyle = ':'
            temperature = 0
            
            
        if "_grc" in do:
            
            prefix = pickle_prefix + "pop_transfer" + "_grc" 
            
            cellimport = ["from GRANULE_Cell import Grc"]
            celltype = ["Grc"]
            cell_exe = ["cell = Grc(np.array([0.,0.,0.]))"]   
         
            color = r1
            linestyle = '-'
            temperature = 37


        if "_cn" in do:
            
            xmax = 20            
            cutf = 20
            sexp = -1 
            prefix = prefix + "_cn"   
            

        if "_wn" in do:
            
            xmax = 100
            cutf = 0
            sexp = 0
            prefix = prefix + "_wn"
            dt = 0.005*ms
            
                        
        method_interpol = np.array(['bin']) 
        
        amp = 0 # absolute value
        fluct_s = [0] # absolute value
        ihold_sigma = [0*nA] # absolute value

        if "_0noise" in do:
            anoise = [1] 
            tau_noise = 0*ms
            prefix = prefix + "_0noise"
            xmax = 40
            dt = 0.025*ms
            amod = 1
            
        bin_width = dt
        jitter = 0*ms
    
        give_freq = True       
        ihold = [ih] 
        
        istart = 0 
        istop = 0.1
        di = 0.005
        
        pop = Population(cellimport = cellimport,  celltype = celltype, cell_exe = cell_exe, N = N, temperature = temperature, ihold = ihold, ihold_sigma = ihold_sigma, amp = amp, amod = amod, give_freq = give_freq, do_run = do_run, pickle_prefix = prefix, istart = istart, istop = istop, di = di, dt = dt)      
        
        pop.bin_width = bin_width
        pop.jitter = jitter
        pop.anoise = anoise
        pop.fluct_s = fluct_s 
        pop.fluct_tau = tau_noise
        pop.method_interpol = method_interpol
        pop.give_freq = give_freq 
        pop.plot_input = False
        pop.simstep = 1*s
        pop.plot_train = False
        pop.data_dir = data_dir
        pop.minimal_dir = minimal_dir
        
        results = pop.fun_cnoise_Stim(t_stim = t_stim, cutf = cutf, sexp = sexp, t_qual = 10)
        
        freq_used, amp_vec, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1') 
        freq_times, spike_freq, fmean, method_interpol, SNR, VAF, Qual = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('SNR'), results.get('VAF'), results.get('Qual')                              
        stim, stim_re_mat, tk, K_mat = results.get('stim'), results.get('stim_re_mat'), results.get('tk'), results.get('K_mat')
        ihold1 = results.get('ihold1')

        if pop.id == 0:
            
            if combine == "yes":
                fig = plt.figure(1)
            elif combine == "no":
                fig = plt.figure()
            
            if (combine != "done") or (combine == "no"):
                
                if "fig2" in do:
                    
                    gs = matplotlib.gridspec.GridSpec(5,2,
                               width_ratios=[1,1],
                               height_ratios=[1,1,1,1,1]
                               )
                    
                    ax1 = plt.subplot(gs[0,0])
                    ax1b = plt.subplot(gs[1,0])
                    ax2 = plt.subplot(gs[2,0])
                    ax3 = plt.subplot(gs[3,0])
                    ax4 = plt.subplot(gs[4,0])
                    
                    ax5 = plt.subplot(gs[0,1])
                    ax5b = plt.subplot(gs[1,1])
                    ax6 = plt.subplot(gs[2,1])
                    ax7 = plt.subplot(gs[3,1])
                    ax8 = plt.subplot(gs[4,1])
                    
                    if "poster" not in do:
                        #x1 = -0.05
                        #y1 = 1.35

                        x1 = -0.08
                        y1 = 1.35
                        ax1.text(x1, y1, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
                        ax1b.text(x1, y1, 'A2', transform=ax1b.transAxes, fontsize=12, va='top', fontproperties=font)
                        ax5.text(x1, y1, 'B1', transform=ax5.transAxes, fontsize=12, va='top', fontproperties=font)
                        ax5b.text(x1, y1, 'B2', transform=ax5b.transAxes, fontsize=12, va='top', fontproperties=font)
                        ax2.text(x1, y1, 'A3', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
                        ax3.text(x1, y1, 'A4', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)
                        ax4.text(x1, y1, 'A5', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)
                        ax6.text(x1, y1, 'B3', transform=ax6.transAxes, fontsize=12, va='top', fontproperties=font)
                        ax7.text(x1, y1, 'B4', transform=ax7.transAxes, fontsize=12, va='top', fontproperties=font)
                        ax8.text(x1, y1, 'B5', transform=ax8.transAxes, fontsize=12, va='top', fontproperties=font)
                
                    gs.update(left=0.10, right=0.96, bottom=0.05, top=0.92, wspace=0.25, hspace=1) 
                    
                
                if "talk" in do:
                    
                    gs = matplotlib.gridspec.GridSpec(3, 2,
                               width_ratios=[1,1],
                               height_ratios=[1,1,1]
                               )
                    
                    ax5 = plt.subplot(gs[0,0])
                    ax5b = plt.subplot(gs[0,1])
                    
                    #plt.savefig("./figs/Pub/Fig2_Methods_" + str(prefix) + "1.pdf", dpi = 300, transparent=True) # save it
                    ax7 = plt.subplot(gs[1,0:2])                    
                    ax6 = plt.subplot(gs[2,0])
                    ax8 = plt.subplot(gs[2,1])
                    
                    
                    gs.update(left=0.11, right=0.96, bottom=0.08, top=0.97, wspace=0.4, hspace=0.8)        
                    
                        

                if combine == "yes":    
                    combine = "done"


            mag[0,:]=mag[0,:]/mag[0,0]
            
            iend = mlab.find(freq_used > xmax)[0]
            
            # plot theoretical 
            tau = 15.7*ms
            H1, H0 = aiftransfer(freq_used[0:iend], tau = tau, f0 = ihold[0])            
            H = H1/H0
            H = H/H[2]
            #print H0
            #print H1
            #H = H/H[0]
            
            magA = abs(H)
            phaA = unwrap(angle(H)) * (180 / pi)
            
            
            if ("talk" in do):
                
                if "wn" in prefix:
                    ax01 = ax5
                    ax01b = ax5b

                elif "cn" in prefix:
                    ax01 = ax5
                    ax01b = ax5b
                    
            else:
                
                if "wn" in prefix:
                    ax01 = ax1
                    ax01b = ax1b
                    ax02 = ax2
                    ax03 = ax3
                    ax04 = ax4

                elif "cn" in prefix:
                    ax01 = ax5
                    ax01b = ax5b
                    ax02 = ax6
                    ax03 = ax7
                    ax04 = ax8 
                    
                    
            
            if ("_noif" in do) and ("_talk" in do):
                pass
            
            else:
                
                if "wn" in prefix:

                    ax01.semilogx(freq_used[0:iend], 20*log10(mag[0,0:iend]), color, linewidth=linewidth, alpha=1, linestyle=linestyle)
                    ax01.semilogx(freq_used[0:iend], 20*log10(magA), 'k--', linewidth=linewidth)
                    
                    adjust_spines(ax01, ['left','bottom'], d_out = d_out, d_down = d_down)
                    
                    ax01.text(0.2, 1.75, 'White noise', transform=ax01.transAxes, fontsize=10, va='top')
                    #ax01.set_title("Transfer WN")
                    
                    ax01.set_title("Gain (dB)")
                    
                    ax01.set_xscale('log')
                    ax01.xaxis.set_ticks(array([1,10, 40,100]))
                    ax01.set_xticklabels(('1', '','40', '100'))
                    ax01.axis(xmin=0.5, xmax=xmax)
                    ax01.axis(ymin=-10, ymax=50)
                    ax01.yaxis.set_ticks(array([0,20,40]))
                    #ax01.set_ylabel("Gain (dB)") 

                    #ax01.plot(freq_used[0:iend], 20*log10(mag[0,0:iend]), color, linewidth=linewidth, alpha=1, linestyle=linestyle)
                    #ax01.plot(freq_used[0:iend], 20*log10(magA), 'k--', linewidth=linewidth)
                    
    
                    
                    ax01b.semilogx(freq_used[0:iend], pha[0,0:iend], color, linewidth=linewidth, alpha=1, linestyle=linestyle)
                    ax01b.semilogx(freq_used[0:iend], phaA, 'k--', linewidth=linewidth)
                    
                    adjust_spines(ax01b, ['left','bottom'], d_out = d_out, d_down = d_down) 
                    
                    ax01b.set_title("Phase ($^\circ$)")
                    
                    ax01b.set_xscale('log')
                    ax01b.xaxis.set_ticks(array([1,10,40,100]))
                    ax01b.set_xticklabels(('1', '', '40', '100'))
                    ax01b.axis(xmin=0.5, xmax=xmax)
                    ax01b.axis(ymin=-100, ymax=100)
                    ax01b.yaxis.set_ticks(array([-100,0,100]))
                    #ax01b.set_ylabel("Phase ($^\circ$)") 
                    
                    
                elif "cn" in prefix:
                    
                    ax01.semilogx(freq_used[0:iend], 20*log10(mag[0,0:iend]), color, linewidth=linewidth, alpha=1, linestyle=linestyle) #
                    if "talk" not in do: ax01.semilogx(freq_used[0:iend], 20*log10(magA), 'k--', linewidth=linewidth)
                    
                    adjust_spines(ax01, ['left','bottom'], d_out = d_out, d_down = d_down) 
                    
                    if "talk" not in do:
                        ax01.text(-0.1, 1.75, 'Low-pass white noise', transform=ax01.transAxes, fontsize=10, va='top')
                        #ax01.set_title("Transfer TF-WN")
    
                    ax01.set_xscale('log')
    
                    if "_talk" in do: 
                        
                        ax01.axis(ymin=-0.4, ymax=2.5)
                        ax01.yaxis.set_ticks(array([0,2]))
                        ax01.xaxis.set_ticks(array([1,10,20]))
                        ax01.set_xticklabels(('1', '10', '20'))
                        
                        ax01.set_ylabel("Gain (dB)") 
                        if "_grc" in do: ax01.text(6, 1.5, "GrC", color=color, fontsize = 6)
                        if "_if" in do: ax01.text(15, 0.8, "IF", color="k", fontsize = 6)
                        if "_resif" in do: ax01.text(1.5, 0.7, "resonant IF", color="k", fontsize = 6) 
                        
                    else:  
                        
                        ax01.set_title("Gain (dB)")
                        
                        ax01.axis(ymin=0.8, ymax=1.1)
                        ax01.axis(ymin=-0.3, ymax=1.2)
                        
                        ax01.yaxis.set_ticks(array([0,1]))
                        #ax01.set_ylabel("Gain (dB)")
                    
                        ax01.xaxis.set_ticks(array([1,10,20]))
                        ax01.set_xticklabels(('1', '10', '20'))
                        
                    ax01.axis(xmin=0.5, xmax=xmax)
                    
                    ax01b.semilogx(freq_used[0:iend], pha[0,0:iend], color, linewidth=linewidth, alpha=1, linestyle=linestyle) #
                    if "talk" not in do: ax01b.semilogx(freq_used[0:iend], phaA, 'k--', linewidth=linewidth)
                    
                    adjust_spines(ax01b, ['left','bottom'], d_out = d_out, d_down = d_down) 
                    
                    #ax01b.set_ylabel("Phase ($^\circ$)")
                    
                    
                    ax01b.set_xscale('log')
                    ax01b.xaxis.set_ticks(array([1,10,20]))
                    ax01b.set_xticklabels(('1', '10', '20'))
                    ax01b.axis(xmin=0.5, xmax=xmax)
                    
                    if "_talk" in do: 
                        
                        ax01b.set_xscale('log')
                        ax01b.xaxis.set_ticks(array([1,10,20]))
                        ax01b.set_xticklabels(('1', '10', '20'))
                        ax01b.axis(xmin=0.5, xmax=xmax)
                        
                        ax01b.axis(ymin=-10, ymax=35)
                        ax01b.yaxis.set_ticks(array([-10,0,10,20,30]))
                        ax01b.set_ylabel("Phase ($^\circ$)") 
                        
                    else:
                        
                        ax01b.set_title("Phase ($^\circ$)")
                        
                        ax01b.axis(ymin=-2, ymax=30)
                        ax01b.yaxis.set_ticks(array([0,10,20,30]))
                        #ax01b.set_ylabel("Phase ($^\circ$)")
                                
    
                ax01.set_xlabel("Hz", labelpad=-5) 
                ax01b.set_xlabel("Hz", labelpad=-5) 
                
                
                if "wn" in prefix:
                    ax01.axvline(x=fmean, color='k', linestyle=':')
                    ax01b.axvline(x=fmean, color='k', linestyle=':')
                    
                    if "talk" in do:
                        ax02 = ax6
                        ax03 = ax7
                        ax04 = ax8
                    else:
                        ax02 = ax2
                        ax03 = ax3
                        ax04 = ax4

                elif "cn" in prefix:
                    ax02 = ax6
                    ax03 = ax7
                    ax04 = ax8
                    
                
                tk2 = tk-tk[-1]/2
                
                K = sin(tk2*2*pi*20)/(tk2*2*pi*20)
                
                ax02.plot(tk2*1e3, K_mat[0,:]/max(K_mat[0,:]), color, linewidth=linewidth, alpha=1, linestyle=linestyle)
                
                ax02.axis(xmin=-100, xmax=100)
                
                
                if "wn" in prefix:
                    adjust_spines(ax02, ['left','bottom'], d_out = d_out, d_down = d_down) 
                    ax02.yaxis.set_ticks(array([0,0.5,1]))
                    
                elif "cn" in prefix:
                    
                    if "talk" in do:
                        adjust_spines(ax02, ['left','bottom'], d_out = d_out, d_down = d_down) 
                        ax02.yaxis.set_ticks(array([0,1]))
                        ax02.xaxis.set_ticks(array([-100,-50,50,100]))
    
                    else:
                        ax02.plot(tk2*1e3, K/max(K), 'k--', linewidth=linewidth)
                        adjust_spines(ax02, ['bottom'], d_out = d_out, d_down = d_down) 
                
                if "talk" in do: 
                    ax02.set_title("Wiener-Kolmogorov filter")
                else:
                    ax02.set_title("Optimal filter K(t)")  

                ax02.set_xlabel("ms", labelpad=3)
                if ("_grc" in do) or ("_talk" not in do):
                    
                    ax02.axis(ymin=-0.25, ymax=1.1)
                                   
                    ax03.plot(np.arange(len(stim))*dt-1, (ihold1+stim)*1e3, 'k-', linewidth=linewidth)
                    ax03.plot(np.arange(len(stim))*dt-1, (ihold1+stim_re_mat[0,:])*1e3, color, linewidth=linewidth, alpha=1)
                    
                    # CHANGE IN ESTIMATION OF stim AND stim_re_mat, MODIFY FOR RERUN!!
                    # stim and stim_re_mat are normalized, multiply by amplitude used in nA!!
                    #ax03.plot(np.arange(len(stim))*dt-1, (ihold1+0.00029416*stim)*1e3, 'k-', linewidth=linewidth)
                    #ax03.plot(np.arange(len(stim))*dt-1, (ihold1+0.00029416*stim_re_mat[0,:])*1e3, color, linewidth=linewidth, alpha=1)
                    
                    #figure(99)
                    #plot(stim, 'b')
                    #plot(stim_re_mat[0,:], 'g')
                    #show()
    
                    
                    if "wn" in prefix:
                        adjust_spines(ax03, ['left','bottom'], d_out = d_out, d_down = d_down) 
                        #ax03.set_ylabel("pA", labelpad=3)
                        
                        #ax03.yaxis.set_ticks(array([110,112,114]))
                        #ax03.set_yticklabels(('110', '112', '114'))
                        
                        ax03.yaxis.set_ticks(array([6.6,6.9,7.2,7.5,7.8]))
                        ax03.set_yticklabels(('6.6', '', '7.2', '', '7.8'))
                        
                    if "cn" in prefix:
                        
                        if "talk" in do:
                            adjust_spines(ax03, ['left','bottom'], d_out = d_out, d_down = d_down) 
                            ax03.set_ylabel("pA", labelpad=2)
                            ax03.axis(ymin=9, ymax=10.3) 
                            ax03.yaxis.set_ticks(array([9, 10]))
                    
                        else:
                            
                            adjust_spines(ax03, ['bottom'], d_out = d_out, d_down = d_down) 
                            
                    ax03.axis(ymin=6.4, ymax=8) 
                        
                    #ax03.axis(ymin=110, ymax=114) 
                                    
                        
                    ax03.axis(xmin=0, xmax=0.3) 
                    ax03.set_title("Reconstr. (pA)") 
                    ax03.set_xlabel("s", labelpad=0)
                    ax03.xaxis.set_ticks(array([0,0.1,0.2,0.3]))
                    
            
            
            
            if "SNR" in do:            
                ax04.semilogx(freq_used, SNR[1][0,:], linewidth=linewidth, color=color, alpha=1)  
                ax04.axhline(y=10, color='k', linestyle=':')
                ax04.set_title("Reconstruction quality (SNR)") 
            else:
                ax04.semilogx(freq_used[:iend], VAF[1][0,:iend]*100, linewidth=linewidth, color=color, alpha=1, linestyle=linestyle)    
                ax04.set_title("VAF (%)") 
             
            if "wn" in prefix:

                adjust_spines(ax04, ['left','bottom'], d_out = d_out, d_down = d_down)
                ax04.set_xscale('log')
                ax04.xaxis.set_ticks(array([1,10, 40,100]))
                ax04.set_xticklabels(('1', '','40', '100'))
                ax04.axis(xmin=0.5, xmax=xmax)
                ax04.axvline(x=fmean, color='k', linestyle=':')
                
                if "SNR" in do:
                    ax04.set_ylabel("SNR")
                    ax04.axis(ymin=-1, ymax=11) 
                else:
                    ax04.yaxis.set_ticks(array([0,50,100]))
                    ax04.set_yticklabels(('0', '50', '100'))
                    ax04.axis(ymin=-1, ymax=105) 
                    ax04.set_ylabel("VAF (%)", labelpad=3 )


            if "cn" in prefix:
                
                if "talk" in do:
                    adjust_spines(ax04, ['left','bottom'], d_out = d_out, d_down = d_down)
                    ax04.yaxis.set_ticks(array([0,100]))
                    ax04.set_yticklabels(('0', '100'))
                    ax04.set_ylabel("VAF (%)", labelpad=2 )
                    
                else:
                    adjust_spines(ax04, ['bottom'], d_out = d_out, d_down = d_down)
                
                ax04.set_xscale('log')
                ax04.xaxis.set_ticks(array([1,10,20]))
                ax04.set_xticklabels(('1', '10', '20'))
                ax04.axis(xmin=0.5, xmax=xmax)
                
                if "SNR" in do:
                    ax04.axis(ymin=-1, ymax=41)    
                else:
                    ax04.axis(ymin=-1, ymax=105) 
                    
                
            ax04.set_xlabel("Hz", labelpad=-4) 
 
            if "poster" in do: prefix = prefix + "_poster"
            
            filename="./figs/Pub/" + str(prefix) 
            savefig(filename + ".png", dpi = 300) # save it
            savefig(filename + ".pdf", dpi = 300) # save it
            #os.system('rsvg-convert -f pdf -o ' + filename +'.pdf ' + filename + '.svg')
            
        pop = None
        

plt.show()    