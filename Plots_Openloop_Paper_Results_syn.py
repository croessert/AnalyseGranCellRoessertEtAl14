# -*- coding: utf-8 -*-
"""
@author: chris

mpiexec -f ~/machinefile -enable-x -n 1 python Plots_Openloop_Paper_Results_syn.py -o fig5 --noplot 2>&1 | tee log/log5.txt
qsub -v J=Plots_Openloop_Paper_Results_syn.py,O=fig5 -pe ompigige 1 PBSinsigneo.sh

mpiexec -f ~/machinefile -enable-x -n 1 python Plots_Openloop_Paper_Results_syn.py -o fig6 --noplot 2>&1 | tee log/log6.txt
qsub -v J=Plots_Openloop_Paper_Results_syn.py,O=fig6 -pe ompigige 1 PBSinsigneo.sh

mpiexec -f ~/machinefile -enable-x -n 96 python Plots_Openloop_Paper_Results_syn.py -o fig7 --noplot 2>&1 | tee log/log7.txt
qsub -v J=Plots_Openloop_Paper_Results_syn.py,O=fig7 -pe ompigige 100 PBSinsigneo.sh

mpiexec -f ~/machinefile -enable-x -n 50 python Plots_Openloop_Paper_Results_syn.py -o fig8b --noplot 2>&1 | tee log/log8b.txt
qsub -v J=Plots_Openloop_Paper_Results_syn.py,O=fig8b -pe ompigige 50 PBSinsigneo.sh

mpiexec -f ~/machinefile -enable-x -n 50 python Plots_Openloop_Paper_Results_syn.py -o fig8a --noplot 2>&1 | tee log/log8a.txt
qsub -v J=Plots_Openloop_Paper_Results_syn.py,O=fig8a -pe ompigige 50 PBSinsigneo.sh

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
pars = parser.parse_args()


import matplotlib
if MPI.COMM_WORLD.rank == 0:
    matplotlib.use('Tkagg', warn=True)
else:
    matplotlib.use('Agg', warn=True)


do_plot = 1
if pars.noplot:  # do not plot to windows
    matplotlib.use('Agg', warn=True)
    if MPI.COMM_WORLD.rank == 0: print "- No plotting"
    do_plot = 0

do_run = 1
if pars.norun:  # do not run again use pickled files!
    print "- Not running, using saved files"
    do_run = 0

use_mpi = True

import numpy as np
import matplotlib.pyplot as plt
#matplotlib.rc('font', **{'sans-serif' : 'Arial'}) #, 'family' : 'sans-serif'})
#matplotlib.rc('font', family='sans-serif')
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)

from matplotlib.font_manager import FontProperties
font0 = FontProperties()


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
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'text.usetex': False,
          'figure.figsize': fig_size}
rcParams.update(params)

o1 = (1,0.5,0)
o2 = (1,0.75,0.5)
ye1 = (1,1,0.2)
ye2 = (1,1,0.6)
g1 = (0.2,1,0)
g2 = (0.7,1,0.55)
b1 = (0.1,0.7,1)
b2 = (0.65,0.93,1)
p1 = (0.4,0.3,1)
p2 = (0.8,0.75,1)
r1 = (0.9,0.1,0.2)
r2 = (1,0.6,0.75)


b1 = '#1F78B4' #377EB8
b2 = '#A6CEE3'
g1 = '#33A02C' #4DAF4A
g2 = '#B2DF8A'
r1 = '#E31A1C' #E41A1C
r2 = '#FB9A99'
o1 = '#FF7F00' #FF7F00
o2 = '#FDBF6F'
p1 = '#6A3D9A' #984EA3
p2 = '#CAB2D6'

#b1 = '#1F78B4'
#b2 = '#45B4FF'
#g1 = '#33A02C'
#g2 = '#51FF45'
#r1 = '#E31A1C'
#r2 = '#FF4547'
#o1 = '#FF7f00'
#o2 = '#FFA245'
#p1 = '#6A3D9A'
#p2 = '#9F45FF'
#gray = 'gray'

ye1 = '#FFFF33'
br1 = '#A65628'
pi1 = '#F781BF'
gr1 = '#999999'



color0 = 'black' # BLACK
color1 = 'blue' # BLUE
color2 = 'red' # RED
color3 = 'gray' # GRAY
color4 = 'purple' # PURPLE
color5 = '#FF6600' # 'orange' # ORANGE
color6 = 'green' # GREEN
color7 = 'yellow' # YELLOW
color8 = 'brown' # brown


linewidth = 1

xmax = 130

dt = 0.025*ms

simstep = 1*s
delay_baseline = 8

plot_train = True

t_stim = 1000*s # only for cnoise
#t_stim = 180*s # only for cnoise
#t_stim = 20*s # only for cnoise

do_if = False

data_dir = "./publish/openloop/fulldata"
minimal_dir = "./publish/openloop/minimal"
export = False

data_dir = "./publish/openloop/minimal"
minimal_dir = False

# FIGURE 6
if pars.opt == "fig5":
    t_stim = 1000*s
    do_if = False
    export = False
    do_vec = np.array([
                       #"pop_transfer_none_synno_cn_a01_fig5_noisesynlow_color1_pos1_label_keep_", "pop_transfer_none_synno_cn_a05_fig5_noisesynlow_color2_pos1_label_keep_",
                       "pop_transfer_none_synno_cn_a1_fig5_noisesynlow_colorb1_pos1_label15_first_",
                       #"pop_transfer_none_synno_cn_a01_nsyn4_varih_N10_fig5_noisesynmed_color1_dashed_pos1_keep_", "pop_transfer_none_synno_cn_a05_nsyn4_varih_N10_fig5_noisesynmed_color2_dashed_pos1_keep_",
                       #"pop_transfer_none_synno_cn_a1_nsyn4_varih_N10_fig5_noisesynmed_color6_dashed_pos1_keep_",
                       #"pop_transfer_none_synno_cn_a01_nsyn4_varih_N10_fig5_noisesynpoiss_color1_dotted_pos1_keep_", "pop_transfer_none_synno_cn_a05_nsyn4_varih_N10_fig5_noisesynpoiss_color2_dotted_pos1_keep_",
                       "pop_transfer_none_synno_cn_a1_nsyn4_varih_N10_fig5_noisesynlow_colorg1_pos1_label16_", #"pop_transfer_none_synno_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_colorg1_dotted_pos1_label17_",
                       #"pop_transfer_none_synno_cn_a10_nsyn4_varih_N5_fig5_noisesynlow_twopop_colorr1_pos1_label16b_keep_",

                       #"pop_transfer_none_synampagr_cn_a01_fig5_noisesynlow_color1_pos2_label_", "pop_transfer_none_synampagr_cn_a05_fig5_noisesynlow_color2_pos2_label_",
                       "pop_transfer_none_synampagr_cn_a1_fig5_noisesynlow_colorb1_pos2_label15_first_",
                       #"pop_transfer_none_synampagr_cn_a01_nsyn4_varih_N10_fig5_noisesynmed_color1_dashed_pos2_", "pop_transfer_none_synampagr_cn_a05_nsyn4_varih_N10_fig5_noisesynmed_color2_dashed_pos2_",
                       #"pop_transfer_none_synampagr_cn_a1_nsyn4_varih_N10_fig5_noisesynmed_color6_dashed_pos2_",
                       #"pop_transfer_none_synampagr_cn_a01_nsyn4_varih_N10_fig5_noisesynpoiss_color1_dotted_pos2_", "pop_transfer_none_synampagr_cn_a05_nsyn4_varih_N10_fig5_noisesynpoiss_color2_dotted_pos2_",
                       "pop_transfer_none_synampagr_cn_a1_nsyn4_varih_N10_fig5_noisesynlow_colorg1_pos2_label16_", #"pop_transfer_none_synampagr_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_colorg1_dotted_pos2_label17_",
                       #"pop_transfer_none_synampagr_cn_a10_nsyn4_varih_N5_fig5_noisesynlow_twopop_colorr1_pos2_label16b_keep_",

                       #"pop_transfer_none_synnmdagr_cn_a01_fig5_noisesynlow_color1_pos3_label_", "pop_transfer_none_synnmdagr_cn_a05_fig5_noisesynlow_color2_pos3_label_",
                       "pop_transfer_none_synnmdagr_cn_a1_fig5_noisesynlow_colorb1_pos3_label15_first_",
                       #"pop_transfer_none_synnmdagr_cn_a01_nsyn4_varih_N10_fig5_noisesynmed_color1_dashed_pos3_", "pop_transfer_none_synnmdagr_cn_a05_nsyn4_varih_N10_fig5_noisesynmed_color2_dashed_pos3_",
                       #"pop_transfer_none_synnmdagr_cn_a1_nsyn4_varih_N10_fig5_noisesynmed_color6_dashed_pos3_",
                       #"pop_transfer_none_synnmdagr_cn_a01_nsyn4_varih_N10_fig5_noisesynpoiss_color1_dotted_pos3_", "pop_transfer_none_synnmdagr_cn_a05_nsyn4_varih_N10_fig5_noisesynpoiss_color2_dotted_pos3_",
                       "pop_transfer_none_synnmdagr_cn_a1_nsyn4_varih_N10_fig5_noisesynlow_colorg1_pos3_label16_", #"pop_transfer_none_synnmdagr_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_colorg1_dotted_pos3_label17_",
                       #"pop_transfer_none_synnmdagr_cn_a10_nsyn4_varih_N5_fig5_noisesynlow_twopop_colorr1_pos3_label16b_keep_",

                       #"pop_transfer_none_syngabagr_cn_a01_fig5_end_noisesynlow_color1_pos4_label_end_", "pop_transfer_none_syngabagr_cn_a05_fig5_end_noisesynlow_color2_pos4_label_end_",
                       "pop_transfer_none_syngabagr_cn_a1_fig5_end_noisesynlow_colorb1_pos4_label15_end_first_",
                       #"pop_transfer_none_syngabagr_cn_a01_nsyn4_varih_N10_fig5_noisesynmed_color1_dashed_pos4_end_", "pop_transfer_none_syngabagr_cn_a05_nsyn4_varih_N10_fig5_noisesynmed_color2_dashed_pos4_end_",
                       #"pop_transfer_none_syngabagr_cn_a1_nsyn4_varih_N10_fig5_noisesynmed_color6_dashed_pos4_end_",
                       #"pop_transfer_none_syngabagr_cn_a01_nsyn4_varih_N10_fig5_noisesynpoiss_color1_dotted_pos4_end_", "pop_transfer_none_syngabagr_cn_a05_nsyn4_varih_N10_fig5_noisesynpoiss_color2_dotted_pos4_end_",
                       "pop_transfer_none_syngabagr_cn_a1_nsyn4_varih_N10_fig5_end_noisesynlow_colorg1_pos4_label16_", #"pop_transfer_none_syngabagr_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_colorg1_dotted_pos4_label17_end_",
                       #"pop_transfer_none_syngabagr_cn_a10_nsyn4_varih_N5_fig5_end_noisesynlow_twopop_colorr1_pos4_label16b_end_keep_",
                       ])

if pars.opt == "fig5a":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array([
                       #"pop_transfer_none_synno_cn_a1_fig5_noisesynlow_colorb1_pos1_label15_first_keep_",
                       #"pop_transfer_none_synno_cn_a1_nsyn4_varih_N10_fig5_noisesynlow_colorg1_pos1_label16_keep_",

                       #"pop_transfer_none_synampagr_cn_a1_fig5_noisesynlow_colorb1_pos2_label15_first_keep_",
                       #"pop_transfer_none_synampagr_cn_a1_nsyn4_varih_N10_fig5_noisesynlow_colorg1_pos2_label16_keep_",

                       "pop_transfer_none_synampagrstoch_cn_a1_nsyn4_varih_N10_fig5_noisesynlow_colorg2_pos2_label16_keep_",

                       #"pop_transfer_none_synnmdagr_cn_a1_fig5_noisesynlow_colorb1_pos3_label15_first_keep_",
                       #"pop_transfer_none_synnmdagr_cn_a1_nsyn4_varih_N10_fig5_noisesynlow_colorg1_pos3_label16_keep_",

                       "pop_transfer_none_synnmdagrstoch_cn_a1_nsyn4_varih_N10_fig5_noisesynlow_colorg2_pos3_label16_keep_",

                       #"pop_transfer_none_syngabagr_cn_a1_fig5_end_noisesynlow_colorb1_pos4_label15_end_first_keep_",
                       #"pop_transfer_none_syngabagr_cn_a1_nsyn4_varih_N10_fig5_end_noisesynlow_colorg1_pos4_label16_end_keep_",

                       "pop_transfer_none_syngabagrstoch_cn_a1_nsyn4_varih_N10_fig5_end_noisesynlow_colorg2_pos4_label16_end_",

                       ])


if pars.opt == "fig5test":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array([
                       "pop_transfer_none_synno_cn_a1_fig5_noisesynlow_color1_pos1_label15_first_keep_",
                       "pop_transfer_none_synno_cn_a1_nsyn4_varih_N10_fig5_noisesynlow_color6_pos1_label16_keep_", "pop_transfer_none_synno_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_color6_dotted_pos1_label17_keep_",

                       "pop_transfer_none_synampagr_cn_a1_fig5_noisesynlow_color1_pos2_label15_first_keep_",
                       "pop_transfer_none_synampagr_cn_a1_nsyn4_varih_N10_fig5_noisesynlow_color6_pos2_label16_keep_", "pop_transfer_none_synampagr_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_color6_dotted_pos2_label17_keep_",
                       "pop_transfer_none_synampagrstoch_cn_a1_nsyn4_varih_N10_fig5_noisesynlow_color2_pos2_label16_", "pop_transfer_none_synampagrstoch_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_color2_dotted_pos2_label17_",

                       "pop_transfer_none_synnmdagr_cn_a1_fig5_noisesynlow_color1_pos3_label15_first_keep_",
                       "pop_transfer_none_synnmdagr_cn_a1_nsyn4_varih_N10_fig5_noisesynlow_color6_pos3_label16_keep_", "pop_transfer_none_synnmdagr_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_color6_dotted_pos3_label17_keep_",
                       "pop_transfer_none_synnmdagrstoch_cn_a1_nsyn4_varih_N10_fig5_noisesynlow_color2_pos3_label16_", "pop_transfer_none_synnmdagrstoch_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_color2_dotted_pos3_label17_",

                       "pop_transfer_none_syngabagr_cn_a1_fig5_end_noisesynlow_color1_pos4_label15_end_first_keep_",
                       "pop_transfer_none_syngabagr_cn_a1_nsyn4_varih_N10_fig5_end_noisesynlow_color6_pos4_label16_end_keep_", "pop_transfer_none_syngabagr_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_color6_dotted_pos4_label17_end_keep_",

                       ])


if pars.opt == "fig5atalk":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array([
                       "pop_transfer_none_synno_cn_a05_fig5_noisesynlow_color1_pos1_label_talk_keep_end_", "pop_transfer_none_synno_cn_a1_fig5_noisesynlow_color2_pos1_label_talk_keep_end_",
                       "pop_transfer_none_synno_cn_a05_nsyn4_varih_N10_fig5_noisesynpoiss_color1_dotted_pos1_talk_keep_end_", "pop_transfer_none_synno_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_color2_dotted_pos1_talk_keep_end_",

                       "pop_transfer_none_synampagr_cn_a05_fig5_noisesynlow_color1_pos2_label_talk_keep_end_", "pop_transfer_none_synampagr_cn_a1_fig5_noisesynlow_color2_pos2_label_talk_keep_end_",
                       "pop_transfer_none_synampagr_cn_a05_nsyn4_varih_N10_fig5_noisesynpoiss_color1_dotted_pos2_talk_keep_end_", "pop_transfer_none_synampagr_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_color2_dotted_pos2_talk_keep_end_",
                       ])

if pars.opt == "fig5btalk":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array([
                       "pop_transfer_none_synnmdagr_cn_a05_fig5_noisesynlow_color1_pos1_label_talk_keep_end_", "pop_transfer_none_synnmdagr_cn_a1_fig5_noisesynlow_color2_pos1_label_talk_keep_end_",
                       "pop_transfer_none_synnmdagr_cn_a05_nsyn4_varih_N10_fig5_noisesynpoiss_color1_dotted_pos1_talk_keep_end_", "pop_transfer_none_synnmdagr_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_color2_dotted_pos1_talk_keep_end_",

                       "pop_transfer_none_syngabagr_cn_a05_fig5_end_noisesynlow_color1_pos2_label_talk_keep_end_", "pop_transfer_none_syngabagr_cn_a1_fig5_end_noisesynlow_color2_pos2_label_talk_keep_end_",
                       "pop_transfer_none_syngabagr_cn_a05_nsyn4_varih_N10_fig5_noisesynpoiss_color1_dotted_pos2_talk_keep_end_", "pop_transfer_none_syngabagr_cn_a1_nsyn4_varih_N10_fig5_noisesynpoiss_color2_dotted_pos2_talk_keep_end_",
                      ])

# FIGURE 7
if pars.opt == "fig6":
    t_stim = 1000*s
    do_if = False
    export = False
    do_vec = np.array([
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo0_fig6_end_noisesynlow_colorb1_pos1_normalize_label11_first_", #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo0_fig6_end_noisesynpoiss_color1_dotted_pos1_",
                       "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N10_CFo0.50_fig6_end_noisesynlow_colorg1_pos1_label12_", #"pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N10_CFo0.50_fig6_end_noisesynpoiss_dotted_color6_pos1_png_",

                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo0_fig6_end_noisesynlow_colorb2_pos1_normalize_label11_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo6_fig6_end_noisesynlow_inhlow_coloro1_label13_pos1_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo6_fig6_end_noisesynlow_inhpoiss_coloro1_dotted_pos1_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo6_fig6_end_noisesynlow_inhlow_colorr1_pos1_label14_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo6_fig6_end_noisesynlow_inhpoiss_dotted_colorr1_pos1_",

                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo6_fig6_end_noisesynlow_inhlow_colorr2_pos1_label14_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N100_CFo0_fig6_end_noisesynlow_colorb1_pos1b_normalize_label11_first_", #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N100_CFo0_fig6_end_noisesynpoiss_color1_dotted_pos1_",
                       "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N100_CFo0.50_fig6_end_noisesynlow_colorg1_pos1b_label12_", #"pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N100_CFo0.50_fig6_end_noisesynpoiss_dotted_color6_pos1_png_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N100_CFo6_fig6_end_plotv20_noisesynlow_inhlow_coloro1_label13_pos1b_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N100_CFo6_fig6_end_noisesynlow_inhpoiss_coloro1_dotted_pos1b_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_CFo6_fig6_end_noisesynlow_inhlow_colorr1_pos1b_label14_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_CFo6_fig6_end_noisesynlow_inhpoiss_dotted_colorr1_pos1b_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_gc10_CFo6_fig6_end_noisesynlow_inhlow_colorp1_pos1b_",

                       #"pop_CF-cutf_none_synno_cn_a1_varih_N10_noisesynlow_fig6_color0_end_label1_pos14_first_label18_",
                       #"pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_noisesynlow_fig6_color2_end_label3_pos14_label19_",

                       #"pop_CF-cutf_none_synno_cn_a1_varih_N100_noisesynlow_inhlow_fig6_color3_end_label1_pos14_first_label20_keep_",
                       #"pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_noisesynlow_inhlow_fig6_color2_end_label3_pos14_label21_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb_N10_CFo4_fig6_noisesynlow_color2_end_pos2_normalize_label9_first_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb_N10_CFo4_fig6_noisesynpoiss_dotted_color2_end_pos2_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N10_CFo4_fig6_noisesynlow_color5_end_pos2_label5_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N10_CFo4_fig6_noisesynpoiss_dotted_color5_end_pos2_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N10_CFo4_fig6_noisesynlow_color1_end_normalize_pos2_label7_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N10_CFo4_fig6_noisesynpoiss_dotted_color1_end_pos2_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo0_fig6_end_noisesynlow_color0_pos1_normalize_label1_first_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo0_fig6_end_noisesynmed_color0_dashed_pos1_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo0_fig6_end_noisesynpoiss_color0_dotted_pos1_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo4_fig6_plotv20_end_noisesynlow_color2_label1_pos1_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo4_fig6_end_noisesynmed_color2_dashed_pos1_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo4_fig6_end_noisesynpoiss_color2_dotted_pos1_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a01_nsyn4_varih_N10_CFo0_fig6_end_noisesynlow_color3_pos1_normalize_ddown25_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_CFo0_fig6_end_noisesynlow_color0_pos1", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_CFo0_fig6_end_noisesynpoiss_color0_dotted_pos1",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_CFo4_fig6_plotv20_end_noisesynlow_color2_pos1", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_CFo4_fig6_end_noisesynpoiss_color2_dotted_pos1",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_lowinh_CFo40_fig6_end_noisesynlow_color1_pos1", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_lowinh_CFo40_fig6_end_noisesynpoiss_color1_dotted_pos1",
                       #"pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_CFo0.6_fig6_end_noisesynlow_color6_pos1", "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_CFo0.6_fig6_end_noisesynpoiss_color6_dotted_pos1",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_lowinh_N10_CFo40_fig6_end_noisesynlow_color1_pos3", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_lowinh_N10_CFo40_fig6_end_noisesynpoiss_color1_dotted_pos3"
                     ])

if pars.opt == "fig6_background":
    t_stim = 10*s
    data_dir = "./data"
    minimal_dir = False
    export = False
    do_if = False
    do_vec = np.array([
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_CFo6_fig6_end_noisesynlow_inhlow_colorr1_pos1b_label14_",
                       ])

if pars.opt == "fig6a":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array([
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_N10_CFo0_fig6_end_noisesynlow_colorb1_pos1_normalize_label11_first_keep_",
                       "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_lowvarih_N10_CFo0.50_fig6_end_noisesynlow_colorg1_pos1_label12_keep_",

                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_lowvarih_N10_CFo0_fig6_end_noisesynlow_colorb2_pos1_normalize_label11_keep_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_N10_CFo6_fig6_end_noisesynlow_inhlow_coloro1_label13_pos1_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo6_fig6_end_noisesynlow_inhpoiss_coloro1_dotted_pos1_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_N10_CFo6_fig6_end_noisesynlow_inhlow_colorr1_pos1_label14_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo6_fig6_end_noisesynlow_inhpoiss_dotted_colorr1_pos1_keep_",

                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_N10_CFo6_fig6_end_noisesynlow_inhlow_colorr2_pos1_label14_keep_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_N100_CFo0_fig6_end_noisesynlow_colorb1_pos1b_normalize_label11_first_keep_",
                       "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_lowvarih_N100_CFo0.50_fig6_end_noisesynlow_colorg1_pos1b_label12_keep_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_N100_CFo6_fig6_end_plotv20_noisesynlow_inhlow_coloro1_label13_pos1b_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N100_CFo6_fig6_end_noisesynlow_inhpoiss_coloro1_dotted_pos1b_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_N100_CFo6_fig6_end_noisesynlow_inhlow_colorr1_pos1b_label14_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_CFo6_fig6_end_noisesynlow_inhpoiss_dotted_colorr1_pos1b_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_N100_gc10_CFo6_fig6_end_noisesynlow_inhlow_colorp1_pos1b_",

                      ])


if pars.opt == "fig6_poster":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array([
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo0_fig6_end_noisesynlow_color1_pos1_normalize_label11_first_keep_",
                       "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N10_CFo0.50_fig6_end_noisesynlow_color6_pos1_label12_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo6_fig6_end_noisesynlow_inhlow_color5_label13_pos1_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo6_fig6_end_noisesynlow_inhpoiss_color5_dotted_pos1_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo6_fig6_end_noisesynlow_inhlow_color2_pos1_label14_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo6_fig6_end_noisesynlow_inhpoiss_dotted_color2_pos1_keep_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N100_CFo0_fig6_end_noisesynlow_color1_pos1b_normalize_label11_first_keep_",
                       "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N100_CFo0.50_fig6_end_noisesynlow_color6_pos1b_label12_keep_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N100_CFo6_fig6_end_plotv20_noisesynlow_inhlow_color5_label13_pos1b_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N100_CFo6_fig6_end_noisesynlow_inhpoiss_color5_dotted_pos1b_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_CFo6_fig6_end_noisesynlow_inhlow_color2_pos1b_label14_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_CFo6_fig6_end_noisesynlow_inhpoiss_dotted_color2_pos1b_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_gc10_CFo6_fig6_end_noisesynlow_inhlow_color4_pos1b_keep_",
                    ])


if pars.opt == "fig6b_poster":
    t_stim = 500*s
    #t_stim = 100*s
    do_if = False
    do_vec = np.array([
                       "pop_CF-cutf_none_synno_cn_a1_varih_N100_noisesynlow_inhlow_fig6_color3_end_label1_pos14_first_label20_keep_",
                       "pop_CF-cutf_none_synno_cn_a1_varih_N50_noisesynlow_inhlow_twopop_fig6_color1_end_label1_pos14_first_label20_keep_",
                       "pop_CF-cutf_none_synno_cn_a10_varih_N50_noisesynlow_inhlow_twopop_fig6_color6_end_label1_pos14_first_label20_keep_",

                       "pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_noisesynlow_inhlow_fig6_color2_end_label3_pos14_label21_keep_",
                       "pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_twopop_N50_noisesynlow_inhlow_fig6_color2_end_label3_pos14_label21_keep_",
                       #"pop_CF-cutf_grc_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_twopop_N50_noisesynlow_inhlow_fig6_color2_end_label3_pos14_label21_",

                       ])

if pars.opt == "fig6test":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array([
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_mf10_CFo6_fig6_end_noisesynlow_inhlow_color0_pos1b_label14_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_gc10_CFo6_fig6_end_noisesynlow_inhlow_color1_pos1b_label14_keep_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_mf10_gc10_CFo6_fig6_end_noisesynlow_inhlow_color2_pos1b_label14_keep_",
                       ])


if pars.opt == "fig6talk":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array([
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo0_fig6_end_noisesynlow_color1_pos1_normalize_label11_first_talk_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo0_fig6_end_noisesynpoiss_color1_dotted_pos1_talk_keep_",
                       "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N10_CFo0.64_fig6_end_noisesynlow_color6_pos1_label12_talk_keep_", "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N10_CFo0.64_fig6_end_noisesynpoiss_dotted_color6_pos1_talk_keep_png_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo4_fig6_end_plotv20_end_noisesynlow_color9_label13_pos1_talk_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_N10_CFo4_fig6_end_noisesynpoiss_color9_dotted_pos1_talk_keep_png_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo4_fig6_end_plotv20_end_noisesynlow_color2_pos1_label14_talk_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo4_fig6_end_noisesynpoiss_dotted_color2_pos1_talk_keep_png_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_CFo0_fig6_end_noisesynlow_color0_pos1", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_CFo0_fig6_end_noisesynpoiss_color0_dotted_pos1",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_CFo4_fig6_plotv20_end_noisesynlow_color2_pos1", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_CFo4_fig6_end_noisesynpoiss_color2_dotted_pos1",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_lowinh_CFo40_fig6_end_noisesynlow_color1_pos1", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_lowinh_CFo40_fig6_end_noisesynpoiss_color1_dotted_pos1",
                       #"pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_CFo0.6_fig6_end_noisesynlow_color6_pos1", "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_CFo0.6_fig6_end_noisesynpoiss_color6_dotted_pos1",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_lowinh_N10_CFo40_fig6_end_noisesynlow_color1_pos3", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_lowinh_N10_CFo40_fig6_end_noisesynpoiss_color1_dotted_pos3"
                     ])

if pars.opt == "fig6btalk":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array([
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb_N100_CFo6_fig6b_noisesynlow_inhlow_color2_end_pos1_normalize_label9_talk_keep_first_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb_N10_CFo4_fig6b_noisesynpoiss_dotted_color2_end_pos1_talk_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N100_CFo6_fig6b_noisesynlow_inhlow_color9_end_pos1_label5_talk_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N10_CFo4_fig6b_noisesynpoiss_dotted_color9_end_pos1_talk_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N100_CFo6_fig6b_noisesynlow_inhlow_color1_end_normalize_pos1_label7_talk_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N10_CFo4_fig6b_noisesynpoiss_dotted_color1_end_pos1_talk_keep_",
                       ])


# FiGURE 8
if pars.opt == "fig7":
    t_stim = 1000*s
    do_if = False
    export = False
    do_vec = np.array([
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N100_CFo6_fig7_noisesynlow_inhlow_coloro1_bottom_notransf_pos2_first_normalize_", # _label5 "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N10_CFo4_fig6_noisesynpoiss_dotted_color5_end_pos2_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_lowinh_N100_CFo40_fig7_noisesynlow_inhlow_coloro1_bottom_notransf_pos2_dotted_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_cutf30_N100_CFo9_fig7_noisesynlow_inhlow_colorr1_pos2_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_cutf5_N100_CFo14_fig7_end_noisesynlow_inhlow_colorbr1_pos2_keep_",


                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N100_CFo6_fig7_noisesynlow_inhlow_colorb1_bottom_notransf_normalize_pos3_first_", # _label7 "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N10_CFo4_fig6_noisesynpoiss_dotted_color1_end_pos2_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_lowinh_N100_CFo40_fig7_noisesynlow_inhlow_colorb1_bottom_notransf_pos3_dotted_",


                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb_N100_CFo6_fig7_noisesynlow_inhlow_colorr1_left_notransf_pos1_normalize_first_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb2_N100_CFo6_fig7_noisesynlow_inhlow_colorr2_left_notransf_pos1_addplot_", # _label9 "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb_N10_CFo4_fig6_noisesynpoiss_dotted_color2_end_pos2_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb2c_N100_CFo6_fig7_noisesynlow_inhlow_colorr2_left_pos1_normalize_first_",


                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb2_cutf30_N100_CFo9_fig7_noisesynlow_inhlow_colorr1_pos1_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb2_cutf5_N100_CFo14_fig7_end_noisesynlow_inhlow_colorbr1_pos1_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo0_fig7_noisesynlow_color0_pos1_label1_normalize_first_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo0_fig7_noisesynpoiss_dotted_color0_pos1_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo0_fig7_noisesynmed_dashed_color0_pos1_keep_",
                       #"pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N10_CFo0.64_fig7_noisesynlow_color6_pos1_label4_keep_", "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N10_CFo0.64_fig7_noisesynpoiss_dotted_color6_pos1_keep_", "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N10_CFo0.64_fig7_noisesynmed_dashed_color6_pos1_keep_",
                       #"pop_transfer_grc_syngr_cn_a1_nsyn3_noinh_varih_N10_CFo1_fig7_noisesynlow_color6_pos1_label4_", "pop_transfer_grc_syngr_cn_a1_nsyn3_noinh_varih_N10_CFo1_fig7_noisesynpoiss_dotted_color6_pos1_", "pop_transfer_grc_syngr_cn_a1_nsyn3_noinh_varih_N10_CFo1_fig7_noisesynmed_dashed_color6_pos1_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo4_fig7_noisesynlow_color2_pos1_label2_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo4_fig7_noisesynpoiss_dotted_color2_pos1_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N10_CFo4_fig7_noisesynmed_dashed_color2_pos1_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lowinh_N10_CFo40_fig7_noisesynlow_color1_pos1_label3_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lowinh_N10_CFo40_fig7_noisesynpoiss_dotted_color1_pos1_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lowinh_N10_CFo40_fig7_noisesynmed_dashed_color1_pos1_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N10_CFo4_fig7_noisesynlow_color2_end_pos2_label5_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N10_CFo4_fig7_noisesynpoiss_dotted_color2_end_pos2_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N10_CFo4_fig7_noisesynmed_dashed_color2_end_pos2_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N10_lowinh_CFo40_fig7_noisesynlow_color1_end_pos2_label6_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N10_lowinh_CFo40_fig7_noisesynpoiss_dotted_color1_end_pos2_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N10_lowinh_CFo40_fig7_noisesynmed_dashed_color1_end_pos2_keep_",
                       #"pop_transfer_grc_syngrfastinh_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N10_CFo40_gsinh0.00025_fig7_noisesynlow_color2_end_pos2_label6_keep_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N10_CFo4_fig7_noisesynlow_color6_end_pos2_label7_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N10_CFo4_fig7_noisesynpoiss_dotted_color6_end_pos2_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N10_CFo4_fig7_noisesynmed_dashed_color6_end_pos2_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N10_lowinh_CFo40_fig7_noisesynlow_color5_end_pos2_label8_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N10_lowinh_CFo40_fig7_noisesynpoiss_dotted_color5_end_pos2_keep_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N10_lowinh_CFo40_fig7_noisesynmed_dashed_color5_end_pos2_keep_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb_N10_CFo0_fig7_noisesynlow_color0_pos3_normalize_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb_N10_CFo0_fig7_noisesynpoiss_dotted_color0_pos3_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb_N10_CFo4_fig7_noisesynlow_color2_pos3_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb_N10_CFo4_fig7_noisesynpoiss_dotted_color2_pos3_",
                      ])


if pars.opt == "fig7a":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array([
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_lincomb_N100_CFo6_fig7_noisesynlow_inhlow_colorr1_left_pos1_normalize_first_keep_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_bothin_N100_CFo6_fig7_noisesynlow_inhlow_coloro1_left_pos2_first_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_bothin_lowinh_N100_CFo40_fig7_noisesynlow_inhlow_coloro1_left_pos2_dotted_keep_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_inhin_N100_CFo6_fig7_noisesynlow_inhlow_colorb1_left_normalize_pos3_first_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_inhin_lowinh_N100_CFo40_fig7_noisesynlow_inhlow_colorb1_left_pos3_dotted_addplot_",

                      ])


if pars.opt == "fig7_poster":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array([
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb_N100_CFo6_fig7_noisesynlow_inhlow_color2_left_pos1_normalize_first_keep_", # _label9 "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lincomb_N10_CFo4_fig6_noisesynpoiss_dotted_color2_end_pos2_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N100_CFo6_fig7_noisesynlow_inhlow_color5_left_pos2_first_keep_", # _label5 "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_N10_CFo4_fig6_noisesynpoiss_dotted_color5_end_pos2_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_bothin_lowinh_N100_CFo40_fig7_noisesynlow_inhlow_color5_left_pos2_dotted_keep_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N100_CFo6_fig7_noisesynlow_inhlow_color1_left_normalize_pos3_first_keep_", # _label7 "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N10_CFo4_fig6_noisesynpoiss_dotted_color1_end_pos2_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_lowinh_N100_CFo40_fig7_noisesynlow_inhlow_color1_left_pos3_dotted_keep_addplot_",

                      ])


if pars.opt == "fig7test":
    t_stim = 100*s
    do_if = False
    do_vec = np.array([
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N100_CFo4_fig7_noisesynlow_color6_end_pos2_label7_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N100_CFo4_fig7_noisesynpoiss_dotted_color6_end_pos2_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N100_CFo4_fig7_noisesynmed_dashed_color6_end_pos2_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N100_lowinh_CFo40_fig7_noisesynlow_color5_end_pos2_label8_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N100_lowinh_CFo40_fig7_noisesynpoiss_dotted_color5_end_pos2_", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin_N100_lowinh_CFo40_fig7_noisesynmed_dashed_color5_end_pos2_",
                    ])


## LOW CF:

# FIGURE 10
if pars.opt == "fig8a":
    t_stim = 100*s                   
    do_if = False
    export = False
    do_vec = np.array([

                       #"pop_CF-cutf_none_synno_cn_a10_varih_N50_noisesynlow_inhlow_twopop_fig8_colorp2_end_label1_pos14_first_label20_keep_",
                       #"pop_transfer_none_synno_cn_a1_varih_ih20_cutf30_N50_fig8_noisesynlow_twopop_coloro1_pos1_label23_first_plotre_normalize_keep_",
                       #"pop_transfer_none_synno_cn_a1_varih_ih20_cutf30_N100_fig8_noisesynlow_colorg1_pos1_label24n_plotre_plotpsth0_keep_",
                       #"pop_transfer_none_synno_cn_a10_varih_ih8_cutf30_N50_fig8_noisesynlow_twopop_coloro2_pos1b_label25_first_plotre_keep_",
                       #"pop_transfer_none_synno_cn_a10_varih_ih8_cutf30_N100_fig8_noisesynlow_colorg2_pos1b_label26n_plotre_plotpsth_keep_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo9_fig8_noisesynlow_inhlow_twopop_colorb1_pos2_label23_first_plotre_normalize_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo9_fig8_noisesynlow_inhlow_colorr1_pos2_label24_plotre_plotpsth_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N50_CFo13_fig8_noisesynlow_inhlow_twopop_colorb2_pos2b_label25_first_plotre_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N100_CFo13_fig8_noisesynlow_inhlow_colorr2_pos2b_label26_plotre_plotpsth_keep_",

                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo9_fig8_noisesynlow_inhlow_twopop_colorb1_pos2_dotted_keep_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo9_fig8_noisesynlow_inhlow_colorr1_pos2_dotted_keep_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N50_CFo15_fig8_noisesynlow_inhlow_twopop_colorb2_pos2b_first_dotted_keep_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N100_CFo15_fig8_noisesynlow_inhlow_colorr2_pos2b_dotted_keep_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo14_fig8_end_noisesynlow_inhlow_twopop_colorb1_pos3_label27_first_plotre_normalize_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo14_fig8_end_noisesynlow_inhlow_colorr1_pos3_label28_plotre_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf4_N50_CFo13_fig8_end_noisesynlow_inhlow_twopop_colorb2_pos3b_label29_first_plotre_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf4_N100_CFo13_fig8_end_noisesynlow_inhlow_colorr2_pos3b_label30_plotre_keep_",

                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo14_fig8_end_noisesynlow_inhlow_twopop_colorb1_pos3_dotted_keep_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo14_fig8_end_noisesynlow_inhlow_colorr1_pos3_dotted_keep_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf4_N50_CFo13_fig8_end_noisesynlow_inhlow_twopop_colorb2_pos3b_dotted_keep_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf4_N100_CFo13_fig8_end_noisesynlow_inhlow_colorr2_pos3b_dotted_keep_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a3_nsyn4_varih_varinhn_cutf30_N100_CFo9_fig8_noisesynlow_inhlow_coloro1_dotted_pos2_label0_plotre_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a3_nsyn4_varih_varinhn_cutf30_N50_CFo9_fig8_end_noisesynlow_inhlow_twopop_dotted_coloro2_pos2b_label0_plotre_",

                       #"pop_CF-cutf_grc_syngr_cn_adjinh_a1_nsyn4_varih_N50_noisesynlow_twopop_fig8_colorb2_end_label3_pos14_label21_keep_",
                       #"pop_CF-cutf_if_syngr_cn_adjinh_a1_nsyn4_varih_N50_noisesynlow_twopop_fig8_colorb2_dotted_star_end_label3_pos14_label21_keep_",

                       "pop_CF-cutf_none_synno_cn_a1_varih_N100_noisesynlow_inhlow_fig8_colorg1_end_label1_pos13_first_label18_",
                       "pop_CF-cutf_none_synno_cn_a1_varih_N50_noisesynlow_inhlow_twopop_fig8_coloro1_end_label1_pos13_label19_",
                       "pop_CF-cutf_none_synno_cn_a10_varih_N50_noisesynlow_inhlow_twopop_fig8_coloro2_end_label1_pos13_label20_",

                       "pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_noisesynlow_inhlow_fig8_colorr1_end_label3_pos14_label21_first_",
                       "pop_CF-cutf_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_noisesynlow_inhlow_fig8_colorr1_dotted_star_end_label3_pos14_label23_",
                       #"pop_CF-cutf_grc_syngr_cn_adjinh_a1_nsyn4_varih_N100_noisesynlow_fig8_colorr2_end_label3_pos14_label25_keep_",
                       #"pop_CF-cutf_if_syngr_cn_adjinh_a1_nsyn4_varih_N100_noisesynlow_fig8_colorr2_dotted_star_end_label3_pos14_label26_keep_",

                       "pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_twopop_N50_noisesynlow_inhlow_fig8_colorb1_end_label3_pos14_label22_",
                       "pop_CF-cutf_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_twopop_N50_noisesynlow_inhlow_fig8_colorb1_dotted_star_end_label3_pos14_label24_",

                     ])


# FIGURE 9
if pars.opt == "fig8b":
    t_stim = 1000*s
    do_if = False
    export = False
    do_vec = np.array([
                       #"pop_transfer_grc_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf4_N50_CFo15_cutf4_fig8_end_noisesynlow_inhlow_twopop_colorp2_pos3b_label29_first_plotre_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf4_N100_CFo15_cutf4_fig8_end_noisesynlow_inhlow_colorp1_pos3b_label30_plotre_",


                       "pop_transfer_none_synno_cn_a1_varih_ih20_cutf30_N100_fig8_noisesynlow_colorg1_pos1_label24n_first_plotre_notransf_plotpsth0_normalize_",
                       "pop_transfer_none_synno_cn_a1_varih_ih20_cutf30_N50_fig8_noisesynlow_twopop_coloro1_pos1_label23_plotre_notransf_",
                       "pop_transfer_none_synno_cn_a10_varih_ih20_cutf30_N100_fig8_noisesynlow_colorg2_pos1b_label26n_plotre_plotpsth_notransf_",
                       "pop_transfer_none_synno_cn_a10_varih_ih20_cutf30_N50_fig8_noisesynlow_twopop_coloro2_pos1b_label25_first_plotre_notransf_",


#                        "pop_transfer_grc_syngr_cn_adjfinh_a05_nsyn4_varih_varinhn_cutf30_N100_CFo9_fig8_noisesynlow_inhlow_colorr1_pos2_label24_plotre_plotpsth_normalize_",
#                        "pop_transfer_grc_syngr_cn_adjfinh_a05_nsyn4_varih_varinhn_cutf30_N50_CFo9_fig8_noisesynlow_inhlow_twopop_colorb1_pos2_label23_first_plotre_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo9_fig8_noisesynlow_inhlow_colorr1_pos2_label24_plotre_first_plotpsth_notransf_normalize_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo9_fig8_noisesynlow_inhlow_twopop_colorb1_pos2_label23_plotre_notransf_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N100_CFo9_is0.14_fig8_noisesynlow_inhlow_colorr2_pos2b_label26_plotre_plotpsth_notransf_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N50_CFo9_is0.14_fig8_noisesynlow_inhlow_twopop_colorb2_pos2b_label25_first_plotre_notransf_",

                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo9_fig8_noisesynlow_inhlow_twopop_colorb1_pos2_dotted_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo9_fig8_noisesynlow_inhlow_colorr1_pos2_dotted_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N50_CFo9_is0.13_fig8_noisesynlow_inhlow_twopop_colorb2_pos2b_first_dotted_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N100_CFo9_is0.13_fig8_noisesynlow_inhlow_colorr2_pos2b_dotted_",

                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo9_is0.19_fig8_noisesynlow_inhlow_colorr1_pos2_dotted_notransf_", #0.2
                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo9_is0.19_fig8_noisesynlow_inhlow_twopop_colorb1_pos2_dotted_notransf_", #0.2
                       "pop_transfer_if_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N100_CFo9_is0.40_fig8_noisesynlow_inhlow_colorr2_pos2b_dotted_notransf_", #0.41
                       "pop_transfer_if_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N50_CFo9_is0.40_fig8_noisesynlow_inhlow_twopop_colorb2_pos2b_first_dotted_notransf_", #0.41
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo9_is0.20_fig8_noisesynlow_inhlow_colorr1_pos2_dotted_", #0.2
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo9_is0.20_fig8_noisesynlow_inhlow_twopop_colorb1_pos2_dotted_", #0.2
                       #"pop_transfer_if_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N100_CFo9_is0.41_fig8_noisesynlow_inhlow_colorr2_pos2b_dotted_", #0.41
                       #"pop_transfer_if_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N50_CFo9_is0.41_fig8_noisesynlow_inhlow_twopop_colorb2_pos2b_first_dotted_", #0.41


                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf5_N100_CFo14_fig8_end_noisesynlow_inhlow_colorbr1_pos3_label28_first_plotre_normalize_notransf_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf5_N50_CFo14_fig8_end_noisesynlow_inhlow_twopop_colorp1_pos3_label27_plotre_notransf_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo14_is0.03_fig8_end_noisesynlow_inhlow_twopop_colorb2_pos3_label27_first_plotre_normalize_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo14_is0.03_fig8_end_noisesynlow_inhlow_colorr2_pos3_label28_plotre_keep_",

                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf5_N50_CFo14_fig8_end_noisesynlow_inhlow_twopop_colorb1_pos3b_dotted_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf5_N100_CFo14_fig8_end_noisesynlow_inhlow_colorr1_pos3b_dotted_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo14_is0.01_fig8_end_noisesynlow_inhlow_twopop_colorb2_pos3_dotted_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo14_is0.01_fig8_end_noisesynlow_inhlow_colorr2_pos3_dotted_",

                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf5_N100_CFo14_is0.19_fig8_end_noisesynlow_inhlow_colorbr1_pos3_label30_dotted_notransf_", #0.2
                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf5_N50_CFo14_is0.19_fig8_end_noisesynlow_inhlow_twopop_colorp1_pos3_label29_dotted_notransf_", #0.2
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf5_N100_CFo14_is0.20_fig8_end_noisesynlow_inhlow_colorr2_pos3_dotted_", #0.2
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf5_N50_CFo14_is0.20_fig8_end_noisesynlow_inhlow_twopop_colorb2_pos3_dotted_", #0.2
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo14_is0.22_fig8_end_noisesynlow_inhlow_colorr2_pos3_dotted_keep_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo14_is0.22_fig8_end_noisesynlow_inhlow_twopop_colorb2_pos3_dotted_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf5_N200_CFo14_fig8_end_noisesynlow_inhlow_colorr1_pos3b_label32_plotre_first_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf5_N100_CFo14_fig8_end_noisesynlow_inhlow_twopop_colorb1_pos3b_label31_plotre_keep_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_highvarih_cutf30_N100_CFo0_fig8_end_noisesynlow_is0.56_colorr1_pos5_label33_notransf_",
                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_highvarih_cutf30_N100_CFo0_fig8_end_noisesynlow_is0.74_colorr1_pos5_label34_dotted_notransf_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_highervarih_cutf5_N100_CFo0_fig8_end_noisesynlow_is0.635_colorbr1_pos5_label35_notransf_",
                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_highervarih_cutf5_N100_CFo0_fig8_end_noisesynlow_is0.84_colorbr1_pos5_label36_dotted_notransf_",

                     ])

if pars.opt == "fig8b_background":
    t_stim = 10*s
    data_dir = "./data"
    minimal_dir = False
    export = False
    do_if = False
    do_vec = np.array([
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo9_fig8_noisesynlow_inhlow_colorr1_pos2_label24_plotre_first_plotpsth_notransf_normalize_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N100_CFo9_is0.14_fig8_noisesynlow_inhlow_colorr2_pos2b_label26_plotre_plotpsth_notransf_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf5_N100_CFo14_fig8_end_noisesynlow_inhlow_colorbr1_pos3_label28_first_plotre_normalize_notransf_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_highervarih_cutf5_N100_CFo0_fig8_end_noisesynlow_is0.635_colorbr1_pos5_label35_notransf_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_highvarih_cutf30_N100_CFo0_fig8_end_noisesynlow_is0.56_colorr1_pos5_label33_notransf_",

                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf5_N100_CFo14_is0.19_fig8_end_noisesynlow_inhlow_colorbr1_pos3_label30_dotted_notransf_",
                       #"pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_highervarih_cutf5_N100_CFo0_fig8_end_noisesynlow_is0.84_colorbr1_pos5_label36_dotted_notransf_",
                       ])

if pars.opt == "fig8b2":
    t_stim = 100*s
    do_if = False
    do_vec = np.array([
                        "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf60_N100_CFo6_fig8_noisesynlow_inhlow_colorr1_pos2_label24_plotre_plotpsth_normalize_",
                        "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf60_N50_CFo6_fig8_noisesynlow_inhlow_twopop_colorb1_pos2_label23_first_plotre_",

                        "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf60_N100_CFo6_is0.19_fig8_noisesynlow_inhlow_colorr1_pos2_dotted_",
                        "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf60_N50_CFo6_is0.19_fig8_noisesynlow_inhlow_twopop_colorb1_pos2_dotted_",

                      ])


if pars.opt == "fig8x":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array([

                       "pop_transfer_none_synno_cn_a1_varih_ih20_cutf30_N50_fig8_noisesynlow_twopop_coloro1_pos1_label23_first_plotre_keep_",
                       "pop_transfer_none_synno_cn_a1_varih_ih20_cutf30_N100_fig8_noisesynlow_colorg1_pos1_label24_plotre_plotpsth0_normalize_keep_",
                       "pop_transfer_none_synno_cn_a10_varih_ih8_cutf30_N50_fig8_noisesynlow_twopop_coloro2_pos1b_label25_first_plotre_keep_",
                       "pop_transfer_none_synno_cn_a10_varih_ih8_cutf30_N100_fig8_noisesynlow_colorg2_pos1b_label26_plotre_plotpsth_keep_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_cutf30_N50_CFo9_fig8_noisesynlow_inhlow_twopop_colorb1_pos2_label23_first_plotre_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_cutf30_N100_CFo9_fig8_noisesynlow_inhlow_colorr1_pos2_label24_plotre_plotpsth_normalize_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a10_nsyn4_lowvarih_lowvarinhn_cutf30_N50_CFo13_fig8_noisesynlow_inhlow_twopop_colorb2_pos2b_label25_first_plotre_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a10_nsyn4_lowvarih_lowvarinhn_cutf30_N100_CFo13_fig8_noisesynlow_inhlow_colorr2_pos2b_label26_plotre_plotpsth_",

                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_cutf30_N50_CFo9_fig8_noisesynlow_inhlow_twopop_colorb1_pos2_dotted_",
                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_cutf30_N100_CFo9_fig8_noisesynlow_inhlow_colorr1_pos2_dotted_",
                       "pop_transfer_if_syngr_cn_adjfinh_a10_nsyn4_lowvarih_lowvarinhn_cutf30_N50_CFo15_fig8_noisesynlow_inhlow_twopop_colorb2_pos2b_first_dotted_",
                       "pop_transfer_if_syngr_cn_adjfinh_a10_nsyn4_lowvarih_lowvarinhn_cutf30_N100_CFo15_fig8_noisesynlow_inhlow_colorr2_pos2b_dotted_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_cutf30_N50_CFo14_fig8_end_noisesynlow_inhlow_twopop_colorb1_pos3_label27_first_plotre_normalize_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_cutf30_N100_CFo14_fig8_end_noisesynlow_inhlow_colorr1_pos3_label28_plotre_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_cutf4_N50_CFo13_cutf4_fig8_end_noisesynlow_inhlow_twopop_colorb2_pos3b_label29_first_plotre_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_cutf4_N100_CFo13_cutf4_fig8_end_noisesynlow_inhlow_colorr2_pos3b_label30_plotre_",

                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_cutf30_N50_CFo14_fig8_end_noisesynlow_inhlow_twopop_colorb1_pos3_dotted_",
                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_cutf30_N100_CFo14_fig8_end_noisesynlow_inhlow_colorr1_pos3_dotted_",
                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_cutf4_N50_CFo13_cutf4_fig8_end_noisesynlow_inhlow_twopop_colorb2_pos3b_dotted_",
                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_cutf4_N100_CFo13_cutf4_fig8_end_noisesynlow_inhlow_colorr2_pos3b_dotted_",

                       "pop_CF-cutf_none_synno_cn_a1_varih_N100_noisesynlow_inhlow_fig8_colorg1_end_label1_pos14_first_label18_",
                       "pop_CF-cutf_none_synno_cn_a1_varih_N50_noisesynlow_inhlow_twopop_fig8_coloro1_end_label1_pos14_label19_",
                       "pop_CF-cutf_none_synno_cn_a10_varih_N50_noisesynlow_inhlow_twopop_fig8_coloro2_end_label1_pos14_label20_",

                       "pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_N100_noisesynlow_inhlow_fig8_colorr1_end_label3_pos14_label21_",
                       "pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_twopop_N50_noisesynlow_inhlow_fig8_colorb1_end_label3_pos14_label22_",

                       "pop_CF-cutf_if_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_N100_noisesynlow_inhlow_fig8_colorr1_dotted_star_end_label3_pos14_label23_",
                       "pop_CF-cutf_if_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_twopop_N50_noisesynlow_inhlow_fig8_colorb1_dotted_star_end_label3_pos14_label24_",

                     ])

if pars.opt == "fig8c":
    t_stim = 10*s
    do_if = False
    do_vec = np.array([

                       #"pop_transfer_none_synno_cn_a1_varih_ih20_cutf30_N100_fig8_noisesynlow_coloro1_pos1_label24_first_plotre_plotpsth_normalize_keep_",
                       #"pop_transfer_none_synno_cn_a10_varih_ih8_cutf30_N100_fig8_noisesynlow_coloro2_pos1b_label26_plotre_plotpsth_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo9_fig8_noisesynlow_inhlow_colorr1_pos2_label24_plotre_plotpsth_normalize_keep_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N100_CFo13_fig8_noisesynlow_inhlow_colorr2_pos2b_label26_plotre_plotpsth_keep_",

                       "pop_CF-cutf_none_synno_cn_a1_varih_N100_noisesynlow_inhlow_fig8_coloro1_end_label1_pos14_first_label18_keep_",
                       "pop_CF-cutf_none_synno_cn_a1_varih_N50_noisesynlow_inhlow_twopop_fig8_colorg1_end_label1_pos14_label19_keep_",
                       "pop_CF-cutf_none_synno_cn_a10_varih_N50_noisesynlow_inhlow_twopop_fig8_colorg2_end_label1_pos14_label20_keep_",

                       "pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_noisesynlow_inhlow_fig8_colorr1_end_label3_pos14_label21_keep_",
                       "pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_twopop_N50_noisesynlow_inhlow_fig8_colorb1_end_label3_pos14_label22_keep_",

                       "pop_CF-cutf_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_noisesynlow_inhlow_fig8_colorr2_end_label3_pos14_label21_keep_",
                       "pop_CF-cutf_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_twopop_N50_noisesynlow_inhlow_fig8_colorb2_end_label3_pos14_label22_keep_",

                     ])

if pars.opt == "fig8d":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array([

                       #"pop_CF-cutf_none_synno_cn_a10_varih_N50_noisesynlow_inhlow_twopop_fig8_colorp2_end_label1_pos14_first_label20_keep_",
                       "pop_transfer_none_synno_cn_a1_varih_ih20_cutf30_N50_fig8_noisesynlow_twopop_colorg1_pos1_label23_plotK_",
                       "pop_transfer_none_synno_cn_a1_varih_ih20_cutf30_N100_fig8_noisesynlow_coloro1_pos1_label24_first_plotK_normalize_keep_",
                       "pop_transfer_none_synno_cn_a10_varih_ih20_cutf30_N50_fig8_noisesynlow_twopop_colorg2_pos1_label25_first_plotK_",
                       "pop_transfer_none_synno_cn_a10_varih_ih20_cutf30_N100_fig8_noisesynlow_coloro2_pos1_label26_plotK_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo9_fig8_noisesynlow_inhlow_twopop_colorb1_pos2_label23_first_plotK_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo9_fig8_noisesynlow_inhlow_colorr1_pos2_label24_plotK_normalize_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N50_CFo13_fig8_noisesynlow_inhlow_twopop_colorb2_pos2_label25_first_plotK_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N100_CFo13_fig8_noisesynlow_inhlow_colorr2_pos2_label26_plotK_keep_",

                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo9_fig8_noisesynlow_inhlow_twopop_colorb1_pos2_label23_first_plotK_dotted_keep_",
                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo9_fig8_noisesynlow_inhlow_colorr1_pos2_label24_plotK_normalize_dotted_keep_",
                       "pop_transfer_if_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N50_CFo15_fig8_noisesynlow_inhlow_twopop_colorb2_pos2_label25_first_plotK_dotted_keep_",
                       "pop_transfer_if_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_cutf30_N100_CFo15_fig8_noisesynlow_inhlow_colorr2_pos2_label26_plotK_dotted_",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo14_fig8_end_noisesynlow_inhlow_twopop_colorb1_pos3_label27_first_plotK_normalize_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo14_fig8_end_noisesynlow_inhlow_colorr1_pos3_label28_plotK_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf4_N50_CFo13_cutf4_fig8_end_noisesynlow_inhlow_twopop_colorb2_pos3_label29_first_plotK_keep_",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf4_N100_CFo13_cutf4_fig8_end_noisesynlow_inhlow_colorr2_pos3_label30_plotK_keep_",

                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N50_CFo14_fig8_end_noisesynlow_inhlow_twopop_colorb1_pos3_label27_first_plotK_normalize_dotted_keep_",
                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf30_N100_CFo14_fig8_end_noisesynlow_inhlow_colorr1_pos3_label28_plotK_dotted_keep_",
                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf4_N50_CFo13_cutf4_fig8_end_noisesynlow_inhlow_twopop_colorb2_pos3_label29_first_plotK_dotted_keep_",
                       "pop_transfer_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_cutf4_N100_CFo13_cutf4_fig8_end_noisesynlow_inhlow_colorr2_pos3_label30_plotK_dotted_keep_",

                       #"pop_transfer_grc_syngr_cn_adjfinh_a3_nsyn4_varih_varinhn_cutf30_N100_CFo9_fig8_noisesynlow_inhlow_coloro1_dotted_pos2_label0_plotre_",
                       #"pop_transfer_grc_syngr_cn_adjfinh_a3_nsyn4_varih_varinhn_cutf30_N50_CFo9_fig8_end_noisesynlow_inhlow_twopop_dotted_coloro2_pos2b_label0_plotre_",

                       "pop_CF-cutf_none_synno_cn_a1_varih_N100_noisesynlow_inhlow_fig8_coloro1_end_label1_pos14_first_label18_keep_",
                       "pop_CF-cutf_none_synno_cn_a1_varih_N50_noisesynlow_inhlow_twopop_fig8_colorg1_end_label1_pos14_label19_keep_",
                       "pop_CF-cutf_none_synno_cn_a10_varih_N50_noisesynlow_inhlow_twopop_fig8_colorg2_end_label1_pos14_label20_keep_",

                       "pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_noisesynlow_inhlow_fig8_colorr1_end_label3_pos14_label21_keep_",
                       "pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_twopop_N50_noisesynlow_inhlow_fig8_colorb1_end_label3_pos14_label22_keep_",

                     ])


if pars.opt == "fig7b":
    t_stim = 100*s
    do_if = False
    do_vec = np.array([
                       #"pop_transfer_none_synno_cn_a1_nsyn4_varih_N1000_ih1_fig7b_noisesynlow_color0_pos1", "pop_transfer_none_synno_cn_a1_nsyn4_varih_N1000_ih1_fig7b_noisesynpoiss_color0_dotted_pos1",

                       "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N1000_CFo0.31_fig7b_noisesynlow_color6", "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N1000_CFo0.30_fig7b_noisesynpoiss_color6_dotted",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_gsex1.15_N1000_CFo40_fig7b_noisesynlow_color2", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N1000_CFo40_fig7b_noisesynpoiss_color2_dotted",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_gsex1.15_samein_N1000_CFo40_fig7b_noisesynlow_color1", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_samein_N1000_CFo40_fig7b_noisesynpoiss_color1_dotted",

                       "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_gsex1.15_lincomb3_N1000_CFo0.32_fig7b_noisesynlow_color3", "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_lincomb3_N1000_CFo0.32_fig7b_noisesynpoiss_color3_dotted",

                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_gsex1.2_lincomb3_N1000_CFo40_fig7b_noisesynlow_color5", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_gsex1.1_lincomb3_N1000_CFo40_fig7b_noisesynpoiss_color5_dotted",

                       # GRC TONIC INHIBITION?
                      ])


if pars.opt == "figX":
    t_stim = 100*s
    do_if = False
    do_vec = np.array([
                       "pop_transfer_prk_synprk_cn_adjfinh_a1_nsyn100_varih_N1_CFo0_figX_end_noisesynlow_colorb1_pos1_normalize_label0_first_",
                     ])



if pars.opt == "samein":
    t_stim = 1000*s
    do_if = False
    do_vec = np.array(["pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_samein_N10_CFo4_fig7_noisesynlow_color2",
                       "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_samein_N10_lowinh_CFo40_fig7_noisesynlow_color1",
                        ])


if pars.opt == "none_4":
    t_stim = 100*s
    do_if = False
    do_vec = np.array(["pop_transfer_none_synno_cn_a1_nsyn4_varih_N100_ih1_fig7b_noisesynlow_color0_pos1",
                       "pop_transfer_none_synno_cn_a1_nsyn4_varih_N100_ih1_fig7b_noisesynpoiss_color0_dotted_pos1",
                       #"pop_transfer_none_synno_cn_a1_nsyn4_N100_ih1_fig7b_noisesynlow_color3_pos1",
                       #"pop_transfer_none_synno_cn_a1_nsyn4_N100_ih1_fig7b_noisesynpoiss_color3_dotted_pos1",
                       ])


if pars.opt == "noinh":
    t_stim = 100*s
    do_if = False
    do_vec = np.array(["pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N100_CFo0.31_fig7b_noisesynlow_color6", "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_N100_CFo0.30_fig7b_noisesynpoiss_color6_dotted",
                      ])

if pars.opt == "inh":
    t_stim = 100*s
    do_if = False
    do_vec = np.array(["pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_gsex1.15_N100_CFo40_fig7b_noisesynlow_color2", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_N100_CFo40_fig7b_noisesynpoiss_color2_dotted",
                      ])

if pars.opt == "inhsamein":
    t_stim = 100*s
    do_if = False
    do_vec = np.array(["pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_gsex1.15_samein_N100_CFo40_fig7b_noisesynlow_color1", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_samein_N100_CFo40_fig7b_noisesynpoiss_color1_dotted",
                      ])

if pars.opt == "inhsamein5":
    t_stim = 100*s
    do_if = False
    do_vec = np.array(["pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_gsex1.15_samein5_N100_CFo40_fig7b_noisesynlow_color4", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_samein5_N100_CFo40_fig7b_noisesynpoiss_color4_dotted",
                      ])

if pars.opt == "lincomb":
    t_stim = 100*s
    do_if = False
    do_vec = np.array(["pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_gsex1.15_lincomb3_N100_CFo0.32_fig7b_noisesynlow_color3", "pop_transfer_grc_syngr_cn_a1_nsyn4_noinh_varih_lincomb3_N100_CFo0.32_fig7b_noisesynpoiss_color3_dotted",
                      ])

if pars.opt == "inhlincomb":
    t_stim = 100*s
    do_if = False
    do_vec = np.array(["pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_gsex1.2_lincomb3_N100_CFo40_fig7b_noisesynlow_color5", "pop_transfer_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_gsex1.1_lincomb3_N100_CFo40_fig7b_noisesynpoiss_color5_dotted",
                      ])


temperature = 37

if MPI.COMM_WORLD.rank == 0:

    if ("poster" in pars.opt) or ("talk" in pars.opt):
        color1 = '#00A0E3' # Cyan
        color2 = '#E5097F' # Magenta
        color3 = '#808080' # Gray
        color4 = '#78317B' # Lila
        color5 = '#EC671F' # Orange
        color6 = '#009A47' # Dark Green
        color7 = '#FFED00' # Yellow
        color8 = '#393476' # Uni Blue
        color9 = '#E42A24' # Red

        linewidth = 1.5

    d_out = 10
    d_down = 10

    # FIGURE 6
    if "fig5" in pars.opt:

        if "talk" in pars.opt:

            fig_size =  [11.7*0.3937,7.4*0.3937]
            params = {'backend': 'ps',
                      'axes.labelsize': 6,
                      'axes.linewidth' : 0.5,
                      'title.fontsize': 8,
                      'text.fontsize': 8,
                      'font.size':6,
                      'axes.titlesize':6,
                      'legend.fontsize': 6,
                      'xtick.labelsize': 6,
                      'ytick.labelsize': 6,
                      'legend.borderpad': 0.2,
                      'legend.linewidth': 0.1,
                      'legend.loc': 3,
                      'legend.ncol': 4,
                      'text.usetex': False,
                      'figure.figsize': fig_size}
            rcParams.update(params)

            linewidth = 1
            d_out = 8
            d_down = 3

            fig1 = plt.figure('results_transfer_syn')

            gs = matplotlib.gridspec.GridSpec(2, 3,
               width_ratios=[1,1,1],
               height_ratios=[1,1]
               )

            #gs.update(bottom=0.23, top=0.93, left=0.08, right=0.97, wspace=0.4, hspace=1.0)
            gs.update(bottom=0.08, top=0.90, left=0.08, right=0.97, wspace=0.4, hspace=0.6)

            ax1 = plt.subplot(gs[0,0])
            ax2 = plt.subplot(gs[0,1])
            ax3 = plt.subplot(gs[0,2])

            ax4 = plt.subplot(gs[1,0])
            ax5 = plt.subplot(gs[1,1])
            ax6 = plt.subplot(gs[1,2])

            font = font0.copy()
            font.set_family('sans-serif')
            font.set_weight('semibold')

        # FIGURE 6
        else:

            fig_size =  [85*0.03937,140*0.03937] # 1-Column
            params['figure.figsize'] = fig_size
            rcParams.update(params)

            fig1 = plt.figure('results_transfer_syn')

            gs = matplotlib.gridspec.GridSpec(4, 3,
               width_ratios=[1,1,1],
               height_ratios=[1,1,1,1]
               )

            #gs.update(bottom=0.23, top=0.93, left=0.08, right=0.97, wspace=0.4, hspace=1.0)
            gs.update(bottom=0.07, top=0.91, left=0.095, right=0.97, wspace=0.5, hspace=1)

            ax1 = plt.subplot(gs[0,0])
            ax2 = plt.subplot(gs[0,1])
            ax3 = plt.subplot(gs[0,2])

            ax4 = plt.subplot(gs[1,0])
            ax5 = plt.subplot(gs[1,1])
            ax6 = plt.subplot(gs[1,2])

            ax7 = plt.subplot(gs[2,0])
            ax8 = plt.subplot(gs[2,1])
            ax9 = plt.subplot(gs[2,2])

            ax10 = plt.subplot(gs[3,0])
            ax11 = plt.subplot(gs[3,1])
            ax12 = plt.subplot(gs[3,2])

            #gs2 = matplotlib.gridspec.GridSpec(1, 1,
            #   width_ratios=[1],
            #   height_ratios=[1]
            #   )

            #gs2.update(bottom=0.07, top=0.15, left=0.10, right=0.97, wspace=0.4, hspace=0.4)
            #ax13 = plt.subplot(gs2[0,0])

            font = font0.copy()
            font.set_family('sans-serif')
            font.set_weight('semibold')

            linewidth = 1.5


        if ("poster" not in pars.opt) and ("talk" not in pars.opt):
            x1 = -0.25
            y1 = 1.3

            ax1.text(x1, y1, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font) # 'A$\mathbf{\mathsf{_1}}$'
            ax2.text(x1, y1, 'A2', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
            ax3.text(x1, y1, 'A3', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)

            ax4.text(x1, y1, 'B1', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)
            ax5.text(x1, y1, 'B2', transform=ax5.transAxes, fontsize=12, va='top', fontproperties=font)
            ax6.text(x1, y1, 'B3', transform=ax6.transAxes, fontsize=12, va='top', fontproperties=font)

            ax7.text(x1, y1, 'C1', transform=ax7.transAxes, fontsize=12, va='top', fontproperties=font)
            ax8.text(x1, y1, 'C2', transform=ax8.transAxes, fontsize=12, va='top', fontproperties=font)
            ax9.text(x1, y1, 'C3', transform=ax9.transAxes, fontsize=12, va='top', fontproperties=font)

            ax10.text(x1, y1, 'D1', transform=ax10.transAxes, fontsize=12, va='top', fontproperties=font)
            ax11.text(x1, y1, 'D2', transform=ax11.transAxes, fontsize=12, va='top', fontproperties=font)
            ax12.text(x1, y1, 'D3', transform=ax12.transAxes, fontsize=12, va='top', fontproperties=font)

            #ax13.text(x1, y1, 'B', transform=ax13.transAxes, fontsize=12, va='top', fontproperties=font)

    # FIGURE 8
    elif "fig7" in pars.opt:

        if "poster" in pars.opt:

            fig_size =  [4.86,5] # 1.5-Column
            params['figure.figsize'] = fig_size
            rcParams.update(params)

            fig1 = plt.figure('results_transfer_syn')
            gs = matplotlib.gridspec.GridSpec(3, 3,
               width_ratios=[1,1,1],
               height_ratios=[1,1,1]
            )

            gs.update(bottom=0.07, top=0.72, left=0.09, right=0.97, wspace=0.4, hspace=0.4)

            ax1 = plt.subplot(gs[0,0])
            ax2 = plt.subplot(gs[1,0])
            ax3 = plt.subplot(gs[2,0])

            ax4 = plt.subplot(gs[0,1])
            ax5 = plt.subplot(gs[1,1])
            ax6 = plt.subplot(gs[2,1])

            ax7 = plt.subplot(gs[0,2])
            ax8 = plt.subplot(gs[1,2])
            ax9 = plt.subplot(gs[2,2])

        # FIGURE 8
        else:

            fig_size =  [85*0.03937,90*0.03937] # 2-Column
            params['figure.figsize'] = fig_size
            rcParams.update(params)

            fig1 = plt.figure('results_transfer_syn')
            gs = matplotlib.gridspec.GridSpec(1, 3,
               width_ratios=[1,1,1],
               height_ratios=[1]
            )

            gs.update(bottom=0.095, top=0.56, left=0.125, right=0.97, wspace=0.2, hspace=0.4)

            ax3 = plt.subplot(gs[0,0])
            #ax2 = plt.subplot(gs[1,0])
            #ax3 = plt.subplot(gs[2,0])

            ax6 = plt.subplot(gs[0,1])
            #ax5 = plt.subplot(gs[1,1])
            #ax6 = plt.subplot(gs[2,1])

            ax9 = plt.subplot(gs[0,2])
            #ax8 = plt.subplot(gs[1,0])
            #ax9 = plt.subplot(gs[2,0])


        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('semibold')

        linewidth = 1.5

        if ("poster" not in pars.opt) and ("talk" not in pars.opt):

            x1 = -0.13
            y1 = 1.1

            #ax1.text(x1, y1, 'A', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font) # 'A$\mathbf{\mathsf{_1}}$'
            #ax2.text(x1, y1, 'A2', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
            ax3.text(x1, y1, 'A', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)

            #ax4.text(x1, y1, 'B', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)
            #ax5.text(x1, y1, 'B2', transform=ax5.transAxes, fontsize=12, va='top', fontproperties=font)
            ax6.text(x1, y1, 'B', transform=ax6.transAxes, fontsize=12, va='top', fontproperties=font)

            #ax7.text(x1, y1, 'C', transform=ax7.transAxes, fontsize=12, va='top', fontproperties=font)
            #ax8.text(x1, y1, 'C2', transform=ax8.transAxes, fontsize=12, va='top', fontproperties=font)
            ax9.text(x1, y1, 'C', transform=ax9.transAxes, fontsize=12, va='top', fontproperties=font)


    # FIGURE 7
    elif "fig6" in pars.opt:


        if "talk" in pars.opt:

            #fig_size =  [4.86,5.4] # 1.5-Column 6.83
            fig_size =  [4.86,3] # 1.5-Column 6.83
            params['figure.figsize'] = fig_size
            rcParams.update(params)

            fig_size =  [11.7*0.3937,7.4*0.3937]
            params = {'backend': 'ps',
                      'axes.labelsize': 6,
                      'axes.linewidth' : 0.5,
                      'title.fontsize': 8,
                      'text.fontsize': 8,
                      'font.size':6,
                      'axes.titlesize':6,
                      'legend.fontsize': 6,
                      'xtick.labelsize': 6,
                      'ytick.labelsize': 6,
                      'legend.borderpad': 0.2,
                      'legend.linewidth': 0.1,
                      'legend.loc': 3,
                      'legend.ncol': 4,
                      'text.usetex': False,
                      'figure.figsize': fig_size}
            rcParams.update(params)

            if "fig6b" in pars.opt:

                linewidth = 1
                d_out = 8
                d_down = 3

                fig1 = plt.figure('results_transfer_syn')

                gs = matplotlib.gridspec.GridSpec(1, 3,
                   width_ratios=[1,1,1],
                   height_ratios=[1]
                   )

                gs.update(bottom=0.11, top=0.65, left=0.08, right=0.97, wspace=0.4, hspace=0.5)

                ax1 = plt.subplot(gs[0,0])
                ax2 = plt.subplot(gs[0,1])
                ax3 = plt.subplot(gs[0,2])

            else:

                linewidth = 1
                d_out = 8
                d_down = 3

                fig1 = plt.figure('results_transfer_syn')

                gs = matplotlib.gridspec.GridSpec(1, 3,
                   width_ratios=[1,1,1],
                   height_ratios=[1]
                   )

                gs.update(bottom=0.39, top=0.92, left=0.08, right=0.97, wspace=0.4, hspace=0.5)

                ax1 = plt.subplot(gs[0,0])
                ax2 = plt.subplot(gs[0,1])
                ax3 = plt.subplot(gs[0,2])

                gs2 = matplotlib.gridspec.GridSpec(1, 1,
                   width_ratios=[1],
                   height_ratios=[1]
                   )

                gs2.update(bottom=0.12, top=0.28, left=0.10, right=0.97, wspace=0.4, hspace=0.4)
                ax13 = plt.subplot(gs2[0,0])
                font = font0.copy()
                font.set_family('sans-serif')
                font.set_weight('semibold')

        elif "fig6b" in pars.opt:

            fig_size =  [4.86,2.5] # 1.5-Column 6.83

            params = {'backend': 'ps',
              'axes.labelsize': 8,
              'axes.linewidth' : 0.5,
              'title.fontsize': 8,
              'text.fontsize': 10,
              'font.size':10,
              'axes.titlesize':8,
              'legend.fontsize': 6,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'legend.borderpad': 0.2,
              'legend.linewidth': 0.1,
              'legend.loc': 'best',
              'legend.ncol': 4,
              'text.usetex': False,
              'figure.figsize': fig_size}
            rcParams.update(params)

            fig1 = plt.figure('results_transfer_syn')

            gs = matplotlib.gridspec.GridSpec(1, 1,
               width_ratios=[1],
               height_ratios=[1]
               )
            gs.update(bottom=0.17, top=0.91, left=0.12, right=0.95, wspace=0.4, hspace=0.4)
            ax14 = plt.subplot(gs[0,0])


            font = font0.copy()
            font.set_family('sans-serif')
            font.set_weight('semibold')


        elif "poster" in pars.opt:

            fig_size =  [4.86,4] # 1.5-Column 6.83

            params = {'backend': 'ps',
              'axes.labelsize': 8,
              'axes.linewidth' : 0.5,
              'title.fontsize': 8,
              'text.fontsize': 10,
              'font.size':10,
              'axes.titlesize':8,
              'legend.fontsize': 6,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'legend.borderpad': 0.2,
              'legend.linewidth': 0.1,
              'legend.loc': 'best',
              'legend.ncol': 4,
              'text.usetex': False,
              'figure.figsize': fig_size}
            rcParams.update(params)

            fig1 = plt.figure('results_transfer_syn')

            gs = matplotlib.gridspec.GridSpec(2, 3,
               width_ratios=[1,1,1],
               height_ratios=[1,1]
               )
            gs.update(bottom=0.30, top=0.95, left=0.1, right=0.97, wspace=0.4, hspace=0.4)
            ax1 = plt.subplot(gs[0:2,0])
            ax2 = plt.subplot(gs[0:2,1])

            ax3 = plt.subplot(gs[0,2])
            ax4 = plt.subplot(gs[1,2])


            gs2 = matplotlib.gridspec.GridSpec(1, 1,
               width_ratios=[1],
               height_ratios=[1]
               )
            gs2.update(bottom=0.1, top=0.2, left=0.10, right=0.96, wspace=0.3, hspace=0.4)
            ax13 = plt.subplot(gs2[0,0])

        # FIGURE 7
        else:

            fig_size =  [180*0.03937,100*0.03937] # 2-Column

            params = {'backend': 'ps',
              'axes.labelsize': 8,
              'axes.linewidth' : 0.5,
              'title.fontsize': 8,
              'text.fontsize': 10,
              'font.size':10,
              'axes.titlesize':8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'legend.borderpad': 0.2,
              'legend.linewidth': 0.1,
              'legend.loc': 'best',
              'legend.ncol': 4,
              'text.usetex': False,
              'figure.figsize': fig_size}
            rcParams.update(params)

            fig1 = plt.figure('results_transfer_syn')

            gs = matplotlib.gridspec.GridSpec(2, 3,
               width_ratios=[1,1,1],
               height_ratios=[1,1]
               )
            gs.update(bottom=0.30, top=0.94, left=0.06, right=0.97, wspace=0.3, hspace=0.4)
            ax1 = plt.subplot(gs[0:2,0])
            ax2 = plt.subplot(gs[0:2,1])

            ax3 = plt.subplot(gs[0,2])
            ax4 = plt.subplot(gs[1,2])

            gs2 = matplotlib.gridspec.GridSpec(1, 1,
               width_ratios=[1],
               height_ratios=[1]
               )
            gs2.update(bottom=0.1, top=0.2, left=0.08, right=0.98, wspace=0.3, hspace=0.4)
            ax13 = plt.subplot(gs2[0,0])

            #gs2 = matplotlib.gridspec.GridSpec(1, 2,
            #   width_ratios=[1,1],
            #   height_ratios=[1]
            #   )
            #gs2.update(bottom=0.1, top=0.3, left=0.10, right=0.96, wspace=0.3, hspace=0.4)
            #ax13 = plt.subplot(gs2[0,0])
            #ax14 = plt.subplot(gs2[0,1])

            #gs3 = matplotlib.gridspec.GridSpec(1, 3,
            #   width_ratios=[1,1,1],
            #   height_ratios=[1]
            #   )
            #gs3.update(bottom=0.06, top=0.29, left=0.1, right=0.97, wspace=0.4, hspace=0.4)
            #ax4 = plt.subplot(gs3[0,0])
            #ax5 = plt.subplot(gs3[0,1])
            ##ax6 = plt.subplot(gs3[0,2])

            font = font0.copy()
            font.set_family('sans-serif')
            font.set_weight('semibold')

            linewidth = 1.5


        if ("poster" not in pars.opt) and ("talk" not in pars.opt):

            x1 = -0.05
            y1 = 1.06
            ax1.text(x1, y1, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font) # 'A$\mathbf{\mathsf{_1}}$'
            ax2.text(x1, y1, 'A2', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
            ax3.text(x1, y1+0.09, 'A3', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)
            ax4.text(x1, y1+0.09, 'A4', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)

            ax13.text(-0.01, 1.3, 'B', transform=ax13.transAxes, fontsize=12, va='top', fontproperties=font)
            #ax14.text(-0.01, 1.1, 'C', transform=ax14.transAxes, fontsize=12, va='top', fontproperties=font)

            #ax4.text(x1, y1, 'D1', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)
            #ax5.text(x1, y1, 'D2', transform=ax5.transAxes, fontsize=12, va='top', fontproperties=font)
            #ax6.text(x1, y1, 'D3', transform=ax6.transAxes, fontsize=12, va='top', fontproperties=font)

    # FIGURE 10
    elif "fig8a" in pars.opt:

        fig_size =  [180*0.03937,130*0.03937] # 2-Column

        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'text.usetex': False,
          'figure.figsize': fig_size}
        rcParams.update(params)

        fig1 = plt.figure('results_transfer_syn')

        #gs4 = matplotlib.gridspec.GridSpec(1, 1,
        #   width_ratios=[1],
        #   height_ratios=[1]
        #   )
        #gs4.update(bottom=0.13, top=0.97, left=0.10, right=0.96, wspace=0.3, hspace=0.2)
        #ax13= plt.subplot(gs4[0,0])

        gs4 = matplotlib.gridspec.GridSpec(2, 1,
           width_ratios=[1],
           height_ratios=[1,1]
           )
        gs4.update(bottom=0.10, top=0.96, left=0.10, right=0.97, wspace=0.2, hspace=0.12)
        ax13= plt.subplot(gs4[0,0])
        ax14= plt.subplot(gs4[1,0])

        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('semibold')

        ax13.text(-0.01, 1.07, 'A', transform=ax13.transAxes, fontsize=12, va='top', fontproperties=font)
        ax14.text(-0.01, 1.07, 'B', transform=ax14.transAxes, fontsize=12, va='top', fontproperties=font)

        #if ("poster" not in pars.opt) and ("talk" not in pars.opt):
        #    x1 = -0.2
        #    y1 = 1.2
        #    ax13.text(x1, y1, 'A', transform=ax13.transAxes, fontsize=12, va='top', fontproperties=font)
        linewidth = 1.5

    # FIGURE 9
    elif "fig8b" in pars.opt:

        fig_size =  [180*0.03937,110*0.03937] # 1-Column 6.83

        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'text.usetex': False,
          'figure.figsize': fig_size}
        rcParams.update(params)

        fig1 = plt.figure('results_transfer_syn')

        gs1 = matplotlib.gridspec.GridSpec(2, 2,
           width_ratios=[1,1],
           height_ratios=[1,1]
           )
        gs1.update(bottom=0.71, top=0.94, left=0.07, right=0.65, wspace=0.3, hspace=0.1)
        #ax1 = plt.subplot(gs1[0:2,0])
        #ax2 = plt.subplot(gs1[0:2,1])
        ax3 = plt.subplot(gs1[0:2,0])
        ax4 = plt.subplot(gs1[0,1])
        ax4b = plt.subplot(gs1[1,1])


        gs2 = matplotlib.gridspec.GridSpec(2, 2,
           width_ratios=[1,1],
           height_ratios=[1,1]
           )
        gs2.update(bottom=0.4, top=0.64, left=0.07, right=0.65, wspace=0.3, hspace=0.1)
        #ax5 = plt.subplot(gs2[0:2,0])
        #ax6 = plt.subplot(gs2[0:2,1])
        ax7 = plt.subplot(gs2[0:2,0])
        ax8 = plt.subplot(gs2[0,1])
        ax8b = plt.subplot(gs2[1,1])


        gs3 = matplotlib.gridspec.GridSpec(2, 2,
           width_ratios=[1,1],
           height_ratios=[1,1]
           )
        gs3.update(bottom=0.09, top=0.33, left=0.07, right=0.65, wspace=0.3, hspace=0.1)
        #ax9 = plt.subplot(gs3[0:2,0])
        #ax10 = plt.subplot(gs3[0:2,1])
        ax11 = plt.subplot(gs3[0:2,0])
        ax12 = plt.subplot(gs3[0:2,1])

        #ax12 = plt.subplot(gs3[0,3])
        #ax12b = plt.subplot(gs3[1,3])


        gs4 = matplotlib.gridspec.GridSpec(2, 1,
           width_ratios=[1],
           height_ratios=[1.7,0.3]
           )
        gs4.update(bottom=0.46, top=0.94, left=0.73, right=0.98, wspace=0.3, hspace=0.2)
        ax14= plt.subplot(gs4[0,0])
        ax14b= plt.subplot(gs4[1,0])

        gs5 = matplotlib.gridspec.GridSpec(1, 1,
           width_ratios=[1],
           height_ratios=[1]
           )
        gs5.update(bottom=0.09, top=0.33, left=0.73, right=0.98, wspace=0.3, hspace=0.2)
        ax15= plt.subplot(gs5[0,0])

        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('semibold')


        if ("poster" not in pars.opt) and ("talk" not in pars.opt):

            x1 = -0.09
            y1 = 1.17
            #ax1.text(x1, y1, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font) # 'A$\mathbf{\mathsf{_1}}$'
            #ax2.text(x1, y1, 'A2', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
            ax3.text(x1, y1, 'A1', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)
            ax4.text(x1, y1+0.21, 'A2', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)

            #ax5.text(x1, y1, 'B1', transform=ax5.transAxes, fontsize=12, va='top', fontproperties=font) # 'A$\mathbf{\mathsf{_1}}$'
            #ax6.text(x1, y1, 'B2', transform=ax6.transAxes, fontsize=12, va='top', fontproperties=font)
            ax7.text(x1, y1, 'B1', transform=ax7.transAxes, fontsize=12, va='top', fontproperties=font)
            ax8.text(x1, y1+0.21, 'B2', transform=ax8.transAxes, fontsize=12, va='top', fontproperties=font)

            #ax9.text(x1, y1, 'C1', transform=ax9.transAxes, fontsize=12, va='top', fontproperties=font) # 'A$\mathbf{\mathsf{_1}}$'
            #ax10.text(x1, y1, 'C2', transform=ax10.transAxes, fontsize=12, va='top', fontproperties=font)
            ax11.text(x1, y1, "C1", transform=ax11.transAxes, fontsize=12, va='top', fontproperties=font)
            ax12.text(x1, y1, "C2", transform=ax12.transAxes, fontsize=12, va='top', fontproperties=font) # +0.21

            ax14.text(-0.09, 1.10, 'D', transform=ax14.transAxes, fontsize=12, va='top', fontproperties=font)
            ax15.text(-0.09, 1.16, 'E', transform=ax15.transAxes, fontsize=12, va='top', fontproperties=font)

        linewidth = 1.5


    elif "fig8" in pars.opt:

        fig_size =  [4.86,6.83] # 1.5-Column 6.83

        params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 10,
          'font.size':10,
          'axes.titlesize':8,
          'legend.fontsize': 8,
          'xtick.labelsize': 8,
          'ytick.labelsize': 8,
          'legend.borderpad': 0.2,
          'legend.linewidth': 0.1,
          'legend.loc': 'best',
          'legend.ncol': 4,
          'text.usetex': False,
          'figure.figsize': fig_size}
        rcParams.update(params)

        fig1 = plt.figure('results_transfer_syn')

        gs1 = matplotlib.gridspec.GridSpec(2, 4,
           width_ratios=[1,1,1,1],
           height_ratios=[1,1]
           )
        gs1.update(bottom=0.83, top=0.96, left=0.1, right=0.97, wspace=0.5, hspace=0.1)
        ax1 = plt.subplot(gs1[0:2,0])
        ax2 = plt.subplot(gs1[0:2,1])
        ax3 = plt.subplot(gs1[0:2,2])
        ax4 = plt.subplot(gs1[0,3])
        ax4b = plt.subplot(gs1[1,3])


        gs2 = matplotlib.gridspec.GridSpec(2, 4,
           width_ratios=[1,1,1,1],
           height_ratios=[1,1]
           )
        gs2.update(bottom=0.66, top=0.79, left=0.1, right=0.97, wspace=0.5, hspace=0.1)
        ax5 = plt.subplot(gs2[0:2,0])
        ax6 = plt.subplot(gs2[0:2,1])
        ax7 = plt.subplot(gs2[0:2,2])
        ax8 = plt.subplot(gs2[0,3])
        ax8b = plt.subplot(gs2[1,3])


        gs3 = matplotlib.gridspec.GridSpec(2, 4,
           width_ratios=[1,1,1,1],
           height_ratios=[1,1]
           )
        gs3.update(bottom=0.49, top=0.62, left=0.1, right=0.97, wspace=0.5, hspace=0.1)
        ax9 = plt.subplot(gs3[0:2,0])
        ax10 = plt.subplot(gs3[0:2,1])
        ax11 = plt.subplot(gs3[0:2,2])
        ax12 = plt.subplot(gs3[0:2,3])
        #ax12 = plt.subplot(gs3[0,3])
        #ax12b = plt.subplot(gs3[1,3])


        gs4 = matplotlib.gridspec.GridSpec(2, 2,
           width_ratios=[0.5,1],
           height_ratios=[1,0.5]
           )
        gs4.update(bottom=0.07, top=0.41, left=0.10, right=0.96, wspace=0.3, hspace=0.2)
        ax13= plt.subplot(gs4[0:2,1])
        ax14= plt.subplot(gs4[0,0])
        ax14b= plt.subplot(gs4[1,0])

        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('semibold')


        if ("poster" not in pars.opt) and ("talk" not in pars.opt):

            x1 = -0.2
            y1 = 1.2
            ax1.text(x1, y1, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font) # 'A$\mathbf{\mathsf{_1}}$'
            ax2.text(x1, y1, 'A2', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
            ax3.text(x1, y1, 'A3', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)
            ax4.text(x1, y1+0.25, 'A4', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)

            ax5.text(x1, y1, 'B1', transform=ax5.transAxes, fontsize=12, va='top', fontproperties=font) # 'A$\mathbf{\mathsf{_1}}$'
            ax6.text(x1, y1, 'B2', transform=ax6.transAxes, fontsize=12, va='top', fontproperties=font)
            ax7.text(x1, y1, 'B3', transform=ax7.transAxes, fontsize=12, va='top', fontproperties=font)
            ax8.text(x1, y1+0.25, 'B4', transform=ax8.transAxes, fontsize=12, va='top', fontproperties=font)

            ax9.text(x1, y1, 'C1', transform=ax9.transAxes, fontsize=12, va='top', fontproperties=font) # 'A$\mathbf{\mathsf{_1}}$'
            ax10.text(x1, y1, 'C2', transform=ax10.transAxes, fontsize=12, va='top', fontproperties=font)
            ax11.text(x1, y1, 'C3', transform=ax11.transAxes, fontsize=12, va='top', fontproperties=font)
            ax12.text(x1, y1+0.25, 'C4', transform=ax12.transAxes, fontsize=12, va='top', fontproperties=font)

            ax13.text(-0.03, 1.06, 'E', transform=ax13.transAxes, fontsize=12, va='top', fontproperties=font)
            ax14.text(-0.04, 1.06, 'D', transform=ax14.transAxes, fontsize=12, va='top', fontproperties=font)

    else:

        fig_size =  [6.83, 6.83] # 2-Column
        params['figure.figsize'] = fig_size
        rcParams.update(params)

        fig1 = plt.figure('results_transfer_syn')

        gs = matplotlib.gridspec.GridSpec(4, 4,
                                   width_ratios=[1,1,1,1],
                                   height_ratios=[1,1,1,1]
                                   )

        gs.update(bottom=0.1, top=0.93, left=0.1, right=0.95, wspace=0.6, hspace=0.6)


        if "_psd" in do_vec[0]:

            ax1 = plt.subplot(gs[0,0:2])
            ax3 = plt.subplot(gs[0,2:4])
            ax5 = plt.subplot(gs[1,0:2])
            ax7 = plt.subplot(gs[1,2:4])
            ax9 = plt.subplot(gs[2,0:2])
            ax11 = plt.subplot(gs[2,2:4])
            ax13 = plt.subplot(gs[3,0:2])
            ax15 = plt.subplot(gs[3,2:4])

        else:

            ax1 = plt.subplot(gs[0,0])
            ax2 = plt.subplot(gs[0,1])
            ax3 = plt.subplot(gs[0,2])
            ax4 = plt.subplot(gs[0,3])
            ax5 = plt.subplot(gs[1,0])
            ax6 = plt.subplot(gs[1,1])
            ax7 = plt.subplot(gs[1,2])
            ax8 = plt.subplot(gs[1,3])
            ax9 = plt.subplot(gs[2,0])
            ax10 = plt.subplot(gs[2,1])
            ax11 = plt.subplot(gs[2,2])
            ax12 = plt.subplot(gs[2,3])
            ax13 = plt.subplot(gs[3,0])
            ax14 = plt.subplot(gs[3,1])
            ax15 = plt.subplot(gs[3,2])
            ax16 = plt.subplot(gs[3,3])

        ax1.text(0.5, 1.5, r'best fit IF ($\mathrm{\mathsf{\tau}}$ = 27 ms)', transform=ax1.transAxes, fontsize=10, va='top')
        ax3.text(0.5, 1.5, 'GrC model', transform=ax3.transAxes, fontsize=10, va='top')


    if "fig7a" in pars.opt:

        fig_size =  [11.7*0.3937,7.4*0.3937]
        params = {'backend': 'ps',
                  'axes.labelsize': 6,
                  'axes.linewidth' : 0.5,
                  'title.fontsize': 8,
                  'text.fontsize': 8,
                  'font.size':6,
                  'axes.titlesize':6,
                  'legend.fontsize': 6,
                  'xtick.labelsize': 6,
                  'ytick.labelsize': 6,
                  'legend.borderpad': 0.2,
                  'legend.linewidth': 0.1,
                  'legend.loc': 3,
                  'legend.ncol': 4,
                  'text.usetex': False,
                  'figure.figsize': fig_size}
        rcParams.update(params)

        linewidth = 1

        plt.figure('results_CF-N')

        gs = matplotlib.gridspec.GridSpec(1, 1,
               width_ratios=[1],
               height_ratios=[1]
               )

        ax_CF_N = plt.subplot(gs[0,0])

        gs.update(bottom=0.15, top=0.93, left=0.15, right=0.95, wspace=0.6, hspace=0.6)

color_v = []
normalize = 1
prev_pos = 0
for d, do in enumerate(do_vec):

    prefix = ""
    if "_fig" in do:
        fig_num = str(do).split("_fig")[1].split("_")[0]
        prefix = prefix + "Fig" + str(fig_num) + "_"

    if "_pos" in do:
        pos = int(str(do).split("_pos")[1].split("_")[0][0])
        #print pos
        if prev_pos is not pos:
            color_v = []
        prev_pos = pos

    if "pop_" in do:

        prefix = prefix + "pop"

        ih = 40
        amod = [0.1]

        noise_syn_tau = [0*ms]
        noise_syn_tau_inh = [0*ms]

        N = [1]
        t_qual = 0

        bypass_cell = False

        method_interpol = np.array(['bin'])
        amp = 0 # absolute value

        fluct_g_e0 = [0]
        fluct_g_i0 = [0]
        fluct_std_e = [0]
        fluct_std_i = [0]
        fluct_std_e = [0]
        fluct_std_i = [0]
        fluct_tau_e = 0
        fluct_tau_i = 0

        ihold_sigma = [0*nA] # absolute value

        dt = 0.025*ms
        bin_width = dt
        jitter = 0*ms

        N = [1]

        Target_fit2 = []

        # CFo
        #color_vec = (np.array([color1, color6, color2, color9, color7, color8]), np.array([color1, color6, color2, color9, color7, color8]))
        #CFo_vec = array([20, 30, 40, 50, 60, 80])

        if "_transfer" in do:
            prefix = prefix + "_transfer"
        elif "_CF-N" in do:
            prefix = prefix + "_CF-N"
        elif "_CF-cutf" in do:
            prefix = prefix + "_CF-cutf"


        if "_if_" in do:

            prefix = prefix + "_if"

            thresh = -41.8
            R = 5227*MOhm
            #tau_passive = 3e-06*5227 = 15.7ms

            cellimport = []
            celltype = ["IfCell"]
            cell_exe = ["cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -71.5*mV)"]

            color = color2

            #if MPI.COMM_WORLD.rank == 0:
            #    ax01 = ax1
            #    if "_psd" not in do: ax02 = ax2

            istart = 0.002
            istop = 0.003
            di = 0.000001

            color_vec = (np.array([color0, color1, color2, color3, color4, color5, color6]), np.array([color0, color1, color2, color3, color4, color5, color6]))
            CFo_vec = array([-2, 20, 30, 40, 50, 60])

            plot_train = False

        if "_resif_" in do:

            prefix = prefix + "_resif"

            thresh = -21.175*mV
            gr = 6.044e-05*uS
            tau_r = 0.0185
            R = 8860*MOhm
            #tau_passive = 3e-06*8860 = 26.6ms

            #thresh: -21.1752 gr: 6.04400329156e-05 tau_r: 0.0185045932005 R: 8860

            cellimport = []
            celltype = "IfCell"
            cell_exe = "cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -60*mV, dgk =" + str(gr) + ", egk = -71.5*mV, ctau =" + str(tau_r) + ")"

            color = color2

            istart = 0.002
            istop = 0.003
            di = 0.000001

            color_vec = (np.array([color0, color1, color2, color3, color4, color5, color6]), np.array([color0, color1, color2, color3, color4, color5, color6]))
            CFo_vec = array([-2, 20, 30, 40, 50, 60])

        if "_grc" in do:

            prefix = prefix + "_grc"

            cellimport = ["from GRANULE_Cell import Grc"]
            celltype = ["Grc"]
            cell_exe = ["cell = Grc(np.array([0.,0.,0.]))"]

            color = color3

            istart = 0
            istop = 0.1
            di = 0.005

            color_vec = (np.array([color0, color1, color2, color3, color4, color5, color6]), np.array([color0, color1, color2, color3, color4, color5, color6]))
            CFo_vec = array([-2, 20, 30, 40, 50, 60])

            plot_train = False

        use_multisplit = False

        if "_prk_" in do:

            prefix = prefix + "_prk"

            cellimport = ["from Purkinje import Purkinje"]
            celltype = ["Prk"]
            cell_exe = ["cell = Purkinje()"]

            color = r1

            istart = 0
            istop = 0.2
            di = 0.01

            use_multisplit = True
            use_mpi = False

            color_vec = (np.array([color0, color1, color2, color3, color4, color5, color6]), np.array([color0, color1, color2, color3, color4, color5, color6]))
            CFo_vec = array([-2, 20, 30, 40, 50, 60])
            delay_baseline = 20

        if "_grcnoisy_" in do:

            prefix = prefix + "noisy"
            cell_exe = ["cell = Grc(np.array([0.,0.,0.]), lkg2_noise=3e-5, lkg2_gbar=6e-5)"]


        if "_grcvar_" in do:

            prefix = prefix + "var"
            cell_exe = ["cell = Grc(np.array([0.,0.,0.]), sigma_L = 0.1)"]


        Nstop = True
        Flip = True

        if "_fig" in do:

            color_vec = (np.array([color0, color1, color2, color6]), np.array([color0, color1, color2, color6]))
            CFo_vec = array([40])
            do_if = False


            if ("pop_CF-N_grc_syngr_cn_a1_nsyn4_noinh_varih" in do):

                x_vec = np.array([0.33,0.37,0.42,0.5,0.6,0.7])[::-1]
                N_vec = np.array([10,100,1000])

                x_vec_type = "ex"
                color_vec = (np.array([color0, color1, color6, color5, color2]), np.array([color0, color1, color6, color5, color2]))
                do_if = False
                Nstop = False


            if ("pop_CF-N_grc_syngr_excn_a1_nsyn4_noinh_varih" in do):

                x_vec = np.array([0.33,0.37,0.42,0.5,0.6,0.7])[::-1]
                N_vec = np.array([10,100,1000])

                x_vec_type = "ex"
                color_vec = (np.array([color0, color1, color6, color5, color2]), np.array([color0, color1, color6, color5, color2]))
                do_if = False
                Nstop = False


            if ("pop_CF-N_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn" in do):

                x_vec = np.array([2,4,6,10,20,40])
                N_vec = np.array([10,100,1000])

                x_vec_type = "inh"
                color_vec = (np.array([color2]), np.array([color2]))
                do_if = False
                Nstop = False


            if ("pop_CF-N_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lowinh" in do):
                x_vec = np.array([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 200, 300, 400, 500, 1000])
                N_vec = np.array([5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 400, 600, 800, 1000, 5000, 10000, 100000])

                x_vec_type = "inh"
                color_vec = (np.array([color2]), np.array([color2]))
                do_if = False
                Nstop = False


            if ("pop_CF-N_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_inhin" in do):

                x_vec = np.array([2,4,6,10,20,40])
                N_vec = np.array([10,100,1000])

                x_vec_type = "inhin"
                color_vec = (np.array([color2]), np.array([color2]))
                do_if = False
                Nstop = False


            if ("pop_CF-N_none_synno" in do):

                x_vec = np.array([1,10,20,30,40,50])[::-1]
                N_vec = np.array([10,100,1000])

                x_vec_type = "in"
                color_vec = (np.array([color0, color1, color6, color5, color2]), np.array([color0, color1, color6, color5, color2]))
                do_if = False
                Nstop = False


            if ("pop_CF-cutf_none_synno" in do):
                x_vec = np.array([1,2,5,10,15,20,25,30,35,40,45,50])[::-1]

                if ("_N100_" in do):
                    N_vec = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90])[::-1]
                else:
                    N_vec = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120])[::-1]

                if ("_a10_" in do):
                    if "_twopop_" in do:
                        N_vec = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130,132,134,136,138,140,142,146,148,150,152,154,156,158,160])[::-1]
                        x_vec = np.array([0.1,0.2,0.4,0.6,0.8,1,2,3,4,5,6])[::-1]

                x_vec_type = "in"
                color_vec = (np.array([color0, color1, color6, color5, color2]), np.array([color0, color1, color6, color5, color2]))
                do_if = False
                Nstop = True


            if ("pop_CF-cutf_grc_syngr_cn_a1_nsyn4_noinh_varih_" in do) or ("pop_CF-cutf_grc_syngr_cn_a1_nsyn4_noinh_lowvarih_" in do):

                x_vec = np.array([0.64])[::-1] #0.33,0.37,0.42,0.45,0.5,0.55,0.6,,0.7,0.75

                if ("_N100_" in do):
                    N_vec = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90])[::-1]
                else:
                    N_vec = np.array([20])[::-1] # 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19, ,21,22,23,24,25,26,27,28,29,30

                x_vec_type = "ex"
                color_vec = (np.array([color0, color1, color6, color5, color2]), np.array([color0, color1, color6, color5, color2]))
                do_if = False
                Nstop = True


            if ("pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_" in do) or ("pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_" in do):

                x_vec = np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,32,34,36,38,40])

                if ("_N100_" in do):
                    N_vec = np.array([0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,44,46,48,50,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70])[::-1]
                else:
                    N_vec = np.array([0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])[::-1]

                if "_twopop_" in do:
                    N_vec = np.array([0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,44,46,48,50,52,54,56,60,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90])[::-1]
                    x_vec = np.array([4,5,6,7,8,9,10,11,12,13,14,14.2,14.4,14.6,14.8,15,16,17,18,19,20,22,24,26,28,30,32])

                x_vec_type = "inh"
                color_vec = (np.array([color0, color1, color6, color5, color2]), np.array([color0, color1, color6, color5, color2]))
                do_if = False
                Nstop = True


            if ("pop_CF-cutf_if_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_" in do) or ("pop_CF-cutf_if_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_" in do):

                x_vec = np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,32,34,36,38,40])

                if ("_N100_" in do):
                    N_vec = np.array([0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,44,46,48,50,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100])[::-1]
                else:
                    N_vec = np.array([0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])[::-1]

                if "_twopop_" in do:
                    N_vec = np.array([0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,44,46,48,50,52,54,56,60,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126,128,130])[::-1]
                    x_vec = np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,32,34,36,38,40])

                x_vec_type = "inh"
                color_vec = (np.array([color0, color1, color6, color5, color2]), np.array([color0, color1, color6, color5, color2]))
                do_if = False
                Nstop = True


            if ("pop_CF-cutf_grc_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_" in do) or ("pop_CF-cutf_if_syngr_cn_adjfinh_a10_nsyn4_varih_varinhn_" in do) or ("pop_CF-cutf_grc_syngr_cn_adjfinh_a10_nsyn4_lowvarih_lowvarinhn_" in do) or ("pop_CF-cutf_if_syngr_cn_adjfinh_a10_nsyn4_lowvarih_lowvarinhn_" in do):

                N_vec = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])[::-1]
                x_vec = np.array([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,32,34,36,38,40])

                x_vec_type = "inh"
                color_vec = (np.array([color0, color1, color6, color5, color2]), np.array([color0, color1, color6, color5, color2]))
                do_if = False
                Nstop = True


            if ("pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_varih_varinhn_lowinh_" in do) or ("pop_CF-cutf_grc_syngr_cn_adjfinh_a1_nsyn4_lowvarih_lowvarinhn_lowinh_" in do):

                x_vec = np.array([1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 200, 300, 400, 500, 1000])
                N_vec = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])[::-1]

                x_vec_type = "inh"
                color_vec = (np.array([color0, color1, color6, color5, color2]), np.array([color0, color1, color6, color5, color2]))
                do_if = False
                Nstop = True


            if ("pop_CF-cutf_grc_syngr_cn_adjinh_a1_nsyn4_varih_" in do) or ("pop_CF-cutf_if_syngr_cn_adjinh_a1_nsyn4_varih_" in do):

                x_vec = np.arange(0.4,0.85,0.02)
                N_vec = np.array([0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72])[::-1]
                # N_vec = np.array([0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,44,46,48,50,52,54,56,60,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90])[::-1]

                if "_if_" in do:
                    x_vec = np.arange(0.5,0.9,0.02)

                if "_twopop_" in do:
                    N_vec = np.array([0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,44,46,48,50,52,54,56,58,60])[::-1]

                    if "_if_" in do:
                        N_vec = np.array([0.7,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,42,44,46,48,50,52,54,56,58,60])[::-1]
                        x_vec = np.arange(0.6,1,0.02)

                x_vec_type = "constinh"
                color_vec = (np.array([color0, color1, color6, color5, color2]), np.array([color0, color1, color6, color5, color2]))
                do_if = False
                Nstop = True




        if "_none_" in do:

            prefix = prefix + "_none"
            plot_train = False

            t_qual = 6
            thresh = -21.175*mV
            R = 8860*MOhm
            #tau_passive = 3e-06*8860 = 26.6ms

            cellimport = []
            celltype = ["IfCell"]
            cell_exe = ["cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -60*mV)"]

            color = color1

            temperature = 37

            #if MPI.COMM_WORLD.rank == 0:

           #     if "_syn" in do: ax01 = ax5; ax02 = ax6
           #     if "_syngr" in do: ax01 = ax7; ax02 = ax8
           #     if "_synno" in do: ax01 = ax1; ax02 = ax2

            istart = 0.002
            istop = 0.003
            di = 0.000001

            method_interpol = np.array(['gsyn_in'])
            CFo_vec = array([40])

            if "_dashed" in do: # make dashed always black!!!
                color_vec = (np.array([color0]), np.array([color0]))
            else:
                color_vec = (np.array([color1]), np.array([color1]))


        if "_syn" in do:

            prefix = prefix + "_syn"

            n_syn_ex = [1]

            g_syn_ex = [0]

            noise_syn = [0]
            noise_syn_inh = [0]

            freq_used = np.array([])

            inh_hold = [0]
            n_syn_inh = [0]
            g_syn_inh = [1]

            tau1_inh = ['gr']
            tau2_inh = ['gr']

            if "_syngr_" in do:
                tau1_ex=['gr']
                tau2_ex=['gr']
                prefix = prefix + "gr"

            elif "_synprk_" in do:
                tau1_ex=['prk']
                tau2_ex=['prk']
                prefix = prefix + "prk"

            elif "_syngrstoch_" in do:
                tau1_ex=['grstoch']
                tau2_ex=['grstoch']
                prefix = prefix + "grstoch"

            elif "_synnostdgr_" in do:
                tau1_ex=['nostdgr']
                tau2_ex=['nostdgr']
                prefix = prefix + "nostdgr"

            elif "_synnomggr_" in do:
                tau1_ex=['nomggr']
                tau2_ex=['nomggr']
                prefix = prefix + "nomggr"

            elif "_synampagr_" in do:
                tau1_ex=['ampagr']
                tau2_ex=['ampagr']
                prefix = prefix + "ampagr"

            elif "_synampagrnopre_" in do:
                tau1_ex=['ampagrnopre']
                tau2_ex=['ampagrnopre']
                prefix = prefix + "ampagrnopre"

            elif "_synampagrstoch_" in do:
                tau1_ex=['ampagrstoch']
                tau2_ex=['ampagrstoch']
                prefix = prefix + "ampagrstoch"

            elif "_synnostdampagr_" in do:
                tau1_ex=['nostdampagr']
                tau2_ex=['nostdampagr']
                prefix = prefix + "nostdampagr"

            elif "_synnmdagr_" in do:
                tau1_ex=['nmdagr']
                tau2_ex=['nmdagr']
                prefix = prefix + "nmdagr"

            elif "_synnmdagrnopre_" in do:
                tau1_ex=['nmdagrnopre']
                tau2_ex=['nmdagrnopre']
                prefix = prefix + "nmdagrnopre"

            elif "_synnmdagrstoch_" in do:
                tau1_ex=['nmdagrstoch']
                tau2_ex=['nmdagrstoch']
                prefix = prefix + "nmdagrstoch"

            elif "_syngabagr_" in do:
                tau1_ex=['gabagr']
                tau2_ex=['gabagr']
                prefix = prefix + "gabagr"

            elif "_syngabagrstoch_" in do:
                tau1_ex=['gabagrstoch']
                tau2_ex=['gabagrstoch']
                prefix = prefix + "gabagrstoch"

            elif "_synnodirect_" in do:
                method_interpol = np.array(['bin'])
                CFo_vec = array([-1])
                g_syn_ex = [-1]
                tau1_ex=[1*ms]
                tau2_ex=[1*ms]
                prefix = prefix + "nodirect"

            elif "_synno_" in do:
                method_interpol = np.array(['bin'])
                CFo_vec = array([-1])
                g_syn_ex = [-1]
                tau1_ex=[0*ms]
                tau2_ex=[0*ms]
                prefix = prefix + "no"

            elif "_synfit_" in do:
                tau1_ex=['fit']
                tau2_ex=['fit']
                prefix = prefix + "fit"

            elif "_syntriexp_" in do:
                tau1_ex=['triexp']
                tau2_ex=['triexp']
                prefix = prefix + "triexp"

            elif "_synnmfit_" in do:
                tau1_ex=['nmfit']
                tau2_ex=['nmfit']
                prefix = prefix + "nmfit"

            elif "_synamfit_" in do:
                tau1_ex=['amfit']
                tau2_ex=['amfit']
                prefix = prefix + "amfit"

            elif "_synnmfitnomg_" in do:
                tau1_ex=[4.394*ms]
                tau2_ex=[4.394*ms]
                prefix = prefix + "nmfitnomg"

            elif "_synlong_" in do:
                tau1_ex=[100*ms]
                tau2_ex=[100*ms]
                prefix = prefix + "long"

            elif "_synshort_" in do:
                tau1_ex=[1*ms]
                tau2_ex=[1*ms]
                prefix = prefix + "short"

            elif "_synnewfit_" in do:
                tau1_ex = [1*ms]
                tau2_ex = [1*ms]
                prefix = prefix + "newfit"
                tau1_inh = [10*ms]
                tau2_inh = [10*ms]

            elif "_syngrfastinh_" in do:
                tau1_ex=['gr']
                tau2_ex=['gr']
                prefix = prefix + "grfastinh"
                tau1_inh = [0*ms]
                tau2_inh = [10*ms]

            if "_nsyn" in do:
                nsyn = str(do).split("_nsyn")[1].split("_")[0]
                prefix = prefix + "_nsyn" + str(int(nsyn))
                n_syn_ex = [int(nsyn)]


        if "_ssine_" in do:

            prefix = prefix + "_ssine"
            CFo_vec = array([-1, 50])
            bin_width = 0.025*ms
            N = [1]

        onf = None


        if "_bn_" in do:

            onf = 10
            cutf = 20
            xmax = 20

            sexp = -1
            prefix = prefix + "_bn"


        if "_cn_" in do:

            xmax = 20
            cutf = 20
            sexp = -1
            prefix = prefix + "_cn"


        if "_excn_" in do:

            xmax = 20
            cutf = 20
            sexp = 4
            prefix = prefix + "_excn"


        if "_wn_" in do:

            xmax = 100
            cutf = 0
            sexp = 0
            prefix = prefix + "_wn"


        if "_cutf" in do:

            sexp = -1
            cutf = float(str(do).split("_cutf")[1].split("_")[0])
            xmax = cutf
            prefix = prefix + "_cutf" + str(int(cutf))


        if "_a00_" in do:

            amod = [0]
            prefix = prefix + "_a00"


        give_psd = False
        if "_psd_" in do:

            give_psd = True
            prefix = prefix + "_psd"
            CFo_vec = array([-1, 50])


        if "_fft_" in do:
            CFo_vec = array([-1, 50])


        if "_a01_" in do:

            amod = [0.1]
            prefix = prefix + "_a01"

            if MPI.COMM_WORLD.rank == 0:

                if "_fig" not in do:

                    if "_if" in do:
                        ax01 = ax1
                        if "_psd" not in do: ax02 = ax2
                    if "_grc" in do:
                        ax01 = ax3
                        if "_psd" not in do: ax02 = ax3
                    if "_resif" in do:
                        ax01 = ax1
                        if "_psd" not in do: ax02 = ax2



        if "_a02_" in do:

            amod = [0.2]
            prefix = prefix + "_a02"

            if MPI.COMM_WORLD.rank == 0:

                if "_fig" not in do:

                    if "_if" in do:
                        ax01 = ax5
                        if "_psd" not in do: ax02 = ax6
                    if "_grc" in do:
                        ax01 = ax7
                        if "_psd" not in do: ax02 = ax8
                    if "_resif" in do:
                        ax01 = ax5
                        if "_psd" not in do: ax02 = ax6


        if "_a05_" in do:

            amod = [0.5]
            prefix = prefix + "_a05"

            if MPI.COMM_WORLD.rank == 0:

                if "_fig" not in do:

                    if "_if" in do:
                        ax01 = ax9
                        if "_psd" not in do: ax02 = ax10
                    if "_grc" in do:
                        ax01 = ax11
                        if "_psd" not in do: ax02 = ax12
                    if "_resif" in do:
                        ax01 = ax9
                        if "_psd" not in do: ax02 = ax10



        if "_a1_" in do:

            amod = [1]
            prefix = prefix + "_a1"

            if MPI.COMM_WORLD.rank == 0:

                if "_fig" not in do:

                    if "_if" in do:
                        ax01 = ax13
                        if "_psd" not in do: ax02 = ax14
                    if "_grc" in do:
                        ax01 = ax15
                        if "_psd" not in do: ax02 = ax16
                    if "_resif" in do:
                        ax01 = ax13
                        if "_psd" not in do: ax02 = ax14

        if "_a10_" in do:

            amod = [10]
            prefix = prefix + "_a10"


        if "_a3_" in do:

            amod = [3]
            prefix = prefix + "_a3"


        if "_a2_" in do:

            amod = [2]
            prefix = prefix + "_a2"


        if "_a15_" in do:

            amod = [1.5]
            prefix = prefix + "_a15"

        if "_fig8" in do:

            if MPI.COMM_WORLD.rank == 0:

                    if "_pos1b_" in do:
                        if "_notransf_" in do:
                            ax01 = None
                            ax01b = None
                        else:
                            ax01 = ax1
                            ax01b = ax2
                        ax02 = ax3
                        ax03 = ax4b

                    elif "_pos1_" in do:
                        if "_notransf_" in do:
                            ax01 = None
                            ax01b = None
                        else:
                            ax01 = ax1
                            ax01b = ax2
                        ax02 = ax3
                        ax03 = ax4

                    elif ("_pos2b_" in do):
                        if "_notransf_" in do:
                            ax01 = None
                            ax01b = None
                        else:
                            ax01 = ax5
                            ax01b = ax6
                        ax02 = ax7
                        ax03 = ax8b

                    elif ("_pos2_" in do):
                        if "_notransf_" in do:
                            ax01 = None
                            ax01b = None
                        else:
                            ax01 = ax5
                            ax01b = ax6
                        ax02 = ax7
                        ax03 = ax8

                    elif ("_pos3b_" in do):
                        if "_notransf_" in do:
                            ax01 = None
                            ax01b = None
                        else:
                            ax01 = ax9
                            ax01b = ax10
                        ax02 = ax11
                        ax03 = ax12b

                    elif ("_pos3_" in do):
                        if "_notransf_" in do:
                            ax01 = None
                            ax01b = None
                        else:
                            ax01 = ax9
                            ax01b = ax10
                        ax02 = ax11
                        ax03 = ax12

                    elif "_pos5" in do:
                        ax01 = None
                        ax01b = None
                        ax02 = ax15
                        ax03 = None


        elif "_fig" in do:

            if MPI.COMM_WORLD.rank == 0:

                    if "_pos14" in do:
                        ax01 = ax14

                    elif "_pos1b" in do:
                        ax01 = None
                        ax01b = None
                        ax02 = ax4

                    elif "_pos1" in do:
                        if "_notransf_" in do:
                            ax01 = None
                            ax01b = None
                        else:
                            ax01 = ax1
                            ax01b = ax2
                        ax02 = ax3


                    elif ("_pos2" in do):
                        if "_notransf_" in do:
                            ax01 = None
                            ax01b = None
                        else:
                            ax01 = ax4
                            ax01b = ax5
                        ax02 = ax6

                    elif "_pos3" in do:
                        if "_notransf_" in do:
                            ax01 = None
                            ax01b = None
                        else:
                            ax01 = ax7
                            ax01b = ax8
                        ax02 = ax9

                    elif "_pos4" in do:
                        ax01 = ax10
                        ax01b = ax11
                        ax02 = ax12


                    elif "_CF-" in do:
                        ax01 = ax1
                        ax01b = ax2
                        ax02 = ax3
                        ax04 = ax7

                    else:
                        ax01 = ax1
                        ax01b = ax2
                        ax02 = ax3


        noise_a = [1e9]
        noise_a_inh = [1e9]

        noise_syn = [0.01]
        noise_syn_tau = [-1]

        noise_syn_inh = [0.01]
        noise_syn_tau_inh = [-1]


        if "_noisesynlow_" in do: # CV: 0.003, FANO: 0.0025

            noise_a = [1e9]
            #noise_a_inh = [1e9]

            noise_syn = [0.01]
            noise_syn_tau = [-1]

            #noise_syn_inh = [0.01]
            #noise_syn_tau_inh = [-1]

            prefix = prefix + "_noisesynlow"


        if "_noisesynmed_" in do: # CV: 0.1, FANO: 0.025

            noise_a = [1e9]
            #noise_a_inh = [1e9]

            noise_syn = [0.5]
            noise_syn_tau = [-1]

            #noise_syn_inh = [0.5]
            #noise_syn_tau_inh = [-1]

            prefix = prefix + "_noisesynmed"


        if "_noisesynhigh_" in do: # CV: 0.1, FANO: 0.025

            noise_a = [1e9]
            #noise_a_inh = [1e9]

            noise_syn = [10]
            noise_syn_tau = [-1]

            #noise_syn_inh = [10]
            #noise_syn_tau_inh = [-1]

            prefix = prefix + "_noisesynhigh"


        if "_noisesyngamma_" in do: # CV: 0.1, FANO: 0.025

            noise_a = [100]
            #noise_a_inh = [100]

            noise_syn = [0]
            noise_syn_tau = [0]

            #noise_syn_inh = [0]
            #noise_syn_tau_inh = [0]

            prefix = prefix + "_noisesyngamma"


        if "_noisesynpoiss_" in do: # CV: 1, FANO: 1

            noise_a = [1]
            #noise_a_inh = [1]

            noise_syn = [0]
            noise_syn_tau = [0]

            #noise_syn_inh = [0]
            #noise_syn_tau_inh = [0]

            prefix = prefix + "_noisesynpoiss"


        if "_inhpoiss_" in do: # CV: 1, FANO: 1

            noise_a_inh = [1]

            noise_syn_inh = [0]
            noise_syn_tau_inh = [0]

            prefix = prefix + "_inhpoiss"


        if "_inhmed_" in do: # CV: 1, FANO: 1

            noise_a_inh = [1e9]

            noise_syn_inh = [0.5]
            noise_syn_tau_inh = [-1]

            prefix = prefix + "_inhmed"


        if "_inhlow_" in do: # CV: 0.003, FANO: 0.0025

            noise_a_inh = [1e9]

            noise_syn_inh = [0.01]
            noise_syn_tau_inh = [-1]

            prefix = prefix + "_inhlow"


        if "_noisesyn10ex_" in do:

            noise_syn = [0.5] #int(40*0.5)
            noise_syn_tau = [10*ms]
            prefix = prefix + "_noisesyn10ex"


        if "_noisesyn10inh_" in do:

            noise_syn_inh = [0.5] #2
            noise_syn_tau_inh = [10*ms]
            prefix = prefix + "_noisesyn10inh"


        if "_noisesyn0_" in do:

            noise_syn = [10]
            noise_syn_tau = [0*ms]
            prefix = prefix + "_noisesyn0"


        if "_addn10_" in do:

            fluct_g_e0 = [0]
            fluct_g_i0 = [0.0002*uS]
            fluct_std_e = [0]
            fluct_std_i = [0.0001*uS]

            fluct_tau_e = 10*ms
            fluct_tau_i = 10*ms

            prefix = prefix + "_addn10"


        if "_addn10b_" in do:

            fluct_g_e0 = [0]
            fluct_g_i0 = [0.0002*uS]
            fluct_std_e = [0]
            fluct_std_i = [0.0001*uS]

            fluct_tau_e = 100*ms
            fluct_tau_i = 100*ms

            prefix = prefix + "b"


        if "_addn10c_" in do:

            fluct_g_e0 = [0]
            fluct_g_i0 = [0.0004*uS]
            fluct_std_e = [0]
            fluct_std_i = [0.0002*uS]

            fluct_tau_e = 1000*ms
            fluct_tau_i = 1000*ms

            prefix = prefix + "c"


        if "_poissex_" in do:

            noise_syn = 1000
            prefix = prefix + "_poissex"

        if "_poissinh_" in do:

            noise_syn_inh = 1000
            prefix = prefix + "_poissinh"

        g_syn_ex_s = [0]
        g_syn_inh_s = [0]

        if "_varg_" in do:
            g_syn_ex_s = [0.25/2]
            g_syn_inh_s = [0.25/2]
            prefix = prefix + "_varg"

        if "_highvarg_" in do:
            g_syn_ex_s = [0.5/2]
            g_syn_inh_s = [0.5/2]
            prefix = prefix + "_highvarg"


        adjinh = False
        if "_adjinh_" in do:

            g_syn_ex = [1]
            fluct_g_i0 = [0]
            prefix = prefix + "_adjinh"
            adjinh = True


        adjfinh = False
        if "_adjfinh_" in do:

            g_syn_ex = [1]
            n_syn_inh = [4]
            g_syn_inh = [1]
            prefix = prefix + "_adjfinh"
            adjfinh = True


        if "_ninhsyn" in do:
            nsyninh = str(do).split("_ninhsyn")[1].split("_")[0]
            prefix = prefix + "_ninhsyn" + str(int(nsyninh))
            n_syn_inh = [int(nsyninh)]


        if "_color" in do:
            nc = str(do).split("_color")[1].split("_")[0]
            #exec 'color_vec = (np.array(['+ str(nc) +']), np.array(['+ str(nc) +']))'
            exec 'color = ' + str(nc)

        CF_var = [[5,10,20]]
        CF_var = False

        give_freq = False

        if "_ih" in do:
            ih = str(do).split("_ih")[1].split("_")[0]
            prefix = prefix + "_ih" + ih
            ih = float(ih)

        ihold = [ih]

        if "_CFo" in do:
            cf = str(do).split("_CFo")[1].split("_")[0]
            CFo_vec = np.array([float(cf)])

        if "_gsex" in do:
            cf = str(do).split("_gsex")[1].split("_")[0]
            prefix = prefix + "_gsex" + cf
            g_syn_ex = [float(cf)]

        if "_gsinh" in do:
            cf = str(do).split("_gsinh")[1].split("_")[0]
            prefix = prefix + "_gsinh" + cf
            g_syn_inh = [float(cf)]

        syn_max_mf = [1] # possible mossy fibres per synapse
        syn_max_inh = [1] # possible Golgi cells per synapse

        sigma_ihold = [0.]
        sigma_inh_hold = [0.]

        if "_varih_" in do:
            sigma_ihold = [0.5/2]
            prefix = prefix + "_varih"

        if "_lowvarih_" in do:
            sigma_ihold = [0.2/2]
            prefix = prefix + "_lowvarih"

        if "_highvarih_" in do:
            sigma_ihold = [0.7/2]
            prefix = prefix + "_highvarih"

        if "_highervarih_" in do:
            sigma_ihold = [1.5/2]
            prefix = prefix + "_highvarih"

        if "_unifih_" in do:
            sigma_ihold = [-1.]
            prefix = prefix + "_unifih"

        if "_varinhn_" in do:
            sigma_inh_hold = [0.5/2]     # 1
            prefix = prefix + "_varinhn"

        if "_lowvarinhn_" in do:
            sigma_inh_hold = [0.2/2]     # 1
            prefix = prefix + "_lowvarinhn"

        if "_N" in do:
            N = str(do).split("_N")[1].split("_")[0]
            prefix = prefix + "_N" + str(int(N))
            N = [int(N)]


        if "_mf" in do:
            mf = str(do).split("_mf")[1].split("_")[0]
            prefix = prefix + "_mf" + str(int(mf))
            syn_max_mf = [int(mf)]
        else:
            syn_max_mf = N # possible mossy fibres per synapse

        if "_gc" in do:
            gc = str(do).split("_gc")[1].split("_")[0]
            prefix = prefix + "_gc" + str(int(gc))
            syn_max_inh = [int(gc)]
        else:
            syn_max_inh = N # possible mossy fibres per synapse

        if "_is" in do:
            is0 = str(do).split("_is")[1].split("_")[0]
            prefix = prefix + "_is" + str(is0)

            fluct_g_e0 = [0]
            fluct_g_i0 = [float(is0)*nS]
            fluct_std_e = [0]
            fluct_std_i = [0]

            fluct_tau_e = 0*ms
            fluct_tau_i = 0*ms

        if "_ddown" in do:
            d_down = int( str(do).split("_ddown")[1].split("_")[0] )

        syn_ex_dist = []
        syn_inh_dist = []
        inh_factor = [1]

        if "_lincomb2_" in do:
            syn_ex_dist = [[1,2,2,2]]
            prefix = prefix + "_lincomb2"

        if "_lincomb2c_" in do:
            syn_ex_dist = [[1,2,2,2]]
            prefix = prefix + "_lincomb2c"
            inh_factor = [0,1]

        if "_lincomb3_" in do:
            syn_inh_dist = [[0,0,0,0]]
            syn_ex_dist = [[1,0,0,0]]
            prefix = prefix + "_lincomb3"

        if "_lincomb4_" in do:
            syn_inh_dist = [[0,0,0,0]]
            syn_ex_dist = [[1,2,3,4]]
            prefix = prefix + "_lincomb4"

        if "_lincomb_" in do:
            syn_ex_dist = [[1,1,2,2]]
            prefix = prefix + "_lincomb"

        if "_inhin_" in do:
            syn_inh_dist = [[1,1,1,1]]
            syn_ex_dist = [[0,0,0,0]]
            prefix = prefix + "_inhin"
            #normalize = 3.56134468312

        if "_bothin_" in do:
            syn_inh_dist = [[1,1,1,1]]
            syn_ex_dist = [[2,2,2,2]]
            prefix = prefix + "_bothin"
            inh_factor = [-1,1]

        inh_delay = 0
        if "_samein" in do:
            syn_inh_dist = [[1,1,1,1]]
            syn_ex_dist = [[1,1,1,1]]
            prefix = prefix + "_samein"

            cf = str(do).split("_samein")[1].split("_")[0]
            if len(cf) > 0:
                prefix = prefix + cf
                inh_delay = float(cf)
                print "inh_delay:", inh_delay, "ms"

        if "_lowinh_" in do:
            #syn_inh_dist = [[0,0,0,0]]
            #syn_ex_dist = [[1,1,1,1]]
            g_syn_inh = [0.45] #0.46
            prefix = prefix + "_lowinh"


        linestyle = "-"
        if "_dashed_" in do:
            linestyle = "--"
        if "_dotted_" in do:
            linestyle = ":"
        if "_dashdot_" in do:
            linestyle = "-."

        markerstyle = "o"
        ms2=3
        if "_star_" in do:
            markerstyle = "^"
            ms2=4

        if "_keep_" in do:
            do_run_now = 0
        else:
            do_run_now = do_run

        a_celltype = [0]
        factor_celltype = [1]

        #if "_if_" in do:
        #    g_syn_ex = [0.65]
        #    g_syn_inh = [0.65]

        if "_twopop_" in do:

            prefix = prefix + "_twopop"
            if cellimport != []:
                cellimport = [cellimport[0],cellimport[0]]

            celltype = [celltype[0],celltype[0]]
            cell_exe = [cell_exe[0],cell_exe[0]]
            N = [N[0],N[0]]
            amod = [amod[0],amod[0]]

            a_celltype = [0,1]     # celltype to analyse
            factor_celltype = [1,-1]

            ihold = [ihold[0],ihold[0]]
            ihold_sigma = [ihold_sigma[0],ihold_sigma[0]]
            sigma_ihold = [sigma_ihold[0],sigma_ihold[0]]
            n_syn_ex = [n_syn_ex[0],n_syn_ex[0]]
            n_syn_inh = [n_syn_inh[0],n_syn_inh[0]]

            fluct_g_e0 = [fluct_g_e0[0], fluct_g_e0[0]]
            fluct_g_i0 = [fluct_g_i0[0],fluct_g_i0[0]]
            fluct_std_e = [fluct_std_e[0], fluct_std_e[0]]
            fluct_std_i = [fluct_std_i[0], fluct_std_i[0]]

            tau1_ex = [tau1_ex[0], tau1_ex[0]]
            tau2_ex = [tau2_ex[0], tau2_ex[0]]
            tau1_inh = [tau1_inh[0], tau1_inh[0]]
            tau2_inh = [tau2_inh[0], tau2_inh[0]]

            g_syn_ex = [g_syn_ex[0], g_syn_ex[0]]
            g_syn_ex_s = [g_syn_ex_s[0], g_syn_ex_s[0]]

            noise_syn = [noise_syn[0], noise_syn[0]]
            noise_syn_tau = [noise_syn_tau[0], noise_syn_tau[0]]
            noise_syn_inh = [noise_syn_inh[0], noise_syn_inh[0]]
            noise_syn_tau_inh = [noise_syn_tau_inh[0], noise_syn_tau_inh[0]]
            noise_a = [noise_a[0], noise_a[0]]
            noise_a_inh = [noise_a_inh[0], noise_a_inh[0]]

            inh_hold = [inh_hold[0], inh_hold[0]]

            g_syn_inh = [g_syn_inh[0], g_syn_inh[0]]
            g_syn_inh_s = [g_syn_inh_s[0], g_syn_inh_s[0]]

            syn_max_mf = [syn_max_mf[0],syn_max_mf[0]]
            syn_max_inh = [syn_max_inh[0], syn_max_inh[0]]
            sigma_inh_hold = [sigma_inh_hold[0], sigma_inh_hold[0]]

            #print syn_ex_dist
            #print syn_inh_dist

            if len(syn_ex_dist) > 0:
                syn_ex_dist = [syn_ex_dist[0], syn_ex_dist[0]]
                syn_inh_dist = [syn_inh_dist[0], syn_inh_dist[0]]
            #else:
            #    syn_ex_dist = [] #[[1,1,1,1],[1,1,1,1]]
            #    syn_inh_dist = [] #[[0,0,0,0],[0,0,0,0]]


        if "_plotre_" in do:
            t_qual = 6

        if "_transfer_" in do:

            for i, CFo in enumerate(CFo_vec):

                prefix = prefix + "_CFo" + str(CFo)

                #print pickle_prefix

                pop = Population(cellimport = cellimport,  celltype = celltype, cell_exe = cell_exe, N = N, temperature = temperature, ihold = ihold, ihold_sigma = ihold_sigma, amp = amp, amod = amod, give_freq = give_freq, do_run = do_run_now, pickle_prefix = prefix, istart = istart, istop = istop, di = di, dt = dt)

                pop.use_mpi = use_mpi
                pop.use_multisplit = use_multisplit

                pop.bin_width = float(bin_width)
                pop.jitter = float(jitter)

                pop.method_interpol = method_interpol[:]
                pop.no_fmean = False

                pop.fluct_g_e0 = fluct_g_e0[:]
                pop.fluct_g_i0 = fluct_g_i0[:]
                pop.fluct_std_e = fluct_std_e[:]
                pop.fluct_std_i = fluct_std_i[:]
                pop.fluct_tau_e = float(fluct_tau_e)
                pop.fluct_tau_i = float(fluct_tau_i)

                pop.CF_var = CF_var
                pop.xmax = xmax

                pop.tau1_ex=tau1_ex[:]
                pop.tau2_ex=tau2_ex[:]
                pop.tau1_inh=tau1_inh[:]
                pop.tau2_inh=tau2_inh[:]

                pop.n_syn_ex = n_syn_ex[:]
                pop.g_syn_ex = g_syn_ex[:]
                pop.g_syn_ex_s = g_syn_ex_s[:]

                pop.noise_syn = noise_syn[:]
                pop.noise_syn_tau = noise_syn_tau[:]
                pop.noise_a = noise_a[:]
                pop.noise_a_inh = noise_a_inh[:]

                pop.noise_syn_inh = noise_syn_inh[:]
                pop.noise_syn_tau_inh = noise_syn_tau_inh[:]
                pop.inh_hold = inh_hold[:]

                pop.n_syn_inh = n_syn_inh[:]
                pop.g_syn_inh_s = g_syn_inh_s[:]

                pop.g_syn_inh = g_syn_inh[:]

                pop.bypass_cell = bypass_cell

                pop.do_if = do_if
                pop.adjinh = adjinh
                pop.adjfinh = adjfinh

                pop.syn_max_mf = syn_max_mf[:]
                pop.syn_max_inh = syn_max_inh[:]
                pop.inh_hold_sigma = sigma_inh_hold[:] #[float(sigma_inh_hold)]
                pop.ihold_sigma = sigma_ihold[:] #[float(sigma_ihold)]

                pop.syn_ex_dist = syn_ex_dist[:]
                pop.syn_inh_dist = syn_inh_dist[:]
                pop.simstep = simstep

                pop.plot_train = plot_train
                pop.inh_delay = inh_delay
                pop.a_celltype = a_celltype
                pop.factor_celltype = factor_celltype

                pop.delay_baseline = delay_baseline
                pop.data_dir = data_dir
                pop.minimal_dir = minimal_dir

                if adjfinh:
                    pop.inh_hold = [CFo]*len(N)
                    if CFo == 0: pop.n_syn_inh = [0]*len(N)

                elif adjinh:
                    pop.fluct_g_i0 = [CFo]*len(N)

                else:
                    pop.g_syn_ex = [CFo]*len(N)



                pop.tmax = 10*s # return full length of signal
                if "_plotv" in do:
                    te = str(do)
                    pop.tmax = float(te.split("_plotv")[1][0:2])*s

                pop.give_psd = give_psd


                if "_ssine" in do:
                    freq_used = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])*Hz
                    freq_used = np.array([4])*Hz
                    results = pop.fun_ssine_Stim(freq_used = freq_used)
                else:
                    results = pop.fun_cnoise_Stim(t_stim = t_stim, cutf = cutf, sexp = sexp, t_qual = t_qual, inh_factor = inh_factor, onf = onf)

                freq_used, amp_vec, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1')
                freq_times, spike_freq, fmean, method_interpol, SNR, VAF, Qual = results.get('freq_times'), results.get('spike_freq'), results.get('fmeanA'), results.get('method_interpol'), results.get('SNR'), results.get('VAF'), results.get('Qual')
                stim, stim_re_mat, t_startstop, current_re = results.get('stim'), results.get('stim_re_mat'), results.get('t_startstop'), results.get('current_re')
                fmax, fmstd, fcvm, inh_factor, K_mat = results.get('fmaxA'), results.get('fmstdA'), results.get('fcvmA'), results.get('inh_factor'), results.get('K_mat')

                #print "fmean:", fmean
                if 'fbaseA' in results:
                    fbase = results.get('fbaseA')
                    fbstd = results.get('fbstdA')
                else:
                    fbase = []
                    fbstd = []

                if "_fft" in do: gsyn_in = results.get('gsyn_in')

                if pop.id == 0:

                    if ("_plotpsth" in do):

                        if ("_plotpsth0_" in do):
                            t_stim2 = 1000
                            t_noise = arange(0, t_stim, dt)
                            noise_data = create_colnoise(t_noise, sexp, cutf, 50, onf = onf)
                            stimulus, t, t_startstop = construct_Stimulus(noise_data, 1/dt, amp=1, ihold = 0, tail_points = 0, delay_baseline = 2)
                            ax14b.plot(t-6.17,stimulus,color='k', linewidth=linewidth)
                            ax14b.axis(xmin=0, xmax=0.1)
                            adjust_spines(ax14b, ['left','bottom'], d_out = d_out, d_down = 10)
                            ax14b.xaxis.set_ticks(array([0, 0.1]))
                            ax14b.set_xticklabels(('0', '0.1'))
                            ax14b.set_xlabel("s", labelpad=-3)
                            ax14b.set_ylabel("a.u.", labelpad=-1)
                            ax14b.axis(ymin=-1, ymax=1)
                            ax14b.yaxis.set_ticks(array([-1, 0, 1]))
                            ax14b.set_yticklabels(('-1', '0', '1'))
                            ax14b.text(0.006, 0.4, "*", color='k', fontsize = params['legend.fontsize'])
                            ax14b.text(0.04, 0.85, "**", color='k', fontsize = params['legend.fontsize'])
                            #plt.show()

                        dtn = 5e-3
                        t2 = np.arange(0,freq_times[-1],dtn)
                        psth = np.zeros(len(t2))

                        for i, ti in enumerate(t2[:-1]):
                            psth[i] = sum(spike_freq[(ti/dt):(t2[i+1])/dt])/sum(N)*dt/dtn

                        if ("_label24_" in do): label = "GrC, a=1"
                        if ("_label26_" in do): label = "GrC, a=10"

                        if ("_label24n_" in do): label = "iIF, a=1"
                        if ("_label26n_" in do): label = "iIF, a=10"

                        ax14.plot(t2-6.17,psth,color=color, linewidth=linewidth, label=label)
                        ax14.axis(xmin=0, xmax=0.1)
                        adjust_spines(ax14, ['left'], d_out = d_out, d_down = 10)
                        ax14.xaxis.set_ticks(array([0, 0.1]))
                        ax14.set_xticklabels(('0', '0.1'))
                        ax14.set_xlabel("s", labelpad=-3)
                        ax14.axis(ymin=0, ymax=170)
                        ax14.yaxis.set_ticks(array([0, 40, 80, 120, 160]))
                        ax14.set_yticklabels(('0', '40', '80', '120', '160'))
                        #ax14.set_ylabel("spikes/s", labelpad=-2)
                        ax14.set_title("spikes/s") #, fontsize=8)

                        #lg = ax14.legend(labelspacing=0.1, loc=1, bbox_to_anchor=(1.1,1.05), handlelength=1.5, handletextpad=0.1) #bbox_to_anchor=(-0.1,-0.1)
                        #fr = lg.get_frame()
                        #fr.set_lw(0.2)


                    if ("_plotre_" in do):
                        #ihold1 = ones(len(stim))*40
                        if ("_first_" in do):
                            if ("_pos3_" in do):
                                ax03.plot(np.arange(len(stim))*dt-0.5, stim, 'k', linewidth=linewidth)
                            else:
                                ax03.plot(np.arange(len(stim))*dt-2, stim, 'k', linewidth=linewidth)


                        if ("_end_" in do) and ("_pos3_" in do):
                            ax03.plot(np.arange(len(stim))*dt-0.5, stim_re_mat[0,:], color=color, alpha=1, linewidth=linewidth, linestyle = linestyle)
                            ax03.axis(xmin=0, xmax=0.5)
                            ax03.axis(ymin=-1.1, ymax=1.1)
                            adjust_spines(ax03, ['left','bottom'], d_out = d_out, d_down = 10)
                            ax03.xaxis.set_ticks(array([0, 0.5]))
                            ax03.set_xticklabels(('0', '0.5'))
                            ax03.set_xlabel("s", labelpad=-3)

                            #plt.show()
                        else:
                            ax03.plot(np.arange(len(stim))*dt-2, stim_re_mat[0,:], color=color, alpha=1, linewidth=linewidth, linestyle = linestyle)
                            ax03.axis(xmin=0, xmax=0.5)
                            ax03.axis(ymin=-1.7, ymax=1.7)
                            adjust_spines(ax03, ['left'], d_out = d_out, d_down = 10)

                        ax03.yaxis.set_ticks(array([-1, 0, 1]))
                        ax03.set_yticklabels(('-1', '0', '1'))

                        pos=np.where(stim>0.5)
                        neg=np.where(stim<-0.5)
                        err = sqrt((stim-stim_re_mat[0,:])**2)

                        print "pos error:", mean(err[pos]), "neg error:", mean(err[neg])

                    if ("_plotK_" in do):
                        K = K_mat[0,:]
                        freqK = fftfreq(len(K),dt)
                        iendK = mlab.find(freqK >= xmax)[0]
                        Kf = fft(fftshift(K))
                        ax03.semilogx(freqK[0:iendK],Kf[0:iendK]*1e3, color=color, alpha=1, linewidth=linewidth, linestyle = linestyle)
                        ax03.axis(xmin=0.3, xmax=30)

                        if ("_end_" in do) and ("_pos3b_" in do):
                            adjust_spines(ax03, ['left','bottom'], d_out = d_out, d_down = 10)
                            ax03.xaxis.set_ticks(array([1, 10, 20, 30]))
                            ax03.set_xticklabels(('1', '10', '', '30'))

                        else:
                            adjust_spines(ax03, ['left'], d_out = d_out, d_down = 10)

                    if ("_plotv" in do): # and (N == [1])

                        if ("_talk" in do): ax13.cla(); ax13.axis('on')
                        t2 = t1-t1[int(6.35/dt)]
                        ax13.plot(t2, voltage[0], color=color, linewidth = linewidth) #voltage[0]

                        adjust_spines(ax13, ['left','bottom'], d_out = d_out, d_down = 10)

                        ax13.axis(xmin=0, xmax=1)
                        ax13.xaxis.set_ticks(array([0, 0.2, 0.4, 0.6, 0.8, 1]))
                        ax13.set_xticklabels(('0', '0.2', '0.4', '0.6', '0.8', '1'))
                        ax13.set_xlabel("s", labelpad=0)

                        ax13.axis(ymin=-65, ymax=40)
                        ax13.yaxis.set_ticks(array([-60, -30, 0, 30]))
                        ax13.set_yticklabels(('-60', '-30', '0', '30'))
                        ax13.set_ylabel("mV", labelpad=2)

                        #plt.show()

                    elif ("_first" in do) and ("_talk" in do):
                        if "_fig6_" in do:
                            ax13.axis('off')

                    if ("_grc_syngr_cn_adjfinh_a01_nsyn4_varih_varinhn_N" in do): # do not plot bode
                        pass
                    else:

                        if "_psd" in do:

                            #pop.set_g(rate = CFo, cutf = cutf, sexp = sexp)

                            plt.figure(93)
                            ax91 = plt.subplot(4,1,1)
                            ax92 = plt.subplot(4,1,2)
                            ax93 = plt.subplot(4,1,3)
                            ax94 = plt.subplot(4,1,4)

                            ax91.plot(t1, voltage[0])
                            ax91.axis(xmin=10, xmax=11)

                            if "_fft" in do:


                                if "_ssine" in do:

                                    t, stimulus, i_startstop, t_startstop = create_singlesine(fu = freq_used[0], amp = amod*ihold[0], ihold = ihold[0] , dt = bin_width, periods = 20, minlength = 2*s, t_prestim = 1*s)
                                    stimulus = stimulus[i_startstop[0]:i_startstop[1]]
                                    t =  t[i_startstop[0]:i_startstop[1]] - t[i_startstop[0]]

                                else:

                                    i_startstop = []
                                    i_startstop.append(find(t1>=t_startstop[0])[0])
                                    i_startstop.append(find(t1<=(t_startstop[0]+10*s))[-1])

                                    t =  t1[i_startstop[0]:i_startstop[1]] - t1[i_startstop[0]]
                                    stimulus = np.zeros(len(t))

                                spike_times = freq_times[where(spike_freq>0)]
                                isi = diff(spike_times)      # differentiate for ISI!
                                isi_times = spike_times[1:]        # Set frequency to second spike for causality!

                                #ax92.plot(isi_times, 1/isi, t, stimulus, 'r')

                                spike_freq = spike_freq[i_startstop[0]:i_startstop[1]]
                                freq_times = freq_times[i_startstop[0]:i_startstop[1]] - t1[i_startstop[0]]

                                spike_f = fft(spike_freq) # transform into frequency domain
                                spike_f_pos = spike_f[0:round(len(spike_f) / 2)] # Take only the first half of the power to avoid redundancy

                                pwr = abs(spike_f_pos) # Impedance Power
                                pha = angle(spike_f_pos, deg=True) # Impedance Phase in degree

                                freq = fftfreq(len(spike_freq), bin_width)[0:round(len(spike_freq) / 2)] # positive frequency vector

                                fstart = find(freq >= 1. / (len(spike_freq) * bin_width))[0]  # lowest frequency depends on window length
                                fend = max(find(freq < 100))  # only use frequencies smaller than 1kHz
                                magz = pwr[fstart:fend]  # don't use freq = 0
                                phaz = pha[fstart:fend]  # don't use freq = 0
                                freq_used = freq[fstart:fend]  # don't use freq = 0

                                ax93.semilogx(freq_used, 20*log10(magz), linewidth=linewidth)

                                mag = zeros((1,len(freq_used)))
                                mag[0,:] = magz / magz[0]

                                if "_ssine" in do:

                                    umean, ramp, rphase, u = fit_sinusoid_fft(freq_used[0], t, spike_freq)
                                    print ramp, magz[find(freq == freq_used[0])]

                                ax94.plot(freq_times, spike_freq/N[0], t, stimulus, 'r')


                            plt.savefig("./figs/dump/Fig" + fig_num + "_latest_test.pdf", dpi = 300)  # save it
                            #plt.clf()
                            #plt.show()

                        # PLOTTING
                        plt.figure('results_transfer_syn')

                        if ("_none_" in do): # and ("_a01" in do) and ("_noisesynlow" in do):
                            normalize = mag[0,0]
                            print "normalizethis:", normalize

                        if ("_normalize_" in do):
                            normalize = mag[0,0]
                            print "normalizethis:", normalize

                        mag[0,:] = mag[0,:] / normalize

                        # SET LABELS
                        label = ""

                        if ("_label1_" in do): label = "FI=0, FC=" + str(int(fmean)) + "Hz, S=Se"
                        if ("_label2_" in do): label = "m(FI)=4sp/s, CF=" + str(int(fmean)) + "Hz, S=Se"
                        if ("_label3_" in do): label = "m(FI)=40sp/s, CF=" + str(int(fmean)) + "Hz, S=Se"
                        if ("_label4_" in do): label = "FI=0, FC=" + str(int(fmean)) + "Hz, S=Se"

                        if ("_label5_" in do): label = "input to inh. and ex." #"m(FI)=4Hz, CF=" + str(int(fmean)) + "Hz, S=Se+" + str(round(inh_factor,2)) + "Si"
                        if ("_label6_" in do): label = "m(FI)=40sp/s, FC=" + str(int(fmean)) + "Hz, S=Se+" + str(round(inh_factor[0],2)) + "Si"

                        if ("_label7_" in do): label = "input to inh." #"m(FI)=4Hz, CF=" + str(int(fmean)) + "Hz, S=Si"
                        if ("_label8_" in do): label = "m(FI)=40sp/s, FC=" + str(int(fmean)) + "Hz, S=Si"

                        if ("_label9_" in do): label = "2 inputs to ex." #"m(FI)=4Hz, CF=" + str(int(fmean)) + "Hz, S=0.5Se1+0.5Se2"

                        if ("_label11_" in do): label = r"100% $\mathsf{g_{ex}}$"
                        if ("_label12_" in do): label = r"50% $\mathsf{g_{ex}}$"
                        if ("_label13_" in do): label = r"$\mathsf{F_{I}}$=6sp/s"
                        if ("_label14_" in do): label = r"m($\mathsf{F_{I}}$)=6sp/s"

                        if ("_label0_" in do): label = "FC=" + str(int(fmean)) + "Hz"

                        # none
                        if ("_label15_" in do): label = "$\mathsf{N_{in}}$=1"
                        if ("_label16_" in do): label = "$\mathsf{N_{in}}$=40"
                        if ("_label16b_" in do): label = "N=40, a=10"
                        if ("_label17_" in do): label = "N=40, Poisson"

                        if ("_label18_" in do): label = "FC=" + str(int(fmean)) + "Hz"

                        if ("_label23_" in do): label = "a=1, p-p"
                        if ("_label24" in do): label = "a=1"

                        if ("_label25_" in do): label = "a=10, p-p"
                        if ("_label26" in do): label = "a=10"

                        if ("_label27_" in do): label = "GrC, p-p"
                        if ("_label28_" in do): label = "GrC"

                        if ("_label29_" in do): label = "IF, p-p"
                        if ("_label30_" in do): label = "IF"

                        if ("_label31_" in do): label = "N=200, p-p"
                        if ("_label32_" in do): label = "N=200"

                        if ("_label33_" in do): label = "GrC, cutoff=30Hz"
                        if ("_label34_" in do): label = "IF, cutoff=30Hz"
                        if ("_label35_" in do): label = "GrC, cutoff=5Hz"
                        if ("_label36_" in do): label = "IF, cutoff=5Hz"

                        #if ("_label2" in do) or ("_label3" in do): label = "FC=" + str(int(fmean)) + "Hz"

                        if ("_none_" in do) and ("_label_" in do) and ("_talk" not in do):
                            label = "a=" + str(amod)

                            #if ("_talk_" in do):
                            #    if ("poiss" in do):
                            #        label = "a=" + str(amod) + ", N=40, Poisson"
                            #    else:
                            #        label = "a=" + str(amod) + ", N=1, iIF"


                        if ("_oldlabel" in do):
                            if ("_N" in do):
                                if ("_noinh_varih" in do) or (CFo==0):
                                    label = "FI=" + str(0) + "Hz, m(CF)=" + str(round(fmean,1)) + 'Hz, std(CF)=' + str(round(fmstd,1))
                                elif "_varinhn" in do:
                                    label = "m(FI)=" + str(CFo) + "Hz, m(CF)=" + str(round(fmean,1)) + 'Hz, std(CF)=' + str(round(fmstd,1))
                                else:
                                    label = "FI=" + str(CFo) + "Hz, m(CF)=" + str(round(fmean,1)) + 'Hz, std(CF)=' + str(round(fmstd,1))
                            else:
                                if ("_noinh_varih" in do) or (CFo==0):
                                    label = "FI=" + str(0) + "Hz, CF=" + str(round(fmean,1)) + 'Hz'
                                elif "_varinhn" in do:
                                    label = "m(FI)=" + str(CFo) + "Hz, CF=" + str(round(fmean,1)) + 'Hz'
                                else:
                                    label = "FI=" + str(CFo) + "Hz, CF=" + str(round(fmean,1)) + 'Hz'


                        # PLOT MAG
                        #print freq_used
                        #print xmax

                        iend = mlab.find(freq_used >= xmax)[0]
                        if (ax01 is not None) and ("_notransf_" not in do):
                            ax01.semilogx(freq_used[0:iend], 20*log10(mag[0,0:iend]), color=color, linewidth=linewidth, linestyle = linestyle, alpha=1, label=label)
                            ax01.set_title("Gain (dB)") #, fontsize=8)

                        if "_fit2" in do:
                            Target_fit2.append([])

                            freq_used2 = freq_used[0:iend]
                            mag2 = mag[0,0:iend] # / mag[0,0]
                            pha2 = pha[0,0:iend]
                            H2 = (mag2 * exp(pi / 180 * 1j * pha2))

                            Target_fit2[-1].append(CFo) # [0]
                            Target_fit2[-1].append(fmean) # [1]
                            Target_fit2[-1].append(freq_used2) # [2]
                            Target_fit2[-1].append(H2) # [3]


                        # PLOT PHASE
                        iend = mlab.find(freq_used >= xmax)[0]
                        if (ax01 is not None) and ("_notransf_" not in do):
                            ax01b.semilogx(freq_used[0:iend], pha[0,0:iend], color=color, linewidth=linewidth, linestyle = linestyle, alpha=1, label=label)
                            ax01b.set_title("Phase ($^\circ$)") #, fontsize=8)


                    # PLOT VAF
                    if ("_psd" in do) or ("_fft" in do) or ("_noqual" in do):
                        pass
                    else:
                        iend = mlab.find(freq_used >= xmax)[0]
                        if "SNR" in do:
                            ax02.semilogx(freq_used[0:iend], SNR[1][0,0:iend], linewidth=linewidth, color=color, alpha=1, label=label)
                            ax02.axhline(y=10, color='k', linestyle=':')
                        else:
                            if ("_fig8" in do) and ("_pos5" in do):
                                ax02.semilogx(freq_used[0:iend], VAF[1][0,0:iend]*100, linewidth=linewidth, color=color, alpha=1, linestyle = linestyle, label=label)
                            elif ("_fig8" in do) and ("_notransf_" in do):
                                ax02.semilogx(freq_used[0:iend], VAF[1][0,0:iend]*100, linewidth=linewidth, color=color, alpha=1, linestyle = linestyle, label=label)
                            elif ("_fig6" in do) or ("_fig8" in do):
                                ax02.semilogx(freq_used[0:iend], VAF[1][0,0:iend]*100, linewidth=linewidth, color=color, alpha=1, linestyle = linestyle)
                            else:
                                ax02.semilogx(freq_used[0:iend], VAF[1][0,0:iend]*100, linewidth=linewidth, color=color, alpha=1, linestyle = linestyle, label=label)

                        if "_fig" not in do:
                            ax02.set_title(do, fontsize=6)
                        elif "_pos1b" not in do:
                            ax02.set_title("VAF (%)") #, fontsize=8)


                    # PLOT DEFAULTS
                    if "_end" in do:
                        if ax01 is not None:
                            adjust_spines(ax01, ['left','bottom'], d_out = d_out, d_down = d_down)
                            ax01.set_xlabel("Hz", labelpad=-3)

                        if ax01b is not None:
                            adjust_spines(ax01b, ['left','bottom'], d_out = d_out, d_down = d_down)
                            ax01b.set_xlabel("Hz", labelpad=-3)

                        adjust_spines(ax02, ['left','bottom'], d_out = d_out, d_down = d_down)
                        ax02.set_xlabel("Hz", labelpad=-3)

                    elif '_left' in do:
                        if "_notransf_" not in do:
                            adjust_spines(ax01, ['left'], d_out = d_out, d_down = d_down)
                            adjust_spines(ax01b, ['left'], d_out = d_out, d_down = d_down)
                        adjust_spines(ax02, ['left','bottom'], d_out = d_out, d_down = d_down)
                        ax02.set_xlabel("Hz", labelpad=-5)

                    elif '_bottom' in do:
                        if ax01 is not None: adjust_spines(ax01, ['bottom'], d_out = d_out)
                        if ax01b is not None: adjust_spines(ax01b, ['bottom'], d_out = d_out)
                        adjust_spines(ax02, ['bottom'], d_out = d_out)
                        ax02.set_xlabel("Hz", labelpad=-5)

                    else:
                        if ax01 is not None: adjust_spines(ax01, ['left'], d_out = d_out)
                        if ax01b is not None: adjust_spines(ax01b, ['left'], d_out = d_out)
                        adjust_spines(ax02, ['left'], d_out = d_out)

                    if ax01 is not None: ax01.set_xscale('log')
                    if ax01b is not None: ax01b.set_xscale('log')
                    ax02.set_xscale('log')

                    if ("_cutf30" in do) or ("_cutf5" in do):

                        if ax01 is not None:
                            ax01.axis(xmin=0.3, xmax=xmax)
                            ax01.xaxis.set_ticks(array([1,10,20,30]))
                            ax01.set_xticklabels(('1', '10', '', '30'))

                        if ax01b is not None:
                            ax01b.axis(xmin=0.3, xmax=xmax)
                            ax01b.xaxis.set_ticks(array([1,10,20,30]))
                            ax01b.set_xticklabels(('1', '10', '', '30'))

                        ax02.axis(xmin=0.3, xmax=xmax)
                        ax02.xaxis.set_ticks(array([1,10,20, 30]))
                        ax02.set_xticklabels(('1', '10', '', '30'))


                    #if ("_cutf5" in do):

                    #    if ax01 is not None:
                    #        ax01.axis(xmin=0.3, xmax=xmax)
                    #        ax01.xaxis.set_ticks(array([0.5,5]))
                    #        ax01.set_xticklabels(('0.5', '5'))

                    #    if ax01b is not None:
                    #        ax01b.axis(xmin=0.3, xmax=xmax)
                    #        ax01b.xaxis.set_ticks(array([0.5,5]))
                    #        ax01b.set_xticklabels(('0.5', '5'))

                    #    ax02.axis(xmin=0.3, xmax=xmax)
                    #    ax02.xaxis.set_ticks(array([0.5,5]))
                    #    ax02.set_xticklabels(('0.5', '5'))

                        #plt.show()


                    elif ("_cn_" in do) or ("_bn_" in do):

                        if "_psd" in do:
                            if ax01 is not None:
                                ax01.axis(xmin=1, xmax=50)
                                ax01.xaxis.set_ticks(array([1,10,20,40]))
                                ax01.set_xticklabels(('1', '10', '20', '40'))

                            if ax01b is not None:
                                ax01b.axis(xmin=1, xmax=50)
                                ax01b.xaxis.set_ticks(array([1,10,20,40]))
                                ax01b.set_xticklabels(('1', '10', '20', '40'))

                            ax02.axis(xmin=1, xmax=50)
                            ax02.xaxis.set_ticks(array([1,10,20,40]))
                            ax02.set_xticklabels(('1', '10', '20', '40'))

                        else:
                            if ax01 is not None:
                                ax01.axis(xmin=0.3, xmax=xmax)
                                if "fig5" in do:
                                    ax01.xaxis.set_ticks(array([1,20]))
                                    ax01.set_xticklabels(('1', '20'))
                                else:
                                    ax01.xaxis.set_ticks(array([1,10,20]))
                                    ax01.set_xticklabels(('1', '10', '20'))

                            if ax01b is not None:
                                ax01b.axis(xmin=0.3, xmax=xmax)
                                if "fig5" in do:
                                    ax01b.xaxis.set_ticks(array([1,20]))
                                    ax01b.set_xticklabels(('1', '20'))
                                else:
                                    ax01b.xaxis.set_ticks(array([1,10,20]))
                                    ax01b.set_xticklabels(('1', '10', '20'))

                            ax02.axis(xmin=0.3, xmax=xmax)
                            if "fig5" in do:
                                ax02.xaxis.set_ticks(array([1,20]))
                                ax02.set_xticklabels(('1', '20'))
                            elif "fig7" in do:
                                ax02.xaxis.set_ticks(array([1,20]))
                                ax02.set_xticklabels(('1', '20'))
                            else:
                                ax02.xaxis.set_ticks(array([1,10,20]))
                                ax02.set_xticklabels(('1', '10', '20'))




                    if "_wn_" in do:

                        if ax01 is not None:
                            ax01.xaxis.set_ticks(array([1,10,100]))
                            ax01.set_xticklabels(('1', '10', '100'))
                            ax01.axvline(x=fmean, color='k', linestyle=':')

                        if ax01b is not None:
                            ax01b.xaxis.set_ticks(array([1,10,100]))
                            ax01b.set_xticklabels(('1', '10', '100'))
                            ax01b.axvline(x=fmean, color='k', linestyle=':')

                        ax02.xaxis.set_ticks(array([1,10,100]))
                        ax02.set_xticklabels(('1', '10', '100'))
                        ax02.axvline(x=fmean, color='k', linestyle=':')


                    # SET YAXIS
                    if ("_label" in do):
                        if "_fig7" in do:
                            if ax01b is not None:
                                lg = ax01b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), handlelength=3, ncol=2, handletextpad=0.5, columnspacing=1)
                                #lg.draw_frame(False)
                                fr = lg.get_frame()
                                fr.set_lw(0.2)

                        elif "_fig6" in do:

                            if "_talk_" in do:
                                pass

                            elif "_fig6b" in do:
                                if ax01b is not None:
                                    lg = ax01b.legend(loc='upper center', bbox_to_anchor=(0.4, -0.05), handlelength=3, ncol=3, handletextpad=0.5, columnspacing=1)
                                    #lg.draw_frame(False)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)

                            else:
                                if ax01b is not None:
                                    #lg = ax01b.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), handlelength=3, ncol=2, handletextpad=0.5, columnspacing=1)
                                    lg = ax01b.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.06, 0), handlelength=0, handletextpad=0.1, numpoints=1)
                                    #lg.draw_frame(False)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)

                                    color_v.append(color)
                                    txt = lg.get_texts()
                                    for i, t in enumerate(txt):
                                        t.set_color(color_v[i])


                            if "_first_" in do:
                                ax02.axhline(y=-1, xmin=0, xmax=0, color='k', linestyle='-', label="iIF", linewidth=linewidth)
                                ax02.axhline(y=-1, xmin=0, xmax=0, color='k', linestyle=':', label="Poisson", linewidth=linewidth)
                                #if "_pos1b_" in do:ax02.axhline(y=-1, xmin=0, xmax=0, color=p1, linestyle='-', label=r"$\mathsf{N_{I}}$=10", linewidth=linewidth)
                                lg = ax02.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.05,-0.05), handlelength=1.5, handletextpad=0.1)
                                #lg.draw_frame(False)
                                fr = lg.get_frame()
                                fr.set_lw(0.2)


                        elif "_talk_" in do:
                            pass
                            #lg = ax01b.legend(labelspacing=0.2, loc=3, handlelength=1.5, handletextpad=0.1)
                            ##lg.draw_frame(False)
                            #fr = lg.get_frame()
                            #fr.set_lw(0.2)

                        elif ("_fig5_" in do):
                            lg = ax02.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.08,-0.2), handlelength=1.5, handletextpad=0.1)
                            #lg.draw_frame(False)
                            fr = lg.get_frame()
                            fr.set_lw(0.2)

                        elif ("_fig8_" in do) and ("_notransf_" not in do):

                            #if "_label31_" in do:
                            #    ax01b.axhline(y=-1, xmin=0, xmax=0, color=p1, linestyle='-', label="N=200, p-p", linewidth=linewidth)
                            #if "_label32_" in do:
                            #    ax01b.axhline(y=-1, xmin=0, xmax=0, color=br1, linestyle='-', label="N=200", linewidth=linewidth)

                            lg = ax01b.legend(labelspacing=0.05, loc=3, bbox_to_anchor=(-0.1,-0.13), handlelength=0, handletextpad=0.1, numpoints=1)
                            #lg.draw_frame(False)
                            fr = lg.get_frame()
                            fr.set_lw(0.2)

                            color_v.append(color)
                            txt = lg.get_texts()
                            for i, t in enumerate(txt):
                                t.set_color(color_v[i])

                        elif ("_fig8_" in do) and ("_pos5_" in do):

                            #lg = ax02.legend(labelspacing=0.2, loc=4, bbox_to_anchor=(1.15,-0.1), handlelength=1.5, handletextpad=0.1)
                            lg = ax02.legend(labelspacing=0.2, loc=4, bbox_to_anchor=(1.0,-0.1), handlelength=1.5, handletextpad=0.1)
                            #lg.draw_frame(False)
                            fr = lg.get_frame()
                            fr.set_lw(0.2)

                            color_v.append(color)
                            txt = lg.get_texts()
                            for i, t in enumerate(txt):
                                t.set_color(color_v[i])

                            ax4.set_title("Sample reconstruction")
                            ax8.set_title("Sample reconstruction")
                            ax12.set_title("Sample reconstruction")

                        elif ("_fig8_" in do) and ("_notransf_" in do):

                            if "none" in do:
                                lg = ax02.legend(labelspacing=0.05, loc=3, bbox_to_anchor=(-0.02,0.24), handlelength=0, handletextpad=0.1, numpoints=1)
                            elif "_pos3_" in do:
                                lg = ax02.legend(labelspacing=0.05, loc=3, bbox_to_anchor=(0.6,-0.1), handlelength=1.5, handletextpad=0.1) # , numpoints=1
                            else:
                                lg = ax02.legend(labelspacing=0.05, loc=3, bbox_to_anchor=(0.6,-0.07), handlelength=0, handletextpad=0.1, numpoints=1)
                            #lg.draw_frame(False)
                            fr = lg.get_frame()
                            fr.set_lw(0.2)

                            color_v.append(color)
                            txt = lg.get_texts()
                            for i, t in enumerate(txt):
                                t.set_color(color_v[i])

                        elif ("_notransf_" not in do):
                            lg = ax01b.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.05,-0.15), handlelength=1.5, handletextpad=0.1)
                            #lg.draw_frame(False)
                            fr = lg.get_frame()
                            fr.set_lw(0.2)

                        if ("_fig5_" in do) and ("_talk_" in do) and (i==0) and (amod[0] == 1):
                            ax02.axhline(y=0, xmin=-1, xmax=-1, color='k', linestyle='-', label="N=1, iIF")
                            ax02.axhline(y=0, xmin=-1, xmax=-1, color='k', linestyle=':', label="N=40, Poisson")
                            lg = ax02.legend(labelspacing=0.2, loc=3, handlelength=1.5, handletextpad=0.1)
                            #lg.draw_frame(False)
                            fr = lg.get_frame()
                            fr.set_lw(0.2)



                    if ("_fig5" in do):

                        if "_synno_" in do:
                            ax01.axis(ymin=-1, ymax=1)
                            ax01.yaxis.set_ticks(array([-1, 0, 1]))
                            ax01.set_yticklabels(('-1', '0', '1'))
                            ax01b.yaxis.set_ticks(array([-10, 0, 10]))
                            ax01b.set_yticklabels(('-10', '0', '10'))
                            dup = 1.5
                            if ("_talk_" in do):
                                dup = 1.25
                            if "_first_" in do: ax01b.text(0.5, dup, 'Spike input', transform=ax01b.transAxes, fontsize = params['text.fontsize'], va='center', ha='center')

                        if "_synampagr_" in do:
                            ax01.axis(ymin=-3, ymax=0.1)
                            ax01.yaxis.set_ticks(array([-3,-2,-1,0]))
                            ax01.set_yticklabels(('-3', '-2', '-1', '0'))
                            ax01b.axis(ymin=-25, ymax=5)
                            ax01b.yaxis.set_ticks(array([-20, -10, 0]))
                            ax01b.set_yticklabels(('-20', '-10', '0'))
                            dup = 1.5
                            if ("_talk_" in do):
                                dup = 1.25
                            if "_first_" in do: ax01b.text(0.5, dup, 'AMPA synapse conductance', transform=ax01b.transAxes, fontsize = params['text.fontsize'], va='center', ha='center')

                        if "_synnmdagr_" in do:
                            ax01.axis(ymin=-5, ymax=0.2)
                            ax01.yaxis.set_ticks(array([-5,-4, -3, -2, -1, 0]))
                            ax01.set_yticklabels(('-5','-4', '-3', '-2', '-1', '0'))
                            ax01b.axis(ymin=-40, ymax=1)
                            ax01b.yaxis.set_ticks(array([-40, -30, -20, -10, 0]))
                            ax01b.set_yticklabels(('-40', '-30', '-20', '-10', '0'))
                            dup = 1.5
                            if ("_talk_" in do): dup = 1.25
                            if "_first_" in do: ax01b.text(0.5, dup, 'NMDA synapse conductance', transform=ax01b.transAxes, fontsize = params['text.fontsize'], va='center', ha='center')

                        if "_syngabagr_" in do:
                            ax01.axis(ymin=-8.5, ymax=0.5)
                            ax01.yaxis.set_ticks(array([-8, -6, -4, -2, 0]))
                            ax01.set_yticklabels(('-8', '-6', '-4', '-2', '0'))
                            ax01b.axis(ymin=-41, ymax=1)
                            ax01b.yaxis.set_ticks(array([-40, -30, -20, -10, 0]))
                            ax01b.set_yticklabels(('-40', '-30', '-20', '-10', '0'))
                            dup = 1.5
                            if ("_talk_" in do): dup = 1.25
                            if "_first_" in do: ax01b.text(0.5, dup, 'GABA synapse conductance', transform=ax01b.transAxes, fontsize = params['text.fontsize'], va='center', ha='center')

                        if ("_talk_" in do):
                            if ("_a05_" in do): ax02.text(0.5, 55, "a=" + str(amod), color=color, fontsize = params['legend.fontsize'])
                            if ("_a1_" in do): ax02.text(0.5, 80, "a=" + str(amod), color=color, fontsize = params['legend.fontsize'])

                        ax02.axis(ymin=50, ymax=100)
                        ax02.yaxis.set_ticks(array([50,60,70,80,90,100]))
                        ax02.set_yticklabels(('50','60','70','80','90','100'))

                    if ("_fig6_" in do) and (("_pos1_" in do) or ("_pos1b_" in do)):

                        if ax01 is not None:
                            ax01.axis(ymin=-7, ymax=0.5)
                            #ax01.yaxis.set_ticks(array([-5, -4, -3, -2, -1, 0, 1]))
                            #ax01.set_yticklabels(('-5','-4', '-3', '-2', '-1', '0', '1'))

                        if ax01b is not None:
                            ax01b.axis(ymin=-40, ymax=11)
                            ax01b.yaxis.set_ticks(array([-40, -30, -20, -10, 0, 10]))
                            ax01b.set_yticklabels(('-40', '-30', '-20', '-10', '0', '10'))

                        if ("_first_" in do): ax02.text(0.4, 80, "N="+str(N[0]) , color='k', fontsize = 10)

                        dup = 1.35
                        if ("_talk_" in do):
                            dup = 1.25
                        else:
                            if ("_first_" in do):
                                pass
                                #ax01b.text(0.5, dup, r'Population (N=10)', transform=ax01b.transAxes, fontsize=10, va='center', ha='center')

                        ax02.axis(ymin=40, ymax=100)
                        ax02.yaxis.set_ticks(array([40,50,60,70,80,90,100]))
                        ax02.set_yticklabels(('40','50','60','70','80','90','100'))

                        if (amod[0] == 0.1): ax02.text(5, 45, "a=0.1", color=color, fontsize = 8)

                        if ("_talk_" in do):
                            if ("_label11_" in do): ax01b.text(0.3, -7, "FI=0, CF=" + str(int(fmean)) + "Hz", color=color, fontsize = params['legend.fontsize'])
                            if ("_label12_" in do): ax01b.text(0.3, -10, "FI=0, CF=" + str(int(fmean)) + "Hz", color=color, fontsize = params['legend.fontsize'])
                            if ("_label13_" in do): ax01b.text(0.3, 7, "FI=4Hz, CF=" + str(int(fmean)) + "Hz", color=color, fontsize = params['legend.fontsize'])
                            if ("_label14_" in do): ax01b.text(0.3, 10, "m(FI)=4Hz, CF=" + str(int(fmean)) + "Hz", color=color, fontsize = params['legend.fontsize'])

                    if (("_fig6b_" in do) and ("_pos1_" in do)) or (("_fig6_" in do) and ("_pos2_" in do)):

                        if ax01 is not None:
                            ax01.axis(ymin=-14, ymax=3)
                            ax01.yaxis.set_ticks(array([-16, -14, -12,-10, -8, -6, -4, -2, 0, 2]))
                            ax01.set_yticklabels(('-16', '-14', '-12','-10','-8', '-6', '-4', '-2', '0', '2'))

                        if ax01b is not None:
                            ax01b.axis(ymin=-80, ymax=11)
                            ax01b.yaxis.set_ticks(array([-90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10]))
                            ax01b.set_yticklabels(('-90', '-80', '-70','-60','-50', '-40','-30','-20','-10', '0', '10'))

                        ax02.axis(ymin=0, ymax=100)
                        ax02.yaxis.set_ticks(array([0,10,20,30,40,50,60,70,80,90,100]))
                        ax02.set_yticklabels(('0','10','20','30','40','50','60','70','80','90','100'))

                        if ("_talk_" in do):
                            if ("_label5_" in do): ax01b.text(0.3, -9, "S=Se+" + str(round(inh_factor[0],2)) + "*Si", color=color, fontsize = params['legend.fontsize']) # "m(FI)=4Hz, CF=" + str(int(fmean)) + "Hz,
                            if ("_label7_" in do): ax01b.text(0.3, -30, "S = Si", color=color_vec[0][i], fontsize = params['legend.fontsize']) # "m(FI)=4Hz, CF=" + str(int(fmean)) + "Hz,
                            if ("_label9_" in do): ax01b.text(0.3, 9,  "S = 0.5*Se1 + 0.5*Se2", color=color, fontsize = params['legend.fontsize'])  # "m(FI)=4Hz, CF=" + str(int(fmean)) + "Hz,



                    if ("_fig7" in do) and ("_pos1_" in do):

                        if "_notransf_" not in do:
                            ax01.axis(ymin=-12, ymax=3)
                            ax01.yaxis.set_ticks(array([-12, -10, -8, -6, -4, -2, 0, 2]))
                            ax01.set_yticklabels(('-12', '', '-8', '', '-4', '', '0', ''))

                            ax01b.axis(ymin=-70, ymax=10)
                            ax01b.yaxis.set_ticks(array([-70, -60, -50, -40, -30, -20, -10, 0, 10]))
                            ax01b.set_yticklabels(('', '-60', '', '-40', '', '-20', '', '0', ''))
                            #if ("_first_" in do): ax01b.text(0.5, 1.3, r'Population (N=10, a=1), Inhibition Frequency (FI) variable', transform=ax01b.transAxes, fontsize=10, va='center', ha='center')

                        ax02.axis(ymin=68, ymax=100)
                        #ax02.yaxis.set_ticks(array([0,10,20,30,40,50,60,70,80,90,100]))
                        #ax02.set_yticklabels(('0','10','20','30','40','50','60','70','80','90','100'))
                        ax02.yaxis.set_ticks(array([70, 80,90,100]))
                        ax02.set_yticklabels(('70', '80','90','100'))

                        #ax01b.text(0.5, 1.3, r'Population (N=10, a=0.5), signal through GABA', transform=ax01b.transAxes, fontsize=10, va='center', ha='center')
                        #ax01b.text(0.5, 1.3, r'Single granule cell (N=1, a=0.5)', transform=ax01b.transAxes, fontsize=10, va='center', ha='center')
                        #ax01b.text(0.5, 1.3, r'Single granule cell (N=1, a=0.5)', transform=ax01b.transAxes, fontsize=10, va='center', ha='center')
                        #ax01b.text(0.5, 1.3, r'Population (N=10, a=0.5)', transform=ax01b.transAxes, fontsize=10, va='center', ha='center')
                        #ax01b.text(0.5, 1.3, r'Population (N=10, a=0.5), two input signals', transform=ax01b.transAxes, fontsize=10, va='center', ha='center')


                    if ("_fig7" in do) and ("_pos2_" in do):

                        if "_notransf_" not in do:
                            ax01.axis(ymin=-12, ymax=3)
                            ax01.yaxis.set_ticks(array([-12, -10, -8, -6, -4, -2, 0, 2]))
                            ax01.set_yticklabels(('-12', '', '-8', '', '-4', '', '0', ''))

                            ax01b.axis(ymin=-70, ymax=10)
                            ax01b.yaxis.set_ticks(array([-70, -60, -50, -40, -30, -20, -10, 0, 10]))
                            ax01b.set_yticklabels(('', '-60', '', '-40', '', '-20', '', '0', ''))
                            #if ("_first_" in do): ax01b.text(0.5, 1.3, r'Population (N=10, a=1), Signal through inhibition', transform=ax01b.transAxes, fontsize=10, va='center', ha='center')

                        #ax02.axis(ymin=0, ymax=100)
                        #ax02.yaxis.set_ticks(array([0,10,20,30,40,50,60,70,80,90,100]))
                        #ax02.set_yticklabels(('0','10','20','30','40','50','60','70','80','90','100'))
                        ax02.axis(ymin=68, ymax=100)
                        #ax02.yaxis.set_ticks(array([70, 80,90,100]))
                        #ax02.set_yticklabels(('70', '80','90','100'))

                        if "_first_" in do:
                            ax02.axhline(y=-1, xmin=0, xmax=0, color=o1, linestyle='-', label="100%,\n"+r"m($\mathsf{F_{I}}$)=6sp/s", linewidth=linewidth)
                            ax02.axhline(y=-1, xmin=0, xmax=0, color=o1, linestyle=':', label="45%,\n"+r"m($\mathsf{F_{I}}$)=40sp/s", linewidth=linewidth)
                            lg = ax02.legend(labelspacing=0.3, loc=3, bbox_to_anchor=(-0.16, -0.06), handlelength=1.5, handletextpad=0)
                            #lg.draw_frame(False)
                            fr = lg.get_frame()
                            fr.set_lw(0.2)


                    if ("_fig7" in do) and ("_pos3_" in do):

                        if "_notransf_" not in do:
                            ax01.axis(ymin=-12, ymax=3)
                            ax01.yaxis.set_ticks(array([-12, -10, -8, -6, -4, -2, 0, 2]))
                            ax01.set_yticklabels(('-12', '', '-8', '', '-4', '', '0', ''))

                            ax01b.axis(ymin=-70, ymax=10)
                            ax01b.yaxis.set_ticks(array([-70, -60, -50, -40, -30, -20, -10, 0, 10]))
                            ax01b.set_yticklabels(('', '-60', '', '-40', '', '-20', '', '0', ''))
                            #if ("_first_" in do): ax01b.text(0.5, 1.3, r'Population (N=10, a=1), Signal through inhibition', transform=ax01b.transAxes, fontsize=10, va='center', ha='center')

                        #ax02.axis(ymin=0, ymax=100)
                        #ax02.yaxis.set_ticks(array([0,10,20,30,40,50,60,70,80,90,100]))
                        #ax02.set_yticklabels(('0','10','20','30','40','50','60','70','80','90','100'))
                        ax02.axis(ymin=68, ymax=100)
                        #ax02.yaxis.set_ticks(array([70, 80,90,100]))
                        #ax02.set_yticklabels(('70', '80','90','100'))

                        if "_first_" in do:
                            ax02.axhline(y=-1, xmin=0, xmax=0, color=b1, linestyle='-', label="100%,\n"+r"m($\mathsf{F_{I}}$)=6sp/s", linewidth=linewidth)
                            ax02.axhline(y=-1, xmin=0, xmax=0, color=b1, linestyle=':', label="45%,\n"+r"m($\mathsf{F_{I}}$)=40sp/s", linewidth=linewidth)
                            lg = ax02.legend(labelspacing=0.3, loc=3, bbox_to_anchor=(-0.14, -0.06), handlelength=1.5, handletextpad=0)
                            #lg.draw_frame(False)
                            fr = lg.get_frame()
                            fr.set_lw(0.2)

                    if ("_fig8" in do) and (("_pos1_" in do) or ("_pos1b_" in do)):

                        if "_notransf_" not in do:
                            ax01.axis(ymin=-1, ymax=1)
                            ax01.yaxis.set_ticks(array([-5, 0, 5]))
                            ax01.set_yticklabels(('-5', '0', '5'))

                            ax01b.axis(ymin=-50, ymax=50)
                            ax01b.yaxis.set_ticks(array([-40, 0, 40]))
                            ax01b.set_yticklabels(('-40', '0', '40'))

                        ax02.axis(ymin=79, ymax=101)
                        ax02.yaxis.set_ticks(array([80,90,100]))
                        ax02.set_yticklabels(('80','90','100'))

                    if ("_fig8" in do) and (("_pos2_" in do) or ("_pos2b_" in do)):

                        if "_notransf_" not in do:
                            ax01.axis(ymin=-10, ymax=12)
                            ax01.yaxis.set_ticks(array([-10,-5, 0, 5, 10]))
                            ax01.set_yticklabels(('-10', '-5','0', '5', '10'))

                            ax01b.axis(ymin=-120, ymax=130)
                            ax01b.yaxis.set_ticks(array([-80,-40, 0, 40, 80, 120]))
                            ax01b.set_yticklabels(('-80', '-40', '0', '40', '80', '120'))

                        ax02.axis(ymin=0, ymax=100)
                        ax02.yaxis.set_ticks(array([0,20,40,60,80,100]))
                        ax02.set_yticklabels(('0','20','40','60','80','100'))

                        #if ("_colorr2_pos2b_dotted" in do):
                        #    ax02.axhline(y=-1, xmin=0, xmax=0, color='k', linestyle='-', label="GrC", linewidth=linewidth)
                        #    ax02.axhline(y=-1, xmin=0, xmax=0, color='k', linestyle=':', label="IF", linewidth=linewidth)
                        #    lg = ax02.legend(labelspacing=0.2, loc=4, bbox_to_anchor=(1.1,0), handlelength=1.5, handletextpad=0.1)
                        #    #lg.draw_frame(False)
                        #    fr = lg.get_frame()
                        #    fr.set_lw(0.2)

                    if ("_fig8" in do) and (("_pos3_" in do) or ("_pos3b_" in do)):

                        if "_notransf_" not in do:
                            ax01.axis(ymin=-3, ymax=16)
                            ax01.yaxis.set_ticks(array([0,4,8,12,16]))
                            ax01.set_yticklabels(('0', '4','8', '12', '16'))

                            ax01b.axis(ymin=-80, ymax=55)
                            ax01b.yaxis.set_ticks(array([-80, -40, 0, 40]))
                            ax01b.set_yticklabels(('-80', '-40', '0', '40'))

                        ax02.axis(ymin=0, ymax=100)
                        ax02.yaxis.set_ticks(array([0,20,40,60,80,100]))
                        ax02.set_yticklabels(('0','20','40','60','80','100'))

                        #if ("_pos3_label29_" in do):
                        #    ax02.axhline(y=-1, xmin=0, xmax=0, color='k', linestyle='-', label="GrC", linewidth=linewidth)
                        #    ax02.axhline(y=-1, xmin=0, xmax=0, color='k', linestyle=':', label="IF", linewidth=linewidth)
                        #    lg = ax02.legend(labelspacing=0.2, loc=4, bbox_to_anchor=(1.1,0), handlelength=1.5, handletextpad=0.1)
                        #    #lg.draw_frame(False)
                        #    fr = lg.get_frame()
                        #    fr.set_lw(0.2)

                    if ("_fig8" in do) and ("_pos5_" in do):

                        ax02.axis(ymin=0, ymax=100)
                        ax02.yaxis.set_ticks(array([0,20,40,60,80,100]))
                        ax02.set_yticklabels(('0','20','40','60','80','100'))


                    if MPI.COMM_WORLD.rank == 0:

                        plotname = "./figs/Pub/" + str(prefix)
                        savefig(plotname + ".pdf", dpi = 300) # save it
                        savefig(plotname + ".png", dpi = 300) # save it
                        #os.system('rsvg-convert -f pdf -o ' + plotname +'.pdf ' + plotname + '.svg')


                        if "_png_" in do:
                            plt.savefig(plotname + ".png", dpi = 300, transparent=True) # save it

                        # save additional information:
                        thisinfo = "Variable=" + str(CFo) + ", m(CF)=" + str(fmean) + 'Hz, std(CF)=' + str(fmstd) + 'Hz, max(CF)=' + str(fmax) + 'Hz, mean(CV)=' + str(fcvm) + ', inh_factor=' + str(inh_factor) + ' ,m(VAF)=' + str(np.mean(VAF[1][0,0:iend]))  + ", m(base)=" + str(fbase) + "Hz, std(base)=" + str(fbstd)

                        fo = open("./figs/txt/" + str(prefix) + ".txt" , "wb")
                        fo.write(thisinfo)
                        fo.close()


                        if ("_talk_" in do) or ("_addplot_"  in do):

                            from pyPdf import PdfFileReader, PdfFileWriter

                            output = PdfFileWriter()
                            pdfOne = PdfFileReader(file(plotname + ".pdf", "rb"))
                            input1 = pdfOne.getPage(0)

                            if "_fig6b_" in do:
                                pdfTwo = PdfFileReader(file("./figs/Inserts/intro_syndiff.pdf", "rb"))
                                input1.mergeTransformedPage(pdfTwo.getPage(0),[1, 0, 0, 1, 10, 150])

                            if "_fig7_" in do:

                                if "poster" in pars.opt:
                                    pdfTwo = PdfFileReader(file("./figs/Inserts/Fig7_clipart_poster.pdf", "rb"))
                                else:
                                    pdfTwo = PdfFileReader(file("./figs/Inserts/Fig7_clipart.pdf", "rb"))
                                input1.mergeTransformedPage(pdfTwo.getPage(0),[1, 0, 0, 1, 0, 162])

                            output.addPage(input1)
                            outputStream = file(plotname + "_merge.pdf", "wb")
                            output.write(outputStream)
                            outputStream.close()

                if do_run_now:
                    pop.delall()
                del pop
                pop = None
                results = None



        elif ("_CF-" in do):

            if MPI.COMM_WORLD.rank == 0:
                if ("_pos13_" in do):
                    axcf = ax13
                else:
                    axcf = ax14

            if ("_cont_" in do):
                use_old = True
            else:
                use_old = False

            t_stim = 100*s # only for cnoise
            CF_vec = nans(len(x_vec)) # create NaN vector
            N_need_vec = nans(len(x_vec))  # create NaN vector
            fmax_v = nans(len(x_vec))  # create NaN vector
            fmstd_v = nans(len(x_vec))  # create NaN vector
            fcvm_v = nans(len(x_vec))  # create NaN vector
            VAF_min_vec = nans((len(N_vec),len(x_vec)))
            CF_min_vec = nans((len(N_vec),len(x_vec)))

            if (Nstop is False):
                CF_min_vec = np.fliplr(CF_min_vec)
                VAF_min_vec = np.fliplr(VAF_min_vec)
                x_vec = x_vec[::-1]


            filepath = data_dir + '/' + str(prefix) + "_results.p"
            if MPI.COMM_WORLD.rank == 0: print filepath

            if do_run_now or (os.path.isfile(filepath) is False):

                ni = 0
                xi = 0

                if use_old and (os.path.isfile(filepath) is True): # start from results saved before!

                    if MPI.COMM_WORLD.rank == 0: print "- using old DATA"

                    results = pickle.load( gzip.GzipFile( filepath, "rb" ) )
                    CF_min_vec, VAF_min_vec, CF_vec, N_need_vec, freq_used, mag, pha, VAF = results.get('CF_min_vec'),results.get('VAF_min_vec'), results.get('CF_vec'), results.get('N_need_vec'), results.get('freq_used'), results.get('mag'), results.get('pha'), results.get('VAF')
                    #xi, ni = results.get('xi'), results.get('ni')
                    #x_vec = results.get('x_vec')
                    fmax_v, fmstd_v, fcvm_v = results.get('fmax_v'),results.get('fmstd_v'), results.get('fcvm_v')
                    if MPI.COMM_WORLD.rank == 0: print np.shape(x_vec), np.shape(N_need_vec), np.shape(CF_vec), np.shape(CF_min_vec), np.shape(VAF_min_vec)
                    if MPI.COMM_WORLD.rank == 0: print np.shape(fmax_v), np.shape(fmstd_v), np.shape(fcvm_v)

                    CF_min_vec = np.delete(CF_min_vec, [11,12,13,14], 1)
                    VAF_min_vec = np.delete(VAF_min_vec, [11,12,13,14], 1)
                    fmax_v = np.delete(fmax_v, [11,12,13,14])
                    fmstd_v = np.delete(fmstd_v, [11,12,13,14])
                    fcvm_v = np.delete(fcvm_v, [11,12,13,14])
                    N_need_vec = np.delete(N_need_vec, [11,12,13,14])
                    CF_vec = np.delete(CF_vec, [11,12,13,14])

                    if MPI.COMM_WORLD.rank == 0: print np.shape(x_vec), np.shape(N_need_vec), np.shape(CF_vec), np.shape(CF_min_vec), np.shape(VAF_min_vec)
                    if MPI.COMM_WORLD.rank == 0: print np.shape(fmax_v), np.shape(fmstd_v), np.shape(fcvm_v)


                    results = {'xi':xi, 'ni':ni, 'x_vec':x_vec, 'CF_vec':CF_vec, 'CF_min_vec':CF_min_vec, 'VAF_min_vec':VAF_min_vec, 'N_need_vec':N_need_vec, 'freq_used':freq_used,'mag':mag,'pha':pha,'VAF':VAF, 'fmax_v':fmax_v, 'fmstd_v':fmstd_v, 'fcvm_v':fcvm_v}
                    pickle.dump( results, gzip.GzipFile( filepath, "wb" ) )

                    if (len(N_vec) > np.shape(VAF_min_vec)[0]) or (len(x_vec) > np.shape(VAF_min_vec)[1]):

                        if MPI.COMM_WORLD.rank == 0: print len(N_vec), np.shape(VAF_min_vec)[0]

                        if MPI.COMM_WORLD.rank == 0: print "- Reshaping CF_min_vec and VAF_min_vec"

                        CF_vec_old = CF_vec
                        N_need_vec_old = N_need_vec
                        CF_min_vec_old = CF_min_vec
                        VAF_min_vec_old = VAF_min_vec
                        fmax_v_old = fmax_v
                        fmstd_v_old = fmstd_v
                        fcvm_v_old = fcvm_v

                        CF_min_vec = nans((len(N_vec), len(x_vec)))   # create NaN vector
                        VAF_min_vec = nans((len(N_vec), len(x_vec)))  # create NaN vector
                        CF_vec = nans(len(x_vec))
                        N_need_vec = nans(len(x_vec))
                        fmax_v = nans(len(x_vec))  # create NaN vector
                        fmstd_v = nans(len(x_vec))  # create NaN vector
                        fcvm_v = nans(len(x_vec))

                        CF_min_vec[0:np.shape(CF_min_vec_old)[0],0:np.shape(CF_min_vec_old)[1]] = CF_min_vec_old
                        VAF_min_vec[0:np.shape(VAF_min_vec_old)[0],0:np.shape(VAF_min_vec_old)[1]] = VAF_min_vec_old
                        fmax_v[0:np.shape(fmax_v_old)[0]] = fmax_v_old
                        fmstd_v[0:np.shape(fmstd_v_old)[0]] = fmstd_v_old
                        fcvm_v[0:np.shape(fcvm_v_old)[0]] = fcvm_v_old
                        CF_vec[0:np.shape(CF_vec_old)[0]] = CF_vec_old
                        N_need_vec[0:np.shape(N_need_vec_old)[0]] = N_need_vec_old

                        if MPI.COMM_WORLD.rank == 0: print np.shape(x_vec), np.shape(N_need_vec), np.shape(CF_vec), np.shape(CF_min_vec), np.shape(VAF_min_vec)
                        if MPI.COMM_WORLD.rank == 0: print np.shape(fmax_v), np.shape(fmstd_v), np.shape(fcvm_v)

                        #if MPI.COMM_WORLD.rank == 0: print CF_min_vec, VAF_min_vec, x_vec, fmax_v, fmstd_v, fcvm_v
                        #if MPI.COMM_WORLD.rank == 0: print np.shape(CF_min_vec), np.shape(VAF_min_vec), np.shape(x_vec), np.shape(fmax_v), np.shape(fmstd_v), np.shape(fcvm_v)

                    if Nstop:
                        xi0 = where(isnan(N_need_vec))[-1][0]-1 # where(N_need_vec>0)[-1][-1] # find last results
                        ni = where(N_vec == N_need_vec[xi0])[-1][0] # find last min N
                        #if MPI.COMM_WORLD.rank == 0: print "xi0:", xi0, "ni:", ni, N_vec, N_need_vec[xi0]

                        #ni = (ni-1) # start at higher value next time, HACK
                        if ni < 0:
                            ni = 0
                        xi = xi0+1


                    if (Nstop is False) and Flip:
                        CF_min_vec = np.fliplr(CF_min_vec)
                        VAF_min_vec = np.fliplr(VAF_min_vec)
                        x_vec = x_vec[::-1]

                    #else:
                    #    ni += 1

                    #if MPI.COMM_WORLD.rank == 0: print "xi,", xi, "ni,", ni, len(N_vec), VAF_min_vec
                    #if MPI.COMM_WORLD.rank == 0: print fmax_v, fmstd_v, fcvm_v:
                    #if MPI.COMM_WORLD.rank == 0: print np.shape(x_vec), np.shape(N_need_vec), np.shape(CF_vec), np.shape(CF_min_vec), np.shape(VAF_min_vec)
                    #if MPI.COMM_WORLD.rank == 0: print np.shape(fmax_v), np.shape(fmstd_v), np.shape(fcvm_v)
                    #if MPI.COMM_WORLD.rank == 0: print CF_vec, N_need_vec, x_vec, fmax_v, fmstd_v, fcvm_v


                while xi < len(x_vec): #for xi, x in enumerate(x_vec): # cycle through input vector x_vec

                    x = x_vec[xi]
                    if MPI.COMM_WORLD.rank == 0: print "x=", x

                    if x_vec_type == "ex":
                        g_syn_ex = [x]*len(N)
                        g_syn_inh = [0]*len(N)
                        n_syn_inh = [0]*len(N)

                    elif x_vec_type == "constinh":
                        g_syn_ex = [1]*len(N)
                        g_syn_inh = [0]*len(N)
                        n_syn_inh = [0]*len(N)
                        fluct_g_e0 = [0]
                        fluct_g_i0 = [x*nS]*len(N)
                        fluct_std_e = [0]
                        fluct_std_i = [0]
                        fluct_tau_e = 0*ms
                        fluct_tau_i = 0*ms
                        if MPI.COMM_WORLD.rank == 0: print "fluct_g_i0:", fluct_g_i0
                        adjinh = True
                        inh_hold = [0]*len(N)

                        if "_twopop_" in do:
                            fluct_g_e0 = [fluct_g_e0[0], fluct_g_e0[0]]
                            fluct_g_i0 = [fluct_g_i0[0],fluct_g_i0[0]]
                            fluct_std_e = [fluct_std_e[0], fluct_std_e[0]]
                            fluct_std_i = [fluct_std_i[0], fluct_std_i[0]]
                            g_syn_ex = [g_syn_ex[0], g_syn_ex[0]]
                            inh_hold = [inh_hold[0], inh_hold[0]]
                            g_syn_inh = [g_syn_inh[0], g_syn_inh[0]]


                    elif x_vec_type == "inhin":
                        g_syn_ex = [1]*len(N)
                        g_syn_inh = [1]*len(N)
                        n_syn_inh = [4]*len(N)
                        inh_hold = [x]*len(N)
                        if MPI.COMM_WORLD.rank == 0: print "inh_hold:", inh_hold
                        adjfinh = True


                    elif x_vec_type == "inh":
                        g_syn_ex = [1]*len(N)
                        g_syn_inh = [1]*len(N)
                        n_syn_inh = [4]*len(N)
                        inh_hold = [x]*len(N)
                        if MPI.COMM_WORLD.rank == 0: print "inh_hold:", inh_hold
                        adjfinh = True

                        if ("_lowinh" in do):
                            g_syn_inh = [0.4]*len(N) #[0.354]


                    elif x_vec_type == "in":
                        ihold = [x]*len(N)
                        g_syn_inh = [0]*len(N)
                        n_syn_inh = [0]*len(N)
                        g_syn_ex = [-1]*len(N)

                    else:
                        g_syn_ex = [1]*len(N)
                        g_syn_inh = [1]*len(N)
                        n_syn_inh = [1] *len(N)

                    do_break = 0

                    #for ni, N in enumerate(N_vec): # cycle through input vector N_vec
                    #    N = [N]

                    while ni < len(N_vec): #in range(ni_start, len(N_vec)):

                        if "_CF-N" in do:
                            N = [N_vec[ni]]*len(N)
                            syn_max_mf = N # possible mossy fibres per synapse
                            syn_max_inh = N # possible Golgi cells per synapse

                        elif "_CF-cutf" in do:
                            cutf = N_vec[ni]
                            xmax = cutf

                        #if MPI.COMM_WORLD.rank == 0:
                        #    print ni
                        #    print xi
                        #    print CF_min_vec
                        #    print CF_min_vec[ni][xi]

                        if isnan(CF_min_vec[ni][xi]) == False:
                            if "_CF-N" in do:
                                print "Jump, value for x=", x, "N=", N, "exists"
                            elif "_CF-cutf" in do:
                                print "Jump, value for x=", x, "cutf=", cutf, "exists"

                            ni += 1

                        else:

                            if do_break == 1:
                                print "BREAK, but why here?"
                                break

                            if MPI.COMM_WORLD.rank == 0:
                                if "_CF-N" in do:
                                    print "N=", N
                                elif "_CF-cutf" in do:
                                    print "cutf=", cutf

                            pickle_prefix = prefix + "_x" + str(x)

                            pop = Population(cellimport = cellimport,  celltype = celltype, cell_exe = cell_exe, N = N, temperature = temperature, ihold = ihold, ihold_sigma = ihold_sigma, amp = amp, amod = amod, give_freq = give_freq, do_run = do_run, pickle_prefix = pickle_prefix, istart = istart, istop = istop, di = di, dt = dt)

                            pop.bin_width = float(bin_width)
                            pop.jitter = float(jitter)

                            pop.method_interpol = method_interpol[:]
                            pop.no_fmean = False

                            pop.fluct_g_e0 = fluct_g_e0[:]
                            pop.fluct_g_i0 = fluct_g_i0[:]
                            pop.fluct_std_e = fluct_std_e[:]
                            pop.fluct_std_i = fluct_std_i[:]
                            pop.fluct_tau_e = float(fluct_tau_e)
                            pop.fluct_tau_i = float(fluct_tau_i)

                            pop.CF_var = CF_var
                            pop.xmax = xmax

                            pop.tau1_ex=tau1_ex[:]
                            pop.tau2_ex=tau2_ex[:]
                            pop.tau1_inh=tau1_inh[:]
                            pop.tau2_inh=tau2_inh[:]

                            pop.n_syn_ex = n_syn_ex[:]
                            pop.g_syn_ex = g_syn_ex[:]
                            pop.g_syn_ex_s = g_syn_ex_s[:]

                            pop.noise_syn = noise_syn[:]
                            pop.noise_syn_tau = noise_syn_tau[:]
                            pop.noise_syn_inh = noise_syn_inh[:]
                            pop.noise_syn_tau_inh = noise_syn_tau_inh[:]
                            pop.noise_a = noise_a[:]
                            pop.noise_a_inh = noise_a_inh[:]

                            pop.inh_hold = inh_hold[:]

                            pop.n_syn_inh = n_syn_inh[:]
                            pop.g_syn_inh = g_syn_inh[:]
                            pop.g_syn_inh_s = g_syn_inh_s[:]

                            pop.bypass_cell = bypass_cell

                            pop.do_if = do_if
                            pop.adjinh = adjinh
                            pop.adjfinh = adjfinh

                            pop.syn_max_mf = syn_max_mf[:]
                            pop.syn_max_inh = syn_max_inh[:]
                            pop.inh_hold_sigma = sigma_inh_hold[:] #[float(sigma_inh_hold)]
                            pop.ihold_sigma = sigma_ihold[:] #[float(sigma_ihold)]

                            pop.syn_ex_dist = syn_ex_dist[:]
                            pop.syn_inh_dist = syn_inh_dist[:]

                            pop.a_celltype = a_celltype
                            pop.factor_celltype = factor_celltype

                            pop.tmax = 10*s # return full length of signal
                            pop.give_psd = give_psd
                            pop.simstep = simstep
                            pop.data_dir = data_dir

                            MPI.COMM_WORLD.Barrier()
                            results = pop.fun_cnoise_Stim(t_stim = t_stim, cutf = cutf, sexp = sexp, t_qual = t_qual)
                            MPI.COMM_WORLD.Barrier()

                            freq_used, amp_vec, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1')
                            freq_times, spike_freq, fmean, method_interpol, SNR, VAF, Qual = results.get('freq_times'), results.get('spike_freq'), results.get('fmeanA'), results.get('method_interpol'), results.get('SNR'), results.get('VAF'), results.get('Qual')
                            stim, stim_re_mat, t_startstop, current_re = results.get('stim'), results.get('stim_re_mat'), results.get('t_startstop'), results.get('current_re')
                            fmax, fmstd, fcvm = results.get('fmaxA'), results.get('fmstdA'), results.get('fcvmA')

                            #pop.del_cells()
                            #pop.pc.gid_clear()
                            #for n in range(pop.n_celltypes):
                            #    for m in pop.cells[n]:
                            #        del m
                            #del pop.cells

                            if do_run_now:
                                pop.delall()
                            del pop
                            pop = None
                            results = None

                            MPI.COMM_WORLD.Barrier()

                            if MPI.COMM_WORLD.rank == 0:


                                if fmean == 0:
                                    do_break = 1
                                else:
                                    do_break = 0

                            do_break = MPI.COMM_WORLD.bcast(do_break, root=0)

                            if do_break == 1:
                                print "BREAK: CF too low!"
                                break

                            if MPI.COMM_WORLD.rank == 0:

                                mag[0,:] = mag[0,:]/mag[0,0]

                                plt.figure('results_transfer_syn') # plot quick status

                                if "_unifih" in do:
                                    fmean = fmax

                                label = "CF=" + str(round(fmean,1))

                                iend = mlab.find(freq_used >= xmax)[0]
                                #VAF_min = min(VAF[1][0,1:iend-1])
                                VAF_min = min(VAF[1][0,0:iend])
                                VAF_min = np.mean(VAF[1][0,0:iend])
                                print "VAF_min: ", VAF_min, " freq_used[0:iend]: ", freq_used[0:iend]
                                VAF_min_vec[ni][xi] = VAF_min
                                print fmean
                                CF_min_vec[ni][xi] = fmean

                                a1 = plt.subplot(3,1,1)
                                a1.semilogx(freq_used[0:iend], 20*log10(mag[0,0:iend]), color=color, linewidth=linewidth, linestyle = linestyle, alpha=1, label=label)

                                if "_CF-N" in do:
                                    plt.text(0.5, 1.1, r'Pop x=' + str(x) + ',N=' + str(N) + ',CF=' + str(round(fmean,1)) + ',fmax=' + str(round(fmax,1)) + ',fmstd=' + str(round(fmstd,1)) + ',VAF_min=' + str(VAF_min), transform=a1.transAxes, fontsize=10, va='center', ha='center')

                                elif "_CF-cutf" in do:
                                    plt.text(0.5, 1.1, r'Pop x=' + str(x) + ',cutf=' + str(cutf) + ',CF=' + str(round(fmean,1)) + ',fmax=' + str(round(fmax,1)) + ',fmstd=' + str(round(fmstd,1)) + ',VAF_min=' + str(VAF_min), transform=a1.transAxes, fontsize=10, va='center', ha='center')

                                ax2 = plt.subplot(3,1,2)
                                ax2.semilogx(freq_used[0:iend], pha[0,0:iend], color=color, linewidth=linewidth, linestyle = linestyle, alpha=1, label=label)

                                lg = ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), handlelength=3, ncol=2, handletextpad=0.5, columnspacing=1)
                                #fr = lg.get_frame()
                                #fr.set_lw(0.2)

                                a3 = plt.subplot(3,1,3)
                                a3.plot(freq_used[0:iend], VAF[1][0,0:iend]*100, linewidth=linewidth, color=color, alpha=1, linestyle = linestyle, marker='*')

                                plt.savefig("./figs/Pub/" + str(prefix) + "_x" + str(x) + "_N" + str(N[0]) +  ".pdf", dpi = 300, transparent=True) # save it
                                plt.clf()

                                do_break = 0
                                CF_vec[xi] = fmean
                                fmax_v[xi] = fmax
                                fmstd_v[xi] = fmstd
                                fcvm_v[xi] = fcvm

                                if VAF_min >= 0.9:
                                    if "_CF-N" in do:
                                        N_need_vec[xi] = N[0]
                                    elif "_CF-cutf" in do:
                                        N_need_vec[xi] = cutf
                                    if Nstop: do_break = 1

                                if Nstop is False:
                                    # Always plot!
                                    plt.figure('results_CF-N_plot')
                                    for ip, N0 in enumerate(N_vec):
                                        plot(CF_min_vec[ip], VAF_min_vec[ip]*100, color=color, linewidth=linewidth, marker='o')
                                    plt.savefig("./figs/Pub/" + str(prefix) + ".pdf", dpi = 300, transparent=True) # save it
                                    plt.clf()

                                    results = {'xi':xi, 'ni':ni, 'x_vec':x_vec, 'CF_vec':CF_vec, 'CF_min_vec':CF_min_vec, 'VAF_min_vec':VAF_min_vec, 'N_need_vec':N_need_vec, 'freq_used':freq_used,'mag':mag,'pha':pha,'VAF':VAF, 'fmax_v':fmax_v, 'fmstd_v':fmstd_v, 'fcvm_v':fcvm_v}
                                    pickle.dump( results, gzip.GzipFile( filepath, "wb" ) )

                            do_break = MPI.COMM_WORLD.bcast(do_break, root=0)

                            ni += 1

                            if do_break == 1:
                                print "BREAK: necessary value found!"
                                break

                    if Nstop:
                        if do_break == 0:
                            print "BREAK: no necessary value found, loop exited without result, break xvec loop!"
                            break


                    if Nstop: # Plot only after succesful N_need finding

                        ni = (ni-1) # start at higher value next time, HACK
                        if ni < 0:
                            ni = 0

                        if MPI.COMM_WORLD.rank == 0:
                            plt.figure('results_CF-N_test')
                            plot(CF_vec, N_need_vec, color=color, linewidth=linewidth, marker='o')
                            plt.savefig("./figs/Pub/" + str(prefix) + ".pdf", dpi = 300, transparent=True) # save it
                            plt.clf()


                    if MPI.COMM_WORLD.rank == 0:
                        results = {'xi':xi, 'ni':ni, 'x_vec':x_vec, 'CF_vec':CF_vec, 'CF_min_vec':CF_min_vec, 'VAF_min_vec':VAF_min_vec, 'N_need_vec':N_need_vec, 'freq_used':freq_used,'mag':mag,'pha':pha,'VAF':VAF, 'fmax_v':fmax_v, 'fmstd_v':fmstd_v, 'fcvm_v':fcvm_v}
                        pickle.dump( results, gzip.GzipFile( filepath, "wb" ) )

                    MPI.COMM_WORLD.Barrier()

                    xi += 1

                    if Nstop:
                        pass
                    else:
                        ni = 0

            else:

                if MPI.COMM_WORLD.rank == 0:
                    results = pickle.load( gzip.GzipFile( filepath, "rb" ) )
                    x_vec, CF_vec, N_need_vec, freq_used, mag, pha, VAF = results.get('x_vec'), results.get('CF_vec'), results.get('N_need_vec'), results.get('freq_used'), results.get('mag'), results.get('pha'), results.get('VAF')
                    VAF_min_vec, CF_min_vec = results.get('VAF_min_vec'), results.get('CF_min_vec')
                    fmax, fmstd, fcvm = results.get('fmax'), results.get('fmstd'), results.get('fcvm')
                    fmax_v, fmstd_v, fcvm_v = results.get('fmax_v'), results.get('fmstd_v'), results.get('fcvm_v')

            if MPI.COMM_WORLD.rank == 0:

                if "talk1" in do:

                    axline, = ax_CF_N.semilogy(CF_vec, N_need_vec, color=color, linewidth=linewidth, marker='o')
                    axline.set_clip_on(False)

                    adjust_spines(ax_CF_N, ['left','bottom'], d_out = d_out, d_down = d_down)
                    ax_CF_N.set_yscale('log')
                    ax_CF_N.yaxis.set_ticks(array([1,10,100,1000,10000]))
                    ax_CF_N.set_yticklabels(('1', '10', '100', '1000', '10000'))

                    if "_CF-N" in do:
                        ax_CF_N.set_ylabel("Population size N")
                    elif "_CF-cutf" in do:
                        ax_CF_N.set_ylabel("Maximum input frequency (Hz)")

                    ax_CF_N.set_xlabel("Mean carrier frequency CF (spikes/s)")

                    ax_CF_N.axis(xmin=0, xmax=80)
                    #ax_CF_N.axis(ymin=0.9, ymax=10000)

                    plt.savefig("./figs/Pub/" + str(prefix) + ".pdf", dpi = 300, transparent=True) # save it

                    from pyPdf import PdfFileReader, PdfFileWriter

                    output = PdfFileWriter()

                    pdfOne = PdfFileReader(file("./figs/Pub/" + str(prefix) + ".pdf", "rb"))
                    pdfTwo = PdfFileReader(file("./figs/Inserts/intro_syn2.pdf", "rb"))

                    input1 = pdfOne.getPage(0)
                    input1.mergeTransformedPage(pdfTwo.getPage(0),[0.9, 0, 0, 0.9, 190, 130])

                    output.addPage(input1)

                    outputStream = file(r"./figs/Pub/" + str(prefix) + "_merge.pdf", "wb")
                    output.write(outputStream)
                    outputStream.close()

                if "_fig8" in do:

                    label=""
                    if ("_label18_" in do): label = "a=1, iIF"
                    if ("_label19_" in do): label = "a=1, iIF, push-pull"
                    if ("_label20_" in do): label = "a=10, iIF, push-pull"
                    if ("_label21_" in do): label = "a=1, GrC,"
                    if ("_label22_" in do): label = "a=1, GrC, push-pull"
                    if ("_label23_" in do): label = "a=1, IF"
                    if ("_label24_" in do): label = "a=1, IF, push-pull"
                    if ("_label25_" in do): label = "a=1, GrC, const. Inh."
                    if ("_label26_" in do): label = "a=1, IF, const. Inh."

                    if "_first_" in do:
                        CF = np.arange(0,100)
                        N = CF/2
                        axcf.plot(CF, N, color="gray", linestyle="--")
                        axcf.text(22, -0.5, "Nyquist\nfrequency", color='gray', fontsize = params['legend.fontsize'])

                        axcf.plot(CF, CF, color="gray", linestyle="--")
                        axcf.text(48, 43, "1:1", color='gray', fontsize = params['legend.fontsize'])

                        axcf.plot(CF/2, CF, color="gray", linestyle="--")
                        axcf.text(35, 75, "1:2", color='gray', fontsize = params['legend.fontsize'])

                        axcf.plot(CF/6, CF, color="gray", linestyle="--")
                        axcf.text(16, 90, "1:6", color='gray', fontsize = params['legend.fontsize'])

                        #if ("_label1_" in do): ax01.text(26, 20, "ideal IF", color=color, fontsize = params['legend.fontsize'])
                        #if ("_label3_" in do): ax01.text(21, 4,  "GrC + Inhibition", color=color, fontsize = params['legend.fontsize'])

                    print shape(CF_vec), shape(N_need_vec), shape(fmstd_v)
                    #axline, = ax_CF_N.plot(N_need_vec, fmax_v, color=color, linewidth=linewidth, marker='o') # CF_vec
                    #axline, = axcf.plot(N_need_vec, CF_vec, marker=markerstyle, color=color, ms=ms2, label=label, linestyle=linestyle)
                    #eb = axcf.errorbar(N_need_vec, CF_vec, yerr=fmstd_v, fmt=linestyle, color=color, elinewidth=linewidth)

                    (_, caps, _) = axcf.errorbar(CF_vec, N_need_vec,  color=color, linewidth=linewidth, marker=markerstyle, linestyle=linestyle, ms=ms2, label=label, elinewidth=linewidth) #

                    # xerr=fmstd_v, fmt=linestyle,
                    #for cap in caps:
                    #    cap.set_linestyle(linestyle)

                    #noclip(ax01)
                    #axline.set_clip_on(False)

                    print N_need_vec
                    print CF_vec

                    lg = axcf.legend(labelspacing=0.2, loc=4, handlelength=3, handletextpad=0.1) # , bbox_to_anchor=(0.5,-0.05)
                    #lg.draw_frame(False)
                    fr = lg.get_frame()
                    fr.set_lw(0.2)

                    if ("_pos13_" in do):
                        adjust_spines(axcf, ['left'], d_out = d_out, d_down = d_down)
                    else:
                        adjust_spines(axcf, ['left','bottom'], d_out = d_out, d_down = d_down)
                    #ax_CF_N.set_yscale('log')
                    #ax_CF_N.yaxis.set_ticks(array([1,10,100,1000,10000]))
                    #ax_CF_N.set_yticklabels(('1', '10', '100', '1000', '10000'))

                    #axcf.set_xlabel("Max. input frequency (Hz) (cutoff)", labelpad=2)
                    #axcf.set_ylabel("Carrier frequency (spikes/s)", labelpad=1)
                    #axcf.axis(xmin=0, xmax=71)
                    #axcf.axis(ymin=-2, ymax=50)

                    axcf.set_ylabel("Max. input frequency (Hz) (cutoff)", labelpad=3)
                    axcf.set_xlabel("Mean effective firing-rate (spikes/s)", labelpad=4)
                    axcf.axis(ymin=0, ymax=101.5)
                    axcf.axis(xmin=-2, xmax=50)

                    filename="./figs/Pub/" + str(prefix)
                    savefig(filename + ".pdf", dpi = 300) # save it
                    savefig(filename + ".png", dpi = 300) # save it


                elif "_fig" in do:

                    if Nstop:

                        #axline, = ax_CF_N.plot(N_need_vec, fmax_v, color=color, linewidth=linewidth, marker='o') # CF_vec
                        ax_CF_N.errorbar(CF_vec, N_need_vec, xerr=fmstd_v, color=color, linewidth=linewidth, marker='o', linestyle='-') #

                        #axline.set_clip_on(False)

                        if "_first_" in do:
                            CF_vec = np.arange(0,60)
                            N_need_vec = CF_vec/2
                            ax_CF_N.plot(CF_vec, N_need_vec, "g:")
                            ax_CF_N.text(49, 23, "Nyquist\nfrequency", color='g', fontsize = params['legend.fontsize'])

                        if ("_talk_" in do):
                            if ("_label1_" in do): ax_CF_N.text(26, 20, "ideal IF", color=color, fontsize = params['legend.fontsize'])
                            if ("_label2_" in do): ax_CF_N.text(40, 14, "GrC", color=color, fontsize = params['legend.fontsize'])
                            if ("_label3_" in do): ax_CF_N.text(21, 4,  "GrC + Inhibition", color=color, fontsize = params['legend.fontsize'])


                        adjust_spines(ax_CF_N, ['left','bottom'], d_out = d_out, d_down = d_down)
                        #ax_CF_N.set_yscale('log')
                        #ax_CF_N.yaxis.set_ticks(array([1,10,100,1000,10000]))
                        #ax_CF_N.set_yticklabels(('1', '10', '100', '1000', '10000'))

                        if "_CF-N" in do:
                            ax_CF_N.set_ylabel("Population size N")
                        elif "_CF-cutf" in do:
                            ax_CF_N.set_ylabel("Maximum input frequency (Hz)")

                        ax_CF_N.set_xlabel("Mean carrier frequency CF (spikes/s)")

                        #ax_CF_N.axis(xmin=0, xmax=60)
                        #ax_CF_N.axis(ymin=0, ymax=30)

                        filename="./figs/Pub/" + str(prefix)
                        savefig(filename + ".pdf", dpi = 300) # save it
                        savefig(filename + ".png", dpi = 300) # save it
                        #os.system('rsvg-convert -f pdf -o ' + filename +'.pdf ' + filename + '.svg')

                    else:

                        for ip, N0 in enumerate(N_vec):
                            ax_CF_N.plot(CF_min_vec[ip], VAF_min_vec[ip]*100, color=color_vec[0][ip], linewidth=linewidth, marker='o')

                        adjust_spines(ax_CF_N, ['left','bottom'], d_out = d_out, d_down = d_down)

                        #plt.savefig("./figs/Pub/" + str(prefix) + ".pdf", dpi = 300, transparent=True) # save it




plt.savefig("./figs/Pub/" + str(prefix) + ".png", dpi = 300) #, transparent=True
plt.show()
