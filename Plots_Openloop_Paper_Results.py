# -*- coding: utf-8 -*-
"""
@author: chris

mpiexec -f ~/machinefile -n 1 python Plots_Openloop_Paper_Results.py -o fig3 --noplot 2>&1 | tee log/log3.txt
qsub -v J=Plots_Openloop_Paper_Results.py,O=fig3 -pe ompigige 96 PBSinsigneo.sh

mpiexec -f ~/machinefile -n 1 python Plots_Openloop_Paper_Results.py -o fig4 --noplot 2>&1 | tee log/log4.txt
qsub -v J=Plots_Openloop_Paper_Results.py,O=fig4 -pe ompigige 1 PBSinsigneo.sh

mpiexec -f ~/machinefile -n 50 python Plots_Openloop_Paper_Results.py -o fig4b --noplot 2>&1 | tee log/log4b.txt
qsub -v J=Plots_Openloop_Paper_Results.py,O=fig4b -pe ompigige 50 PBSinsigneo.sh

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

from neuron import h

from units import *
from Population import *
from Stimulation import *
from Plotter import *
from Stimhelp import *
from cells.IfCell import *

# PLOS: Fonts: 8 - 12, Multi-panel figures labels: 12

fig_size =  [28*0.3937, 10*0.3937]
params = {'backend': 'ps',
          'axes.labelsize': 8,
          'axes.linewidth' : 0.5,
          'title.fontsize': 8,
          'text.fontsize': 8,
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

#matplotlib.rc('font', **{'sans-serif' : 'Arial'}) #, 'family' : 'sans-serif'})
#matplotlib.rc('font', family='sans-serif')

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
gray = 'gray'
black = 'black'

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

color0 = 'black' # BLACK
color1 = 'blue' # BLUE
color2 = 'red' # RED
color3 = 'gray' # GRAY
color4 = 'purple' # PURPLE
color5 = 'orange' # ORANGE
color6 = 'green' # GREEN

linewidth = 1

xmax = 40
#xmax = 100

t_stim = 1000*s # only for cnoise

delay_baseline = 8
dt = 0.025*ms
tstop_if = 1*s
plot_train = False

# Used for conversion, ignore
data_dir = "./publish/openloop/fulldata"
minimal_dir = "./publish/openloop/minimal"
export = False

data_dir = "./publish/openloop/minimal"
minimal_dir = False

# FIGURE 3
if results.opt == "fig3":
    t_stim = 1000*s # only for cnoise
    #do_vec = np.array(["cell_results_transfer_grc", "pop_transfer_if_wn", "pop_transfer_grc_wn"])
    #do_vec = np.array(["cell_results_transfer_grc", "pop_transfer_if_cn_SNR_addn10", "pop_transfer_grc_cn_SNR_addn10"])
    export = False
    do_vec = np.array([
                        "cell_results_transfer_grc_alt_fig3_normalize_first_",
                        "cell_results_transfer_resif_alt_fig3_normalize_",
                        "pop_transfer_if_cn_alt_full_fig3_normalize_",
                        "pop_transfer_grc_cn_alt_full_fig3_normalize_",
                        "pop_transfer_resif_cn_alt_full_fig3_normalize_",
                        ])


if results.opt == "fig3test":
    t_stim = 10*s # only for cnoise
    data_dir = "./data"
    minimal_dir = False
    export = False
    #plot_train = True
    do_vec = np.array([
                        "pop_transfer_grc_cn_alt_full_fig3_normalize_",
                        ])


if results.opt == "fig3b":
    t_stim = 1000*s # only for cnoise
    do_vec = np.array([

                       "pop_transfer_if_bn_N1_consta20_colorb1_fig4_normalize_keep_",
                       "pop_transfer_if_bn_N1_consta40_colorr1_fig4_keep_",
                       "pop_transfer_if_bn_N1_consta60_colorp1_fig4_keep_",

                       ])

if results.opt == "fig3c":
    t_stim = 100*s # only for cnoise
    do_vec = np.array([

                       "pop_transfer_if_bn_N1_ihold20_a01_colorb1_fig4b_normalize_",
                       "pop_transfer_if_bn_N1_ihold40_a005_colorr1_fig4b_",
                       "pop_transfer_if_bn_N1_ihold60_a0033_colorp1_fig4b_",

                       ])

if results.opt == "fig3d":
    t_stim = 100*s # only for cnoise
    do_vec = np.array([

                       "pop_transfer_if_cn_N1_ihold20_a01_colorb1_fig4b_normalize_",
                       "pop_transfer_if_cn_N1_ihold40_a005_colorr1_fig4b_",
                       "pop_transfer_if_cn_N1_ihold80_a0025_colorp1_fig4b_",

                       ])

# FIGURE 4
if results.opt == "fig4":
    t_stim = 100*s # only for cnoise
    export = False
    do_vec = np.array([
                        "pop_transfer_grc_cn_a1_N100_alt_colorblack_fig4_addspe_normalize_",
                        #"pop_transfer_grc_cn_a1_N100_ihsigma0.1_alt_colorgray_fig4_addspe_normalize_",
                        #"pop_transfer_resif_cn_a1_N100_alt_colorblack_fig4_addspe_normalize_",
                        #"pop_transfer_if_cn_a1_N100_alt_colorblack_fig4_addspe_normalize_",

                        #"pop_transfer_if_cn_addn100_alt_full_colorg2_fig4_normalize_",
                        "pop_transfer_grc_cn_addn100_alt_full_colorg2_fig4_tqual_normalize_first_",
                        #"pop_transfer_resif_cn_addn100_alt_full_colorg2_fig4_normalize_",

                        #"pop_transfer_if_cn_addn1_alt_full_colorg1_fig4_normalize_",
                        "pop_transfer_grc_cn_addn1_alt_full_colorg1_fig4_tqual1_normalize_",
                        #"pop_transfer_resif_cn_addn1_alt_full_colorg1_fig4_normalize_",


                        ])


if results.opt == "fig4ih":
    t_stim = 1000*s # only for cnoise
    do_vec = np.array([
                        "pop_transfer_grc_cn_a1_N100_alt_colorblack_fig4_addspe_normalize_keep_",
                        "pop_transfer_grc_cn_a1_N100_ihsigma0.05_alt_colorgray_fig4_addspe_",
                        "pop_transfer_grc_cn_a1_N100_F046_alt_colorp1_fig4_addspe_",

                        ])



if results.opt == "fig4a":
    t_stim = 1000*s # only for cnoise
    do_vec = np.array([
                        #"pop_transfer_if_cn_a01_N100_alt_colorgray_fig4_addspe_normalize_",
                        #"pop_transfer_resif_cn_a01_N100_alt_colorgray_fig4_addspe_normalize_",
                        #"pop_transfer_grc_cn_a01_N100_alt_colorgray_fig4_addspe_normalize_",

                        #"pop_transfer_grc_cn_a01_addn1_alt_full_colorg1_fig4_tqual_normalize_",
                        #"pop_transfer_if_cn_a01_addn1_alt_full_colorg1_fig4_normalize_",
                        #"pop_transfer_resif_cn_a01_addn1_alt_full_colorg1_fig4_normalize_",

                        "pop_transfer_if_cn_a01_addn100_alt_full_colorg2_fig4_normalize_",
                        "pop_transfer_resif_cn_a01_addn100_alt_full_colorg2_fig4_normalize_",
                        "pop_transfer_grc_cn_a01_addn100_alt_full_colorg2_fig4_tqual_normalize_",

                        ])


#if results.opt == "fig4b":
#    t_stim = 1000*s # only for cnoise
#    do_vec = np.array([
#                       "pop_transfer_if_cn_N10000_lowcf_fig4b_",
#                       "pop_transfer_if_cn_N10000_lowcf_slownoise_fig4b_",
#                       "pop_transfer_grc_cn_N10000_lowcf_fig4b_",
#                       "pop_transfer_grc_cn_N10000_lowcf_slownoise_fig4b_",
#                       "pop_transfer_grc_cn_N1000_lowcf_slownoise_fig4b_",
#                       "pop_transfer_resif_cn_N10000_lowcf_fig4b_",
#                       "pop_transfer_resif_cn_N10000_lowcf_slownoise_fig4b_",
#                       "pop_transfer_ifpass_cn_N10000_lowcf_fig4b_",
#                       "pop_transfer_ifpass_cn_N10000_lowcf_slownoise_fig4b_",
#                       ])


if results.opt == "fig4b_old":
    t_stim = 1000*s # only for cnoise
    do_vec = np.array([
                       "pop_transfer_if_twopop_cn_N500_lowcf_slownoise_colorr1_fig4b_first_normalize_",

                       "pop_transfer_resif_twopop_cn_N500_lowcf_slownoise_colorr1_fig4b_",

                       "pop_transfer_grc_twopop_cn_N500_lowcf_slownoise_colorr1_fig4b_",

                       "pop_transfer_if_cn_N1000_lowcf_slownoise_colorb1_fig4b_",

                       "pop_transfer_resif_cn_N1000_lowcf_slownoise_colorb1_fig4b_",

                       "pop_transfer_grc_cn_N1000_lowcf_slownoise_colorb1_fig4b_",



                       "pop_transfer_if_twopop_cn_N500_lowcf_colorr1_fig4b_first_normalize_",

                       "pop_transfer_resif_twopop_cn_N500_lowcf_colorr1_fig4b_",

                       "pop_transfer_grc_twopop_cn_N500_lowcf_colorr1_fig4b_",

                       "pop_transfer_if_cn_N1000_lowcf_colorb1_fig4b_",

                       "pop_transfer_resif_cn_N1000_lowcf_colorb1_fig4b_",

                       "pop_transfer_grc_cn_N1000_lowcf_colorb1_fig4b_",


                       #"pop_transfer_grc_twopop_cn_N100_lowcf_slownoise_color2_fig4b_",

                       #"pop_transfer_ifpass_twopop_cn_N100_lowcf_color2_fig4b_",
                       #"pop_transfer_ifpass_twopop_cn_N100_lowcf_slownoise_color2_fig4b_",
                       ])

# FIGURE 5
if results.opt == "fig4b":
    t_stim = 1000*s # only for cnoise
    export = False
    do_vec = np.array([
                       "pop_transfer_if_twopop_cn_N50_lowcf_colorr1_fig4b_normalize_",
                       "pop_transfer_resif_twopop_cn_N50_lowcf_colorr1_fig4b_",
                       "pop_transfer_grc_twopop_cn_N50_lowcf_colorr1_fig4b_tqual_first_",
                       "pop_transfer_if_cn_N100_lowcf_colorb1_fig4b_",
                       "pop_transfer_resif_cn_N100_lowcf_colorb1_fig4b_",
                       "pop_transfer_grc_cn_N100_lowcf_colorb1_fig4b_tqual_",

                       "pop_transfer_if_twopop_cn_N50_lowcf_slownoise_colorr1_fig4b_normalize_",
                       "pop_transfer_resif_twopop_cn_N50_lowcf_slownoise_colorr1_fig4b_",
                       "pop_transfer_grc_twopop_cn_N50_lowcf_slownoise_colorr1_fig4b_tqual_first_",
                       "pop_transfer_if_cn_N100_lowcf_slownoise_colorb1_fig4b_",
                       "pop_transfer_resif_cn_N100_lowcf_slownoise_colorb1_fig4b_",
                       "pop_transfer_grc_cn_N100_lowcf_slownoise_colorb1_fig4b_tqual_",

                       #"pop_transfer_grc_twopop_cn_N100_lowcf_slownoise_color2_fig4b_",
                       #"pop_transfer_ifpass_twopop_cn_N100_lowcf_color2_fig4b_",
                       #"pop_transfer_ifpass_twopop_cn_N100_lowcf_slownoise_color2_fig4b_",
                       ])

if results.opt == "fig4btest":
    t_stim = 20*s # only for cnoise
    data_dir = "./data"
    minimal_dir = False
    export = False
    plot_train = True
    do_vec = np.array([
                       #"pop_transfer_grc_twopop_cn_N50_lowcf_colorr1_fig4b_tqual_first_",
                       "pop_transfer_grc_cn_N100_lowcf_colorb1_fig4b_tqual_",

                       #"pop_transfer_grc_twopop_cn_N50_lowcf_slownoise_colorr1_fig4b_tqual_first_",
                       "pop_transfer_grc_cn_N100_lowcf_slownoise_colorb1_tqual_fig4b_",

                       "pop_transfer_if_cn_N100_lowcf_slownoise_colorb1_fig4b_",
                       "pop_transfer_if_cn_N100_lowcf_colorb1_fig4b_",

                       ])


if results.opt == "fig4l":
    t_stim = 1000*s # only for cnoise
    do_vec = np.array([

                       "pop_transfer_if_bn_N100_a01_ihold20_ihsigma0.5_colorb1_fig4b_normalize_keep_",
                       "pop_transfer_if_twopop_bn_N50_a01_ihold20_ihsigma0.5_colorr1_fig4b_normalize_keep_",

                       "pop_transfer_if_bn_N100_a10_ihold20_ihsigma0.5_colorg1_fig4b_normalize_keep_",
                       "pop_transfer_if_twopop_bn_N50_a10_ihold20_ihsigma0.5_colorp1_fig4b_normalize_keep_",

                       "pop_transfer_if_bn_N100_a1_ihold20_ihsigma0.5_coloro1_fig4b_normalize_keep_",
                       "pop_transfer_if_twopop_bn_N50_a1_ihold20_ihsigma0.5_colorblack_fig4b_normalize_keep_",


                       "pop_transfer_if_bn_N100_a01_ihold20_colorb2_fig4b_normalize_keep_",
                       "pop_transfer_if_twopop_bn_N50_a01_ihold20_colorr2_fig4b_normalize_keep_",

                       "pop_transfer_if_bn_N100_a10_ihold20_colorg2_fig4b_normalize_keep_",
                       "pop_transfer_if_twopop_bn_N50_a10_ihold20_colorp2_fig4b_normalize_keep_",

                       "pop_transfer_if_bn_N100_a1_ihold20_coloro2_fig4b_normalize_keep_",
                       "pop_transfer_if_twopop_bn_N50_a1_ihold20_colorgray_fig4b_normalize_keep_",


                       "pop_transfer_if_bn_N100_a01_ihold20_addn1b_ihsigma0.5_colorb1_fig4b_normalize_keep_",
                       "pop_transfer_if_twopop_bn_N50_a01_ihold20_addn1b_ihsigma0.5_colorr1_fig4b_normalize_keep_",

                       "pop_transfer_if_bn_N100_a10_ihold20_addn1d_ihsigma0.5_colorg1_fig4b_normalize_keep_",
                       "pop_transfer_if_twopop_bn_N50_a10_ihold20_addn1d_ihsigma0.5_colorp1_fig4b_normalize_keep_",

                       "pop_transfer_if_bn_N100_a1_ihold20_addn1c_ihsigma0.5_coloro1_fig4b_normalize_keep_",
                       "pop_transfer_if_twopop_bn_N50_a1_ihold20_addn1c_ihsigma0.5_colorblack_fig4b_normalize_keep_",


                       "pop_transfer_if_bn_N100_a01_ihold20_addn1b_colorb2_fig4b_normalize_keep_",
                       "pop_transfer_if_twopop_bn_N50_a01_ihold20_addn1b_colorr2_fig4b_normalize_keep_",

                       "pop_transfer_if_bn_N100_a10_ihold20_addn1d_colorg2_fig4b_normalize_keep_",
                       "pop_transfer_if_twopop_bn_N50_a10_ihold20_addn1d_colorp2_fig4b_normalize_keep_",

                       "pop_transfer_if_bn_N100_a1_ihold20_addn1c_coloro2_fig4b_normalize_",
                       "pop_transfer_if_twopop_bn_N50_a1_ihold20_addn1c_colorgray_fig4b_normalize_",

                       ])


if results.opt == "fig4btalk":
    t_stim = 1000*s # only for cnoise
    do_vec = np.array([
                       "pop_transfer_if_cn_N10000_lowcf_color0_fig4b_talk_first",
                       "pop_transfer_if_cn_N10000_lowcf_slownoise_color0_fig4b_talk_first_",

                       "pop_transfer_grc_cn_N10000_lowcf_color2_fig4b_talk_",
                       "pop_transfer_grc_cn_N10000_lowcf_slownoise_color2_fig4b_talk_",

                       "pop_transfer_resif_cn_N10000_lowcf_color0_fig4b_talk_",
                       "pop_transfer_resif_cn_N10000_lowcf_slownoise_color0_fig4b_talk_",

                       "pop_transfer_grc_cn_N1000_lowcf_slownoise_color9_fig4b_talk_noplot_",
                       ])


if results.opt == "if":
    do_vec = np.array(["pop_transfer_resif_cn_alt", "pop_transfer_if_cn_alt"])


if results.opt == "prk":
    t_stim = 100*s # only for cnoise
    do_vec = np.array(["pop_transfer_prk_pn_figX_"])

if results.opt == "prk_equi":
    t_stim = 50*s # only for cnoise
    do_vec = np.array(["pop_transfer_equi_prk_pn_figX_"])

if results.opt == "goc":
    t_stim = 100*s # only for cnoise
    #do_vec = np.array(["pop_transfer_goc_cn_figX_"])
    do_vec = np.array(["cell_results_transfer_goc_alt_fig3_"])

if results.opt == "stl":
    t_stim = 100*s # only for cnoise
    #do_vec = np.array(["pop_transfer_stl_cn_figX_"])
    do_vec = np.array(["cell_results_transfer_stl_alt_fig3_"])

if results.opt == "if0":
    do_vec = np.array(["pop_transfer_if0_cn_alt"])


#do_vec = np.array(["pop_transfer_grc_cn_poster_alt" , "pop_transfer_resif_cn_poster_alt", "pop_transfer_if_cn_poster_alt"])
#do_vec = np.array(["pop_transfer_if_cn_addn10_poster_alt", "pop_transfer_grc_cn_addn10_poster_alt"])
#do_vec = np.array(["pop_transfer_grc_cn_alt"])


if MPI.COMM_WORLD.rank == 0:

    if ("_poster_" in do_vec[0]) or ("_talk_" in do_vec[0]):
        color0 = '#000000' # Black
        color1 = '#00A0E3' # Cyan
        color2 = '#E5097F' # Magenta
        color8 = '#FFED00' # Yellow
        color4 = '#393476' # Uni Blue
        color5 = '#E42A24' # Red
        color6 = '#009A47' # Dark Green
        color7 = '#78317B' # Lila
        color3 = '#BFB5B1' # Gray
        color9 = '#EC671F' # Orange

        linewidth = 1.5

    d_out = 10
    d_down = 10

    # FIGURE 3,4
    if ("_fig3_" in do_vec[0]) or ("_fig4_" in do_vec[0]):
        linewidth = 1.5

        if "alt" in do_vec[0]:
            fig_size =  [180*0.03937,100*0.03937] # 2-Column
        else:
            fig_size =  [180*0.03937,6.83] # 1.5-Column

        params['figure.figsize'] = fig_size
        rcParams.update(params)

        fig1 = plt.figure('results_transfer')

        font = font0.copy()
        font.set_family('sans-serif')
        font.set_weight('semibold')

        if "_addn" in do_vec[-1]:

            if "poster" not in do_vec[0]:

                gs = matplotlib.gridspec.GridSpec(2, 3,
                   width_ratios=[1,1,1],
                   height_ratios=[1,1]
                   )

                ax1 = plt.subplot(gs[0,2])
                ax1b = plt.subplot(gs[1,2])
                ax2 = plt.subplot(gs[0:2,0])
                ax2b = plt.subplot(gs[0:2,1])

                x1 = -0.08
                y1 = 1.11
                ax1.text(-0.08, y1+0.16, 'B', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
                ax2.text(x1, y1, 'A1', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
                ax2b.text(x1, y1, 'A2', transform=ax2b.transAxes, fontsize=12, va='top', fontproperties=font)

            else:

                gs = matplotlib.gridspec.GridSpec(1, 1,
                   width_ratios=[1],
                   height_ratios=[1]
                   )

                ax1 = plt.subplot(gs[0,0])

        else:

            if "poster" not in do_vec[0]:

                gs = matplotlib.gridspec.GridSpec(1, 4,
                   width_ratios=[1,1,1,1],
                   height_ratios=[1]
                   )

                ax1 = plt.subplot(gs[0,0])
                ax1b = plt.subplot(gs[0,1])
                ax2 = plt.subplot(gs[0,2])
                ax2b = plt.subplot(gs[0,3])

                x1 = -0.13
                y1 = 1.22
                ax1.text(x1, y1, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
                ax1b.text(x1, y1, 'A2', transform=ax1b.transAxes, fontsize=12, va='top', fontproperties=font)
                ax2.text(x1, y1, 'B1', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
                ax2b.text(x1, y1, 'B2', transform=ax2b.transAxes, fontsize=12, va='top', fontproperties=font)

            else:

                gs = matplotlib.gridspec.GridSpec(1, 2,
                   width_ratios=[1,1],
                   height_ratios=[1]
                   )

                ax2 = plt.subplot(gs[0,0])
                ax2b = plt.subplot(gs[0,1])


        if "_addn" in do_vec[-1]:
            if "_alt" not in do_vec[0]:
                gs.update(left=0.14, right=0.97, bottom=0.80, top=0.92, wspace=0.3, hspace=0.2)
            else:
                gs.update(left=0.065, right=0.97, bottom=0.57, top=0.94, wspace=0.3, hspace=0.3)
        else:
            if "_alt" not in do_vec[0]:
                gs.update(left=0.10, right=0.97, bottom=0.75, top=0.92, wspace=0.65, hspace=0.4)
            else:
                gs.update(left=0.065, right=0.97, bottom=0.65, top=0.89, wspace=0.4, hspace=0.4)

        if "_alt" not in do_vec[0]:
            gs2 = matplotlib.gridspec.GridSpec(3, 2,
                   width_ratios=[1,1],
                   height_ratios=[1,1,1]
                   )
        else:
            gs2 = matplotlib.gridspec.GridSpec(1, 3,
                   width_ratios=[1,1,1],
                   height_ratios=[1]
                   )

        if "_addn" in do_vec[-1]:
            if "_alt" not in do_vec[0]:
                gs2.update(bottom=0.08, top=0.64, left=0.13, right=0.97, wspace=0.1, hspace=0.4)
            else:
                gs2.update(bottom=0.1, top=0.42, left=0.065, right=0.97, wspace=0.1, hspace=0.3)

        else:
            if "_alt" not in do_vec[0]:
                gs2.update(bottom=0.08, top=0.55, left=0.13, right=0.97, wspace=0.1, hspace=0.4)
            else:
                gs2.update(bottom=0.1, top=0.40, left=0.065, right=0.97, wspace=0.1, hspace=0.4)


        if "_alt" not in do_vec[0]:
            ax3 = plt.subplot(gs2[0,0])
            ax4 = plt.subplot(gs2[0,1])
            ax4.text(0.3, 1.5, 'GrC model', transform=ax4.transAxes, fontsize=10, va='top')
            ax5 = plt.subplot(gs2[1,0])
            ax6 = plt.subplot(gs2[1,1])
            ax7 = plt.subplot(gs2[2,0])
            ax8 = plt.subplot(gs2[2,1])

        else:
            ax4 = plt.subplot(gs2[0,0])
            ax6 = plt.subplot(gs2[0,1])
            ax8 = plt.subplot(gs2[0,2])

        if "_poster" not in do_vec[0]:
            if "_alt" not in do_vec[0]:
                x1 = -0.05
                y1 = 1.25
                ax3.text(x1, y1, 'C1', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)
                ax3.text(0.2, 1.5, r'IF ($\tau$ = 15.7 ms)', transform=ax3.transAxes, fontsize=10, va='top')
                ax4.text(x1, y1, 'D1', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)
                ax5.text(x1, y1, 'C2', transform=ax5.transAxes, fontsize=12, va='top', fontproperties=font)
                ax6.text(x1, y1, 'D2', transform=ax6.transAxes, fontsize=12, va='top', fontproperties=font)
                ax7.text(x1, y1, 'C3', transform=ax7.transAxes, fontsize=12, va='top', fontproperties=font)
                ax8.text(x1, y1, 'D3', transform=ax8.transAxes, fontsize=12, va='top', fontproperties=font)
            else:
                x1 = -0.07
                y1 = 1.15
                ax4.text(x1, y1, 'C1', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)
                ax6.text(x1, y1, 'C2', transform=ax6.transAxes, fontsize=12, va='top', fontproperties=font)
                ax8.text(x1, y1, 'C3', transform=ax8.transAxes, fontsize=12, va='top', fontproperties=font)

    # FIGURE 5
    elif ("_fig4b_" in do_vec[0]) or ("_figX_" in do_vec[0]):

        if ("_talk_" in do_vec[0]):

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

            fig1 = plt.figure('results_transfer')

            font = font0.copy()
            font.set_family('sans-serif')
            font.set_weight('semibold')

            gs = matplotlib.gridspec.GridSpec(2, 2,
               width_ratios=[1,1],
               height_ratios=[1,1]
               )

            gs.update(bottom=0.12, top=0.88, left=0.1, right=0.97, wspace=0.1, hspace=0.4)

            ax1 = plt.subplot(gs[0,0])
            ax2 = plt.subplot(gs[1,0])

            ax5 = plt.subplot(gs[0,1])
            ax6 = plt.subplot(gs[1,1])


            fig2 = plt.figure('results_transfer2')

            font = font0.copy()
            font.set_family('sans-serif')
            font.set_weight('semibold')

            gs = matplotlib.gridspec.GridSpec(2, 2,
               width_ratios=[1,1],
               height_ratios=[1,1]
               )

            gs.update(bottom=0.125, top=0.86, left=0.1, right=0.97, wspace=0.1, hspace=0.2)

            ax3 = plt.subplot(gs[0,0])
            ax4 = plt.subplot(gs[1,0])

            ax7 = plt.subplot(gs[0,1])
            ax8 = plt.subplot(gs[1,1])


        # FIGURE 5
        else:

            fig_size =  [85*0.03937,120*0.03937] # 1.5-Column
            params['figure.figsize'] = fig_size
            rcParams.update(params)

            fig1 = plt.figure('results_transfer')

            font = font0.copy()
            font.set_family('sans-serif')
            font.set_weight('semibold')

            gs = matplotlib.gridspec.GridSpec(3, 2,
               width_ratios=[1,1],
               height_ratios=[1,1,1]
               )

            gs.update(left=0.17, right=0.97, bottom=0.35, top=0.89, wspace=0.1, hspace=0.4)

            ax1 = plt.subplot(gs[0,0])
            ax2 = plt.subplot(gs[1,0])
            ax3 = plt.subplot(gs[2,0])

            ax5 = plt.subplot(gs[0,1])
            ax6 = plt.subplot(gs[1,1])
            ax7 = plt.subplot(gs[2,1])


            gs2 = matplotlib.gridspec.GridSpec(1, 2,
               width_ratios=[1,1],
               height_ratios=[1]
               )

            gs2.update(left=0.17, right=0.96, bottom=0.09, top=0.24, wspace=0.1, hspace=0.2)

            ax4 = plt.subplot(gs2[0,0])
            ax8 = plt.subplot(gs2[0,1])

        if ("_poster_" not in do_vec[0]) and ("_talk_" not in do_vec[0]):
            x1 = -0.1
            y1 = 1.23
            ax1.text(x1, y1, 'A1', transform=ax1.transAxes, fontsize=12, va='top', fontproperties=font)
            ax2.text(x1, y1, 'A2', transform=ax2.transAxes, fontsize=12, va='top', fontproperties=font)
            ax3.text(x1, y1, 'A3', transform=ax3.transAxes, fontsize=12, va='top', fontproperties=font)
            ax4.text(x1, y1, 'A4', transform=ax4.transAxes, fontsize=12, va='top', fontproperties=font)

            x1 = -0.1
            y1 = 1.23
            ax5.text(x1, y1, 'B1', transform=ax5.transAxes, fontsize=12, va='top', fontproperties=font)
            ax6.text(x1, y1, 'B2', transform=ax6.transAxes, fontsize=12, va='top', fontproperties=font)
            ax7.text(x1, y1, 'B3', transform=ax7.transAxes, fontsize=12, va='top', fontproperties=font)
            ax8.text(x1, y1, 'B4', transform=ax8.transAxes, fontsize=12, va='top', fontproperties=font)

else:
    ax1 = None
    ax1b = None

normalize = 1
for d, do in enumerate(do_vec):

    if "_keep_" in do:
        do_run_now = 0
    else:
        do_run_now = do_run

    pickle_prefix = ""
    if "_fig" in do:
        fig_num = str(do).split("_fig")[1].split("_")[0]
        pickle_prefix = pickle_prefix + "Fig" + str(fig_num) + "_"

    if "cell_" in do:

        ihold = 40
        amp = 0
        amod = 0.1

        anoise = 0
        tau_noise = 0*ms
        delta_t = 0*ms

        if "_grc_" in do:

            istart = 0.002
            istop = 0.05
            di = 0.0001

            sexp = 0
            cutf = 0

            cellimport = "from GRANULE_Cell import Grc"
            celltype = "Grc"
            cell_exe = "cell = Grc(np.array([0.,0.,0.]))"
            # Passive prop: R: 1049.03039562 MOhm, C: 3.38407734878e-06 uF, tau: 0.00355000000001 s
            # From Model: 3 pF!!!

            #-71.4794082315
            #-0.000951879443107
            #R: 1049.03039562 MOhm, C: 3.38407734878e-06 uF, tau: 0.00355000000001 s

            #tau_passive = 3e-06*1049 = 3.15ms

            pickle_prefix = pickle_prefix + "cell_grc"

            exec cellimport
            exec cell_exe

            temperature = 37
            give_freq = True
            SNR = None
            NI = None

            synout_tau1 = 100*ms
            synout_tau2 = 100*ms

            icloc = "soma(0.5)"


        if "_goc_" in do:

            istart = 0.002
            istop = 0.05
            di = 0.0001

            sexp = 0
            cutf = 0

            cellimport = "from templates.golgi.Golgi_template import Goc"
            celltype = "Goc"
            cell_exe = "cell = Goc(np.array([0.,0.,0.]))"

            pickle_prefix = pickle_prefix + "cell_goc"

            exec cellimport
            exec cell_exe

            temperature = 37
            give_freq = True
            SNR = None
            NI = None

            synout_tau1 = 100*ms
            synout_tau2 = 100*ms

            icloc = "soma(0.5)"


        if "_stl_" in do:

            istart = 0.001
            istop = 0.03
            di = 0.001

            sexp = 0
            cutf = 0

            cellimport = "from templates.mli.stellate import Stellate"
            celltype = "Stl"
            cell_exe = "cell = Stellate(np.array([0.,0.,0.]))"

            pickle_prefix = pickle_prefix + "cell_stl"

            exec cellimport
            exec cell_exe

            temperature = 37
            give_freq = True
            SNR = None
            NI = None

            synout_tau1 = 100*ms
            synout_tau2 = 100*ms

            icloc = "soma(0.5)"


        if "_resif_" in do:

            istart = 0.002
            istop = 0.05
            di = 0.0001

            sexp = 0
            cutf = 0

            #OLD:
            #thresh = -21.175*mV
            #gr = 6.044e-05*uS
            #tau_r = 0.0185
            #R = 8860*MOhm
            #tau_passive = 3e-06*8860 = 26.6ms
            #thresh: -21.1752 gr: 6.04400329156e-05 tau_r: 0.0185045932005 R: 8860
            #gr = 3.72e-05*uS
            #tau_r = 0.0201
            #R = 7097*MOhm

            gr = 5.56e-05*uS
            tau_r = 19.6*ms
            R = 5227*MOhm
            delta_t = 4.85*ms
            thresh = (0.00568*nA * R) - 71.5*mV #
            thresh = -41.8

            cellimport = []
            celltype = "IfCell"
            cell_exe = "cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -71.5*mV, dgk =" + str(gr) + ", egk = -71.5*mV, ctau =" + str(tau_r) + ")"

            pickle_prefix = pickle_prefix + "cell_resif"

            #exec cellimport
            exec cell_exe

            temperature = 0
            give_freq = True
            SNR = None
            NI = None

            synout_tau1 = 100*ms
            synout_tau2 = 100*ms

            icloc = "soma(0.5)"


        if "_grD_" in do:

            istart = 0
            istop = 0.02
            di = 0.0001

            sexp = 0
            cutf = 0

            celltype = "GrcDiwakar"
            cell_exe = "cell = GrcDiwakar()"
            cellimport = "from cells." + celltype + " import *"

            pickle_prefix = pickle_prefix + "cell_grD"

            exec cellimport
            exec cell_exe

            temperature = 30
            give_freq = True
            SNR = None
            NI = None

            synout_tau1 = 100*ms
            synout_tau2 = 100*ms

            icloc = "dendrites[0][3](0.5)"


        if "_grVSCS_" in do:

            istart = 0
            istop = 0.05
            di = 0.0005

            sexp = 0
            cutf = 0

            celltype = "GrcVSCS"
            cell_exe = "cell = GrcVSCS()"
            cellimport = "from cells." + celltype + " import *"

            pickle_prefix = pickle_prefix + "cell_grVSCS"

            exec cellimport
            exec cell_exe

            temperature = 37
            give_freq = True
            SNR = None
            NI = None

            synout_tau1 = 100*ms
            synout_tau2 = 100*ms

            icloc = "soma(0.5)"


        if "results_transfer" in do:

            pickle_prefix = pickle_prefix + "_transfer"

            freq_used0 = concatenate(( arange(0.5, xmax+0.5, 0.5), array([]) )) #, arange(210, 1010, 10) ))

            sim = Stimulation(cell, celltype = celltype, cell_exe = cell_exe, temperature = temperature, do_run = do_run_now, pickle_prefix = pickle_prefix, give_freq = give_freq, istart = istart, istop = istop, di = di)
            #if MPI.COMM_WORLD.rank == 0: rm, cm, taum = sim.get_RCtau() #method = "vc", dv=0.0001*mV)

            sim.spikes_from_neuron = False

            sim.del_freq = array([20])
            sim.ihold = ihold
            sim.amp = amp
            sim.amod = amod
            sim.anoise = anoise
            sim.tau_noise = tau_noise

            sim.synout_tau1 = synout_tau1
            sim.synout_tau2 = synout_tau2
            sim.delta_t = delta_t
            sim.data_dir = data_dir
            sim.minimal_dir = minimal_dir

            currtitle = "single sinusoid, " + celltype + ", amp: " + str(sim.amp) + ", ihold: " + str(sim.ihold)

            sim.linewidth = linewidth

            method_interpol = np.array(["none"])

            if "_grc_" in do:
                sim.color_vec = (array([color2,color2,color2,color2,color2]),array([color2,color2,color2,color2,color2]))
                opt_plot = np.array(["only_mag", "normalize", "dB"]) # "dB"
            if "_goc_" in do:
                sim.color_vec = (array([color2,color2,color2,color2,color2]),array([color2,color2,color2,color2,color2]))
                opt_plot = np.array(["only_mag", "normalize", "dB"]) # "dB"
            if "_stl_" in do:
                sim.color_vec = (array([color2,color2,color2,color2,color2]),array([color2,color2,color2,color2,color2]))
                opt_plot = np.array(["only_mag", "normalize", "dB"]) # "dB"
            if "_resif_" in do:
                sim.color_vec = (array(['k','k','k','k','k']),array(['k','k','k','k','k']))
                opt_plot = np.array(["only_mag", "normalize", "dB", "dotted"]) # "dB"
            if "_grD_" in do:
                sim.color_vec = (array([color3,color3,color3,color3,color3]),array([color3,color3,color3,color3,color3]))
                opt_plot = np.array(["only_mag", "normalize", "dB"]) # "dB"
            if "_grVSCS_" in do:
                sim.color_vec = (array([p1,p1,p1,p1,p1]),array([p1,p1,p1,p1,p1]))
                opt_plot = np.array(["only_mag", "normalize", "dB"]) # "dB"

            _, _, mag, pha, freq_used = sim.fun_plot(currtitle, "ssine", freq_used = freq_used0, method_interpol = method_interpol, method_interpol_plot = method_interpol, ax = ax1, axP = ax1b, SNR = 0, VAF = 0, opt_plot = opt_plot, xmax=xmax)

            if MPI.COMM_WORLD.rank == 0:
                adjust_spines(ax1, ['left','bottom'], d_out = 10)
                adjust_spines(ax1b, ['left','bottom'], d_out = 10)

                if "_grc_" in do: # old: 27.4

                    #freq_used = freq_used0
                    #H_goal = (mag[0,:] * exp(pi / 180 * 1j * pha[0,:]))

                    #tau_fit, scale_fit, H_fit = fit_aiftransfer(freq_used, H_goal, f0 = 40, i0 = 0)

                    #print "fit theor., tau=" + str(tau_fit/ms) + "ms, scale_fit:" + str(scale_fit)
                    #H_fit, H0_fit = aiftransfer(freq_used, tau = tau_fit, f0 = 40, i0 = 0.000591817847824*nA)  # compute again with all frequencies

                    #magA_fit = abs(H_fit)
                    #phaA_fit = unwrap(angle(H_fit)) * (180 / pi)
                    #scale_fit = 1/magA_fit[0]

                    #ax1.semilogx(freq_used, 20*log10(magA_fit*scale_fit), 'b--', linewidth = linewidth)
                    #ax1b.semilogx(freq_used, phaA_fit, 'b--' , linewidth = linewidth)


                    H, H0 = aiftransfer(freq_used0, tau = 15.7*1e-3, f0 = 40, i0 = 0)
                    magA = abs(H)
                    scale = 1/magA[0]
                    ax1.semilogx(freq_used0, 20*log10(magA*scale), '--' , color = 'k', linewidth = linewidth)

                    #ax1.text(20, 0, r'GrC model', color=r1, fontsize = params['text.fontsize'])
                    #ax1.text(1, -4, r"IF ($\tau$ = 26.6 ms)", color='k', fontsize = params['text.fontsize'])

                    phaA = unwrap(angle(H)) * (180 / pi)
                    ax1b.semilogx(freq_used0, phaA, '--' , color = 'k', linewidth = linewidth)

                    #plt.figure(99)
                    #plt.plot(concatenate((real(H_goal), imag(H_goal))), 'r')
                    #plt.plot(concatenate((real(H_fit)*scale_fit, imag(H_fit)*scale_fit)), 'g')
                    #plt.plot(concatenate((real(H)*scale_fit, imag(H)*scale_fit)), 'b')

                    #plt.figure(96)
                    #plt.plot(concatenate((real(H), imag(H))), 'b')
                    #plt.show()

                if "_resif_" in do:
                    pass
                    #ax1.text(1, -10, r"resonant IF", color='k', fontsize = params['text.fontsize'])


                if "_grD_" in do:
                    H, H0 = aiftransfer(freq_used0, tau = 4.8*1e-3, f0 = 40, i0 = 0)
                    magA = abs(H)
                    scale = 1/magA[0]
                    ax1.semilogx(freq_used0,  20*log10(magA*scale), '--', color = b1, linewidth = linewidth)

                    ax1.text(1.2, 1.0, r'GrC model (Diwakar et al. 2009)', color=g1, fontsize = params['text.fontsize'])
                    ax1.text(1.2, -4, r"IF ($\tau$ = 4.8 ms)", color=b1, fontsize = params['text.fontsize'])


                if "_grVSCS_" in do:
                    H, H0 = aiftransfer(freq_used0, tau = 4.8*1e-3, f0 = 40, i0 = 0)
                    magA = abs(H)
                    scale = 1/magA[0]
                    ax1.semilogx(freq_used0,  20*log10(magA*scale), '--', color = b1, linewidth = linewidth)

                    ax1.text(1.2, 1.0, r'GrC model (Steuber)', color=g1, fontsize = params['text.fontsize'])
                    ax1.text(1.2, -4, r"IF ($\tau$ = 3.5 ms)", color=b1, fontsize = params['text.fontsize'])


                #ax1.set_ylabel("Gain (dB)")
                ax1.set_title("Gain (dB)") #, fontsize=8)
                ax1.set_xscale('log')
                ax1.xaxis.set_ticks(array([1, 10, 20]))
                ax1.set_xticklabels(('1', '','20'))
                ax1.axis(xmin=0.5, xmax=21)
                ax1.axis(ymin=-4, ymax=1)
                ax1.yaxis.set_ticks(array([-4, -2,  0, 1]))
                ax1.set_xlabel("Hz", labelpad=-4)

                #ax1.set_title("Transfer   fit")
                if "first" in do:
                    ax1.text(0.6, 1.4, 'Sinusoidal fit', transform=ax1.transAxes, fontsize=10, va='top')

                ax1b.set_title("Phase ($^\circ$)")
                ax1b.set_xscale('log')
                ax1b.xaxis.set_ticks(array([1, 10 ,20]))
                ax1b.set_xticklabels(('1', '', '20'))
                ax1b.axis(xmin=0.5, xmax=21)
                ax1b.axis(ymin=-100, ymax=0)
                ax1b.yaxis.set_ticks(array([-100, -50, 0]))
                ax1b.set_xlabel("Hz", labelpad=-4)

                plotname = "./figs/Pub/" + str(pickle_prefix)
                savefig(plotname + ".png", dpi = 300) # save it
                savefig(plotname + ".pdf", dpi = 300) # save it
                #os.system('rsvg-convert -f pdf -o ' + plotname +'.pdf ' + plotname + '.svg')


    sim = None
    cell = None


    if "pop_transfer" in do:

        #pop = None

        ihold = [40]
        amod = [0.1]
        anoise = [0]
        fluct_tau = 0*ms
        N = [1]
        #do_run = 1

        istart = 0.002
        istop = 0.05
        di = 0.0001

        delta_t = 0*ms

        use_multisplit = False

        linestyle = '-'

        simstep = 1*s

        method_interpol = np.array(['bin'])

        amp = 0 # absolute value
        fluct_s = [0] # absolute value
        ihold_sigma = [0*nA] # absolute value

        dt = 0.025*ms
        bin_width = dt
        jitter = 0*ms

        give_freq = True

        if "_if0_" in do:

            prefix = pickle_prefix + "pop_transfer" + "_if0"

            istart = 0.001
            istop = 0.015
            di = 0.0001

            istart = 0.01
            istop = 0.15
            di = 0.001

            cellimport = []
            celltype = ["IfCell"]
            cell_exe = ["cell = IfCell(C = 107e-06*uF, R = 235*MOhm, e = -58*mV, thresh = -50*mV, vrefrac = -58*mV)"]

            color = 'k'
            linestyle = '--'

            temperature = 0


        if "_if_" in do:

            prefix = pickle_prefix + "pop_transfer" + "_if"

            thresh = -41.8
            R = 5227*MOhm
            #tau_passive = 3e-06*5227 = 15.7ms

            cellimport = []
            celltype = ["IfCell"]
            cell_exe = ["cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -71.5*mV)"]

            color = 'k'
            linestyle = '--'

            temperature = 0

            if "_lowcf_slownoise_" in do:

                give_freq = False
                #ihold = [0.00185]
                #ihold_sigma = [0.42/2] # 0.1/2 0.01 realtive value
                #amod = [None]
                #amp = 0.001
                #anoise = [None]
                #fluct_s = [0.002]  # .005
                #fluct_tau = 100*ms

                ihold = [0.0041]
                ihold_sigma = [0.2/2] # 0.1/2 0.01 realtive value
                amod = [None]
                amp = 0.002
                anoise = [None]
                fluct_s = [0.001]  # .005
                fluct_tau = 100*ms

                prefix = prefix + "_lowcf_slownoise"

                if MPI.COMM_WORLD.rank == 0:
                    ax01 = ax5
                    ax02 = ax6
                    ax03 = ax7
                    ax04 = ax8


            elif "_lowcf_" in do:

                give_freq = False
                #ihold = [0.00487]
                #ihold_sigma = [0.04/2] # 0.1/2 0.01 realtive value
                #amod = [None]
                #amp = 0.001
                #anoise = [None]
                #fluct_s = [0.00]  # .005
                #fluct_tau = 0*ms

                ihold = [0.00465]
                ihold_sigma = [0.12/2] # 0.1/2 0.01 realtive value
                amod = [None]
                amp = 0.002
                anoise = [None]
                fluct_s = [0.00]  # .005
                fluct_tau = 0*ms

                prefix = prefix + "_lowcf"

                if MPI.COMM_WORLD.rank == 0:
                    ax01 = ax1
                    ax02 = ax2
                    ax03 = ax3
                    ax04 = ax4

            elif "_consta20_" in do:

                give_freq = False

                ihold = [0.00592898]
                ihold_sigma = [0] # 0.1/2 0.01 realtive value
                amod = [None]
                amp = 0.000145
                anoise = [None]
                fluct_s = [0.00]  # .005
                fluct_tau = 0*ms

                prefix = prefix + "_consta20"

                if MPI.COMM_WORLD.rank == 0:
                    ax01 = ax1
                    ax02 = ax2
                    ax03 = ax3
                    ax04 = ax4

            elif "_consta40_" in do:

                give_freq = False

                ihold = [0.0071345]
                ihold_sigma = [0] # 0.1/2 0.01 realtive value
                amod = [None]
                amp = 0.000145
                anoise = [None]
                fluct_s = [0.00]  # .005
                fluct_tau = 0*ms

                prefix = prefix + "_consta40"

                if MPI.COMM_WORLD.rank == 0:
                    ax01 = ax1
                    ax02 = ax2
                    ax03 = ax3
                    ax04 = ax4

            elif "_consta60_" in do:

                give_freq = False

                ihold = [0.008687875]
                ihold_sigma = [0] # 0.1/2 0.01 realtive value
                amod = [None]
                amp = 0.000145
                anoise = [None]
                fluct_s = [0.00]  # .005
                fluct_tau = 0*ms

                prefix = prefix + "_consta60"

                if MPI.COMM_WORLD.rank == 0:
                    ax01 = ax1
                    ax02 = ax2
                    ax03 = ax3
                    ax04 = ax4

            else:

                if MPI.COMM_WORLD.rank == 0:

                    if "_alt" not in do:
                        ax01 = ax3
                        ax02 = ax5
                        ax03 = ax7
                    else:
                        ax01 = ax4
                        ax02 = ax6
                        ax03 = ax8


        if "_ifpass_" in do:

            prefix = pickle_prefix + "pop_transfer" + "_ifpass"

            R = 737*MOhm
            thresh = (0.00568*nA * R) - 71.5*mV #
            thresh = -67.3

            #tau_passive = 3e-06*737 = 2.2ms

            cellimport = []
            celltype = ["IfCell"]
            cell_exe = ["cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -71.5*mV)"]

            color = 'gray'
            linestyle = '--'

            temperature = 0

            if "_lowcf_slownoise_" in do:

                give_freq = False
                ihold = [0.0016]
                ihold_sigma = [0.2/2] # 0.1/2 0.01 realtive value
                amod = [None]
                amp = 0.001
                anoise = [None]
                fluct_s = [0.0015]  # .005
                fluct_tau = 100*ms

                prefix = prefix + "_lowcf_slownoise"

                if MPI.COMM_WORLD.rank == 0:
                    ax01 = ax5
                    ax02 = ax6
                    ax03 = ax7
                    ax04 = ax8

            elif "_lowcf_" in do:

                give_freq = False
                ihold = [0.00445]
                ihold_sigma = [0.05/2] # 0.1/2 0.01 realtive value
                amod = [None]
                amp = 0.001
                anoise = [None]
                fluct_s = [0.00]  # .005
                fluct_tau = 0*ms

                prefix = prefix + "_lowcf"

                if MPI.COMM_WORLD.rank == 0:
                    ax01 = ax1
                    ax02 = ax2
                    ax03 = ax3
                    ax04 = ax4

            else:

                if MPI.COMM_WORLD.rank == 0:

                    if "_alt" not in do:
                        ax01 = ax3
                        ax02 = ax5
                        ax03 = ax7
                    else:
                        ax01 = ax4
                        ax02 = ax6
                        ax03 = ax8


        if "_resif" in do:

            prefix = pickle_prefix + "pop_transfer" + "_resif"

            #OLD:
            #thresh = -21.175*mV
            #gr = 6.044e-05*uS
            #tau_r = 0.0185
            #R = 8860*MOhm
            #tau_passive = 3e-06*8860 = 26.6ms
            #thresh: -21.1752 gr: 6.04400329156e-05 tau_r: 0.0185045932005 R: 8860

            gr = 5.56e-05*uS
            tau_r = 19.6*ms
            R = 5227*MOhm
            delta_t = 4.85*ms

            if "_resif2" in do:
                delta_t = 0
                prefix = prifix + "2"

            thresh = (0.00568*nA * R) - 71.5*mV #
            thresh = -41.8

            cellimport = []
            celltype = ["IfCell"]
            cell_exe = ["cell = IfCell(C = 3e-06*uF, R = " + str(R) + ", e = -71.5*mV, thresh =" + str(thresh) + ", vrefrac = -71.5*mV, dgk =" + str(gr) + ", egk = -71.5*mV, ctau =" + str(tau_r) + ")"]

            color = 'k'
            linestyle = ':'

            temperature = 0

            if "_lowcf_slownoise_" in do:

                give_freq = False
                #ihold = [0.00185]
                #ihold_sigma = [0.42/2] # 0.1/2 0.01 realtive value
                #amod = [None]
                #amp = 0.001
                #anoise = [None]
                #fluct_s = [0.002]  # .005
                #fluct_tau = 100*ms

                ihold = [0.0042]
                ihold_sigma = [0.2/2] # 0.1/2 0.01 realtive value
                amod = [None]
                amp = 0.002
                anoise = [None]
                fluct_s = [0.001]  # .005
                fluct_tau = 100*ms

                prefix = prefix + "_lowcf_slownoise"

                if MPI.COMM_WORLD.rank == 0:
                    ax01 = ax5
                    ax02 = ax6
                    ax03 = ax7
                    ax04 = ax8


            elif "_lowcf_" in do:

                give_freq = False
                #ihold = [0.00487]
                #ihold_sigma = [0.04/2] # 0.1/2 0.01 realtive value
                #amod = [None]
                #amp = 0.001
                #anoise = [None]
                #fluct_s = [0.00]  # .005
                #fluct_tau = 0*ms

                ihold = [0.0047]
                ihold_sigma = [0.12/2] # 0.1/2 0.01 realtive value
                amod = [None]
                amp = 0.002
                anoise = [None]
                fluct_s = [0.00]  # .005
                fluct_tau = 0*ms

                prefix = prefix + "_lowcf"

                if MPI.COMM_WORLD.rank == 0:
                    ax01 = ax1
                    ax02 = ax2
                    ax03 = ax3
                    ax04 = ax4

            else:

                if MPI.COMM_WORLD.rank == 0:
                    if "_alt" not in do:
                        ax01 = ax3
                        ax02 = ax5
                        ax03 = ax7
                    else:
                        ax01 = ax4
                        ax02 = ax6
                        ax03 = ax8


        if "_grc_" in do:

            prefix = pickle_prefix + "pop_transfer" + "_grc"

            cellimport = ["from GRANULE_Cell import Grc"]
            celltype = ["Grc"]
            cell_exe = ["cell = Grc(np.array([0.,0.,0.]))"]

            color = r1

            temperature = 37

            if "_lowcf_slownoise_" in do:

                give_freq = False
                #ihold = [0.002873]
                #ihold_sigma = [0.29/2] # 0.1/2 0.01 realtive value
                #amod = [None]
                #amp = 0.001
                #anoise = [None]
                #fluct_s = [0.002]  # .005
                #fluct_tau = 100*ms

                ihold = [0.0051]
                ihold_sigma = [0.18/2] # 0.1/2 0.01 realtive value
                amod = [None]
                amp = 0.002
                anoise = [None]
                fluct_s = [0.001]  # .005
                fluct_tau = 100*ms

                prefix = prefix + "_lowcf_slownoise"

                if MPI.COMM_WORLD.rank == 0:
                    ax01 = ax5
                    ax02 = ax6
                    ax03 = ax7
                    ax04 = ax8

            elif "_lowcf_" in do:

                give_freq = False
                #ihold = [0.00541]
                #ihold_sigma = [0.03/2] # 0.1/2 0.01 realtive value
                #amod = [None]
                #amp = 0.001
                #anoise = [None]
                #fluct_s = [0.00]  # .005
                #fluct_tau = 0*ms

                ihold = [0.0055]
                ihold_sigma = [0.1/2] # 0.1/2 0.01 realtive value
                amod = [None]
                amp = 0.002
                anoise = [None]
                fluct_s = [0.00]  # .005
                fluct_tau = 0*ms

                prefix = prefix + "_lowcf"

                if MPI.COMM_WORLD.rank == 0:
                    ax01 = ax1
                    ax02 = ax2
                    ax03 = ax3
                    ax04 = ax4

            else:

                if MPI.COMM_WORLD.rank == 0:
                    ax01 = ax4
                    ax02 = ax6
                    ax03 = ax8


        use_mpi = True

        if "_prk_" in do:

            amod = [0.1]
            ihold = [60]

            #give_freq = False
            #ihold = [0.0042]
            #ihold_sigma = [0]
            #amod = [None]
            #amp = 0.002

            prefix = pickle_prefix + "pop_transfer" + "_prk"

            cellimport = ["from Purkinje import Purkinje"]
            celltype = ["Prk"]
            cell_exe = ["cell = Purkinje()"]

            color = o1

            temperature = 37

            istart = 0
            istop = 0.2
            di = 0.01

            use_multisplit = True
            use_mpi = False

            if MPI.COMM_WORLD.rank == 0:
                ax01 = ax1
                ax02 = ax2
                ax03 = ax3
                ax04 = ax4

        if "_goc_" in do:

            amod = [0.2]
            ihold = [40]

            prefix = pickle_prefix + "pop_transfer" + "_goc"

            cellimport = ["from templates.golgi.Golgi_template import Goc"]
            celltype = ["Goc"]
            cell_exe = ["cell = Goc(np.array([0.,0.,0.]))"]

            color = o1

            temperature = 37

            istart = 0
            istop = 0.1
            di = 0.005


            if MPI.COMM_WORLD.rank == 0:
                ax01 = ax1
                ax02 = ax2
                ax03 = ax3
                ax04 = ax4


        a_celltype = [0]
        factor_celltype = [1]

        if "_cn_" in do:

            xmax = 20
            cutf = 20
            sexp = -1
            prefix = prefix + "_cn"


        if "_pn_" in do:

            xmax = 60
            cutf = 60
            sexp = -1
            prefix = prefix + "_pn"


        if "_bn_" in do:

            xmax = 30
            cutf = 30
            sexp = -1
            prefix = prefix + "_bn"


        if "_wn_" in do:

            xmax = 100
            cutf = 0
            sexp = 0
            prefix = prefix + "_wn"


        if "_N" in do:
            N = str(do).split("_N")[1].split("_")[0]
            N = [int(N)]



        if "_addn100_" in do:

            anoise = [2]
            fluct_tau = 100*ms
            prefix = prefix + "_addn100"
            N = [100]
            amod = [1]

            if "_grc_" in do: color = g1


        if "_addn100b_" in do:

            anoise = [1]
            fluct_tau = 100*ms
            prefix = prefix + "_addn100b"
            N = [100]
            amod = [1]

            if "_grc_" in do: color = g1


        if "_addn1_" in do:

            anoise = [4]
            fluct_tau = 1*ms
            prefix = prefix + "_addn1"
            N = [100]
            amod = [1]

            if "_grc_" in do: color = g1


        if "_addn10_" in do:

            anoise = [0.1]
            fluct_tau = 10*ms
            prefix = prefix + "_addn10"


        if "_addn0_" in do:

            anoise = [0.1]
            fluct_tau = 0*ms
            prefix = prefix + "_addn0"

        if "_a1_" in do:
            amod = [1]

        if "_a05_" in do:
            amod = [0.5]

        if "_a01_" in do:
            amod = [0.1]

        if "_a10_" in do:
            amod = [10]

        if "_a2_" in do:
            amod = [2]

        if "_a0133_" in do:
            amod = [0.133]

        if "_a005_" in do:
            amod = [0.05]

        if "_a0033_" in do:
            amod = [1/30]

        if "_a0025_" in do:
            amod = [0.025]


        if "_ihold" in do:
            iho = str(do).split("_ihold")[1].split("_")[0]
            prefix = prefix  + "_ihold" + str(iho)
            ihold = [float(iho)]

            if MPI.COMM_WORLD.rank == 0:
                ax01 = ax1
                ax02 = ax2
                ax03 = ax3
                ax04 = ax4

        if "_F0" in do:
            iho = str(do).split("_F0")[1].split("_")[0]
            prefix = prefix  + "_F0" + str(iho)
            ihold = [float(iho)]

        if "_addn1b_" in do:

            anoise = [0.4]
            fluct_tau = 1*ms
            prefix = prefix + "_addn1b"

            if MPI.COMM_WORLD.rank == 0:
                ax01 = ax5
                ax02 = ax6
                ax03 = ax7
                ax04 = ax8


        if "_addn1c_" in do:

            anoise = [4]
            fluct_tau = 1*ms
            prefix = prefix + "_addn1c"

            if MPI.COMM_WORLD.rank == 0:
                ax01 = ax5
                ax02 = ax6
                ax03 = ax7
                ax04 = ax8


        if "_addn1d_" in do:

            anoise = [40]
            fluct_tau = 1*ms
            prefix = prefix + "_addn1d"

            if MPI.COMM_WORLD.rank == 0:
                ax01 = ax5
                ax02 = ax6
                ax03 = ax7
                ax04 = ax8

        CF_var = False

        if "_CFvar" in do:
            prefix = prefix  + "_CFvar"
            CF_var = [[10,5,30]]

        if "_ihsigma" in do:
            iho = str(do).split("_ihsigma")[1].split("_")[0]
            prefix = prefix  + "_ihsigma" + str(iho)
            ihold_sigma = [float(iho)/2]

        if "_twopop_" in do:

            prefix = prefix + "_twopop"
            if cellimport != []:
                cellimport = [cellimport[0],cellimport[0]]

            celltype = [celltype[0],celltype[0]]
            cell_exe = [cell_exe[0],cell_exe[0]]
            N = [N[0],N[0]]

            a_celltype = [0,1]     # celltype to analyse
            factor_celltype = [1,-1]

            ihold = [ihold[0],ihold[0]]
            ihold_sigma = [ihold_sigma[0],ihold_sigma[0]]
            fluct_s = [fluct_s[0],fluct_s[0]]

            amod = [amod[0],amod[0]]
            anoise = [anoise[0],anoise[0]]

            if CF_var is not False:
                CF_var = [CF_var[0],CF_var[0]]

            #sigma_ihold = [sigma_ihold[0],sigma_ihold[0]]
            #noise_a = [noise_a[0], noise_a[0]]
            #noise_a_inh = [noise_a_inh[0], noise_a_inh[0]]


        if "_color" in do:
            nc = str(do).split("_color")[1].split("_")[0]
            #exec('color = color'+ str(nc))
            exec 'color = ' + str(nc)

        t_qual = 0
        if "_tqual" in do:
            t_qual = 10

        equi = 0
        if "_equi_" in do:
            give_freq = False
            ihold = [0]
            ihold_sigma = [0] # 0.1/2 0.01 realtive value
            amod = [None]
            amp = 0
            equi = 1


        pickle_prefix0 = prefix + "_N" + str(N) + "_CF" + str(ihold) + "_amod" + str(amod)
        pop = Population(cellimport = cellimport,  celltype = celltype, cell_exe = cell_exe, N = N, temperature = temperature, ihold = ihold, ihold_sigma = ihold_sigma, amp = amp, amod = amod, give_freq = give_freq, do_run = do_run_now, pickle_prefix = pickle_prefix0, istart = istart, istop = istop, di = di, dt = dt, use_mpi = use_mpi)

        pop.bin_width = bin_width
        pop.jitter = jitter
        pop.anoise = anoise
        pop.fluct_s = fluct_s
        pop.fluct_tau = fluct_tau
        pop.method_interpol = method_interpol
        pop.delta_t = delta_t
        pop.use_multisplit = use_multisplit
        pop.simstep = simstep
        pop.plot_input = False
        pop.a_celltype = a_celltype
        pop.factor_celltype = factor_celltype
        pop.delay_baseline = delay_baseline
        pop.tstop_if = tstop_if
        pop.plot_train = plot_train
        pop.xmax = xmax
        pop.CF_var = CF_var
        pop.data_dir = data_dir
        pop.minimal_dir = minimal_dir

        results = pop.fun_cnoise_Stim(t_stim = t_stim, cutf = cutf, sexp = sexp, t_qual = t_qual, equi = equi)
        t_qual = 0

        freq_used, amp_vec, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1')
        freq_times, spike_freq, fmean, method_interpol, SNR, VAF, Qual = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('SNR'), results.get('VAF'), results.get('Qual')
        stim, stim_re_mat, current_re, tk, K_mat = results.get('stim'), results.get('stim_re_mat'), results.get('current_re'), results.get('tk'), results.get('K_mat')
        ihold1 = results.get('ihold1')
        fmstd, fmax ,fcvm = results.get('fmstd'), results.get('fmax'), results.get('fcvm')

        if 'fbaseA' in results:
            fbase = results.get('fbaseA')
            fbstd = results.get('fbstdA')
        else:
            fbase = []
            fbstd = []

        if pop.id == 0:

            iend = mlab.find(freq_used > xmax)[0]


            # save additional information:
            if isinstance(fmean, (list)):
                thisinfo = "m(CF)=" + str(fmean[0]) + 'Hz, std(CF)=' + str(fmstd[0]) + 'Hz, max(CF)=' + str(fmax[0]) + 'Hz, mean(CV)=' + str(fcvm[0]) + ' ,m(VAF)=' + str(np.mean(VAF[1][0,0:iend])) + ' ,m(SNR)=' + str(np.mean(SNR[1][0,0:iend]))  +', m(base)=' + str(fbase) + ' Hz' +', std(base)=' + str(fbstd) + ' Hz'
            else:
                thisinfo = "m(CF)=" + str(fmean) + 'Hz,   std(CF)=' + str(fmstd) + 'Hz, max(CF)=' + str(fmax) + 'Hz, mean(CV)=' + str(fcvm) + ' ,m(VAF)=' + str(np.mean(VAF[1][0,0:iend])) + ' ,m(SNR)=' + str(np.mean(SNR[1][0,0:iend]))  +', m(base)=' + str(fbase) + ' Hz' +', std(base)=' + str(fbstd) + ' Hz'

            fo = open("./figs/txt/" + str(pickle_prefix0) + ".txt" , "wb")
            fo.write(thisinfo)
            fo.close()


            #tbin = 100*ms
            #tb = np.arange(0,t1[-1],tbin)
            #[all_rate, _] = neuronpy.util.spiketrain.get_histogram(t_all_cut, bins = tb)
            #all_rate = np.concatenate((np.zeros(1),all_rate)) / self.N[a] / tbin

            #plt.figure('results_psth')
            #plt.plot(tb,all_rate)
            #plotname = "./figs/Pub/PSTH_" + str(prefix)
            #plt.savefig(plotname + ".pdf", dpi = 300, transparent=True) # save it
            #plt.clf()

            if ("_fig4b_" in do) or ("_figX_" in do):

                if ("_normalize_" in do):
                    normalize = mag[0,0]
                    print "normalizethis:", normalize

                mag[0,:] = mag[0,:] / normalize

                if ("_noplot_" not in do):

                    ax01.semilogx(freq_used[0:iend], 20*log10(mag[0,0:iend]), color, linestyle=linestyle, linewidth=linewidth, alpha=1)

                    if "_slownoise_" in do:
                        if "_talk_" in do:
                            adjust_spines(ax01, ['bottom'], d_out = 10)
                        else:
                            adjust_spines(ax01, [], d_out = 10)
                    else:

                        if "_talk_" in do:
                            adjust_spines(ax01, ['left','bottom'], d_out = 10)
                        else:
                            adjust_spines(ax01, ['left'], d_out = 10)

                        ax01.set_xscale('log')
                        ax01.set_ylabel("Gain (dB)", labelpad=0)
                        #ax01.set_title("Gain (dB)", fontsize=8)
                        ax01.yaxis.set_ticks(array([-25, -20,-15,-10,-5,0,5]))
                        ax01.set_yticklabels(('', '-20', '', '-10' , '', '0', ''))

                    if "_talk_" in do:
                        ax01.set_xlabel("Hz", labelpad=-2)
                        ax01.set_xscale('log')
                        ax01.xaxis.set_ticks(array([1,10,20]))
                        ax01.set_xticklabels(('1', '10', '20'))

                    if ("_slownoise_" in do) and ("_first_" in do):
                        if ("_talk_" in do) :
                            ax01.text(0.5, 1.3, 'FC=1sp/s, N=10000 \n + slow noise', transform=ax01.transAxes, fontsize=10, va='top', ha='center')
                            ax03.text(0.5, 1.3, 'FC=1sp/s, N=10000 \n + slow noise', transform=ax01.transAxes, fontsize=10, va='top', ha='center')
                        else:
                            ax01.text(0.5, 1.7, r"$\mathsf{F_{eff}}$=4sp/s, N=100" + "\n + slow noise", transform=ax01.transAxes, fontsize=10, va='top', ha='center')

                            ax01.axhline(y=-1e9, xmin=0, xmax=0, color=b1, linestyle='-', label="standard")
                            ax01.axhline(y=-1e9, xmin=0, xmax=0, color=r1, linestyle='-', label="push-pull")


                            lg = ax01.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.07,-0.1), handlelength=0, handletextpad=0.1, numpoints=1)
                            #lg.draw_frame(False)
                            fr = lg.get_frame()
                            fr.set_lw(0.2)
                            txt = lg.get_texts()
                            for i, t in enumerate(txt):
                                if i==0: t.set_color(b1)
                                if i==1: t.set_color(r1)


                    elif ("_first_" in do):
                        if "_talk_" in do:
                            ax01.text(0.5, 1.3, 'FC=1sp/s, N=10000 \n ', transform=ax01.transAxes, fontsize=10, va='top', ha='center')
                            ax03.text(0.5, 1.3, 'FC=1sp/s, N=10000 \n ', transform=ax01.transAxes, fontsize=10, va='top', ha='center')
                        else:
                            ax01.text(0.4, 1.7, r"$\mathsf{F_{eff}}$=4sp/s, N=100", transform=ax01.transAxes, fontsize=10, va='top', ha='center')

                            ax01.axhline(y=-1e9, xmin=0, xmax=0, color='k', linestyle='-', label="GrC")
                            ax01.axhline(y=-1e9, xmin=0, xmax=0, color='k', linestyle='--', label="IF")
                            ax01.axhline(y=-1e9, xmin=0, xmax=0, color='k', linestyle=':', label="resonant IF")

                            lg = ax01.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.07,-0.1), handlelength=2.5, handletextpad=0.1)
                            #lg.draw_frame(False)
                            fr = lg.get_frame()
                            fr.set_lw(0.2)

                    if ("_figX_" in do):
                        pass
                    else:
                        ax01.axis(ymin=-25, ymax=5)

                    ax01.axis(xmin=0.3, xmax=xmax)

                    ax02.semilogx(freq_used[0:iend], pha[0,0:iend], color, linestyle=linestyle, linewidth=linewidth, alpha=1)

                    if "_slownoise_" in do:
                        if "_talk_" in do:
                            adjust_spines(ax02, ['bottom'], d_out = 10)
                        else:
                            adjust_spines(ax02, [], d_out = 10)

                    else:
                        if "_talk_" in do:
                            adjust_spines(ax02, ['left','bottom'], d_out = 10)
                        else:
                            adjust_spines(ax02, ['left'], d_out = 10)

                        ax02.set_xscale('log')
                        #ax02.set_title("Phase ($^\circ$)", fontsize=8)
                        ax02.set_ylabel("Phase ($^\circ$)", labelpad=-2)
                        ax02.yaxis.set_ticks(array([-140, -120, -100, -80, -60, -40, -20, 0, 20]))
                        ax02.set_yticklabels(('', '-120', '', '-80', '', '-40', '', '0', ''))

                    if "_talk_" in do:
                        ax02.set_xlabel("Hz", labelpad=-2)
                        ax02.set_xscale('log')
                        ax02.xaxis.set_ticks(array([1,10,20]))
                        ax02.set_xticklabels(('1', '10', '20'))

                        if "_grc_" in do:
                            ax02.axhline(y=-1e9, xmin=0, xmax=0, color=r1, linestyle='-', label="GrC")
                            ax02.axhline(y=-1e9, xmin=0, xmax=0, color='k', linestyle='--', label="IF")
                            ax02.axhline(y=-1e9, xmin=0, xmax=0, color='k', linestyle=':', label="resonant IF")
                            lg = ax02.legend(labelspacing=0.2, loc=3, handlelength=1.5, handletextpad=0.1)
                            #lg.draw_frame(False)
                            fr = lg.get_frame()
                            fr.set_lw(0.2)

                    if ("_figX_" in do):
                        pass
                    else:
                        ax02.axis(ymin=-140, ymax=20)

                    ax02.axis(xmin=0.3, xmax=xmax)

                ax03.semilogx(freq_used[0:iend], VAF[1][0,0:iend]*100, color, linewidth=linewidth, linestyle=linestyle, alpha=1)
                print "mean VAF:", np.mean(VAF[1][0,0:iend]*100)

                ax03.set_xscale('log')

                ax03.axis(ymin=0, ymax=100)
                ax03.axis(xmin=0.3, xmax=xmax)

                if "_slownoise_" in do:
                    adjust_spines(ax03, ['bottom'], d_out = d_out, d_down = d_down)
                    if ("_grc_" in do) and ("_talk_" in do) and ("_N1000_" in do): ax03.text(6, 50, "N=1000", color=color, fontsize = params['legend.fontsize'])
                else:
                    adjust_spines(ax03, ['left','bottom'], d_out = d_out, d_down = d_down)
                    ax03.yaxis.set_ticks(array([0,50,100]))
                    ax03.set_yticklabels(('0', '50', '100'))
                    ax03.set_ylabel("VAF (%)", labelpad=0)

                ax03.set_xlabel("Hz", labelpad=-4)
                ax03.set_xscale('log')
                ax03.xaxis.set_ticks(array([1,10,20]))
                ax03.set_xticklabels(('1', '10', '20'))

                if ("_grc_" in do) and ("_talk_" in do) and ("_slownoise_" not in do):
                    ax03.axhline(y=-1e9, xmin=0, xmax=0, color=r1, linestyle='-', label="GrC")
                    ax03.axhline(y=-1e9, xmin=0, xmax=0, color='k', linestyle='--', label="IF")
                    ax03.axhline(y=-1e9, xmin=0, xmax=0, color='k', linestyle=':', label="resonant IF")
                    lg = ax03.legend(labelspacing=0.2, loc=1, handlelength=1.5, handletextpad=0.1)
                    #lg.draw_frame(False)
                    fr = lg.get_frame()
                    fr.set_lw(0.2)


                if ("_grc_" in do) and ("_noplot_" not in do):
                #if True:
                    #print ihold1

                    if "_slownoise_" in do:
                        ih0 = 5.1
                        a0 = 2
                    else:
                        ih0 = 5.5
                        a0 = 2

                    print stim
                    print np.shape(stim)
                    if ("_first_" in do): ax04.plot(np.arange(len(stim))*dt-1, a0*(stim)+ih0, 'k-', linewidth=linewidth)
                    #print np.shape(stim_re_mat)
                    #print stim_re_mat
                    ax04.plot(np.arange(len(stim))*dt-1, a0*(stim_re_mat[0,:])+ih0, color, linewidth=linewidth, alpha=1)

                    ax04.axis(xmin=0, xmax=1)
                    ax04.axis(ymin=3, ymax=8.5)

                    if "_slownoise_" in do:
                        adjust_spines(ax04, ['bottom'], d_out = 10)

                    else:
                        adjust_spines(ax04, ['left','bottom'], d_out = 10)
                        ax04.yaxis.set_ticks(array([4,6,8]))
                        ax04.set_ylabel("pA", labelpad=2)

                    ax04.xaxis.set_ticks(array([0,0.5,1]))
                    ax04.set_xticklabels(('0', '0.5', '1'))
                    ax04.set_xlabel("s", labelpad=3)


            else:

                if "_addn10_poster" not in do:

                    if ("_normalize_" in do):
                        normalize = mag[0,0]
                        print "normalizethis:", normalize

                    mag[0,:] = mag[0,:] / normalize

                    iend = mlab.find(freq_used > xmax)[0]

                    label = "None"
                    #if ("cn" in do):
                    #    if ("_grc" in do): label = r'GrC model'
                    #    if ("_resif" in do): label = r'resonant IF'
                    #    if ("_if" in do): label = r'IF ($\tau$ = 15.7 ms)'
                    #    if ("_prk" in do): label = r'Purkinje model'

                    if ("fig4" in do) and ("grc" not in do):
                        if "_addspe_" in do:
                            ax01.semilogx(freq_used[0:iend], VAF[1][0,0:iend]*100, linewidth=linewidth, linestyle=linestyle, color=color)
                    else:

                        ax2.semilogx(freq_used[0:iend], 20*log10(mag[0,0:iend]), color=color, linestyle=linestyle, linewidth=linewidth, alpha=1) # , label = label
                        adjust_spines(ax2, ['left','bottom'], d_out = 10)

                        ax2.set_xscale('log')
                        #ax2.set_ylabel("Gain (dB)")
                        ax2.set_title("Gain (dB)") #, fontsize=8)
                        ax2.set_xlabel("Hz", labelpad=-4)
                        ax2.axis(xmin=0.5, xmax=xmax)

                        ax2b.semilogx(freq_used[0:iend], pha[0,0:iend], color=color, linestyle=linestyle, linewidth=linewidth, alpha=1)
                        adjust_spines(ax2b, ['left','bottom'], d_out = 10)
                        ax2b.set_xscale('log')
                        ax2b.set_title("Phase ($^\circ$)")
                        ax2b.set_xlabel("Hz", labelpad=-4)

                        if "_addspe_" in do:
                            ax01.semilogx(freq_used[0:iend], VAF[1][0,0:iend]*100, linewidth=linewidth, linestyle=linestyle, color=color)

                        if "cn" in do:

                            if "addn" in do:

                                ax2.axis(xmin=0.5, xmax=xmax)
                                ax2.axis(ymin=-0.2, ymax=1.5)

                                ax2.yaxis.set_ticks(array([0,0.5,1,1.5]))
                                ax2.set_yticklabels(('0', '0.5','1','1.5'))

                                ax2.xaxis.set_ticks(array([1,10,20]))
                                ax2.set_xticklabels(('1', '','20'))

                                ax2b.axis(xmin=0.5, xmax=xmax)
                                ax2b.axis(ymin=-12, ymax=5)

                                ax2b.yaxis.set_ticks(array([-10,-5,0,5]))
                                ax2b.set_yticklabels(('-10', '-5', '0' ,'5'))

                                ax2b.xaxis.set_ticks(array([1,10,20]))
                                ax2b.set_xticklabels(('1', '' ,'20'))

                            else:

                                ax2.axis(ymin=-0.2, ymax=2.5)
                                ax2.axis(xmin=0.5, xmax=xmax)

                                ax2.yaxis.set_ticks(array([0,0.5,1,1.5,2,2.5]))
                                ax2.set_yticklabels(('0', '0.5','1','1.5','2','2.5'))

                                ax2.xaxis.set_ticks(array([1,10,20]))
                                ax2.set_xticklabels(('1', '','20'))

                                ax2b.axis(xmin=0.5, xmax=xmax)
                                ax2b.axis(ymin=-1, ymax=35)
                                ax2b.yaxis.set_ticks(array([0,10,20,30]))

                                ax2b.xaxis.set_ticks(array([1,10,20]))
                                ax2b.set_xticklabels(('1', '' ,'20'))


                        else:

                            ax2.xaxis.set_ticks(array([1,10,40,100]))
                            ax2.set_xticklabels(('1', '10', '40', '100'))
                            ax2.yaxis.set_ticks(array([-20,0,20,40]))
                            ax2.axvline(x=fmean, color='k', linestyle=':')
                            ax2.axis(ymin=-20, ymax=50)

                        if ("cn" in do) and ("_if" in do) and ("_addn" not in do) and ("fig4" not in do):

                            #ax2.set_title("Transfer LP-WN")

                            if "poster" in do:
                                #ax2.text(0.9, 1.4, 'Low-pass white noise', transform=ax2.transAxes, fontsize=10, va='top')
                                lg = ax2.legend(loc='upper center', bbox_to_anchor=(1.3, -0.4), handlelength=3, ncol=3)
                                #lg.draw_frame(False)
                                fr = lg.get_frame()
                                fr.set_lw(0.2)
                            else:
                                ax2.text(0.3, 1.4, 'Low-pass white noise', transform=ax2.transAxes, fontsize=10, va='top')

                                #ax2.text(3, 2, r'GrC model', color=r1, fontsize = params['text.fontsize'])
                                #ax2.text(1.7, 1, r'resonant IF', color='k', fontsize = params['text.fontsize'])
                                #ax2.text(1, -0.5, r"IF ($\tau$ = 26.6 ms)", color='k', fontsize = params['text.fontsize'])

                                ax2.axhline(y=-1e9, xmin=0, xmax=0, color=r1, linestyle='-', label="GrC model")
                                ax2.axhline(y=-1e9, xmin=0, xmax=0, color='k', linestyle='--', label=r"IF ($\tau$ = 15.7 ms)")
                                ax2.axhline(y=-1e9, xmin=0, xmax=0, color='k', linestyle=':', label="resonant IF")

                                lg = ax2.legend(loc='upper center', bbox_to_anchor=(-0.5, -0.4), handlelength=3, ncol=3)
                                #lg.draw_frame(False)
                                fr = lg.get_frame()
                                fr.set_lw(0.2)

                        elif ("_grc" in do) and ("_addn" in do) and ("first" in do):

                            #ax2.set_title("Transfer WN")
                            #ax2.text(1.2, 1.8, r'GrC model', color=g1, fontsize = params['text.fontsize'])
                            #ax2.text(0.5, 1, r'resonant IF', color='k', fontsize = params['text.fontsize'])
                            #ax2.text(15, -0.5, r"IF", color='k', fontsize = params['text.fontsize'])

                            ax2.text(2, -0.2, r'$\tau$ = 1 ms', color=g1, fontsize = params['text.fontsize'])
                            ax2.text(0.5, 0.3, r'$\tau$ = 100 ms', color=g2, fontsize = params['text.fontsize'])
                            ax2.text(3, 1, r"no noise", color='black', fontsize = params['text.fontsize'])

                            #ax2.axhline(y=-1e9, xmin=0, xmax=0, color='k', linestyle='-', label="GrC model")
                            #ax2.axhline(y=-1e9, xmin=0, xmax=0, color='k', linestyle='--', label=r"IF ($\tau$ = 15.7 ms)")
                            #ax2.axhline(y=-1e9, xmin=0, xmax=0, color='k', linestyle=':', label="resonant IF")

                            #lg = ax2.legend(loc='upper center', bbox_to_anchor=(0.4, -0.24), handlelength=3, ncol=3)
                            ##lg.draw_frame(False)
                            #fr = lg.get_frame()
                            #fr.set_lw(0.2)


                if ("_tqual_" in do):

                    #if "_addn10_poster" not in do: ax2.text(0.3, 1.5, 'Low-pass white noise', transform=ax2.transAxes, fontsize=10, va='top')

                    ax1.plot(np.arange(len(stim))*dt-1, current_re*1e3, 'gray', linewidth=linewidth)
                    # stim and stim_re_mat are normalized, multiply by amplitude used in nA!!
                    ax1.plot(np.arange(len(stim))*dt-1, (ihold1+0.00517416*stim)*1e3, 'k-', linewidth=linewidth)
                    ax1.plot(np.arange(len(stim))*dt-1, (ihold1+0.00517416*stim_re_mat[0,:])*1e3, color, linewidth=linewidth, alpha=1)

                    #figure(99)
                    #plot(stim, 'b')
                    #plot(stim_re_mat[0,:], 'g')
                    #show()

                    adjust_spines(ax1, ['left'], d_out = 10)
                    ax1.axis(xmin=0, xmax=0.5)

                    #ax1.axis(ymin=8.7, ymax=10.7)
                    #ax1.yaxis.set_ticks(array([9,9.5,10,10.5]))
                    #ax1.set_title("Reconstruction")

                    if "_addn10_poster" not in do:
                        #ax1.text(0.1, 1.5, 'Reconstruction', transform=ax1.transAxes, fontsize=10, va='top')
                        ax1.set_title("Reconstruction") #, fontsize=8)
                    #else:
                    #    ax1.text(0.3, 1.5, 'Reconstruction', transform=ax1.transAxes, fontsize=10, va='top')

                    ax1.set_ylabel("pA", labelpad=0)

                    #ax1.text(0.3, -10, "Input current", color='gray', fontsize = 8)
                    #ax1.text(0.1, -5, "Signal", color="#000000", fontsize = 8)
                    #ax1.text(0.0, 25, "Reconstruction", color=g1, fontsize = 8)

                    ax1.text(0.2, 21, r"$\tau$ = 100 ms", color='gray', fontsize = 8)

                    ax1.axis(xmin=0, xmax=1)
                    ax1.axis(ymin=-5, ymax=25)

                    ax1.yaxis.set_ticks(array([0,10,20]))
                    ax1.set_yticklabels(('0', '10','20'))

                    ax1.xaxis.set_ticks(array([0,0.5]))
                    ax1.set_xticklabels(('0', '0.5'))

                elif ("_tqual1_" in do):

                    ax1b.plot(np.arange(len(stim))*dt-1, current_re*1e3, 'gray', linewidth=linewidth)
                    # stim and stim_re_mat are normalized, multiply by amplitude used in nA!!
                    ax1b.plot(np.arange(len(stim))*dt-1, (ihold1+0.00517416*stim)*1e3, 'k-', linewidth=linewidth)
                    ax1b.plot(np.arange(len(stim))*dt-1, (ihold1+0.00517416*stim_re_mat[0,:])*1e3, color, linewidth=linewidth, alpha=1)

                    adjust_spines(ax1b, ['left','bottom'], d_out = 10)

                    ax1b.set_xlabel("s", labelpad=-1)
                    ax1b.set_ylabel("pA", labelpad=-4)

                    ax1b.text(0.2, 55, r"$\tau$ = 1 ms", color='gray', fontsize = 8)

                    ax1b.axis(xmin=0, xmax=0.5)
                    ax1b.axis(ymin=-30, ymax=55)

                    ax1b.yaxis.set_ticks(array([-20,0,20,40]))
                    ax1b.set_yticklabels(('-20', '0','20','40'))

                    ax1b.xaxis.set_ticks(array([0,0.5]))
                    ax1b.set_xticklabels(('0', '0.5'))


            plt.figure('results_transfer')
            plotname = "./figs/Pub/" + str(prefix)
            savefig(plotname + ".png", dpi = 300) # save it
            savefig(plotname + ".pdf", dpi = 300) # save it
            #os.system('rsvg-convert -f pdf -o ' + plotname +'.pdf ' + plotname + '.svg')

            if ("_fig4b_" in do) and ("_talk_" in do):
                plt.figure('results_transfer2')
                plotname = "./figs/Pub/" + str(prefix)
                savefig(plotname + ".png", dpi = 300) # save it
                savefig(plotname + ".pdf", dpi = 300) # save it
                #os.system('rsvg-convert -f pdf -o ' + plotname +'.pdf ' + plotname + '.svg')

            if do_run_now:
                pop.delall()
            del pop
            pop = None
            results = None


        if ("_fig3_" in do) or ("_fig4_" in do):

            color_vec = (np.array([r1, b1, g1]), np.array([r1, b1, g1]))
            if "_addn100_" in do: color_vec = (np.array([r2, b2, g2]), np.array([r2, b2, g2]))

            #if "_resif" not in do:
            if ("_full_" in do):

                N_vec = array([1, 10, 100]) #
                #N_vec = array([]) #

                for i, No in enumerate(N_vec):

                    pickle_prefix = prefix + "_N" + str([No]) + "_CF" + str(ihold) + "_amod" + str(amod)

                    if pickle_prefix == pickle_prefix0:
                        do_run_now0 = 0
                    else:
                        do_run_now0 = do_run_now

                    pop = Population(cellimport = cellimport,  celltype = celltype, cell_exe = cell_exe, N = [No], temperature = temperature, ihold = ihold, ihold_sigma = ihold_sigma, amp = amp, amod = amod, give_freq = give_freq, do_run = do_run_now0, pickle_prefix = pickle_prefix, istart = istart, istop = istop, di = di, dt = dt)

                    pop.bin_width = bin_width
                    pop.jitter = jitter
                    pop.anoise = anoise
                    pop.fluct_s = fluct_s
                    pop.fluct_tau = fluct_tau
                    pop.method_interpol = method_interpol
                    pop.simstep = simstep
                    pop.plot_input = False
                    pop.delta_t = delta_t
                    pop.plot_train = plot_train
                    pop.xmax = xmax
                    pop.data_dir = data_dir
                    pop.minimal_dir = minimal_dir
                    pop.delay_baseline = delay_baseline

                    results = pop.fun_cnoise_Stim(t_stim = t_stim, cutf = cutf, sexp = sexp, t_qual = 0)

                    freq_used, amp_vec, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1')
                    freq_times, spike_freq, fmean, method_interpol, SNR, VAF, Qual = results.get('freq_times'), results.get('spike_freq'), results.get('fmeanA'), results.get('method_interpol'), results.get('SNR'), results.get('VAF'), results.get('Qual')
                    stim, stim_re_mat = results.get('stim'), results.get('stim_re_mat')
                    fmstd, fmax ,fcvm = results.get('fmstdA'), results.get('fmaxA'), results.get('fcvmA')

                    if 'fbaseA' in results:
                        fbase = results.get('fbaseA')
                        fbstd = results.get('fbstdA')
                    else:
                        fbase = []
                        fbstd = []

                    if pop.id == 0:

                        if ("_normalize_" in do):
                            normalize = mag[0,0]
                            print "normalizethis:", normalize

                        mag[0,:] = mag[0,:] / normalize

                        iend = mlab.find(freq_used > xmax)[0]

                        # save additional information:
                        if isinstance(fmean, (list)):
                            thisinfo = "m(CF)=" + str(fmean[0]) + 'Hz, std(CF)=' + str(fmstd[0]) + 'Hz, max(CF)=' + str(fmax[0]) + 'Hz, mean(CV)=' + str(fcvm[0]) + ' ,m(VAF)=' + str(np.mean(VAF[1][0,0:iend])) +', m(base)=' + str(fbase) + ' Hz' +', std(base)=' + str(fbstd) + ' Hz'
                        else:
                            thisinfo = "m(CF)=" + str(fmean) + 'Hz,   std(CF)=' + str(fmstd) + 'Hz, max(CF)=' + str(fmax) + 'Hz, mean(CV)=' + str(fcvm) + ' ,m(VAF)=' + str(np.mean(VAF[1][0,0:iend])) +', m(base)=' + str(fbase) + ' Hz' +', std(base)=' + str(fbstd) + ' Hz'

                        fo = open("./figs/txt/" + str(pickle_prefix) + ".txt" , "wb")
                        fo.write(thisinfo)
                        fo.close()

                        if "SNR" in do:
                            #ax01.semilogx(freq_used, 10*log10(SNR[1][0,:]), linewidth=linewidth, color=color_vec[0][i], label = "N=" + str(N))
                            ax01.semilogx(freq_used[0:iend], SNR[1][0,0:iend], linewidth=linewidth, linestyle=linestyle, color=color_vec[0][i], label = "N=" + str(No))
                            ax01.axhline(y=10, color='k', linestyle=':')
                        elif ("grc" in do) and ("_alt" in do) and (("_addn1_" in do) or ("fig3" in do)):
                            ax01.semilogx(freq_used[0:iend], VAF[1][0,0:iend]*100, linewidth=linewidth, linestyle=linestyle, color=color_vec[0][i], label = "N=" + str(No))
                        else:
                            ax01.semilogx(freq_used[0:iend], VAF[1][0,0:iend]*100, linewidth=linewidth, linestyle=linestyle, color=color_vec[0][i])
                        ax01.set_title("Population size N")

                        print "mean VAF:", np.mean(VAF[1][0,0:iend]*100)

                        if "_alt" in do:
                            ax01.set_xscale('log')
                            adjust_spines(ax01, ['left','bottom'], d_out = 10)

                            ax01.yaxis.set_ticks(array([0,50,100]))
                            ax01.set_yticklabels(('0', '50', '100'))
                            ax01.set_ylabel("VAF (%)")
                            ax01.axis(ymin=0, ymax=105)

                            if "grc" in do:
                                if "_addn1_" in do:
                                    lg = ax01.legend(labelspacing=0.2, loc=4, bbox_to_anchor=(0.27,0.35), handlelength=0, handletextpad=0.1, numpoints=1)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)

                                    txt = lg.get_texts()
                                    for i, t in enumerate(txt):
                                        t.set_color(color_vec[0][i])

                                    ax01b = ax01.twinx()
                                    adjust_spines(ax01b, [], d_out = 10)
                                    ax01b.axhline(y=-1, xmin=0, xmax=0, color='k', linestyle='-', label="N=100\nno noise", linewidth=linewidth)
                                    lg2 = ax01b.legend(labelspacing=0.2, loc=4, bbox_to_anchor=(0.97,0.44), handlelength=0, handletextpad=0.1, numpoints=1)
                                    fr2 = lg2.get_frame()
                                    fr2.set_lw(0.2)


                                elif "fig3" in do:
                                    lg = ax01.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.00,-0.05), handlelength=0, handletextpad=0.1, numpoints=1)
                                    #lg.draw_frame(False)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)

                                    txt = lg.get_texts()
                                    for i, t in enumerate(txt):
                                        t.set_color(color_vec[0][i])


                        elif "if" in do:
                            ax01.set_xscale('log')
                            adjust_spines(ax01, ['left'], d_out = 10)

                            if "SNR" in do:

                                ax01.set_ylabel("SNR")
                                ax01.axis(ymin=-1, ymax=50)
                                ax01.yaxis.set_ticks(array([0,10,20,30,40,50]))

                                if "cn" in do:
                                    ax01.text(1.5, 28, "N=1", color=b1, fontsize = 8)
                                    ax01.text(1.2, 45, "N=10", color=g1, fontsize = 8)
                                    ax01.text(11, 45, "N=100", color=r1, fontsize = 8)

                            else:

                                ax01.yaxis.set_ticks(array([0,50,100]))
                                ax01.set_yticklabels(('0', '50', '100'))
                                ax01.set_ylabel("VAF (%)")

                                ax01.axis(ymin=0, ymax=105)

                                if "wn" in do:
                                    ax01.text(2, 59, "N=1", color=b1, fontsize = 8)
                                    ax01.text(6, 78, "N=10", color=g1, fontsize = 8)
                                    ax01.text(30, 90, "N=100", color=r1, fontsize = 8)

                                if "cn" in do:
                                    lg = ax01.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.1,-0.05))
                                    #lg.draw_frame(False)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)

                        else:
                            adjust_spines(ax01, [], d_out = 10)

                            if "SNR" in do:

                                ax01.axis(ymin=-1, ymax=50)

                                if "cn" in do:
                                    ax01.text(1.5, 32, "N=1", color=b1, fontsize = 8)
                                    ax01.text(1.2, 47, "N=10", color=g1, fontsize = 8)
                                    ax01.text(11, 45, "N=100", color=r1, fontsize = 8)

                            else:

                                ax01.axis(ymin=0, ymax=105)

                                if "wn" in do:
                                    ax01.text(2, 59, "N=1", color=b1, fontsize = 8)
                                    ax01.text(6, 78, "N=10", color=g1, fontsize = 8)
                                    ax01.text(30, 90, "N=100", color=r1, fontsize = 8)

                                if "cn" in do:
                                    lg = ax01.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.1,-0.05))
                                    #lg.draw_frame(False)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)

                        ax01.axis(xmin=0.5, xmax=xmax)
                        ax01.set_xscale('log')

                        if "_alt" in do:
                            ax01.xaxis.set_ticks(array([1,10,20]))
                            ax01.set_xticklabels(('1', '10', '20'))
                            ax01.set_xlabel("Hz", labelpad=-4)

                        if "wn" in do: ax01.axvline(x=fmean, color='k', linestyle=':')

                        plt.figure('results_transfer')
                        plotname = "./figs/Pub/" + str(prefix)
                        savefig(plotname + ".png", dpi = 300) # save it
                        savefig(plotname + ".pdf", dpi = 300) # save it
                        #os.system('rsvg-convert -f pdf -o ' + plotname +'.pdf ' + plotname + '.svg')

                    if do_run_now0:
                        pop.delall()
                    del pop
                    pop = None
                    results = None

                color_vec = (np.array([b1, r1, g1]), np.array([b1, r1, g1]))
                if "_addn1_" in do: color_vec = (np.array([b1, g1, r1]), np.array([b1, g1, r1]))
                if "_addn100_" in do: color_vec = (np.array([b2, g2, r2]), np.array([b2, g2, r2]))

                #MPI.COMM_WORLD.Barrier()

                CF_vec = array([20, 40, 80])
                #CF_vec = array([])

                for i, CF in enumerate(CF_vec):

                    pickle_prefix = prefix + "_N" + str(N) + "_CF" + str([CF]) + "_amod" + str(amod)

                    if pickle_prefix == pickle_prefix0:
                        do_run_now0 = 0
                    else:
                        do_run_now0 = do_run_now

                    pop = Population(cellimport = cellimport,  celltype = celltype, cell_exe = cell_exe, N = N, temperature = temperature, ihold = [CF], ihold_sigma = ihold_sigma, amp = amp, amod = amod, give_freq = give_freq, do_run = do_run_now0, pickle_prefix = pickle_prefix, istart = istart, istop = istop, di = di, dt = dt)
                    pop.bin_width = bin_width
                    pop.jitter = jitter
                    pop.anoise = anoise
                    pop.fluct_s = fluct_s
                    pop.fluct_tau = fluct_tau
                    pop.method_interpol = method_interpol
                    pop.simstep = simstep
                    pop.plot_input = False
                    pop.delta_t = delta_t
                    pop.plot_train = plot_train
                    pop.xmax = xmax
                    pop.data_dir = data_dir
                    pop.minimal_dir = minimal_dir
                    pop.delay_baseline = delay_baseline

                    results = pop.fun_cnoise_Stim(t_stim = t_stim, cutf = cutf, sexp = sexp, t_qual = 0)

                    freq_used, amp_vec, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1')
                    freq_times, spike_freq, fmean, method_interpol, SNR, VAF, Qual = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('SNR'), results.get('VAF'), results.get('Qual')
                    stim, stim_re_mat = results.get('stim'), results.get('stim_re_mat')
                    fmstd, fmax ,fcvm = results.get('fmstd'), results.get('fmax'), results.get('fcvm')

                    if 'fbaseA' in results:
                        fbase = results.get('fbaseA')
                        fbstd = results.get('fbstdA')
                    else:
                        fbase = []
                        fbstd = []

                    if pop.id == 0:

                        if ("_normalize_" in do):
                            normalize = mag[0,0]
                            print "normalizethis:", normalize

                        mag[0,:] = mag[0,:] / normalize

                        iend = mlab.find(freq_used > xmax)[0]


                        # save additional information:
                        if isinstance(fmean, (list)):
                            thisinfo = "m(CF)=" + str(fmean[0]) + 'Hz, std(CF)=' + str(fmstd[0]) + 'Hz, max(CF)=' + str(fmax[0]) + 'Hz, mean(CV)=' + str(fcvm[0]) + ' ,m(VAF)=' + str(np.mean(VAF[1][0,0:iend])) +', m(base)=' + str(fbase) + ' Hz' +', std(base)=' + str(fbstd) + ' Hz'
                        else:
                            thisinfo = "m(CF)=" + str(fmean) + 'Hz,   std(CF)=' + str(fmstd) + 'Hz, max(CF)=' + str(fmax) + 'Hz, mean(CV)=' + str(fcvm) + ' ,m(VAF)=' + str(np.mean(VAF[1][0,0:iend])) +', m(base)=' + str(fbase) + ' Hz' +', std(base)=' + str(fbstd) + ' Hz'

                        fo = open("./figs/txt/" + str(pickle_prefix) + ".txt" , "wb")
                        fo.write(thisinfo)
                        fo.close()

                        if "SNR" in do:
                            #ax02.semilogx(freq_used, 10*log10(SNR[1][0,:]), linewidth=linewidth, color=color_vec[0][i], label = "CF=" + str(CF))
                            ax02.semilogx(freq_used[0:iend], SNR[1][0,0:iend], linewidth=linewidth, linestyle=linestyle, color=color_vec[0][i], label = "FC=" + str(CF))
                            ax02.axhline(y=10, color='k', linestyle=':')
                        elif ("grc" in do) and ("_alt" in do) and (("_addn1_" in do) or ("fig3" in do)):
                            ax02.semilogx(freq_used[0:iend], VAF[1][0,0:iend]*100, linewidth=linewidth, linestyle=linestyle, color=color_vec[0][i], label = r"$\mathsf{F_0}$=" + str(CF) + "sp/s")
                        else:
                            ax02.semilogx(freq_used[0:iend], VAF[1][0,0:iend]*100, linewidth=linewidth, linestyle=linestyle, color=color_vec[0][i])

                        if "wn" in do: ax02.axvline(x=fmean, color='k', linestyle=':')

                        ax02.set_title(r"Carrier-rate $\mathsf{F_0}$")

                        if "_alt" in do:
                            ax02.set_xscale('log')
                            adjust_spines(ax02, ['bottom'], d_out = 10)
                            ax02.axis(ymin=0, ymax=105)

                            if "grc" in do:
                                if "_addn1_" in do:
                                    lg = ax02.legend(labelspacing=-0.1, loc=3, bbox_to_anchor=(-0.00,0.0), handlelength=0, handletextpad=0.1, numpoints=1)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)

                                    txt = lg.get_texts()
                                    for i, t in enumerate(txt):
                                        t.set_color(color_vec[0][i])

                                elif "fig3" in do:
                                    lg = ax02.legend(labelspacing=-0.1, loc=3, bbox_to_anchor=(-0.00,-0.05), handlelength=0, handletextpad=0.1, numpoints=1)
                                    #lg.draw_frame(False)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)

                                    txt = lg.get_texts()
                                    for i, t in enumerate(txt):
                                        t.set_color(color_vec[0][i])


                        elif "if" in do:
                            adjust_spines(ax02, ['left'], d_out = 10)

                            if "SNR" in do:

                                ax02.set_ylabel("SNR")
                                ax02.axis(ymin=-1, ymax=50)
                                ax02.yaxis.set_ticks(array([0, 10,20,30,40,50]))

                                if "cn" in do:
                                    ax02.text(1.5, 0, "FC=20", color=g1, fontsize = 8)
                                    ax02.text(3, 40, "FC=40", color=b1, fontsize = 8)
                                    ax02.text(11, 40, "FC=80", color=r1, fontsize = 8)

                            else:

                                ax02.yaxis.set_ticks(array([0,50,100]))
                                ax02.set_yticklabels(('0', '50', '100'))
                                ax02.set_ylabel("VAF (%)")

                                ax02.axis(ymin=0, ymax=105)

                                if "wn" in do:
                                    ax02.text(1.5, 42, "FC=20", color=g1, fontsize = 8)
                                    ax02.text(2, 85, "FC=40", color=b1, fontsize = 8)
                                    ax02.text(35, 50, "FC=80", color=r1, fontsize = 8)

                                if "cn" in do:
                                    lg = ax02.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.00,-0.05))
                                    #lg.draw_frame(False)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)


                        else:
                            adjust_spines(ax02, [], d_out = 10)

                            if "SNR" in do:

                                ax02.axis(ymin=-1, ymax=50)

                                if "cn" in do:
                                    ax02.text(1.5, 0, "FC=20", color=g1, fontsize = 8)
                                    ax02.text(3, 30, "FC=40", color=b1, fontsize = 8)
                                    ax02.text(11, 40, "FC=80", color=r1, fontsize = 8)

                            else:

                                ax02.axis(ymin=0, ymax=105)

                                if "wn" in do:
                                    ax02.text(1.5, 42, "FC=20", color=g1, fontsize = 8)
                                    ax02.text(2, 85, "FC=40", color=b1, fontsize = 8)
                                    ax02.text(35, 50, "FC=80", color=r1, fontsize = 8)

                                if "cn" in do:
                                    lg = ax02.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.00,-0.05))
                                    #lg.draw_frame(False)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)

                        ax02.set_xscale('log')
                        ax02.axis(xmin=0.5, xmax=xmax)

                        if "_alt" in do:
                            ax02.xaxis.set_ticks(array([1,10,20]))
                            ax02.set_xticklabels(('1', '10', '20'))
                            ax02.set_xlabel("Hz", labelpad=-4)

                        plt.figure('results_transfer')
                        plotname = "./figs/Pub/" + str(prefix)
                        savefig(plotname + ".png", dpi = 300) # save it
                        savefig(plotname + ".pdf", dpi = 300) # save it
                        #os.system('rsvg-convert -f pdf -o ' + plotname +'.pdf ' + plotname + '.svg')

                    if do_run_now0:
                        pop.delall()
                    del pop
                    pop = None
                    results = None


                color_vec = (np.array([b1, r1, g1]), np.array([b1, r1, g1]))
                if "_addn100_" in do: color_vec = (np.array([b2, r2, g2]), np.array([b2, r2, g2]))
                amod_vec = array([0.05, 0.1, 1])
                #amod_vec = array([0.5])

                for i, a in enumerate(amod_vec):

                    if a == 1.0: a = 1
                    print a
                    pickle_prefix = prefix + "_N" + str(N) + "_CF" + str(ihold) + "_amod[" + str(a) + "]"

                    if pickle_prefix == pickle_prefix0:
                        do_run_now0 = 0
                    else:
                        do_run_now0 = do_run_now

                    pop = Population(cellimport = cellimport,  celltype = celltype, cell_exe = cell_exe, N = N, temperature = temperature, ihold = ihold, ihold_sigma = ihold_sigma, amp = amp, amod = a, give_freq = give_freq, do_run = do_run_now0, pickle_prefix = pickle_prefix, istart = istart, istop = istop, di = di, dt = dt)

                    pop.bin_width = bin_width
                    pop.jitter = jitter
                    pop.anoise = anoise
                    pop.fluct_s = fluct_s
                    pop.fluct_tau = fluct_tau
                    pop.method_interpol = method_interpol
                    pop.simstep = simstep
                    pop.plot_input = False
                    pop.delta_t = delta_t
                    pop.plot_train = plot_train
                    pop.xmax = xmax
                    pop.data_dir = data_dir
                    pop.minimal_dir = minimal_dir
                    pop.delay_baseline = delay_baseline

                    results = pop.fun_cnoise_Stim(t_stim = t_stim, cutf = cutf, sexp = sexp, t_qual = 0)

                    freq_used, amp_vec, mag, pha, ca, voltage, current, t1 = results.get('freq_used'), results.get('amp'), results.get('mag'), results.get('pha'), results.get('ca'), results.get('voltage'), results.get('current'), results.get('t1')
                    freq_times, spike_freq, fmean, method_interpol, SNR, VAF, Qual = results.get('freq_times'), results.get('spike_freq'), results.get('fmean'), results.get('method_interpol'), results.get('SNR'), results.get('VAF'), results.get('Qual')
                    stim, stim_re_mat = results.get('stim'), results.get('stim_re_mat')
                    fmstd, fmax ,fcvm = results.get('fmstd'), results.get('fmax'), results.get('fcvm')

                    if 'fbaseA' in results:
                        fbase = results.get('fbaseA')
                        fbstd = results.get('fbstdA')
                    else:
                        fbase = []
                        fbstd = []

                    if pop.id == 0:

                        if ("_normalize_" in do):
                            normalize = mag[0,0]
                            print "normalizethis:", normalize

                        mag[0,:] = mag[0,:] / normalize

                        iend = mlab.find(freq_used > xmax)[0]

                        # save additional information:
                        if isinstance(fmean, (list)):
                            thisinfo = "m(CF)=" + str(fmean[0]) + 'Hz, std(CF)=' + str(fmstd[0]) + 'Hz, max(CF)=' + str(fmax[0]) + 'Hz, mean(CV)=' + str(fcvm[0]) + ' ,m(VAF)=' + str(np.mean(VAF[1][0,0:iend])) +', m(base)=' + str(fbase) + ' Hz' +', std(base)=' + str(fbstd) + ' Hz'
                        else:
                            thisinfo = "m(CF)=" + str(fmean) + 'Hz,   std(CF)=' + str(fmstd) + 'Hz, max(CF)=' + str(fmax) + 'Hz, mean(CV)=' + str(fcvm) + ' ,m(VAF)=' + str(np.mean(VAF[1][0,0:iend])) +', m(base)=' + str(fbase) + ' Hz' +', std(base)=' + str(fbstd) + ' Hz'

                        fo = open("./figs/txt/" + str(pickle_prefix) + ".txt" , "wb")
                        fo.write(thisinfo)
                        fo.close()

                        if "SNR" in do:
                            #ax03.semilogx(freq_used, 10*log10(SNR[1][0,:]), linewidth=linewidth, color=color_vec[0][i], label = "a=" + str(a))
                            ax03.semilogx(freq_used[0:iend], SNR[1][0,0:iend], linewidth=linewidth, linestyle=linestyle, color=color_vec[0][i], label = "a=" + str(a))
                            ax03.axhline(y=10, color='k', linestyle=':')
                        elif ("grc" in do) and ("_alt" in do) and (("_addn1_" in do) or ("fig3" in do)):
                            ax03.semilogx(freq_used[0:iend], VAF[1][0,0:iend]*100, linewidth=linewidth, linestyle=linestyle, color=color_vec[0][i], label = "a=" + str(a))
                        else:
                            ax03.semilogx(freq_used[0:iend], VAF[1][0,0:iend]*100, linewidth=linewidth, linestyle=linestyle, color=color_vec[0][i])

                        ax03.set_title("Modulation amplitude a")

                        if "_alt" in do:
                            ax03.set_xscale('log')
                            adjust_spines(ax03, ['bottom'], d_out = 10)
                            ax03.axis(ymin=0, ymax=105)

                            #ax03.axis(xmin=0.5, xmax=xmax)
                            #ax03.xaxis.set_ticks(array([1,10,20]))
                            #ax03.set_xticklabels(('1', '10', '20'))
                            #ax03.set_xlabel("Hz", labelpad=0)

                            if "grc" in do:
                                if "_addn1_" in do:
                                    lg = ax03.legend(labelspacing=0.2, loc=4, bbox_to_anchor=(0.25,0.34), handlelength=0, handletextpad=0.1, numpoints=1)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)

                                    txt = lg.get_texts()
                                    for i, t in enumerate(txt):
                                        t.set_color(color_vec[0][i])

                                elif "fig3" in do:
                                    lg = ax03.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.00,-0.05), handlelength=0, handletextpad=0.1, numpoints=1)
                                    #lg.draw_frame(False)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)

                                    txt = lg.get_texts()
                                    for i, t in enumerate(txt):
                                        t.set_color(color_vec[0][i])


                        elif "if" in do:
                            adjust_spines(ax03, ['left','bottom'], d_out = 10)

                            if "SNR" in do:

                                ax03.set_ylabel("SNR")
                                ax03.axis(ymin=-1, ymax=50)
                                ax03.yaxis.set_ticks(array([0, 10,20,30,40,50]))

                                if "cn" in do:
                                    ax03.text(1.5, 25, "a=0.05", color=g1, fontsize = 8)
                                    ax03.text(3, 40, "a=0.1", color=b1, fontsize = 8)
                                    ax03.text(10, 15, "a=1", color=r1, fontsize = 8)

                            else:

                                ax03.yaxis.set_ticks(array([0,50,100]))
                                ax03.set_yticklabels(('0', '50', '100'))
                                ax03.set_ylabel("VAF (%)")

                                ax03.axis(ymin=0, ymax=105)

                                if "wn" in do:
                                    ax03.text(1.5, 36, "a=0.05", color=g1, fontsize = 8)
                                    ax03.text(2, 59, "a=0.1", color=b1, fontsize = 8)
                                    ax03.text(3, 92, "a=1", color=r1, fontsize = 8)

                                if "cn" in do:
                                    lg = ax03.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.00,-0.05))
                                    #lg.draw_frame(False)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)

                        else:

                            adjust_spines(ax03, ['bottom'], d_out = 10)

                            if "SNR" in do:

                                ax03.axis(ymin=-1, ymax=50)

                                if "cn" in do:
                                    ax03.text(1.5, 30, "a=0.05", color=g1, fontsize = 8)
                                    ax03.text(3, 45, "a=0.1", color=b1, fontsize = 8)
                                    ax03.text(10, 15, "a=1", color=r1, fontsize = 8)

                            else:

                                ax03.axis(ymin=0, ymax=105)

                                if "wn" in do:
                                    ax03.text(1.5, 36, "a=0.05", color=g1, fontsize = 8)
                                    ax03.text(2, 59, "a=0.1", color=b1, fontsize = 8)
                                    ax03.text(3, 92, "a=1", color=r1, fontsize = 8)

                                if "cn" in do:
                                    lg = ax03.legend(labelspacing=0.2, loc=3, bbox_to_anchor=(-0.00,-0.05))
                                    #lg.draw_frame(False)
                                    fr = lg.get_frame()
                                    fr.set_lw(0.2)

                        ax03.set_xscale('log')
                        ax03.axis(xmin=0.5, xmax=xmax)

                        if "wn" in do:
                            ax03.xaxis.set_ticks(array([1,10,40,100]))
                            ax03.set_xticklabels(('1', '10', '40', '100'))
                            ax03.axvline(x=fmean, color='k', linestyle=':')
                        else:
                            ax03.xaxis.set_ticks(array([1,10,20]))
                            ax03.set_xticklabels(('1', '10', '20'))

                        ax03.set_xlabel("Hz", labelpad=-4)

                    if do_run_now0:
                        pop.delall()
                    del pop
                    pop = None
                    results = None



        if MPI.COMM_WORLD.rank == 0:

            if "poster" in do: prefix = prefix + "_poster"

            plt.figure('results_transfer')
            plotname = "./figs/Pub/" + str(prefix)
            savefig(plotname + ".png", dpi = 300) # save it
            savefig(plotname + ".pdf", dpi = 300) # save it
            #os.system('rsvg-convert -f pdf -o ' + plotname +'.pdf ' + plotname + '.svg')


plt.show()
