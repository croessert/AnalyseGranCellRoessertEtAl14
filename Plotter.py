# -*- coding: utf-8 -*-
"""
@author: chris
"""

from __future__ import division

from pylab import *

import os

from Stimhelp import *

# SET DEFAULT VALUES FOR THIS PLOT
fig_size =  [11.7, 8.3]
params = {'backend': 'ps', 'axes.labelsize': 9, 'axes.linewidth' : 0.5, 'title.fontsize': 8, 'text.fontsize': 9,
    'legend.borderpad': 0.2, 'legend.fontsize': 8, 'legend.linewidth': 0.1, 'legend.loc': 'best', # 'lower right'    
    'legend.ncol': 4, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'text.usetex': False, 'figure.figsize': fig_size}
rcParams.update(params)


def plot_if(currlabel, current_vector, freq_vector, freq_onset_vector, ax=None, color_vec=None):
    
    if color_vec == None:
        color_vec = (array(["Blue"]), array(["Blue"]))    
        
    if ax is None:
        ax1 = subplot(1,1,1)
        
    else:
        ax1 = ax

    ax1.plot(current_vector, freq_vector, color=color_vec[0][0], label = currlabel + " adapted rate")
    ax1.plot(current_vector, freq_onset_vector, '--', color=color_vec[1][0], label = currlabel + " onset rate")
    xlabel("Current [nA]")
    ylabel("Frequency [Hz]")
    
    if ax is None:  
        title("I/F")
        lg = legend()
        lg.get_frame().set_linewidth(0.5)

    axis(xmin=0, xmax=current_vector[-1])


def plot_iv(currlabel, current_vector, v_vector, r_vector, tau_vector):

    subplot(1,3,1)  
    plot(current_vector, v_vector, label = currlabel)
    title("I/V")
    xlabel("Current [nA]")
    ylabel("Voltage [mV]")
    
    lg = legend()
    lg.get_frame().set_linewidth(0.5)
    
    axis(xmin=current_vector[0], xmax=current_vector[-1])    

    subplot(1,3,2)
    plot(current_vector, r_vector, label = currlabel)
    title("I/R")
    xlabel("Current [nA]")
    ylabel("Resistance [MOhm]")    
        
    lg = legend()
    lg.get_frame().set_linewidth(0.5)

    axis(xmin=current_vector[0], xmax=current_vector[-1])    
    
    subplot(1,3,3)
    plot(current_vector, tau_vector, label = currlabel)
    title("I/Tau")
    xlabel("Current [nA]")
    ylabel("Tau [s]")    
        
    lg = legend()
    lg.get_frame().set_linewidth(0.5)

    axis(xmin=current_vector[0], xmax=current_vector[-1])    
    

def plot_refrac_onset(currlabel, latency_vec, late_amp_vec, amp_init, late_eff_lat_vec):

    subplot(1,2,1)  
                
    plot(1. / latency_vec, late_amp_vec / amp_init, '+-', label = currlabel)
    plot(1. / latency_vec, (late_amp_vec / late_amp_vec) , 'k--') 
         
    xlabel("Interpulse frequency [Hz]")
    ylabel("Relative threshold current")
                
    lg = legend()
    lg.get_frame().set_linewidth(0.5)
            
    subplot(1,2,2)
                
    plot(1. / latency_vec, 1. / late_eff_lat_vec, '+-', label = currlabel)
                 
    xlabel("Interpulse frequency [Hz]")
    ylabel("Resulting interspike frequency")
                
    lg = legend()
    lg.get_frame().set_linewidth(0.5)
                
    suptitle('Dual pulse test of refractory period')


def plot_refrac_train(currlabel, pulses, latency_vec, amp_vec, amp_init, eff_lat_vec, stim_type = "i"):

    subplot(1,2,1)
            
    plot(1. / latency_vec, amp_vec / amp_init, '+-', label = currlabel)
    plot(1. / latency_vec, (amp_vec / amp_vec) , 'k--')
         
    xlabel("Interpulse frequency [Hz]")
    ylabel("Relative threshold")
                
    lg = legend()
    lg.get_frame().set_linewidth(0.5)
            
    subplot(1,2,2)
                
    plot(1. / latency_vec, 1. / eff_lat_vec, '+-', label = currlabel)
                 
    xlabel("Interpulse frequency [Hz]")
    ylabel("Resulting interspike frequency")
                
    lg = legend()
    lg.get_frame().set_linewidth(0.5)
    
    if stim_type == "i":            
        suptitle("Pulse train (" + str(pulses) + " pulses) test of refractory period")                
    else: 
        suptitle("Synaptic train (" + str(pulses) + " events) test of refractory period")
        

def plot_transfer(currlabel=[], freq_used=[], mag=[], pha=[], t1=[], current=[], voltage=[], freq_times=[], spike_freq=[], tau=0*ms, f0=100*Hz, i0=1*nA, rm=1*MOhm, Vreset=0, Vth=2, Vrest=0, method_interpol=array(["none"]), method_interpol_plot=[], ymax=0, SNR=None, VAF=None, NI=None, ax=None, axP=None, linewidth=2, color_vec=None, alpha=1, opt_plot = np.array([]) ):

    if color_vec == None:
        color_vec = (array(["Blue", "Green", "Red", "Orange", "DarkGoldenRod", "DimGray", "HotPink", "Indigo", "Magenta", "CadetBlue", "OrangeRed"]), array(["Blue", "Green", "Red", "Orange", "DarkGoldenRod", "DimGray", "HotPink", "Indigo", "Magenta", "CadetBlue", "OrangeRed"]))    
    
    if len(method_interpol_plot) == 0:
        method_interpol_plot = method_interpol    
    
    if "only_mag" not in opt_plot: 
        ax1 = subplot(2,3,1)
        ax2 = subplot(2,3,2)
        ax3 = subplot(2,3,3)
        ax4 = subplot(2,3,4)
        ax5 = subplot(2,3,5)
        ax6 = subplot(2,3,6)
    elif ax is None:
        ax1 = subplot(1,1,1)
        adjust_spines(ax1, ['left', 'bottom']) 
    else:
        ax1 = ax
        
        if axP is not None:
            ax2 = axP
    
    linestyle = '-'
    if ("dotted" in opt_plot):
        linestyle = ':'
    if ("dashed" in opt_plot):
        linestyle = '--'
        
                
    if ("do_fit" in opt_plot) or ("div_fit" in opt_plot):
        
        H_goal = (mag[0,:] * exp(pi / 180 * 1j * pha[0,:]))
        new_end = find(freq_used >= 39)[0]  # do not used all frequencies
        #new_end=-1
        tau_fit, scale_fit, H_fit = fit_aiftransfer(freq_used[0:new_end], H_goal[0:new_end], f0, i0)
        
        f0=40
        print "fit theor., tau=" + str(tau_fit/ms) + "ms"
        H_fit, H0_fit = aiftransfer(freq_used, tau = tau_fit, f0 = f0, i0 = i0)  # compute again with all frequencies
        
        print tau_fit
        print f0
        print i0
        print H_fit
                
        magA_fit = abs(H_fit)
        phaA_fit = unwrap(angle(H_fit)) * (180 / pi)
        
        if "normalize" in opt_plot: 
            scale_fit = 1/magA_fit[0]
            print "normalized magnitude"
        
        if ("do_fit" in opt_plot):
        
            if "loglog" in opt_plot:
                ax1.loglog(freq_used, magA_fit*scale_fit, 'k--', label = "fit theor., tau=" + str(tau_fit/ms) + "ms, scale_fit=" + str(scale_fit), linewidth = linewidth, alpha = alpha)
            else:
                if "dB" in opt_plot:
                    ax1.semilogx(freq_used, 20*log10(magA_fit*scale_fit), 'k--', label = "fit theor., tau=" + str(tau_fit/ms) + "ms, scale_fit=" + str(scale_fit), linewidth = linewidth, alpha = alpha)
                else:
                    ax1.semilogx(freq_used, magA_fit*scale_fit, 'k--', label = "fit theor., tau=" + str(tau_fit/ms) + "ms, scale_fit=" + str(scale_fit), linewidth = linewidth, alpha = alpha)
                
            if "only_mag" not in opt_plot:
                ax2.semilogx(freq_used, phaA_fit, 'k--', label = "fit theor.", linewidth = linewidth, alpha = alpha)
                
        if "div_fit" in opt_plot:
            if "loglog" in opt_plot:
                ax1.loglog(freq_used, (mag[0,:]/mag[0,0])/(magA_fit*scale_fit), 'r--', label = "mag/fit", linewidth = linewidth, alpha = alpha)
            else:
                if "dB" in opt_plot:
                    ax1.semilogx(freq_used, 20*log10((mag[0,:]/mag[0,0])/(magA_fit*scale_fit)), 'r--', label = "mag/fit", linewidth = linewidth, alpha = alpha)
                else:
                    ax1.semilogx(freq_used, (mag[0,:]/mag[0,0])/(magA_fit*scale_fit), 'r--', label = "mag/fit", linewidth = linewidth, alpha = alpha)
 
    if (f0 != None) and (f0 > 0): # and (np.isnan(f0) is False): 
        ax1.axvline(x=f0, color='k', linestyle=':')
    
    import shlex
    print method_interpol_plot
    print method_interpol
    
    for l, m in enumerate(method_interpol):
        
        if shlex.split(m)[0] in method_interpol_plot: 
            
            if "normalize" in opt_plot:  
                mag[l,:]=mag[l,:]/mag[l,0]
                                            
            if "loglog" in opt_plot:
                ax1.loglog(freq_used, mag[l,:], linestyle=linestyle, color=color_vec[0][l], label=currlabel + ", interp: " + method_interpol[l], linewidth=linewidth, alpha=alpha)
            else:
                if "dB" in opt_plot:
                    ax1.semilogx(freq_used, 20*log10(mag[l,:]), linestyle=linestyle, color=color_vec[0][l], label=currlabel + ", interp: " + method_interpol[l], linewidth=linewidth, alpha=alpha)  # swith between magf and phaf , rasterized=True
                else:        
                    ax1.semilogx(freq_used, mag[l,:], linestyle=linestyle, color=color_vec[0][l], label=currlabel + ", interp: " + method_interpol[l], linewidth=linewidth, alpha=alpha)  # swith between magf and phaf , rasterized=True
                    #print method_interpol[l], mag[l,:]
    
    if tau > 0:
        
        H = aiftransfer(freq = freq_used, tau = tau, f0 = f0, i0 = i0, rm = rm, Vreset = Vreset, Vth = Vth, Vrest = Vrest)[0]
        
        magA = abs(H)
        phaA = unwrap(angle(H)) * (180 / pi)
        
        scaling = mag[0,0]/magA[0]
        print "theoretical aif transfer scaling: " + str(scaling)
        
        if "normalize" in opt_plot: 
            scaling = 1/magA[0]
            print "normalized magnitude"
        
        #print "freq_used: " + str(freq_used) + " tau: " + str(tau) + " f0: " + str(f0) + " i0: " + str(i0) + " rm: " + str(rm) + " Vreset: " + str(Vreset) + " Vth: " + str(Vth) + " Vrest: " + str(Vrest)
        
        if "loglog" in opt_plot:
            ax1.loglog(freq_used, magA*scaling, 'k--', label = "theor. function, scaling=" + str(scaling), linewidth = linewidth)
        else:
            if "dB" in opt_plot:
                ax1.semilogx(freq_used, 20*log10(magA*scaling), 'k--', label = "theor. function, scaling=" + str(scaling), linewidth = linewidth)          
            else:
                ax1.semilogx(freq_used, magA*scaling, 'k--', label = "theor. function, scaling=" + str(scaling), linewidth = linewidth)    
            
        #ax1.axvline(x=f0/2, color='k', linestyle='--')
        
        if "only_mag" not in opt_plot:        
            ax2.semilogx(freq_used, phaA, 'k--', label = "theor. function", linewidth = linewidth)
            
    if ymax == 0:
        if "dB" not in opt_plot:
            ax1.axis(ymin=0, ymax=2*mag[0,0])
        if "loglog" in opt_plot:
            ax1.axis(ymin=10e-3, ymax=100*mag[0,0])
    else:
        ax1.axis(ymin=0, ymax=ymax) 
    

    if ax is None: ax1.set_xlabel("freq [Hz]")
    
    if "normalize" in opt_plot: 
        ax1.set_ylabel("Normalized Transfer Magnitude")
    else:
        ax1.set_ylabel("Transfer Magnitude [Hz/nA]")
    
    if ax is None:  # only add legend if no axe is given
        lg = legend(loc='upper center', shadow = True, bbox_to_anchor = (0.5, 1.1), fancybox = True, ncol = 2)
        #lg.get_frame().set_linewidth(0.5)
        

    if SNR != None:
        
        ax1snr = ax1.twinx()
        
        for l in range(len(method_interpol)):
            
            SNR_ = SNR[1][l,:]
            ax1snr.semilogx(SNR[0], SNR_, '-.', color=color_vec[1][l], linewidth=linewidth, alpha=alpha) 

        adjust_spines(ax1snr, ['right'], color=color_vec[1][l], d_out = 0) 
        ax1snr.set_ylabel('SNR (dB)')
        ax1snr.axis(ymin=0, ymax=2*SNR_[0])

            
    if VAF != None: 
        
        ax1vaf = ax1.twinx()
        
        for l in range(len(method_interpol)):
            
            VAF_ = VAF[1][l,:]
            ax1vaf.semilogx(VAF[0], VAF_, ':', color=color_vec[1][l], linewidth=linewidth, alpha=alpha) 
        
        d_out = 0            
        if SNR != None:  d_out = 40 
        adjust_spines(ax1vaf, ['right'], color=color_vec[1][l], d_out = d_out)
        ax1vaf.set_ylabel('VAF')
        ax1vaf.axis(ymin=0, ymax=2) 
        ax1vaf.yaxis.set_ticks(array([0,0.5,1]))
        ax1vaf.set_yticklabels(('0', '0.5', '1'))
            
 
    if NI != None: 
        
        ax1ni = ax1.twinx()
        
        NI_ = -1*NI[1][0,:]
        axni.semilogx(NI[0], NI_, 'r-.', linewidth=linewidth, alpha=alpha) 
        NI_ = -1*NI[1][1,:]
        axni.semilogx(NI[0], NI_, 'g-.', linewidth=linewidth, alpha=alpha) 
        
        d_out = 0            
        if SNR != None:  d_out = 40 
        if (SNR != None) & (VAF != None):  d_out = 80 
        adjust_spines(axni, ['right'], color=color_vec[1][l], d_out = d_out)
        
        axni.set_ylabel('NI (-log10)')
        axni.axis(ymin=0, ymax=2*NI_[0])
    
    if axP is not None:

        for l, m in enumerate(method_interpol):
            
            if shlex.split(m)[0] in method_interpol_plot:
                
                phaX = unwrap( pha[l,:] * (pi / 180)) * (180 / pi)
                
                ax2.semilogx(freq_used, phaX, color=color_vec[0][l], label=currlabel + ", interp: " + method_interpol[l], linestyle=linestyle, linewidth=linewidth, alpha=alpha)
                
                if tau > 0:
                    ax2.semilogx(freq_used, phaA, 'k--', linewidth=linewidth, alpha=alpha)
         
        if (f0 != None) and (f0 > 0): # and (np.isnan(f0) is False): 
            axP.axvline(x=f0, color='k', linestyle=':')
        

    if "only_mag" not in opt_plot: 
        for l in range(len(method_interpol)):
            ax2.semilogx(freq_used, pha[l,:], color=color_vec[0][l], label=currlabel + ", interp: " + method_interpol[l], linewidth=linewidth, alpha=alpha)
            
        ax2.set_xlabel("freq [Hz]")
        ax2.set_ylabel("Transfer Phase [degree]")       

        ax4.plot(t1, current, linewidth=linewidth, alpha=alpha)
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Current (nA)")
        
        ax5.plot(t1, voltage, linewidth=linewidth, alpha=alpha)
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Voltage (mV)")
                
        ax6.plot(freq_times, spike_freq, linewidth=linewidth, alpha=alpha)
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("ISF (Hz)")

    if ax is None: subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)
    
    ax1.xaxis.set_ticks(array([1,10,100,1000]))
    ax1.set_xticklabels(('1', '10', '100', '1000'))
       
    return ax1
    

def plot_impedance(currlabel, freq_used, magz, phaz, ca, t1, current, voltage, rm = 0, cm = 0, gr = 0, tau_r = 0):
        
    if rm > 0:
        #if gr > 0:
        Yn = 1 / rm + 2 * pi * 1j * freq_used * cm + gr * 1/(2 * pi * 1j * freq_used * tau_r + 1)
        #else
        #    Yn = 1 / rm + 2 * pi * 1j * freq_used * cm #/ 1000 # rm [MOhm], cm [nF] TEST CELL
            
        admM = abs(Yn)
        admP = unwrap(angle(Yn)) * (180 / pi)
        impM = 1. / admM
        impP = -admP
    
        subplot(2,3,1)  
        semilogx(freq_used, impM, '--', label = "analytical cell")
        subplot(2,3,2)  
        semilogx(freq_used, impP, '--', label = "analytical cell")
        subplot(2,3,3)
        plot(real(Yn), imag(Yn), '--', label = "analytical cell")
            
    subplot(2,3,1)  
    semilogx(freq_used, magz, '-', label=currlabel) 
    xlabel("freq [Hz]")
    ylabel("Impedance Magnitude [MOhm]")
  
    subplot(2,3,2)  
    semilogx(freq_used, phaz, '-', label=currlabel)
    xlabel("freq [Hz]")
    ylabel("Impedance Phase [degree]")
            
    lg = legend(loc = 'upper center', shadow = True, bbox_to_anchor = (0.5, 1.0), fancybox = True, ncol = 2)
    lg.get_frame().set_linewidth(0.5)    
    
    subplot(2,3,3)
    plot(real(ca), imag(ca), '-', label=currlabel)
    xlabel("Admittance Real Part [uS]")
    ylabel("Admittance Imaginary Part [uS]")    
   
    subplot(2,3,4) 
    plot(t1, current)
    xlabel("Time (s)")
    ylabel("Current (nA)")
    
    subplot(2,3,5)
    plot(t1, voltage)
    xlabel("Time (s)")
    ylabel("Voltage (mV)")

    subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.1)
    
    
def adjust_spines(ax, spines, color = 'k', d_out = 10, d_down = []):
    
    if d_down == []:
        d_down = d_out
        
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    
    for loc, spine in ax.spines.iteritems():
        if loc in spines:
            if loc == 'bottom': 
                spine.set_position(('outward',d_down)) # outward by 10 points
            else:
                spine.set_position(('outward',d_out)) # outward by 10 points
            #spine.set_smart_bounds(True)
        else:
            spine.set_visible(False) # set_color('none') # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
        
        if color is not 'k':
            
            ax.spines['left'].set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis='y', colors=color)
        
        
    elif 'right' not in spines:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
        
    if 'right' in spines:
        ax.yaxis.set_ticks_position('right')
        
        if color is not 'k':
            
            ax.spines['right'].set_color(color)
            ax.yaxis.label.set_color(color)
            ax.tick_params(axis='y', colors=color)
        
#        s = ax.transAxes.transform((1, 1))  # transform into display coordinates
#        e = ax.transAxes.transform((1, 0))  # transform into display coordinates
#        line = Line2D((s[0]+10,e[0]+10), (s[1],e[1]), color=color, linewidth=rcParams['axes.linewidth'], transform=None) # ax.transAxes
#        line.set_clip_on(False) # show line!
#        ax.add_line(line)
        
        #second_right = matplotlib.spines.Spine(ax, 'right', ax.spines['left']._path)
        #second_right.set_position(('outward', 10))
        #ax.spines['second_right'] =  second_right
        #ax.spines['second_right'].set_color('k')
        #ax.spines['right'].set_color('k')

       
    if 'bottom' in spines:
        pass
        ax.xaxis.set_ticks_position('bottom')
        #ax.axes.get_xaxis().set_visible(True)
        
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])
        ax.axes.get_xaxis().set_visible(False)

