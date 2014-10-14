# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 19:12:05 2011

@author: chris

"""

from __future__ import division

#import sys
#sys.path.insert(0, "/home/chris/lib/python")  # for new matplotlib!!!

from pylab import *
from numpy import round, random, any
from units import *
import time
from NeuroTools import stgen
import h5py
import os
import shutil


def noclip(ax): 
     "Turn off all clipping in axes ax; call immediately before drawing" 
     ax.set_clip_on(False) 
     artists = [] 
     artists.extend(ax.collections) 
     artists.extend(ax.patches) 
     artists.extend(ax.lines) 
     artists.extend(ax.texts) 
     artists.extend(ax.artists) 
     for a in artists: 
         a.set_clip_on(False) 
         
def make_colormap(seq):
    import matplotlib.colors as mcolors
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
    
def create_colnoise(t, sexp, cutf = None, seed = None, onf = None):
    '''x = create_colnoise(t, sexp, cutf)
    Make coloured noise signal
    t    = vector of times
    sexp = spectral exponent - Power ~ 1 / f^sexp
    cutf = frequency cutoff  - Power flat (white) for f <~ cutf
    
    cutf == None OR sexp == 0: white noise
    
    output:  mean = 0, std of signal = 1/2 => max 95% is 1
    '''
    
    from scipy import signal
    
    nt = len(t)
    dt = (t[-1] - t[0]) / (nt - 1)
    random.seed(seed)
    
    #if sexp == 0:  # no smooth cutoff
    #    N = 10001   # number of filter taps
    #    xp = int((N-1)/2)
    #    x = random.standard_normal(size = len(t)+xp)   
    #else:
    #    x = random.standard_normal(size = shape(t)) 
        
    x = random.standard_normal(size = shape(t))/2
    #print std(x)
    if cutf == None:
        pass
    else:
        
        if sexp == 0:  # no cutoff
            pass
        
            # OLD: sharp cutoff
            
            #x = fft(x)
            #f = fftfreq(nt, dt)
            #x[nonzero(f == 0)] = 0               # remove zero frequency contribution
        
            ##i = nonzero(f != 0)              # find indices i of non-zero frequencies
            ##x[i]=x[i] / (cutf ** 2 + f[i] ** 2) ** (sexp / 4)  # using i allows cutf = 0
            
            #x = real(ifft(x))      # ignore imaginary part (numerical error)
            #x = x / std(x)         # std of signal = 1/2, mean = 0
            
            #fs = 1/dt                                            # sampling rate 
            #fc = cutf/(0.5*fs)                                   # cutoff frequency
            
            #a = 1                                                # filter denominator
            #b = signal.firwin(N, cutoff=fc, window='hamming') #'blackmanharris')    # filter numerator
            #x = signal.lfilter(b, a, x)                          # filtered output
            #x = x[xp:]
            
            ##w,h = signal.freqz(b,a)
            ##h_dB = 20 * log10 (abs(h))
            ##figure(77)
            ##semilogx(w/max(w),h_dB)
            ##show()

        elif sexp == -1:  # sharp cutoff
            x, freq, freq_wp, freq_used = create_multisines(t, freq_used=None, cutf=cutf, onf=onf)
            #print std(x)
            x = x / std(x) / 2         # std of signal = 1, mean = 0
                        
        else:  # smooth cutoff   
            
            x = fft(x)
            f = fftfreq(nt, dt)
            x[nonzero(f == 0)] = 0               # remove zero frequency contribution
        
            i = nonzero(f != 0)              # find indices i of non-zero frequencies
            x[i]=x[i] / (cutf ** 2 + f[i] ** 2) ** (sexp / 4)  # using i allows cutf = 0
        
            x = real(ifft(x))      # ignore imaginary part (numerical error)
            x = x / std(x) / 2         # std of signal = 1/2, mean = 0
            
    return x


def create_multisines(t, freq_used=array([1]), cutf = None, onf = None):
    """
    This function will produce a colored noise signal using the time points 
    as defined in array t. The signal will be constructed using sinosoids with 
    frequencies as defined in the array freq_used and randomized phases.

    The output consists of signal x        
    """

    tstop = t[-1]
    df = 1 / tstop # frequency stepsize
    dt = t[2] - t[1]
    data_points = len(t) # number of time or frequency steps  

    vector_freq = zeros(data_points) 
    vector_phase = zeros(data_points)

    if cutf != None:
        f = arange(0,data_points)*df
        if onf != None:
            freq_used = f[nonzero((f <= cutf) & (f >= onf))]
            #print freq_used
        else:            
            freq_used = f[nonzero((f <= cutf) & (f > 0))]
        
    index_f_used = round(freq_used / df).astype('int') # get indices of used frequencies in frequency vector
    
    index_fneg_used = (data_points - index_f_used).astype('int') # indices of negative frequencies 

    index_fall_used = concatenate((index_f_used, index_fneg_used)) # indices of pos+neg frequencies

    vector_freq[index_fall_used] = data_points / 2 # each frequency used ???
    
    phase = 2*pi*(random.rand(len(freq_used),1)-0.5) # pick phases randomly shifted by +-pi (sould there by another 2* to shift +-2pi???)

    vector_phase[index_f_used] = phase # assign positive phases to full vector
    vector_phase[index_fneg_used] = -phase # assign negative phases to full vector

    freqtemp = vector_freq * exp(1j * vector_phase) # generate frequency domain response
    x = real(ifft(freqtemp)) # convert into time domain

    #print "- Number of msine frequencies: " + str(2 * std(x) ** 2)
    
    noise_data = x/max(abs(x)) # scale so that signal amplitude is 1
    
    freq = fftfreq(data_points, dt)[ 0:round(data_points / 2) ] # positive frequency vector
   
    noise_power = abs(fft(noise_data))[ 0:round(data_points / 2) ] # compute noise power
    freq_wp = find(noise_power > 2 * std(noise_power)) # threshold to discriminate the indexes of peak frequencies
    freq_used = freq[freq_wp] # vector of used frequencies [Hz]
    
    return noise_data, freq, freq_wp, freq_used 
    

def create_singlesine(fu = 5, amp = 0.5, ihold = 1, dt = 0.025*ms, periods = 10, minlength = 1*s, t_prestim = 2*s, l_zeros = 2):
    """
    This function will produce a single sine signal of frequency fu with holding current ihold
    Signal has at least the length periods*T (s) or minlength (s).
    Use stimulate with pre stimulus of length t_prestim (s)        
    """
    
    fs = 1 / dt  # sampling rate 
    
    tnext = 0
    # delay for no noise input
    start_zeros =  zeros(l_zeros * fs)   
    t_zeros = tnext + arange(0, l_zeros, dt)
    
    tnext = t_zeros[-1] + dt
    l_pre_signal = ceil(t_prestim / (1. / fu)) * 1. / fu # length of pre stimulus should to be at least t_prestim seconds but with length of full periods
    t_pre_signal = arange(0, l_pre_signal, dt) # create pre time vector
    
    pre_signal = amp * sin(2 * pi * t_pre_signal * fu) # create pre signal vector
    t_pre_signal = t_pre_signal + tnext
    
    tnext = t_pre_signal[-1] + dt
    l_t = max(minlength, periods * 1 / fu) # length of input_signal: stimulate for at least periods*T or minlength
    t_input_signal = arange(0, l_t, dt) # create stimulus time vector
    
    #window = sin(2 * pi * t_input_signal * 1/l_t/2)  # not really good if nonlinear membrane function!!!!
    input_signal = amp * sin(2 * pi * t_input_signal * fu)
    t_input_signal = t_input_signal + tnext
    
    i_start = len(start_zeros) + len(pre_signal)
    i_stop = len(start_zeros) + len(pre_signal) + len(input_signal) 
    
    tnext = t_input_signal[-1] + dt
    l_post_signal = 1 # length of post stimulus should only be 1 s, equivalent to 1 Hz lower bound for spike cutoff 
    t_post_signal = arange(0, l_post_signal, dt) # create pre time vector
    post_signal = amp * sin(2 * pi * t_post_signal * fu) # create pre signal vector
    t_post_signal = t_post_signal + tnext

    all_data = concatenate((start_zeros, pre_signal, input_signal, post_signal)) # combine all
    t = concatenate((t_zeros, t_pre_signal, t_input_signal, t_post_signal)) # combine all
    t1 = arange(0, size(all_data) * dt, dt) # time vector of stimulus [s]  
    
    i_startstop = array([i_start, i_stop])
    t_startstop = array([t[i_start], t[i_stop]])
    
    iholdvec = concatenate((zeros(1 * fs), ones(len(all_data) - 1 * fs) * ihold)) # create holding current vector
    #iholdvec = concatenate((zeros(1 * fs), ones(len(all_data) - 2 * fs) * ihold, zeros(1 * fs))) # create holding current vector
    
    stimulus = all_data + iholdvec # create full stimulus vector
    
    return t, stimulus, i_startstop, t_startstop


def create_ZAP(dt=0.025*ms, ihold=1*nA, amp=0.1*nA, fmax=100, t_stim=30*s, ex=2):

    t = arange(0,t_stim,dt)
    zap = amp*sin(2*pi*((fmax*t**ex)/((ex+1)*t_stim**ex))*t) + ihold
    f=(fmax*t**ex)/(t_stim**ex)
    
    return t, zap, f
        

def aiftransfer(freq = arange(0,1000,1), tau = 20*ms, f0 = 100*Hz, i0 = 0*nA, rm = 10*MOhm, Vreset = 0, Vth = 1, Vrest = 0, delta_t = 0):
    """
    Create theoretical transfer function of a simplified integrate and fire neuron
    from Knight 1972
    """

    gamma = 1/tau
    # theoretical transfer function H(2*pi*i*freq)
    z = (gamma + 2 * pi * freq * 1j) / f0
    
    if i0 == 0:
        s0 = gamma / (1 - exp(-gamma / f0))   # theoretical current for f0
        H = f0/s0 * exp(gamma / f0) * (1 - exp(-z)) / z * exp(- 2*pi*delta_t*1j)
    else:
        s0 = (i0*rm + (Vrest - Vreset)) / (Vth - Vreset)
        H = rm/(Vth - Vreset) * f0/s0 * exp(gamma / f0) * (1 - exp(-z)) / z * exp(- 2*pi*delta_t*1j)
        
    # useful to return TF for zero gamma
    z = (2 * pi * freq * 1j) / f0
    H0 = f0/s0 * (1 - exp(-z)) / z
        
    return H, H0
    

def syn_kernel(t, tau1 = 5, tau2 = 10):    

    if tau1 == 0:
        
        G = exp(-t/tau2)    

    else:
        
        if (tau1/tau2 > .9999):
            tau1 = .9999*tau2
    
        tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
        factor = -exp(-tp/tau1) + exp(-tp/tau2)
        factor = 1/factor                        
        G = factor * (exp(-t/tau2) - exp(-t/tau1))
    
    return G
    
    
#def pop_giftransfer(freq = arange(0,1000,1), C = 0.5*1e-9, g = 0.025*1e-6, gr = 0.025*1e-6, tau_r = 100*1e-3, Vreset = -56*1e-3, Vth = -50*1e-3):
#    """
#    Create theoretical population transfer function of a general integrate and fire neuron with resonance
#    from Magnus et al. 2003
#    """
#
#    omega = 2*pi*freq
#    
#    gamma = gr/g
#    tau_m = C/g
#
#    def Y(omega):
#        (1 - g * tau_m * (Vth - Vreset) * A_LIF(omega)) / (1 + 1j * omega * tau_m)
#        
#    def A_LIF(omega):
#    
#    def A(omega):
#        return (1 - gamma * Y(omega) / (1 + gamma * Y(omega) + 1j * omega * tau_r)) * A_LIF(omega)
#           
#    return A(omega)
    

def fit_aiftransfer(freq, H0, f0, i0):
    """
    Fit theoretical transfer function of a simplified integrate and fire neuron to actual transfer function
    freq = used frequencies
    H = transfer function
    """
    
    from scipy.optimize import fmin, leastsq
    
    def peval(freq, f0, i0, p):
        H, _ = aiftransfer(freq, tau = p[0], f0 = f0, i0 = i0, delta_t = p[2])
        return p[1] * H
                
    def residuals(p, H0, freq, f0, i0):
        H = peval(freq, f0, i0, p)
        #err =  sum( abs(abs(H0) - abs(H)) + abs(unwrap(angle(H0)) - unwrap(angle(H0))) * 180/pi )**2 # 
        err = sum( ( concatenate((real(H0), imag(H0))) - concatenate((real(H), imag(H))) )**2 )
        #err =  sum( (abs(H0) - abs(H))**2 )
        
        return err
    
    p0 = array([20*ms, 1, 0])  # tau, rm 
    
    plsq = fmin(residuals, p0, args=(H0, freq, f0, i0))
    
    #print plsq
    tau = plsq[0] #[0] #plsq[0][0]
    scale = plsq[1] #[1]
    delta_t = plsq[2]
    
    H, _ = aiftransfer(freq, tau = tau, f0 = f0, i0 = i0, delta_t = delta_t)
    
    H = scale*H
 
    return tau, scale, H, delta_t
    

def fit_sinusoid(f, t0, u0, t = None):
    """
    Fit sinusoid of known frequency to signal
    f = known frequency
    t0 = timebase
    u0 = signal
    t = optional interpolation timebase
    """
    
    e = ones(size(t0))
    c = sin(2 * pi * f * t0) 
    s = cos(2 * pi * f * t0)
    
    a = array((e, c, s)).T
    b = linalg.lstsq(a, u0.T)[0]

    umean = b[0]
    uamp = sqrt(b[1] ** 2 + b[2] ** 2)
    uphase = math.atan2(b[2], b[1]) # convention: cos(2*pi*f*t+uphase)
    
    if t is not None:
        u = umean + uamp * sin(2 * pi * f * t + uphase)
    else:
        u = umean + uamp * sin(2 * pi * f * t0 + uphase)    
        t = t0

    
    # from scipy import io
    #if f == 1:
    #    figure(9)
    #    plot(t0, u0, 'b.', t, u, 'r')
    #    #ylim(25, 40) 
    #    from datetime import datetime
    #    idate = datetime.now().strftime('%Y%m%d_%H%M%S')  # %S
    #    savefig("./figs/" + idate + "-sin_" + str(f) + ".png", dpi = 300) # save it
    #    txt=array((t0,u0,u)).T
    #    savetxt("./figs/" + idate + "-sin_" + str(f) + ".txt", txt)
    #    show()
    #    clf()
    
    return umean, uamp, uphase, u
    

def fit_sinusoid2(f, t0, u0, t = None):
    """
    Fit sinusoid of known frequency to signal
    f = known frequency
    t0 = timebase
    u0 = signal
    t = optional interpolation timebase
    """
    
    from scipy.optimize import leastsq
    
    def peval(t, f, p):
        return p[0] + p[1] * sin(2 * pi * f * t + p[2])
        
    def residuals(p, u, t, f):
        err = u-peval(t, f, p)
        return err
    
    p0 = array([0, 0, 0])
    
    plsq = leastsq(residuals, p0, args=(u0, t0, f))
    
    umean = plsq[0][0]
    uamp = plsq[0][1]
    uphase = plsq[0][2]
    
    if t is not None:
        u = umean + uamp * sin(2 * pi * f * t + uphase)
    else:
        u = umean + uamp * sin(2 * pi * f * t0 + uphase)    
        t = t0
   
    return umean, uamp, uphase, u
    

def fit_sinusoid_fft(f, t0, u0, t = None, i = None):
    """
    Fit sinusoid of known frequency to signal using fft
    f = known frequency
    t0 = timebase
    u0 = signal
    t = optional interpolation timebase
    i = optional stimulus current (will return magnitude)
    """
    
    e = ones(size(t0))
    c = sin(2 * pi * f * t0) 
    s = cos(2 * pi * f * t0)
   
    a = array((e, c, s)).T
    b = linalg.lstsq(a, u0.T)[0]

    umean = b[0]
    umean = mean(u0)    
    
    if i is not None:
        
        b = linalg.lstsq(a, i.T)[0]
        imean = b[0]    
        
        expb = exp(-2 * pi * 1j * f * t0) # exponetial basis for Fourier series
        
        G = sum((u0 - umean) * expb) / sum((i - imean) * expb)
    
        uamp = abs(G)
        uphase = angle(G)
        u = []        
            
    else:
        
        expb = exp(-2 * pi * 1j * f * t0 / len(t0)) # exponetial basis for Fourier series

        G = sum((u0 - umean) * expb)
    
        uamp = abs(G)
        uphase = angle(G)-pi/2
    
        if t is not None:
            u = umean + uamp * sin(2 * pi * f * t + uphase)
        else:
            u = umean + uamp * sin(2 * pi * f * t0 + uphase)    
            t = t0
   
    
    return umean, uamp, uphase, u


def fit_exp(t0, u0, p0 = array([0, 0]), delay = 0):
    """
    Fit exponential to signal
    t0 = timebase
    u0 = signal
    """
    
    from scipy.optimize import leastsq
    
    def peval(t, p, delay = 0):
        dt = t[1]-t[0]
        
        u = p[0] * (1 - exp(-(t-delay)/p[1])) 
        
        u[0:delay/dt]=0 
        return u 
        
    def residuals(p, u, t, delay = 0):
        err = u-peval(t, p, delay)
        return err
    
    plsq = leastsq(residuals, p0, args=(u0, t0, delay))
    
    udecay = plsq[0][0]
    utau = plsq[0][1]
    
    u = peval(t0, array([udecay,utau]), delay)
    
    return udecay, utau, u
    

def fit_dualexp(t0, u0, p0 = array([0, 0, 0]), delay = 0):
    """
    Fit exponential to signal
    t0 = timebase
    u0 = signal
    """
    
    from scipy.optimize import leastsq
    
    def peval(t, p, delay = 0):
        dt = t[1]-t[0]
        
        tp = (p[1]*p[2])/(p[2] - p[1]) * log(p[2]/p[1])
        factor = 1 / (-exp(-tp/p[1]) + exp(-tp/p[2]))
        u = factor * p[0] * ( -exp(-(t-delay)/p[1]) + exp(-(t-delay)/p[2]) )
        
        u[0:delay/dt]=0 
        return u 
        
    def residuals(p, u, t, delay = 0):
        err = u-peval(t, p, delay)
        return err
    
    plsq = leastsq(residuals, p0, args=(u0, t0, delay))
    
    umax = plsq[0][0]
    utau1 = plsq[0][1]
    utau2 = plsq[0][2]
    
    u = peval(t0, array([umax,utau1,utau2]), delay)
    
    return umax, utau1, utau2, u
    

def fit_twodualexp(t0, u0, p0 = array([0, 0, 0, 0, 0, 0]), delay = 0):
    """
    Fit exponential to signal
    t0 = timebase
    u0 = signal
    """
    
    from scipy.optimize import leastsq
    
    def peval(t, p, delay = 0):
        dt = t[1]-t[0]
        u = p[0] * ( -exp(-(t-delay)/p[1]) + exp(-(t-delay)/p[2]) )
        v = p[3] * ( -exp(-(t-delay)/p[4]) + exp(-(t-delay)/p[5]) )
        
        tp = (p[1]*p[2])/(p[2] - p[1]) * log(p[2]/p[1])
        factor = 1 / (-exp(-tp/p[1]) + exp(-tp/p[2]))
        u = factor * p[0] * ( -exp(-(t-delay)/p[1]) + exp(-(t-delay)/p[2]) )
        
        tp = (p[4]*p[5])/(p[5] - p[4]) * log(p[5]/p[4])
        factor = 1 / (-exp(-tp/p[4]) + exp(-tp/p[5]))
        v = p[3] * ( -exp(-(t-delay)/p[4]) + exp(-(t-delay)/p[5]) )
        
        u = u+v
        u[0:delay/dt]=0 
        return u 
        
    def residuals(p, u, t, delay = 0):
        err = u-peval(t, p, delay)
        return err
    
    plsq = leastsq(residuals, p0, args=(u0, t0, delay))
    
    umax = plsq[0][0]
    utau1 = plsq[0][1]
    utau2 = plsq[0][2]
    
    vmax = plsq[0][3]
    vtau1 = plsq[0][4]
    vtau2 = plsq[0][5]
    
    u = peval(t0, array([umax,utau1,utau2,vmax,vtau1,vtau2]), delay)
    
    return umax, utau1, utau2, u, vmax,vtau1,vtau2
    
    
def fit_threedualexp(t0, u0, p0 = array([0, 0, 0, 0, 0, 0, 0, 0, 0]), delay = 0):
    """
    Fit exponential to signal
    t0 = timebase
    u0 = signal
    """
    
    from scipy.optimize import leastsq
    
    def peval(t, p, delay = 0):
        dt = t[1]-t[0]
        
        tp = (p[1]*p[2])/(p[2] - p[1]) * log(p[2]/p[1])
        factor = 1 / (-exp(-tp/p[1]) + exp(-tp/p[2]))
        u = factor * p[0] * ( -exp(-(t-delay)/p[1]) + exp(-(t-delay)/p[2]) )
        
        tp = (p[4]*p[5])/(p[5] - p[4]) * log(p[5]/p[4])
        factor = 1 / (-exp(-tp/p[4]) + exp(-tp/p[5]))
        v = p[3] * ( -exp(-(t-delay)/p[4]) + exp(-(t-delay)/p[5]) )
        
        tp = (p[7]*p[8])/(p[8] - p[7]) * log(p[8]/p[7])
        factor = 1 / (-exp(-tp/p[7]) + exp(-tp/p[8]))
        z = p[6] * ( -exp(-(t-delay)/p[7]) + exp(-(t-delay)/p[8]) )
        
        u = u+v+z
        u[0:delay/dt]=0 
        return u 
        
    def residuals(p, u, t, delay = 0):
        err = u-peval(t, p, delay)
        return err
    
    plsq = leastsq(residuals, p0, args=(u0, t0, delay))
    
    umax = plsq[0][0]
    utau1 = plsq[0][1]
    utau2 = plsq[0][2]
    
    vmax = plsq[0][3]
    vtau1 = plsq[0][4]
    vtau2 = plsq[0][5]
    
    zmax = plsq[0][6]
    ztau1 = plsq[0][7]
    ztau2 = plsq[0][8]
    
    u = peval(t0, array([umax,utau1,utau2,vmax,vtau1,vtau2,zmax,ztau1,ztau2]), delay)
    
    return umax, utau1, utau2, u, vmax, vtau1, vtau2, zmax, ztau1, ztau2

    
def fit_tripleexp(t0, u0, p0 = array([0, 0, 0, 0]), delay = 0):
    """
    Fit exponential to signal
    t0 = timebase
    u0 = signal
    """
    
    from scipy.optimize import leastsq
    
    def peval(t, p, delay = 0):
        dt = t[1]-t[0]
        u = p[0] * ( -exp(-(t-delay)/p[1]) + exp(-(t-delay)/p[2]) + exp(-(t-delay)/p[3]) )
        u[0:delay/dt]=0 
        return u 
        
    def residuals(p, u, t, delay = 0):
        err = u-peval(t, p, delay)
        return err
    
    plsq = leastsq(residuals, p0, args=(u0, t0, delay))
    
    umax = plsq[0][0]
    utau1 = plsq[0][1]
    utau2 = plsq[0][2]
    utau3 = plsq[0][3]
    
    u = peval(t0, array([umax,utau1,utau2,utau3]), delay)
    
    return umax, utau1, utau2, utau3, u
    

def shannon_interp(t0, y0, t, tu=None, chunk_size=0):
    """ 
    function y = shannon_interp(t0, y0, t, tu)
    
    band-passed interpolation of non-uniform sampled data to
    done by brute-force n x n matrix inversion, so only viable for
    small sample numbers n.
    
    t0 = non-uniform sample times
    with t0 = [] uses linspaced data with same range as tu
    y0 = sample data

    tu = uniform sampling times (same length as t0)
    t  = timebase for interpolation

    y  = values at interpolated times t
    """
    #print size(y0)
    
    y = []
    tn = []
    it_start = 0    
    
    if chunk_size == 0:
        chunk_size = len(t0)
        
    for i in xrange(0, len(t0), chunk_size):  # split with given chunk_size

        t0_chunk = t0[i:i + chunk_size]
        y0_chunk = y0[i:i + chunk_size]   
        
        if i == int((ceil((len(t0) / double(chunk_size)) - 1) * chunk_size)):  # if this the last run
            it_end = len(t)  # use full rest of t 
        else:    
            it_end = t.searchsorted(t0_chunk[-1]) + 1 
        
        ## test searchsorted() 
        # t = array([1,2,3,3.9,4.1,5,6])
        # t.searchsorted(4)
        # print str(it_start) + " " + str(it_end)
        
        t_offset = t[it_start]
        t0_chunk = t0_chunk - t_offset       
        t_chunk = t[it_start:it_end+1] - t_offset
        it_start = it_end + 1
        
        nsamp = len(t0_chunk)
        if tu == None:
            tu = linspace(t0_chunk[0], t0_chunk[-1], nsamp)
        
        T = tu[2]-tu[1]
        G = zeros((nsamp, nsamp))
    
        for l in range(nsamp):
            for k in range(nsamp):
                G[l, k] = sinc((t0_chunk[l] - tu[k]) / T)
    
        yu = linalg.solve(G,y0_chunk)
    
        y_chunk = 0
        for l in range(nsamp):
            y_chunk = y_chunk + yu[l] * sinc((t_chunk - tu[l]) / T)
                    
        y.append(y_chunk)
        tn.append(t_chunk)
    
    y = concatenate(y) 
    tn = concatenate(tn)
    
    #plot(t0, y0, "*b", t, y, "-r")
    #axis(ymin=min(y0), ymax=max(y0))
    #show()
    
    ## check missed values
    #print set(t).difference(set(tn))
    
    return y


def get_spikes(v, vthres, t):
    '''
    s, ts = get_spikes(v, vthres, t)
    Spike detection implemented as voltage threshold crossings
    v = membrane potential
    vthres = threshold
    t = time vector (in s)
    -
    s = spike 'bit' vector s
    ts = (interpolated) spike times
    '''
    
    s = zeros(len(v))
    s[1:] = ((v[:-1] <= vthres) & (v[1:] > vthres)).astype(float)   
    i = nonzero(s)[0]
    ts = t[i-1]+(t[i]-t[i-1])*(vthres-v[i-1])/(v[i]-v[i-1])
    
    return s, ts
    
    
def get_spikefreq(spike_times, stimlength = 0, compute_mean = 0, change_factor = 0.3):
    '''
    Compute spike frequency, set to second spike for causality, 
    also returns mean and onset frequency
    Note: time in s now!
    '''
    
    if len(np.array(spike_times)) >= 2:  # compute only if at least 2 spikes
        spike_diff = diff(spike_times)      # differentiate for spike intervals!
        spike_freq = 1 / spike_diff         # spike_times is in s
        freq_times = spike_times[1:]        # Set frequency to second spike for causality!
        freq_onset = spike_freq[0]          # Onset frequency only

        a = [0]
        if compute_mean > 1:
            a = where(freq_times > (stimlength - 1)) 
            print a
            
        elif change_factor == 0:
            freq_mean = mean(spike_freq[int(np.ceil(len(spike_freq)*9/10.)):])
            
        elif any(spike_times > (stimlength / 2)) & compute_mean: # compute only if there is sustained spiking and a request is actually made
            freq_change = abs(diff(spike_freq)) # check how the firing rate is changing
            a = where(freq_change < change_factor) # check if rate is adapting
        
        if any(a):  # only compute mean firing rate if spiking adapts
            freq_stable = a[0][0]    # get first tuple and first element. here the firing rate has adapted
            freq_mean = mean(spike_freq[freq_stable:]) # compute mean firing rate
        else:
            freq_mean = float('NaN')


    else:  
        freq_times = np.array([])
        spike_freq = np.array([])
        freq_mean = float('NaN')
        freq_onset = float('NaN')
        
    return freq_times, spike_freq, freq_mean, freq_onset
    

def get_magphase(input_signal, t, output_signal, t_out, method = "peak", f = None, deb = 0):
    """
    Compute the mean amplitude, mean magnitude and mean phase from an input and 
    output signal with a single frequency. 

    Default method *peak": peak detection    
    Magnitude is computed by: full oscillation out / full oscillation in 
    Minimum length of signal and response has to be 1.5 * 1 / f
    
    Other method "fit": least-square fit, f has to be given!
    """
    
    if method == "peak":  # t and t_out have to be identical!
    
        if size(t) != size(t_out):  
            raise ValueError('Input and output need same timebase! Use method "fit" instead!')
        
        dt = t[2] - t[1]
        
        input_signal_temp = input_signal - mean(input_signal) # remove holding current for amplitude detection
        pos_halfwaves = concatenate((where(input_signal_temp > 0.75 * max(input_signal_temp))[0], array([len(input_signal_temp)]))) # detect the positive half waves
        iend = pos_halfwaves[where(diff(pos_halfwaves) > 2)[0]] # detect where the half waves end, limit of 2 is arbitrary, 1 would also work
        f = 1 / (mean(diff(iend)) * dt) # compute frequency 
    
        peak_vec_in = zeros(len(iend)-1) # construct vectors for peak values 
        peak_vec_out = zeros(len(iend)-1) # construct vectors for peak values 
        peak_vec_in_index = zeros(len(iend)-1, int) # construct vectors for peak indexes
        peak_vec_out_index = zeros(len(iend)-1, int) # construct vectors for peak indexes
        amptwice_vec_in = zeros(len(iend)-1) # construct vectors for size of full oscillation
        amptwice_vec_out = zeros(len(iend)-1) # construct vectors for size of full oscillation
        mean_vec_out = zeros(len(iend)-1) # construct vectors for size of full oscillation
    
        for i in range(len(iend)-1): # iterate through all sections
            wave_cut_in = input_signal[iend[i]:iend[i+1]] # cut out input section
            peak_vec_in[i], i_in, min_in = wave_cut_in.max(0), wave_cut_in.argmax(0), wave_cut_in.min(0) # get peak positions, indexes and minimas
            amptwice_vec_in[i] = peak_vec_in[i] - min_in
            peak_vec_in_index[i] = int(iend[i] + i_in) # convert to global indexes
            
            wave_cut_out = output_signal[iend[i]:iend[i+1]] # cut out output section
            peak_vec_out[i], i_out, min_out = wave_cut_out.max(0), wave_cut_out.argmax(0), wave_cut_out.min(0) # get peak positions, indexes and minimas
            amptwice_vec_out[i] = peak_vec_out[i] - min_out         
            peak_vec_out_index[i] = int(iend[i] + i_out) # convert to global indexes
            
            mean_vec_out[i] = amptwice_vec_out[i]*0.5 + min_out
    
        phase_mean = mean((peak_vec_in_index-peak_vec_out_index) * dt / (1 / f) * 360) # compute phase
        
        mag_mean = mean(amptwice_vec_out / amptwice_vec_in) # compute magnitude 
        amp_mean = mean(peak_vec_out) # just compute mean of amplitude
        umean =  mean(mean_vec_out[i])
        
        
    if method == "fit":
        
        print "- fitting"
        imean, iamp, iphase, i = fit_sinusoid(f, t, input_signal)
        umean, uamp, uphase, u = fit_sinusoid(f, t_out, output_signal, t)
        
        phase_mean = (uphase - iphase) * (180 / pi)
        mag_mean = uamp / iamp 
        amp_mean = uamp + umean  # for voltage amplitude DAngelo 2001
        
        
    if method == "fft": 
        
        if any(abs(t_out - t) > 0.00000001):  # bloody floating point precision       
            raise ValueError('Time base of input (t) and output (t_out) have to be identical')
        
        imean, iamp, iphase, i = fit_sinusoid(f, t, input_signal)
        umean, ramp, rphase, u = fit_sinusoid_fft(f, t_out, output_signal, i = input_signal) 
        
        phase_mean = rphase * (180 / pi)
        mag_mean = ramp 
        amp_mean = ramp * iamp 
        
        print f, amp_mean
    
    #plt.figure('FIT') 
    #ax99 = plt.subplot(1,1,1)
    #ax99.plot(t_out, output_signal,'b')
    #ax99.plot(t_out, u,'r')
    #plt.savefig("./figs/fit_" + str(f) + ".pdf", dpi = 300, transparent=True) # save it  
    #plt.clf()
              
                                  
    return amp_mean, mag_mean, phase_mean, umean, u


def construct_Stimulus(noise_data, fs, amp = 1, ihold = 0, tail_points = 2, delay_baseline = 4):
    """
    Construct Stimulus from cnoise/msine input and other parameters.
    """

    #inin = 8 # stimulate before with 10 s of signal
    inin = np.array((len(noise_data)/fs)*0.1).clip(max=8)

    stim_data = concatenate((noise_data[-inin*fs:], noise_data))  # increase length of stimulus # no normalization here: / max(abs(noise_data))
    
    stimulus = concatenate((concatenate((zeros(round(delay_baseline*fs)), amp * stim_data)), zeros(round(tail_points*fs))))  # construct stimulus
            
    iholdvec = concatenate((zeros(round(fs)), ones(round(len(stimulus) - 1 * fs)) * ihold))
    stimulus = stimulus + iholdvec
    
    dt = 1 / fs
    t = arange(0, len(stimulus) * dt,dt)  # time vector of stimulus [s]
            
    t_startstop = np.array([inin+delay_baseline, inin+delay_baseline+len(noise_data)/fs])
    
    return stimulus, t, t_startstop
    

def construct_Pulsestim(dt = 0.025e-3, pulses = 1, latency = 10e-3, stim_start = 0.02, stim_end = 0.02, len_pulse = 0.5e-3, amp_init = 1, amp_next = None):
    """
    Construct a pulse stimulus in the form of |---stim_start---|-len_pulse-|--(latency-len_pulse)--|...|-len_pulse-|--(latency-len_pulse)--|--stim_end--|
    For stim_end shorter than pulse: stim_end = stim_end + len_pulse
    """
    #print dt
    
    fs = 1 / dt
    
    if len(np.shape(amp_next)) > 0:
        pulses = len(amp_next)
        amp_vec = amp_next
    else:
        if amp_next == None:
            amp_vec = np.ones(pulses)*amp_init
        else:
            amp_vec = np.ones(pulses)*amp_next
            amp_vec[0] = amp_init

    if len(np.shape(latency)) > 0: 
        pass
    else:
        latency = np.ones(pulses)*latency
        
    if len(amp_vec) != len(latency):    
        raise ValueError('amp_vec and latency vectors do not have the same size!!!')
        
     
    if stim_end < len_pulse:
        print "From construct_Pulsestim: stim_end shorter than pulse, setting stim_end = stim_end + len_pulse"
        stim_end = stim_end + len_pulse
           
    ivec = zeros(round((stim_start + sum(latency) + stim_end)*fs))  # construct zero vector to begin with 
    t = arange(0, len(ivec))*dt  # construct time vector
    
    ivec[round(stim_start*fs):round((stim_start+len_pulse)*fs)] = amp_vec[0]
                    
    for i in range(1, pulses):
        
        ivec[round((stim_start+sum(latency[0:i]))*fs):round((stim_start+sum(latency[0:i])+len_pulse)*fs)] = amp_vec[i]
        
        #ivec_new = concatenate((zeros(round((stim_start + (i - 1) * latency[i-1]) * fs)), ones(round(len_pulse * fs)) * amp))  # stimulus delay + spike
        # 
        #ivec_new = concatenate((ivec_new, zeros(round(((pulses - i + 1) * latency[i-1] - len_pulse + stim_end) * fs))))  # rest of stimulus
        # 
        #if len(ivec) > len(ivec_new):
        #    ivec_new =  concatenate(( ivec_new, zeros(len(ivec)-len(ivec_new)) ))
        #    
        #if len(ivec) < len(ivec_new):
        #    rem = -1*(len(ivec_new)-len(ivec))
        #    ivec_new =  ivec_new[:rem]
            
        #ivec = ivec + ivec_new
    
    #if len(t) != len(ivec):    
        #raise ValueError('Both vectors do not have the same size!!!')
                               
    return t, ivec


def compute_Impedance(voltage, current, t1, stimulus, t, noise_data_points, freq_wp = None, do_csd = 0, w_length = 5):
    """
    Compute the impedance using input and output and cross spectral density method, 
    cut out relevant parts with knowledge of noise_data_points
    w_length: desired length of csd window
    """

    from matplotlib.mlab import csd
    
    response = voltage
    #stimulus_out = current

    # no interpolation is best !!
    #stimulus_out_ds = interp(t,t1[:-1],stimulus_out[:-1]) # interpolate (downsample) to be eqivalent with input
    response_ds = interp(t,t1[:-1],response[:-1]) # interpolate (downsample) to be eqivalent with input

    #stimulus_out_ds = stimulus_out[:end-1] # no interpolation
    #response_ds = response[:end-1]
    
    #noiseinput = stimulus_out_ds[:end-1] # interplation introduces error!!!
    noiseinput = stimulus # interplation introduces error: use original input

    dt = t[2] - t[1]
    fdataend = max(find(noiseinput - noiseinput[round(1.5 / dt)])) # get rid of constant input for detection
    noiseinput_cut = noiseinput[(fdataend - noise_data_points):fdataend] # cut out relevant part
    response_ds_cut = response_ds[(fdataend - noise_data_points):fdataend] # cut out relevant part
    
   
    if do_csd:

        # power spectrum & transfer function using cross spectral density     
        fs = 1 / dt
        nfft = 2 ** int(ceil(log2(w_length * fs)))    # nfft is the number of data points used 
                                                    # in each block for the FFT. Must be even; 
                                                    # a power 2 is most efficient.
        #w_true_length = nfft * dt   # true length window
        
        c_in_out, freq = csd(noiseinput_cut,  response_ds_cut, Fs = fs, NFFT = nfft, noverlap = nfft/4)
        c_in_in, freq = csd(noiseinput_cut,  noiseinput_cut, Fs = fs, NFFT = nfft, noverlap = nfft/4)
    
        #c_in_out = abs(c_in_out)  # phas einformation is lost by doing this
        #c_in_in = abs(c_in_in)
        
        zdata_pos = c_in_out / c_in_in
    
        pwr = abs(zdata_pos) # Impedance Power
        pha = angle(zdata_pos, deg = True) # Impedance Phase in degree
        
        if freq_wp == None: # check if specific frequencies are used 
        
            fstart = find(freq >= 1. / w_length)[0] # lowest frequency depends on window length
            fend = max(find(freq < 1000))  # only use frequencies smaller than 1kHz        
            magz = pwr[fstart:fend] # don't use freq = 0 
            phaz = pha[fstart:fend] # don't use freq = 0
            freq_used = freq[fstart:fend] # don't use freq = 0
            #print "- number of cnoise frequencies: " + str(len(freq_used)) + " Hz"
            
        else:   # multi sine stimulation
        
            freq_was = fftfreq(noise_data_points, dt)[ 0:round(noise_data_points / 2) ]  # full frequency vector without windowing
            freq_used = freq_was[freq_wp] # the frequencies that were used
            
            #freq_wp_new = freq.searchsorted(freq_used)  
            freq_wp_new = [abs(freq-i).argmin() for i in freq_used] # find corresponding indexes to sued frequencies
            
            error = sum(abs(freq_used - freq[freq_wp_new]))
            print "- frequency conversion error due to csd: " + str(error) + " Hz"
                        
            magz = pwr[freq_wp_new]
            phaz = pha[freq_wp_new]
            freq_used = freq[freq_wp_new]  # freq_used is changed here to new values !
                   
    else:
        
        noiseinput_f = fft(noiseinput_cut) # transform into frequency domain
        response_f = fft(response_ds_cut) # transform into frequency domain
    
        zdata = response_f / noiseinput_f # mV/nA = MOhms
        zdata_pos = zdata[0:round(len(noiseinput_f) / 2)] # Take only the first half of the power to avoid redundancy
    
        pwr = abs(zdata_pos) # Impedance Power
        pha = angle(zdata_pos, deg=True) # Impedance Phase in degree
        
        freq = fftfreq(noise_data_points, dt)[0:round(noise_data_points / 2)] # positive frequency vector
        
        if freq_wp == None: # check if specific frequencies are used 
        
            fstart = find(freq >= 1. / (noise_data_points * dt))[0]  # lowest frequency depends on window length
            fend = max(find(freq < 1000))  # only use frequencies smaller than 1kHz        
            magz = pwr[fstart:fend]  # don't use freq = 0 
            phaz = pha[fstart:fend]  # don't use freq = 0
            freq_used = freq[fstart:fend]  # don't use freq = 0
            #print "- number of cnoise frequencies: " + str(len(freq_used)) + " Hz"
            
        else:   # multi sine stimulation
            magz = pwr[freq_wp]
            phaz = pha[freq_wp]
            freq_used = freq[freq_wp]
            
            
    ca = 1 / (magz * exp(pi / 180 * 1j * phaz))
    
    #figure; plot(t1, current); show()
    #figure; semilogx(freq_used, magz); show()
            
    return magz, phaz, ca, freq, freq_used
    
    
def compute_Transfer(spike_freq, freq_times, stimulus, t, noise_data_points, gsyn, gsyn_in = None, freq_wp = None, method_interpol = array(["linear", "quadratic", "shannon", "syn"]), do_csd = 0, nc_delay = 0, w_length = 2*s, t_kernel = 0, t_qual = 0, K_mat_old = np.array([]), t_startstop = [], give_psd = False):
    """
    Compute the transfer function using cross spectral density method, 
    cut out relevant parts with knowledge of noise_data_points
    w_length: desired length of csd window
    t_kernel: length of final kernel, default = full length
    t_qual: length of quality comparison, default: 0*s comparison
    """
    from matplotlib.mlab import csd
    from scipy.interpolate import InterpolatedUnivariateSpline 
    import neuronpy.util.spiketrain
    import nitime.algorithms as tsa
    #from csdmt import csdmt
        
    dt = t[2] - t[1]
    shift = 0  # only important for syn if nc_delay was used
    
    fmean = None
    for i, ii in enumerate(method_interpol):    
        
        # Here we have to interpolate!!
        if "linear" in ii:   
            # linear interpolation
            response_ds = interp(t, freq_times, spike_freq, left = 0, right = 0) # interpolate (upsample) to be eqivalent with input, set zero beginning and end


        if "bin" in ii:
            
            dt0 = dt
            t0 = t
            
            response_ds = spike_freq
            t = freq_times 
            dt = t[2] - t[1]
            
            if dt != dt0: #adjust stimulus
                stimulus = interp(t,t0,stimulus) 
                print "Adjust Stimulus", len(response_ds), len(stimulus)
            
        
        if "dslin" in ii:  
            # linear interpolation
            response_ds = interp(t, freq_times, spike_freq, left = 0, right = 0) # interpolate (upsample) to be eqivalent with input, set zero beginning and end

            # test downsampling
            dt2 = 5e-3    
            dt_fac = dt/dt2
            t2 = arange(0, t[-1], dt2)
            
            response_ds = interp(t2,t,response_ds) 

            # downsample to bin size dt!
            stimulus = interp(t2,t,stimulus) 
            t = t2
            dt = dt2
            noise_data_points = noise_data_points*dt_fac 
            
            #figure(44)
            #plot(freq_times, ones(len(freq_times)), '*')
            #plot(t, response_ds)

    
        if "binary" in ii:  
            # binary spike code
            dt2 = 1e-3
            dt_fac = dt/dt2
            t2 = arange(0, t[-1], dt2)  # define edges, otherwise get_histogram does not define edges consistently!!!
            
            [response_ds, _] = neuronpy.util.spiketrain.get_histogram(freq_times, bins = t2)
            response_ds = concatenate((zeros(1),response_ds))
            
            # downsample to bin size dt!
            stimulus = interp(t2,t,stimulus) 
            t = t2
            dt = dt2
            noise_data_points = noise_data_points*dt_fac 
            

        if "dt" in ii:  
            # binary spike code using dt as bin size!
            [response_ds, _] = neuronpy.util.spiketrain.get_histogram(freq_times, bins = t)
            response_ds = concatenate((zeros(1),response_ds))
            
            
        if "quadratic" in ii:  
            # quadratic interpolation
            sfun = InterpolatedUnivariateSpline(freq_times, spike_freq, k=2)
            response_ds = sfun(t)
            
        
        if "shannon" in ii:                               
            # shannon interpolation
            response_ds = shannon_interp(freq_times, spike_freq, t)

                
        if "syn" in ii:    
            # synaptic integration                
            response_ds = gsyn
            shift = int(nc_delay/dt)  # shift response by the nc delay to remove offset
            
            
        if "gsyn_in" in ii:    
            # synaptic integration                
            response_ds = gsyn_in
            shift = (2*ms)/dt  # shift response by the delay of 1*ms spike play + 1*ms synapse nc to remove offset
            
        noiseinput = stimulus  # interplation introduces error: use original input
        
    
        if len(t_startstop) < 2:
            fdataend = max(find(noiseinput-noiseinput[round(1.5 / dt)]))+1  # get rid of constant input for detection
            noiseinput_cut = noiseinput[(fdataend-noise_data_points):fdataend]  # cut out relevant part
            response_ds_cut = response_ds[(fdataend-noise_data_points)+shift:fdataend+shift]  # cut out relevant part
        else:
            print "- t_startstop given!"            
            #i_start = mlab.find(t >= t_startstop[0])
            #i_stop = mlab.find(t <= t_startstop[1])
            #noiseinput_cut = noiseinput[i_start:i_stop]  # cut out relevant part
            #response_ds_cut = response_ds[i_start+shift:i_stop+shift]  # cut out relevant part
            
            noiseinput_cut = noiseinput[int(t_startstop[0]/dt):int(t_startstop[1]/dt)]  # cut out relevant part
            response_ds_cut = response_ds[int(t_startstop[0]/dt)+shift:int(t_startstop[1]/dt)+shift]  # cut out relevant part
            
        if "linear" in ii:
            fmean = mean(response_ds_cut)  # take fmean from linear interpolation
        

        rmean = mean(response_ds_cut) 
        smean = mean(noiseinput_cut) 
        
        if do_csd:  # Use CSD for colored/white noise AND multi sine estimation 
            
            print "- using csd for transfer function estimation! \r"
                        
            # power spectrum & transfer function using cross spectral density     
            fs = 1 / dt
            nfft = 2 ** int(ceil(log2(w_length * fs)))  # nfft is the number of data points used 
                                                        # in each block for the FFT. Must be even; 
                                                        # a power 2 is most efficient.
            w_true_length = nfft * dt   # true length window
            
            r = response_ds_cut-rmean  # response, no mean
            s = noiseinput_cut-smean  # signal, no mean 

            if "NOT_USED" in ii:  
                
                sr = np.array([s,r])
                BW = None #0.05
                
                #NW = BW / (2 * fs) * len(s)
                #print NW
                
                freq, P = tsa.multi_taper_csd(sr, Fs=fs, BW=BW, low_bias=True, adaptive=True, sides='twosided')
                P_ss = P[0,0,:]
                P_rr = P[1,1,:]
                P_sr = P[0,1,:]
                P_rs = P[1,0,:]
                 
                # results are returned like a = array([0, 1, 2, -3, -2, -1])
                # already ordered for ifft!
                # but freq is wrong!!
                
            else:
                
                noverlap = nfft / 4
                
                #print np.shape(r)
                P_rr, freq = csd(r, r, Fs = fs, NFFT = nfft, noverlap = noverlap, sides = 'twosided')  # , window =  mlab.window_none
                P_sr, freq = csd(s, r, Fs = fs, NFFT = nfft, noverlap = noverlap, sides = 'twosided')
                P_rs, freq = csd(r, s, Fs = fs, NFFT = nfft, noverlap = noverlap, sides = 'twosided')
                #P_rs = concatenate([array([0]),P_sr[1:][::-1]])  # P_rs(f) = P_sr(-f)
                #P_rs, freq = csd(r, s, Fs = fs, NFFT = nfft, noverlap = nfft / 4, sides = 'twosided')
                P_ss, freq = csd(s, s, Fs = fs, NFFT = nfft, noverlap = noverlap, sides = 'twosided')
                
                # results are returned like a = array([-3, -2, -1, 0, 1, 2])
                # for ifft it should be: ifftshift(a) = array([ 0,  1,  2, -3, -2, -1])
                # reorder here!!
                P_rr = ifftshift(P_rr)
                P_sr = ifftshift(P_sr)
                P_rs = ifftshift(P_rs)
                P_ss = ifftshift(P_ss)
                freq = ifftshift(freq)
            
            if sum(real(P_ss))<=0: # there is no power in the input, just check the output!
                give_psd = True
                P_ss = np.ones(len(P_ss)) 
                P_sr = np.ones(len(P_sr))
                P_rs = np.ones(len(P_rs))
                print "- return PSD of output only!"

            P_nn = (P_ss - abs(P_sr)**2/P_rr)
            SNR = (P_ss / P_nn)[0:len(freq)/2] # only positive frequencies
            VAFf = (abs(P_sr)**2/(P_rr*P_ss))[0:len(freq)/2]
            
            #fend = max(find(freq[0:len(freq)/2] < 100)) # only frequencies up to 100 Hz
            #print "Total VAF: " + str(sum(( abs(P_sr)**2 / P_ss )[0:len(freq)/2][0:fend]) / sum(P_rr[0:len(freq)/2][0:fend]))
            
            if give_psd:
                fdata_pos = (P_rr)[0:len(freq)/2] # only positive frequencies
            else:
                fdata_pos = (P_sr/P_ss)[0:len(freq)/2] # only positive frequencies
            
            Kf0 = (P_rs/P_rr) 
            
            P_ss = P_ss[0:len(freq)/2]
            
            freq =  freq[0:len(freq)/2]  # only positive frequencies
            
            #figure(33)                
            #plot(freq, P_ss[0:len(P_ss)/2])
            #figure(34)
            #plot(freq, P_rr[0:len(P_rr)/2])
            #figure(35)
            #plot(freq, P_sr[0:len(P_sr)/2])
            #figure(36)
            #plot(freq, P_rs[0:len(P_rs)/2])
                                
            #figure(37)
            #semilogx(freq, 10*log10(SNR))
            #show()
            
            pwr = abs(fdata_pos) # Transfer Power
            pha = angle(fdata_pos, deg = True) # Transfer Phase in degree
            
            if t_kernel == 0:
                n_kernel = len(Kf0)
            else:
                n_kernel = int(t_kernel/dt)
                
            tk = arange(n_kernel)*dt 
            
            if freq_wp == None: # colored/white noise: check if specific frequencies are used 
            
                #fstart = find(freq >= 1. / w_length)[0] # lowest frequency depends on window length
                fstart = 1
                fend = max(find(freq < 1000))  # only use frequencies smaller than 1kHz
                #freq_used = freq[fstart:fend] # don't use freq = 0
                freq_used = freq[fstart:fend]
                    
                if i == 0:
                    # create matrix to hold all different interpolation methods:
                    ca_mat = zeros((len(method_interpol),len(fdata_pos)), dtype=complex)
                    mag_mat = zeros((len(method_interpol),len(freq_used)))  # impedance magnitude vector
                    pha_mat = zeros((len(method_interpol),len(freq_used)))  # imedance phase vector           
                    K_mat = zeros((len(method_interpol),n_kernel), dtype=complex) 
                    SNR_mat = zeros((len(method_interpol),len(freq_used))) 
                    VAFf_mat = zeros((len(method_interpol),len(freq_used)))
                    Qual_mat = zeros((len(method_interpol),3,3)) 
            
                mag_mat[i,:] = pwr[fstart:fend].T  # don't use freq = 0 
                pha_mat[i,:] = pha[fstart:fend].T  # don't use freq = 0
              
                # USE FULL FREQ FOR THIS
                ca_mat[i,:] = fdata_pos.T #[fstart:fend].T
                SNRi = 10*log10(real(SNR[fstart:fend].T))
                SNR_mat[i,:] = SNRi
                
                VAFi = real(VAFf[fstart:fend].T)
                VAFf_mat[i,:] = VAFi
            
                P_ss = P_ss[fstart:fend]

                #K_mat[i,:] = ifftshift(ifft(Kf0,n_kernel))  
                #K_mat[i,:] = ifftshift(ifft(concatenate((Kf0,Kf0[::-1][:-1])),n_kernel))   
                K_mat[i,:] = ifftshift(ifft(Kf0))[len(Kf0)//2-n_kernel//2:len(Kf0)//2-n_kernel//2+n_kernel] # cutting is better than "downsampling"
               
                #print "- number of cnoise frequencies: " + str(len(freq_used))
                
            else:   # multi sine stimulation
            
                freq_was = fftfreq(noise_data_points, dt)[ 0:round(noise_data_points / 2) ]  # full frequency vector without windowing
                freq_used = freq_was[freq_wp] # the frequencies that were used
                
                #freq_wp_new = freq.searchsorted(freq_used)  
                freq_wp_new = [abs(freq-i).argmin() for i in freq_used] # find corresponding indexes to sued frequencies
                
                error = sum(abs(freq_used - freq[freq_wp_new]))
                print "- frequency conversion error due to csd: " + str(error) + " Hz"
                print freq_used
                print freq[freq_wp_new]
                
                freq_used = freq[freq_wp_new]  # freq_used is changed here to new values !!
                
                if i == 0:
                    # create matrix to hold all different interpolation methods:
                    ca_mat = zeros((len(method_interpol),len(freq_used_new)), dtype=complex)    
                    mag_mat = zeros((len(method_interpol),len(freq_used_new)))  # impedance magnitude vector
                    pha_mat = zeros((len(method_interpol),len(freq_used_new))) # imedance phase vector           
                    K_mat = zeros((len(method_interpol),n_kernel), dtype=complex) 
                    SNR_mat = zeros((len(method_interpol),len(freq_used_new))) 
                    VAFf_mat = zeros((len(method_interpol),len(freq_used_new))) 
                    Qual_mat = zeros((len(method_interpol),3,3)) 
                    
                mag_mat[i,:] = pwr[freq_wp_new]
                pha_mat[i,:] = pha[freq_wp_new]
                ca_mat[i,:] = fdata_pos[freq_wp_new]
                
                K_mat[i,:] = ifftshift(ifft(Kf0,n_kernel))[freq_wp_new]
                
                
                SNRi = 10*log10(real(SNR[freq_wp_new]))
                SNR_mat[i,:] = SNRi
                
                VAFi = real(VAFf[freq_wp_new])
                VAFf_mat[i,:] = VAFi                
                
                P_ss = []
                
            
            # get SNR and VAF quality 
            SNR0 = mean(SNRi[0:3])
            iSNRc = find(SNRi <= (SNR0-3))
            if len(iSNRc) == 0:
                SNR_cutff = float('NaN')
            else:
                SNR_cutff = freq_used[iSNRc[0]];
            SNR_mean = mean(SNRi)
                
            Qual_mat[i,0,0] = SNR0
            Qual_mat[i,0,1] = SNR_cutff
            Qual_mat[i,0,2] = SNR_mean
            
            #print 'SNR0:', SNR0
                
            VAF0 = mean(VAFi[0:3])
            iVAFc = find(VAFi <= 0.9*VAF0)
            if len(iVAFc) == 0:
                VAF_cutff = float('NaN')
            else:
                VAF_cutff = freq_used[iVAFc[0]];
            VAF_mean = mean(VAFi)
                
            Qual_mat[i,1,0] = VAF0
            Qual_mat[i,1,1] = VAF_cutff
            Qual_mat[i,1,2] = VAF_mean                 
                   
            if t_qual > 0:  # compare signal to estimated signal 
            
                resp = r[0:t_qual/dt]  # used response     
                stim = s[0:t_qual/dt]  # used signal
                t2 = t[0:t_qual/dt]
                
                if len(K_mat_old) > 0:
                    K_mat = K_mat_old
                
                tc, resp_cc, stim_cc, stim_re_cc, noise_cc, CF, VAF = reconstruct_Stimulus(K_mat[i,:], resp, stim, t2)

              
                if i == 0:
                    # create matrix to hold all different interpolation methods:
                    stim_re_mat = zeros((len(method_interpol),len(stim_re_cc)))
                    resp_mat = zeros((len(method_interpol),len(resp_cc)))
                    noise_mat = zeros((len(method_interpol),len(noise_cc)))
                    CF_mat = zeros(len(method_interpol)) 
                    VAF_mat = zeros(len(method_interpol)) 

                stim_re_mat[i,:] = stim_re_cc
                stim = stim_cc
                resp_mat[i,:] = resp_cc
                noise_mat[i,:] = noise_cc
                CF_mat[i] = CF   
                VAF_mat[i] = VAF 
                
                Qual_mat[i,2,0] = CF
                Qual_mat[i,2,1] = VAF

            else:  # no comparison
                
                stim_re_mat = []
                stim = []
                resp_mat = []
                noise_mat = []
                CF_mat = [] 
                VAF_mat = []
                tc = []        
        else:  # NO CSD
            
            noiseinput_f = fft(noiseinput_cut)  # transform into frequency domain
            response_f = fft(response_ds_cut)  # transform into frequency domain
            
            fdata = response_f / noiseinput_f  # Hz / nA
            fdata_pos = fdata[0:round(len(noiseinput_f) / 2)]  # Take only the first half of the power to avoid redundancy
        
            pwr = abs(fdata_pos)  # Transfer Power
            pha = angle(fdata_pos, deg = True)  # Transfer Phase in degree
            
            freq = fftfreq(noise_data_points, dt)[0:round(noise_data_points / 2)]  # positive frequency vector
            
            if freq_wp == None:  # colored/white noise: check if specific frequencies are used 
            
                fstart = find(freq >= 1. / (noise_data_points*dt))[0] # lowest frequency depends on window length
                fend = max(find(freq < 1000))  # only use frequencies smaller than 1kHz
                freq_used = freq[fstart:fend] # don't use freq = 0
                
                if i == 0:
                    # create matrix to hold all different interpolation methods:
                    ca_mat = zeros((len(method_interpol),len(freq_used)), dtype=complex)    
                    mag_mat = zeros((len(method_interpol),len(freq_used)))  # impedance magnitude vector
                    pha_mat = zeros((len(method_interpol),len(freq_used))) # imedance phase vector   
                    K_mat = zeros((len(method_interpol)), dtype=complex) 
                    tk = [] 
                    
                mag_mat[i,:] = pwr[fstart:fend] # don't use freq = 0 
                pha_mat[i,:] = pha[fstart:fend] # don't use freq = 0
                ca_mat[i,:] = 1 / (mag_mat[i,:] * exp(pi / 180 * 1j * pha_mat[i,:]))
                #print "- number of cnoise frequencies: " + str(len(freq_used)) + " Hz"
                
            else:   # multi sine stimulation
                
                freq_used = freq[freq_wp]
          
                if i == 0:
                    # create matrix to hold all different interpolation methods:
                    ca_mat = zeros((len(method_interpol),len(freq_used)), dtype=complex)    
                    mag_mat = zeros((len(method_interpol),len(freq_used)))  # impedance magnitude vector
                    pha_mat = zeros((len(method_interpol),len(freq_used))) # imedance phase vector   
                    K_mat = zeros((len(method_interpol)), dtype=complex) 
                    tk = [] 
                    
                mag_mat[i,:] = pwr[freq_wp]
                pha_mat[i,:] = pha[freq_wp]
                ca_mat[i,:] = 1 / (mag_mat[i,:] * exp(pi / 180 * 1j * pha_mat[i,:]))   
                
            stim = []            
            stim_re_mat = []
            resp_mat = []
            noise_mat = []
            CF_mat = []
            VAF_mat = []
            VAFf_mat = []
            SNR_mat = []
            tc = [] 
            P_ss = []
            Qual_mat = []
    
    SNR_mat = (freq_used, SNR_mat)
    VAFf_mat = (freq_used, VAFf_mat)   
     
    return {'mag_mat':mag_mat,'pha_mat':pha_mat,'ca_mat':ca_mat,'freq':freq,'freq_used':freq_used,
        'fmean':fmean,'tk':tk,'K_mat':K_mat,'SNR_mat':SNR_mat,'Qual_mat':Qual_mat,
        'stim':stim,'stim_re_mat':stim_re_mat,'resp_mat':resp_mat,
        'noise_mat':noise_mat,'CF_mat':CF_mat,'VAF_mat':VAF_mat, 'VAFf_mat':VAFf_mat, 'tc':tc, 'P_ss':P_ss }


def reconstruct_Stimulus(Ker, resp, stim, t):
                
    dt = t[2] - t[1]
    
#    if len(Ker)%2==0:
#        Ker_conv = Ker[0:-1-1]
#    else:
#        Ker_conv = Ker  

    Ker_conv = Ker
    n_kernel = len(Ker_conv) # NO SHIFT NECESSARY!!!    
    
    start_time = time.time()  
    stim_re_ = real(convolve(Ker_conv, resp))  # reconstructed signal
    process_time = time.time() - start_time
    
    print "Convolution took ", process_time

    #n_shift = floor(n_kernel/2)
    #n_shift = 1  # JUST SHIFT ONE TIMESTEP, IDKY???
    #stim_re = concatenate([stim_re_[n_shift:], zeros(n_shift)])  # acausal filter was used: shift to reconstrcuted signal
    #stim_re = concatenate([zeros(n_shift), stim_re_[0:(len(stim_re_)-n_shift)]])  # acausal filter was used: shift to reconstrcuted signal
    istart = int(len(Ker_conv)//2) #+ 0.01/dt

    stim_re = stim_re_            
    resp_cc = resp[int(n_kernel):int(len(resp))-int(n_kernel)]  #  cut at beginning and end to eliminate conv artefacts
    stim_cc = stim[int(n_kernel):int(len(resp))-int(n_kernel)]  # cut at beginning and end to eliminate conv artefacts
    stim_re_cc = stim_re[istart:istart+int(len(resp))][int(n_kernel):int(len(resp))-int(n_kernel)]
    
    #figure(99)
    #plot(stim_cc, 'b')
    #plot(stim_re_cc, 'g')
    #show()

    # old
    #stim_re_ = real(convolve(Ker, resp, 'same'))  # reconstructed signal
    #stim_re = stim_re_  
    #resp_cc = resp[2*n_kernel:len(resp)-2*n_kernel]  
    #stim_cc = stim[2*n_kernel:len(stim)-2*n_kernel]  
    #stim_re_cc = stim_re[2*n_kernel:len(stim_re)-2*n_kernel]
    
    #resp_cc = resp_cc[1/dt:len(resp_cc)-1/dt]  
    #stim_cc = stim_cc[1/dt:len(stim_cc)-1/dt]  
    #stim_re_cc = stim_re_cc[1/dt:len(stim_re_cc)-1/dt]
    
#    if n_kernel%2==0:
#        Ker_conv = np.r_[np.zeros(len(Ker)-1),Ker[0:-1-1]]
#    else:
#        Ker_conv = np.r_[np.zeros(len(Ker)),Ker[0:-1]]    
#    
#    window_len = len(Ker_conv) 
#
#    print "Kernel length:", window_len, n_kernel
#    
#    resp_conv = np.r_[resp[window_len-1:0:-1],resp,resp[-1:-window_len:-1]]
#    resp_conv = resp
#
#    start_time = time.time()      
#    stim_re_ = np.convolve(Ker_conv,resp_conv,mode='valid')
#    process_time = time.time() - start_time
#    
#    print "Convolution took ", process_time
#    
#    #stim_re =  stim_re_[window_len//2:len(stim_re_)-window_len//2]
#    stim_re =  stim_re_#[window_len:len(stim_re_)-window_len]
#    
#    resp_cc = resp[window_len//2:len(resp)-window_len//2]
#    stim_cc = stim[window_len//2:len(stim)-window_len//2]
#    stim_re_cc = stim_re
       
    noise_cc = stim_re_cc - stim_cc # noise
    
    #P_nn_, freq_ = csd(noise_cc, noise_cc, Fs = fs, NFFT = nfft, noverlap = nfft / 4, sides = 'twosided') 
    #P_ss_, freq_ = csd(stim_cc, stim_cc, Fs = fs, NFFT = nfft, noverlap = nfft / 4, sides = 'twosided') 
    #SNR_ = P_ss_/P_nn_

    err = mean(noise_cc**2)
    CF = 1 - (sqrt(err)/std(stim_cc))
    VAF = 1 - (err/var(stim_cc))
    
    tc = arange(len(stim_cc))*dt
    
    return tc, resp_cc, stim_cc, stim_re_cc, noise_cc, CF, VAF

    
def est_quality(t, f, R, amp, pha, fmean):
    
    dt = t[2]-t[1]
    
    n = len(R)
    FT = fft(R)[0:round(n / 2)]  # transform into frequency domain
    freq = fftfreq(n, dt)[0:round(n / 2)]  # positive frequency vector

#    figure(66)
#    subplot(2,1,1)
#    plot(freq[1:100], abs(FT[1:100])**2,'*') 
#    subplot(2,1,2)
#    plot(t, R)  
#    show()
    
    i_f = abs(freq-f).argmin()  # find corresponding indexes to frequencies
    i_3f = abs(freq-3*f).argmin() # find corresponding indexes to frequencies
   
    NI = log10(abs(FT[i_3f])**2 / abs(FT[i_f])**2)
    
    # Variance Accounted For 
    
    R_est = amp*sin(2*pi*f*t+pha)+fmean
    N = (R-R_est)
    
    VAF = 1 - (mean(N**2)/var(R))
    #figure(99)    
    #plot(t, R, t, R_est)
    #show()
                
    return {'NI':NI, 'VAF':VAF, 'R_est':R_est, 'N':N}
    
    
def if_extrap(x, xp, yp):
    """interp function with linear extrapolation"""
    
    #print type(x), type(xp), type(yp)
    #print shape(x), shape(xp), shape(yp)
    
    if np.sum((np.diff(xp) < 0)) > (len(xp)/2): # reverse interpolation
       xp = xp[::-1]
       yp = yp[::-1]
           
    y = interp(x, xp, yp, left = 0, right = 0)
    m = mean(diff(yp)) / mean(diff(xp))
    y = where(x > xp[-1], yp[-1]+(x-xp[-1])*m, y)
    
    return y
    
    
#def adjust_spines(ax,spines, color='k'):
#    for loc, spine in ax.spines.iteritems():
#        if loc in spines:
#            spine.set_position(('outward',10)) # outward by 10 points
#            #spine.set_smart_bounds(True)
#        else:
#            spine.set_color('none') # don't draw spine
#
#    # turn off ticks where there is no spine
#    if 'left' in spines:
#        ax.yaxis.set_ticks_position('left')
#    elif 'right' not in spines:
#        # no yaxis ticks
#        ax.yaxis.set_ticks([])
#        
#    if 'right' in spines:
#        ax.yaxis.set_ticks_position('right')
#        s = ax.transAxes.transform((1, 1))  # transform into display coordinates
#        e = ax.transAxes.transform((1, 0))  # transform into display coordinates
#        line = Line2D((s[0]+10,e[0]+10), (s[1],e[1]), color=color, linewidth=rcParams['axes.linewidth'], transform=None) # ax.transAxes
#        line.set_clip_on(False) # show line!
#        ax.add_line(line)
#        
#        #second_right = matplotlib.spines.Spine(ax, 'right', ax.spines['left']._path)
#        #second_right.set_position(('outward', 10))
#        #ax.spines['second_right'] =  second_right
#        #ax.spines['second_right'].set_color('k')
#        #ax.spines['right'].set_color('k')
#
#       
#    if 'bottom' in spines:
#        ax.xaxis.set_ticks_position('bottom')
#    else:
#        # no xaxis ticks
#        ax.xaxis.set_ticks([])
#        ax.axes.get_xaxis().set_visible(False)
        

def if_spike_train(frequency=100, duration=1000, start_time=0, dt=0.1, seed=None, noise=0, refrac=0, tau=0):
    """
    generate constant spike train with memoryless integrate and fire, add noise 
    """
    
    ni = noise/2. # adjusted so that noise = +-2*std 
    
    t_refrac = -1
    rangen = np.random.mtrand.RandomState()
    
    if seed is not None:
        rangen.seed(seed)   
       
       
    if np.size(frequency) > 1:
        duration = frequency[0][-1]
        start_time = frequency[0][0]
        time_vec = frequency[0] 
        freq_vec = frequency[1]
        dt = time_vec[2]-time_vec[1]
    else:
        time_vec = np.arange(start_time,duration,dt) 
        freq_vec = frequency * np.ones(len(time_vec))
         
    train = np.array([])
    
    u = np.zeros(len(time_vec))
    u[0] = rangen.uniform()  # start potential
    
    n = np.zeros(len(time_vec))
    n[0] = 0  
    
    thresh_v = np.zeros(len(time_vec))

    if u[0] == 1:       # neuron at threshold
        train = np.append(train, time_vec[0])  # record spike times
        u[0] = 0  # reset    
    
    #if tau < 0:
    #    thresh = 1 + noise * rangen.normal()
    #else:
    #    thresh = 1
    
    thresh = 1    
    for i, t in enumerate(time_vec):
            
        freq = freq_vec[i]

        if tau == 0:
            n[i] = freq/1000. + (ni/1000.) * rangen.normal()
        elif tau < 0:
            n[i] = freq/1000.
        else:
            n[i] = n[i-1] + (1. - exp(-dt/tau)) * (freq/1000. - n[i-1]) + sqrt(1.-exp(-2.*dt/tau)) * (ni/1000.) * rangen.normal() 
             
        u[i] = u[i-1] + dt*n[i]
                
        if t < t_refrac:
          
            if tau < 0:
                u[i] = noise * rangen.normal()
            else:
                u[i] = 0  
            
            #if tau < 0:
            #    thresh = 1 + noise * rangen.normal()
            #    #if thresh < 0: thresh = 1e-6
            
            
        elif (u[i] >= thresh):  # treshold crossing (u[i-1] < thresh) &
            #dts = (u[i]-1)/(u[i]-u[i-1])*dt  # interpolate spike time
            t_spike = t#+dts
            train = np.append(train, t_spike)  # spike times
            #u[i] = 0
            t_refrac = t + refrac
            
            if tau < 0:
                u[i] = noise * rangen.normal()
                #if u[i] > 1: u[i] = 1-1e-6
            else:
                u[i] = 0
            
            #if tau < 0:
            #    thresh = 1 + noise * rangen.normal()
            #    #if thresh < 0: thresh = 1e-12
                            
        #thresh_v[i] = thresh
       
        
    #print std(n)*1000
    
    #plt.figure(255)
    #plt.plot(time_vec, n*1000)
        
    #import matplotlib.pyplot as plt
    #freq_times, spike_freq, freq_mean, freq_interp = get_spikefreq(train, time_vec)
    #plt.figure(1)    
    #plt.plot(time_vec,freq_interp, time_vec, freq_vec)
    
    #plt.figure(2)
    #plt.plot(time_vec, u)
    #plt.show()
    
    #print 'freq_mean: ',freq_mean
    
    #n = thresh_v
    return train, u, n # *1000 # freq_vec 
    
    
def if_spike_train2(frequency=100, duration=1000, start_time=0, dt=0.1, seed=None, noise=0, refrac=0, tau=0):
    """
    generate constant spike train with memoryless integrate and fire, add noise 
    """
    
    ni = noise/2 # adjusted so that noise = +-2*std 
    
    t_refrac = -1
    rangen = np.random.mtrand.RandomState()
    
    if seed is not None:
        rangen.seed(seed)   
       
       
    if np.size(frequency) > 1:
        duration = frequency[0][-1]
        start_time = frequency[0][0]
        time_vec = frequency[0] 
        freq_vec = frequency[1]
        dt = time_vec[2]-time_vec[1]
    else:
        time_vec = np.arange(start_time,duration,dt) 
        freq_vec = frequency * np.ones(len(time_vec))
    
    #print time_vec
    #print freq_vec
     
    train = np.array([])
    
    u = np.zeros(len(time_vec))
    cur_u = rangen.uniform()  # start potential
    
    n = np.zeros(len(time_vec))
    cur_n = 0    

    if cur_u == 1:       # neuron at threshold
        train = np.append(train, time_vec[0])  # record spike times
        cur_u = 0  # reset
     
    u[0] = cur_u
    
    for i, cur_time in enumerate(time_vec):
            
        #freq = freq_vec[mlab.find(time_vec >= cur_time)[0]]/1e3
        freq = freq_vec[i]
        
        if tau == 0:
            next_n = freq/1000. + (ni/1000.) * rangen.normal()
        else:
            next_n = cur_n + (1. - exp(-dt/tau)) * (freq/1000. - cur_n) + sqrt(1.-exp(-2.*dt/tau)) * (ni/1000.) * rangen.normal() 
        
        stim = cur_n    
        #stim = freq/1000. + (ni/1000.) * rangen.normal()
        
        next_u = cur_u + dt*stim
                
        if cur_time < t_refrac:
            next_u = 0            
            
        elif (cur_u < 1) & (next_u >= 1):  # treshold crossing
            dts = (next_u-1)/(next_u-cur_u)*dt  # interpolate spike time
            train = np.append(train, cur_time+dts)  # spike times
            next_u = 0
            t_refrac = cur_time + refrac
        
        cur_u = next_u
        u[i] = next_u  
        
        #print next_n
        cur_n = next_n
        n[i] = next_n
        
    #print std(n)*1000
    
    #plt.figure(255)
    #plt.plot(time_vec, n*1000)
        
    #import matplotlib.pyplot as plt
    #freq_times, spike_freq, freq_mean, freq_interp = get_spikefreq(train, time_vec)
    #plt.figure(1)    
    #plt.plot(time_vec,freq_interp, time_vec, freq_vec)
    
    #plt.figure(2)
    #plt.plot(time_vec, u)
    #plt.show()
    
    #print 'freq_mean: ',freq_mean
        
    return train, u, n*1000 # freq_vec 
    

def spike_train(frequency=100, duration=1000, start_time=0, seed=None, noise=0, jitter=0):
    """
    generate constant spike train and add noise and/or jitter 
    """

    rangen = np.random.mtrand.RandomState()
    
    if seed is not None:
        rangen.seed(seed)

    if np.size(frequency) > 1:
        duration = frequency[0][-1]
        start_time = frequency[0][0]
        time_vec = frequency[0] 
        freq_vec = frequency[1]
        freq_vec_max = max(freq_vec)/1000.
    else:
        freq_vec_max = freq_vec/1000.
    
    train = np.array([])
    cur_time = duration-10*rangen.uniform() # start with end
        
    while cur_time >= start_time:

        train = np.append(train,np.array([cur_time]))
        
        if np.size(frequency) > 1: 
            freq = freq_vec[mlab.find(time_vec >= cur_time)[0]]/1000.
        else:
            freq = frequency/1000.                    
        
        isi = 1/freq  
       
        cur_time -= (1. - noise)*isi + noise*isi*rangen.exponential()
        
    train = train[::-1]  # reverse

    if jitter > 0:  # jitter in ms!!!
        x = rangen.normal(0, jitter, len(train))             
        train = train + x
        train = np.sort(train)
        train = np.delete(train, [i for i,x in enumerate(train) if x < start_time or x > duration])
    
    #print '- method: spike_train'    
    return train  


def mod_spike_train(modulation, noise = 0, seed=None, noise_tau = 0, noise_a=1e9): 
    """
    generate sinusoidal modulated spike train with and without noise 
    modulation = (time, rate) with time in seconds, rate in Hz!!
    """
    
    modulation = (modulation[0]/ms, modulation[1].clip(min=0)) # has to be in ms! No negative values!
    u = []
    
    if noise_a >= 1e9:

        #print "--- train generate from integrate and fire"
        train, u, n = if_spike_train(frequency=modulation, seed=seed, noise=noise, tau=noise_tau/ms, refrac=1)
        
    elif noise_a == 1:

        #print "--- train generate poisson"
        gen = stgen.StGen(rng = np.random.mtrand.RandomState(), seed=seed)
        train = gen.inh_poisson_generator(rate=modulation[1], t=modulation[0], t_stop=modulation[0][-1], array=True)
        
    else:

        #print "--- train generate gamma"
        a = np.ones_like(modulation[0])*noise_a
        b = 1/a/modulation[1]
        t = modulation[0]   
        
        #print shape(a),shape(b),shape(t)
        
        gen = stgen.StGen(rng = np.random.mtrand.RandomState(), seed=seed)
        train = gen.inh_gamma_generator(a, b, t=t, t_stop=t[-1], array=True)

        
    return train, u # output in ms


def oscill_spike_train(sor = 4, spike_prob = 0.25, noise_fraction = 4, end_time = 200.e3, seed = 1): 
    
    print sor, spike_prob, noise_fraction, end_time
    
    # Subthreshold oscill rate
    sor_isi = 1000/sor
    spt = np.arange(sor_isi,end_time,sor_isi)
    
    np.random.seed(seed)
    
    rnd = np.random.uniform(0.,1.,spt.size)
    
    spt = np.array([s for (s,r) in zip(spt,rnd) if r < spike_prob])

    # Add uniformely distributed noise to spt times
    # noise mean = 0 added on the peak of oscillation
    # noise amplitude = 1/4 of cycle
    spt = np.array([s+np.random.uniform(-1.,1.)*sor_isi/noise_fraction/2 for s in spt])
    
    return spt # output in ms
    
    
def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    
    From Django's "django/template/defaultfilters.py".
    """
    
    #import string
    #valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    
    #return ''.join(c for c in value if c in valid_chars)
    
    for c in r'[]/\;,><&*:%=+@!#^\'()$| ?^':
        value = value.replace(c,'')
        
    return value

    #import re
    #_slugify_strip_re = re.compile(r'[^\w\s-]')
    #_slugify_hyphenate_re = re.compile(r'[-\s]+')

    #import unicodedata
    #if not isinstance(value, unicode):
    #    value = unicode(value)
    #value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore')
    #value = unicode(_slugify_strip_re.sub('', value).strip().lower())
    #return _slugify_hyphenate_re.sub('_', value)


def gauss_func(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))
    
    
def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a
    
def nanmean(a, dim=0):
    return np.mean(np.ma.masked_array(a,np.isnan(a)),dim).filled(np.nan)  
    
def r_squared(actual, ideal):
    
    actual_mean = np.mean(actual)
    ideal_dev = np.sum([(val-actual_mean)**2])
    actual_dev = np.sum([(val-actual_mean)**2])
    return ideal_dev / actual_dev  

    #slope, intercept, r_value, p_value; std_err = scipy.stats.linregress(actual, ideal)
    #return r_value**2

def rw_hdf5(filepath = "data_dict.hdf5", data_dict = None, export = False):
    
    if data_dict == None: # load
        
        data_dict = {}
        
        print "rw_hdf5 load:", filepath
        f = h5py.File(filepath, 'r')

        for i in f.items():
            data_dict[i[0]]  = np.array(i[1])    
            
        f.close() 
            
        if export:
            if not os.path.exists(export):
                os.makedirs(export)
            shutil.copy(filepath, export)

    else:
        
        f = h5py.File(filepath, 'w')
        
        for name  in data_dict:

            data0 = data_dict[name]
            
            if (type(data0) == np.ndarray):
                pass
            else:
                data0 = np.array(data0)
            
            #print data0
            if len(np.shape(data0)) < 1:
                f.create_dataset(name, data=data0)
            else:
                f.create_dataset(name, data=data0, compression='lzf')
                

        f.close() 
    
    return data_dict
    

# test code
if __name__ == '__main__': 
    
    test = array(["get_magphase", "fit_sinusoid", "create_multisines", "create_colnoise"])
    test = array(["fit_exp"])
    test = array(["fit_dualexp"])
    test = array(["fit_tripleexp"])
    #test = array(["shannon_interp"])
    test = array(["construct_Pulsestim"])
    test = array(["construct_Test"])
    #test = array(["fit_aiftransfer"])
    test = array(["create_colnoise"])
    #test = array(["test_amp"])
    #test = array(["create_ZAP"])
    #test = array(["est_quality"])
    #test = array(["create_multisines"])
    test = array(["syn_kernel"])
    #test = array(["aiftransfer"])
    #test = array(["create_singlesine"])
    #test =  array(["if_spike_train_compare_dttest"])
    #test =  array(["if_spike_train_compare"])
    #test = array(["get_spikefreq"])
    #test = array(["test_syn"])
    #test = array(["hdf5"])
    #test = array(["fit_expnew"])
    
    if "test_syn" in test:
        
        duration=1000
        start_time=0
        dt=0.1
       
        time_vec = np.arange(start_time,duration,dt) 
        
        g = np.zeros(len(time_vec))
         
        g[0] = 1
        cur_g = 1
        
        tau = 2.86
        for i, cur_time in enumerate(time_vec):
            
            s = 0
            if (cur_time >= 500) and (cur_time <= 600):
                s = 10
                
            next_g = cur_g + dt*(s*(1-cur_g)-cur_g*tau)
            cur_g = next_g
            g[i] = next_g 
             
        
        figure(1)
        plot(time_vec, g)
        show()
    

    if "get_spikefreq" in test:
        
        spike_times = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2,2.2,2.4,2.6,2.8,3,3.2])
        freq_times, spike_freq, freq_mean, freq_onset = get_spikefreq(spike_times, stimlength = 3, compute_mean = 2, change_factor = 0.3)
        
        print freq_times, spike_freq, freq_mean, freq_onset
      
    if "get_magphase" in test:
        
        tstart = 0; dt = 0.025*ms; tstop = 7.5 # s 
        fs = 1/dt # sampling rate
        fmax = fs/2 # maximum frequency (nyquist)
        t = arange(tstart,tstop,dt) # create stimulus time vector
        fu = 0.2 # [Hz]
        
        input_signal = sin(2 * pi * t * fu)
        
        t_out = sort(random.uniform(tstart, tstop, size(t)/10000))  # different timebase! only fit possible!  
        output_signal = (10 * sin(2 * pi * t_out * fu + pi / 2)) - 70 
        
        amp_mean, mag_mean, phase_mean, umean = get_magphase(input_signal, t, output_signal, t_out, method = "fit", f = fu)   
        
        print "mean amplitude output signal " + str(amp_mean)
        print "mean magnitude " + str(mag_mean) + " ,mean phase = " + str(phase_mean)
        
        plot(t, input_signal ,'--', t_out, output_signal, '*-')
    
        show()
        
        
    if "fit_sinusoid" in test:
    
        a0 = 3; b0 = 2; c0 = 0; f = 3
        t0 = linspace(0, 1, 1000)
        noise = rand(1000).T * 0 
        u0 = a0 + b0 * sin(2 * pi * f * t0 + c0) + noise
        t = linspace(0, 1, 1000)
        
        a, b, c, u = fit_sinusoid_fft(f, t0, u0, t)
        print a
        print b
        print c
    
        figure(1)
        plot(t0, u0, '.', t, u)
        show()
        
    
    if "fit_exp" in test:

        t0 = arange(0,1,0.001)
        decay = -10
        u0 = decay * (1 - exp(-t0/20e-3)) 
        un0 = u0 + 0.2*np.random.normal(size=len(t0))

        udecay, utau, u = fit_exp(t0, un0, p0 = array([-1, 1e-3]))

        print udecay, utau
        figure(1)
        plot(t0, un0, '.', t0, u0, '--', t0, u)
        show()
        
        
        
    if "fit_dualexp" in test:

        dt = 0.025*ms
        t0 = arange(0,1,dt)
        
        tau1 = 1*ms
        tau2 = 100*ms
        maxim = 1
        delay = 100*ms
         
        tau_rise=(tau1*tau2)/(tau1-tau2)
        tau_fall=tau1
        B = 1/( ((tau2/tau1)**(tau_rise/tau1))-((tau2/tau1)**(tau_rise/tau2)) )
        
        print B
        
        u0 = ( maxim*( -exp(-(t0-delay)/tau1)+exp(-(t0-delay)/tau2) ) )
        u0[0:delay/dt]=0;
        #F = F + ydata(floor(delay/dt))
    
        un0 = u0 + 0.1*np.random.normal(size=len(t0))

        umax, utau1, utau2, u = fit_dualexp(t0, un0, p0 = array([1, 1*ms, 1*ms]),delay=delay)

        print umax, utau1, utau2
        
        figure(1)
        plot(t0, un0, '.', t0, u0, '--', t0, u)
        show()
        
        
    if "fit_tripleexp" in test:

        dt = 0.025*ms
        t0 = arange(0,0.2,dt)
        
        tau1 = 1*ms
        tau2 = 10*ms
        tau3 = 1000*ms
        maxim = 1
        delay = 100*ms
        
        u0 = ( maxim*( -exp(-(t0-delay)/tau1)+exp(-(t0-delay)/tau2)+exp(-(t0-delay)/tau3) ) )
        u0[0:delay/dt]=0;
        #F = F + ydata(floor(delay/dt))
    
        un0 = u0 + 0.1*np.random.normal(size=len(t0))

        umax, utau1, utau2, utau3, u = fit_tripleexp(t0, un0, p0 = array([1, 1*ms, 1*ms, 1*ms]),delay=delay)

        print umax, utau1, utau2, utau3
        
        figure(1)
        plot(t0, un0, '.', t0, u0, '--', t0, u)
        show()
        
        
    if "fit_expnew" in test:

        tau = 500e-3
        cutf = 20
        sexp = -1 
        dt = 0.001
        t_noise = arange(0, 100, dt)
        noise_data = create_colnoise(t_noise, sexp, cutf)
        stimulus, t, t_startstop = construct_Stimulus(noise_data, 1/dt, amp=1, ihold = 0, tail_points = 0, delay_baseline = 10) 
                    
        t_kernel = arange(0, tau*10, dt)
        kernel = syn_kernel(t_kernel, 0, tau)
        kernel = np.concatenate((zeros(len(kernel)-1),kernel))

        filtered = np.convolve(stimulus, kernel, mode='same')
        filtered1 = filtered[int(t_startstop[0]/dt):int(t_startstop[1]/dt)] 
        stimulus1 = stimulus[int(t_startstop[0]/dt):int(t_startstop[1]/dt)] 
        t1 = t[int(t_startstop[0]/dt):int(t_startstop[1]/dt)] 
        
        w_length = 2
        fs = 1 / dt
        nfft = 2 ** int(ceil(log2(w_length * fs)))  
        w_true_length = nfft * dt   # true length window
        noverlap = nfft / 4

        r = stimulus1
        P_rr, freq = psd(r, Fs = 1/dt, NFFT = nfft, noverlap = noverlap, sides = 'twosided')  # , window =  mlab.window_none
        P_rr = ifftshift(P_rr)
        fdata_pos = (P_rr)[1:len(freq)/2] # only positive frequencies
        pwr_s = abs(fdata_pos) # Transfer Power
        pha_s = angle(fdata_pos, deg = True) # Transfer Phase in degree
            
        r = filtered1
        P_rr, freq = psd(r, Fs = 1/dt, NFFT = nfft, noverlap = noverlap, sides = 'twosided')  # , window =  mlab.window_none
        P_rr = ifftshift(P_rr)
        fdata_pos = (P_rr)[1:len(freq)/2] # only positive frequencies
        pwr_f = abs(fdata_pos) # Transfer Power
        pha_f = angle(fdata_pos, deg = True) # Transfer Phase in degree
        
        freq = ifftshift(freq)
        freq = freq[1:len(freq)/2]

        figure(1)
        subplot(2,1,1)
        plot(t1, stimulus1/max(stimulus1), 'k-', t1, filtered1/max(filtered1), 'r-')
        subplot(2,1,2)
        semilogx(freq, pwr_s/pwr_s[0], 'k-', freq, pwr_f/pwr_f[0], 'r-')
        show()

        
    if "shannon_interp" in test:

        a0 = 3; b0 = 2; c0 = 1.234; freq = 10;
        t0 = array([0.00459605074693903,0.00726375896667564,0.00784973441063397,0.0127224903712386,0.0171270655461151,0.0393663919020232,0.0609900368303785,0.0635371421307821,0.0654594032207050,0.0999845349934256,0.119188624525719,0.131158588842870,0.134423875905300,0.138989575327831,0.139516447247645,0.182021984615913,0.189528321172140,0.201667933001159,0.202604832168432,0.218847864233656,0.219479711282702,0.226652337702447,0.228111424603092,0.239969946662225,0.246105341192989,0.250848554299275,0.254989299897020,0.275066518258535,0.296418212747334,0.301828987940621,0.332499026325882,0.348105182140276,0.351266689715571,0.354722452499086,0.359590326763347,0.374299226239647,0.383307652959454,0.383364191948955,0.389899994370318,0.404459485959467,0.421426947556085,0.442676684844202,0.449368223144463,0.456522496631530,0.469306274142778,0.490669133318145,0.492682694168636,0.504719391132661,0.507205621367988,0.518527193799826,0.521561514445141,0.534135096228315,0.568392189742432,0.568484472347038,0.573023005985662,0.577029656618231,0.582603276061774,0.584055978860854,0.587796282178598,0.588501277932966,0.608464044116267,0.625893464986574,0.626008271991061,0.627958771815969,0.630538464601416,0.648639994155523,0.651808804576273,0.660791197220462,0.680866786193684,0.690596392893720,0.699105430999923,0.704644674745085,0.707497758866246,0.724810615824970,0.726322828528511,0.752762163986078,0.758617951982347,0.774559534114423,0.780486968166248,0.791867080004023,0.799655776201573,0.802171594064305,0.848515015994186,0.856540805722978,0.864847431324938,0.876360704800671,0.877835683926097,0.883152342556065,0.884554440010395,0.885558670718519,0.929895247208519,0.941655353501460,0.945441426048917,0.946800026376379,0.948273065241326,0.957117920800973,0.978062788376808,0.989045885911051,0.994412760780681,0.995238107661316])
        t = linspace(0, 1, 1000)
        
        u0 = a0 + b0 * sin(2 * pi * freq * t0 + c0)
        
        u = shannon_interp(t0, u0, t, chunk_size=10)
        
        print len(u)
        
        plot(t0, u0, "*b", t, u, "-r") 
        axis(ymin=0, ymax=6)
        
        show()    
        

    if "create_multisines" in test:
        
        fstart = 1; fstop = 100; fsteps = 1 # Hz
        freq_used = arange(fstart, fstop, fsteps) # frequencies we want to examine
                    
        tstart = 0; t_stim = 10; dt = 0.025*ms # s 
        fs = 1/dt # sampling rate
        
        t_noise = arange(tstart, t_stim, dt) # create stimulus time vector, make sure stimulus is even!!!
            
        noise_data, f1, freq_wp, f_used_check = create_multisines(t_noise, freq_used)  # create multi sine signal
    
        pxx, freq = psd(noise_data, Fs = 1 / dt) # compute power spectrum
        
        data_points = len(t_noise)
        noise_power = abs(fft(noise_data))[ 0:round(data_points / 2) ] # compute noise power
        
        figure(1); clf()
        subplot(3, 1, 1)
        plot(t_noise, noise_data)
        
        subplot(3, 1, 2)
        loglog(freq, pxx / pxx[1])
        
        subplot(3, 1, 3)
        semilogx(f1, noise_power)
        
        show()
        
        
    if "create_colnoise" in test:
        
        dt = 0.025e-3
        fs = 1/dt
        t = arange(0, 100, dt)
        
#        sexp = 0   # reasonably smooth
#        cutf= 10 # 0.01kHz = 10Hz frequency cutoff
#        #cutf= None
#        x = create_colnoise(t, sexp, cutf)
#        
#        cutf= 10
#        dt = t[2] - t[1]
#                
#        # display power spectrum
#        figure(2)
#        pxx, f = psd(x, Fs=1/dt)
#        figure(1); #clf()
#        subplot(2, 1, 1)
#        plot(t, x)
#        subplot(2, 1, 2)
#        semilogx(f, pxx / pxx[1],
#               [f[1], cutf],  [1, 1], 'r:',                     # white
#               [cutf, f[-1]], [1,  (cutf / f[-1]) ** sexp], 'r:')   # coloured
               
        w_length = 1
        nfft = 2 ** int(ceil(log2(w_length * fs))) 
        
        
        sexp = 0 #-1   
        cutf= 10 # 0.01kHz = 10Hz frequency cutoff
        #cutf= None
        
        sexp = 4 #-1   
        cutf= 20 # 0.01kHz = 10Hz frequency cutoff
        
        x = create_colnoise(t, sexp, cutf)

        figure(3)
        # the histogram of the data
        n, bins, patches = hist(x, 100, normed=True, facecolor='green', alpha=0.75)

        # add a 'best fit' line
        y = normpdf(bins, 0, 1)
        plot(bins, y, 'r--', linewidth=1)
    
        
       
        # display power spectrum
        figure(2)
        P, f = psd(x, Fs = fs, NFFT = nfft, noverlap = nfft / 4, sides = 'twosided')
        
        P = ifftshift(P)[0:len(f)/2]
        f =  ifftshift(f)[0:len(f)/2]  # only positive frequencies
        
        figure(1)
        subplot(2, 1, 1)
        print shape(t), shape(x)
        plot(t, x)
        
        subplot(2, 1, 2)
        semilogx(f,P)
        
        sexp = -1   
        cutf= 20 # 0.01kHz = 10Hz frequency cutoff
        #cutf= None
        onf = 10
        x = create_colnoise(t, sexp, cutf, onf=onf)
    
        figure(3)
        n, bins, patches = hist(x, 100, normed=True, facecolor='blue', alpha=0.75) 
    
        # display power spectrum
        figure(2)
        P, f = psd(x, Fs = fs, NFFT = nfft, noverlap = nfft / 4, sides = 'twosided')
        P = ifftshift(P)[0:len(f)/2]
        f =  ifftshift(f)[0:len(f)/2]  # only positive frequencies
        
        figure(1)
        subplot(2, 1, 1)
        plot(t, x)
        
        subplot(2, 1, 2)
        semilogx(f,P)
               
        show()
        
        
        #from scipy import signal

#FS = 1000.0                                          # sampling rate
#FC = 0.05/(0.5*FS)                                   # cutoff frequency at 0.05 Hz
#N = 1001                                             # number of filter taps
#a = 1                                                # filter denominator
#b = signal.firwin(N, cutoff=FC, window='hamming')    # filter numerator
#
#M = FS*60                                            # number of samples (60 seconds)
#n = arange(M)                                        # time index
#x1 = cos(2*pi*n*0.025/FS)                            # signal at 0.025 Hz
#x = x1 + 2*rand(M)                                   # signal + noise
#y = signal.lfilter(b, a, x)                          # filtered output
#
#plot(n/FS, x); plot(n/FS, y, 'r')                    # output in red
#grid()


    if "test_amp" in test:
        
        dt = 0.025e-3
        fs = 1/dt
        t1 = arange(0, 5, dt)
        
        sexp = -1 
        cutf= 100 # 0.01kHz = 10Hz frequency cutoff
        
        #sexp = 0 
        #cutf= 0 # 0.01kHz = 10Hz frequency cutoff

        x = create_colnoise(t1, sexp, cutf)
        print std(x)
        print var(x)
        print std(x)**2
        
        a = 0.1
        x = x*a

        t, stimulus, i_startstop, t_startstop = create_singlesine(fu = 5, amp = a, ihold = 0, dt = dt, periods = 1, minlength = 1, t_prestim = 1)

        figure(1)
        plot(t1, x, t, stimulus)
        
        show()
        
 
    if "create_singlesine" in test:  
     
        t, stimulus, i_startstop, t_startstop = create_singlesine(fu = 100, amp = 0.5, ihold = 1, dt = 0.1*ms, periods = 1, minlength = 1, t_prestim = 1)
        plot(t, stimulus)
        
        show()
            

    if "aiftransfer" in test:
        
        freq = arange(0,1000,1)

        H, H0 = aiftransfer(freq = freq, tau = 10*ms, f0 = 100, i0 = 1*nA, rm = 10*MOhm, Vreset = -70, Vth = -60, Vrest = -70)
        
        H1, H0 = aiftransfer(freq = freq, tau = 10*ms, f0 = 100, i0 = 1*nA, rm = 10*MOhm, Vreset = -70, Vth = -60, Vrest = -70, delta_t = 10e-3)
        
        magfA = abs(H)
        phafA = unwrap(angle(H)) * (180 / pi)
        
        magfA1 = abs(H1)
        phafA1 = unwrap(angle(H1)) * (180 / pi)
        
        subplot(2,1,1)  
        semilogx(freq, magfA, '--', label = "analytical model")
        semilogx(freq, magfA1, 'r--', label = "analytical model")
        subplot(2,1,2)  
        semilogx(freq, phafA, '--', label = "analytical model")
        semilogx(freq, phafA1, 'r--', label = "analytical model")
        
        figure(2)
        H = H/H0
        mag = abs(H)
        loglog(freq, mag)
        
        show()
        
        
    if "syn_kernel" in test:
    
        t = arange(-2,2,1e-3)
        G = syn_kernel(t, 0.2, 0.2)  
        G[0:2/1e-3] = 0
        plot(t,G)
        show()
    
    if "fit_aiftransfer" in test:
        
        freq = arange(0,1000,1)

        tau0 = 10*ms
        f0 = 80*Hz
        i0 = 0.010*nA
        
        H0 = aiftransfer(freq = freq, tau = tau0, f0 = f0, i0 = i0, rm = 10*MOhm, Vreset = -70, Vth = -60, Vrest = -70)
        
        tau, scale, H = fit_aiftransfer(freq, H0, f0, i0)

        print tau0
        print tau 
        print scale
                
        magfA0 = abs(H0)
        phafA0 = unwrap(angle(H0)) * (180 / pi)
        
        magfA = abs(H)
        phafA = unwrap(angle(H)) * (180 / pi)
        
        subplot(2,1,1)  
        semilogx(freq, magfA0, freq, magfA, '--', label = "analytical model")
        subplot(2,1,2)  
        semilogx(freq, phafA0, freq, phafA, '--', label = "analytical model") 
        
        show()
 
     
    if "construct_Pulsestim" in test:
        
        #t, ivec = construct_Pulsestim(pulses = 2, dt = 2.5e-05, latency = 0.0091, stim_start = 0.01, stim_end = 0.01, len_pulse = 0.0002, amp_init = 0.64, amp_next = 0.24)
        
        t, ivec = construct_Pulsestim(dt = 1e-3, latency = [1,2,3,4], stim_start = 1.5, stim_end = 0.5, len_pulse = 0.1, amp_next = [1,-1,0.5,0.3])
        
        print len(t)
        print len (ivec)
        
        plot(t, ivec)
        show()
        
#        t = arange(0,1,1e-3)
#        x = zeros(len(t))
#        x[0:100] = 1
#        x[900:990] = -1    
#        plot(t,x)
#        
#        X = fft(x)
#        plot(fftshift(X))
#        
#        x2 = ifft(X)
#        plot(x2)
#        
#        show()

    if "construct_Test" in test:
        
        import scipy
        from scipy import io
        
        dt = 1*ms
        
        pulses = 50
        istep = np.random.rand(pulses)-0.5
        print istep
        tstep = np.arange(0,pulses*0.5,0.5) 
        tstep = tstep + np.random.rand(pulses)*(0.3)   
            
        t, ivec = construct_Pulsestim(dt = dt, latency = np.append(diff(tstep), 0), stim_start = 0*s, stim_end = 1*s, len_pulse = 0.01*s, amp_next = istep)
        ivec = ivec + 0.5
        
        istep = -1*istep
        t, ivec2 = construct_Pulsestim(dt = dt, latency = np.append(diff(tstep), 0), stim_start = 0*s, stim_end = 1*s, len_pulse = 0.01*s, amp_next = istep)
        ivec2 = ivec2 + 0.5
        
        #cutf = 20
        #sexp = -1 
        
        #t = arange(0, 12, dt)
        #seed = 3
        #ivec = create_colnoise(t, sexp, cutf, seed)*0.2 + 0.5      
        
        
        tau = 0.06*s
        t_kernel = arange(0, 0.5*s, dt)
        kernel = syn_kernel(t_kernel, tau, tau)
        kernel = np.concatenate((zeros(len(kernel)-1),kernel))
                        
        ovec = np.convolve(ivec, kernel, mode='same')
        
        
        ivec = ivec[int(1/dt):int(21/dt)]
        ivec2 = ivec2[int(1/dt):int(21/dt)]
        ovec = ovec[int(1/dt):int(21/dt)]
        t = t[int(1/dt):int(21/dt)]
        
        ovec = ovec/max(ovec)
        
        print len(ovec), len(ivec)
        
        print "Saving .mat"
        data = {}
        data['inputSequence'] = np.array([ones(len(ivec)), ivec]) #
        data['outputSequence'] = ovec
        scipy.io.savemat('./input.mat',data)                
                        
        plot(t, ivec, 'k', t, ovec, 'r',)
        show()
                
            
        
        
        


    if "create_ZAP" in test:
        
        ihold = 1
        t, zap, f = create_ZAP(ihold=ihold, ex=2)
    
        # Find all indices right before a rising-edge zero crossing
        zap_ = zap - ihold
        indices = find((zap_[1:] >= 0) & (zap_[:-1] < 0))
    
        # More accurate, using linear interpolation to find intersample
        # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
        crossings = [i - zap_[i] / (zap_[i+1] - zap_[i]) for i in indices]
        f_est = 1/diff(t[crossings])
        t_est = t[crossings[:-1]]
       
        plot(t, zap)
        twinx()
        plot(t, f, 'b', t_est, f_est, 'r')
        show()
     
     
    if "if_spike_train" in test: 
         
        train, u = if_spike_train(frequency=100, duration=10/ms, start_time=0, dt=0.025, seed=30, noise=10, tau=0.1)
        rate = 1/diff(train*ms)
        
        figure(1)
        plot(train[:-1], rate, 'b')
        
        figure(2)
        plot(train, ones(len(train)), 'b|')
        
        #train, u = if_spike_train(frequency=100, duration=10/ms, start_time=0, dt=0.0025, seed=30, noise=10, tau=100)
        #rate = 1/diff(train*ms)
        
        #figure(1)
        #plot(train[:-1], rate, 'r')
        
        #figure(2)
        #plot(train, ones(len(train)), 'r|')
        
        
        show()
        
        
    if "if_spike_train_compare_dttest" in test:
       
        
        import numpy
        from NeuroTools import stgen, signals
        from pylab import zeros_like, plot
        import sys
        import matplotlib.pyplot as plt
        import time 
        
        tsim = 100000
        fmean = 40
        noise = 0.01*40
        tau = 5
        
        #noise = 0.36*40
        #tau = 5
        
        #noise = 1*40
        #tau = 5
        
        #noise = 40
        #tau = 0
        
        noise = 0.1
        tau = -1
        
        
        dt = 0.025 #10.0
        t = numpy.arange(0,tsim,dt)

        rate = numpy.ones_like(t)*fmean
        
        modulation = (t, rate) # has to be in ms!
    
        t0 = time.time()
        train, u, n = if_spike_train(frequency=modulation, seed=1, noise=noise, tau=tau)
        tsolve = time.time() - t0
        
        values, xaxis = numpy.histogram(train, bins=50)
        fano = numpy.var(values)/numpy.mean(values)
        
        print numpy.mean(1/diff(train)), numpy.std(diff(train))/numpy.mean(diff(train)), tsolve, fano
        
        plt.figure(1)
        plt.plot(t, n, 'b')

        plt.figure(2)
        plt.plot(train[1:], 1e3/diff(train),"*-b")
        
        figure(4)
        values, xaxis = numpy.histogram(diff(train), bins=50)
        values = values/float(values.sum())
        xaxis = xaxis[:-1]
        plt.plot(xaxis,values, color="b")
            
        
        dt = 0.1 #10.0
        
        t = numpy.arange(0,tsim,dt)

        rate = numpy.ones_like(t)*fmean
        
        modulation = (t, rate) # has to be in ms!
    
        t0 = time.time()
        train, u, n = if_spike_train(frequency=modulation, seed=1, noise=noise, tau=tau)
        tsolve = time.time() - t0
        
        values, xaxis = numpy.histogram(train, bins=50)
        fano = numpy.var(values)/numpy.mean(values)
        
        print numpy.mean(1/diff(train)), numpy.std(diff(train))/numpy.mean(diff(train)), tsolve, fano
        
        plt.figure(1)
        plt.plot(t, n, 'r')
        
        plt.figure(2)
        plt.plot(train[1:], 1e3/diff(train),"*-r")
        
        figure(4)
        values, xaxis = numpy.histogram(diff(train), bins=50)
        values = values/float(values.sum())
        xaxis = xaxis[:-1]
        plt.plot(xaxis,values, color="r")
        
        
        fmean = 60
        
        dt = 0.01 #10.0
        
        t = numpy.arange(0,tsim,dt)

        rate = numpy.ones_like(t)*fmean
        
        modulation = (t, rate) # has to be in ms!
    
        t0 = time.time()
        train, u, n = if_spike_train(frequency=modulation, seed=1, noise=noise, tau=tau)
        tsolve = time.time() - t0
        
        values, xaxis = numpy.histogram(train, bins=50)
        fano = numpy.var(values)/numpy.mean(values)
        
        print numpy.mean(1/diff(train)), numpy.std(diff(train))/numpy.mean(diff(train)), tsolve, fano
        
        plt.figure(1)
        plt.plot(t, n, 'm')
        
        plt.figure(2)
        plt.plot(train[1:], 1e3/diff(train),"*-m")
        
        figure(4)
        values, xaxis = numpy.histogram(diff(train), bins=50)
        values = values/float(values.sum())
        xaxis = xaxis[:-1]
        plt.plot(xaxis,values, color="m")
        
        
        # NEURON FASTER:
#        import numpy as np
#        from neuron import h
#        from cells.IfCell import *
#        
#        dt = 0.1
#        
#        t0 = time.time()
#        
#        cell = IfCell(C = 1, R = 1e12, e = 0, thresh = 1, vrefrac = 0)
#        
#        gid = 0
#        noiseRandObj = h.Random()  # provides NOISE with random stream                       
#        fluct = h.Ifluct2(cell.soma(0.5))
#        fluct.m = 0
#        fluct.s = noise/2
#        fluct.tau = tau
#        fluct.noiseFromRandom(noiseRandObj)  # connect random generator!
#        noiseRandObj.MCellRan4(1, gid+1)  # set lowindex to gid+1, set highindex to > 0 
#        noiseRandObj.normal(0,1)
#        
#        stim = h.IClamp(cell.soma(0.5)) 
#        stim.amp = fmean
#        stim.delay = 0
#        stim.dur = tsim
#        
#        tstop = tsim
#        
#        rec_t = h.Vector()
#        rec_t.record(h._ref_t)
#        
#        rec_v = h.Vector()
#        rec_v.record(cell.soma(0.5)._ref_v) 
#        
#        rec_s = h.Vector()
#        nc = cell.connect_target(None)  # threshold is set in neuron definition, or here!
#        nc.record(rec_s)  # record indexes of the positive zero crossings
#
#        
#        h.load_file("stdrun.hoc")
#        h.celsius = 0              
#        h.init()
#        h.tstop = tstop
#        h.dt = dt
#        h.steps_per_ms = 1 / dt
#        
#        h.v_init = 0
#                
#        h.run()  
#        
#        t1 = np.array(rec_t)
#        voltage = np.array(rec_v)
#        train = np.array(rec_s)
#        
#        tsolve = time.time() - t0
#        
#        values, xaxis = numpy.histogram(train, bins=50)
#        fano = numpy.var(values)/numpy.mean(values)
#        
#        print numpy.mean(1/diff(train)), numpy.std(diff(train))/numpy.mean(diff(train)), tsolve, fano
#        
#        #plt.figure(1)
#        #plt.plot(t, n, 'g')
#        
#        plt.figure(2)
#        plt.plot(train[1:], 1e3/diff(train),"*-g")
#        
#        plt.figure(3)
#        plt.plot(t1, voltage,"g")
#        
#        figure(4)
#        values, xaxis = numpy.histogram(diff(train), bins=50)
#        values = values/float(values.sum())
#        xaxis = xaxis[:-1]
#        plt.plot(xaxis,values, color="g")
#        
        plt.show()

        
    if "if_spike_train_compare" in test:
        
        # Generate the PSTH for an inhomogeneous gamma renewal process
        # with a step change in the rate (b changes, a stays fixed)
        
        # This script generates Figure 5 in:

        #       Eilif Muller, Lars Buesing, Johannes Schemmel, and Karlheinz Meier 
        #       Spike-Frequency Adapting Neural Ensembles: Beyond Mean Adaptation and Renewal Theories
        #       Neural Comput. 2007 19: 2958-3010.
        
        # i.e. the PSTH for a 2D adapting markov process undergoing
        # a step change in statistics due to a step stimulus.
        

        import numpy
        from NeuroTools import stgen, signals
        from pylab import zeros_like, plot
        import sys
        import matplotlib.pyplot as plt
        import time
        
        tsim = 10000.0      
        dt = 0.025 #10.0
        t = numpy.arange(0,tsim,dt)
        bins = numpy.arange(0,tsim,10)
        trials = 1 #5000
        fmean = 40 
        
        # stepup
        i_start = t.searchsorted(4000.0,'right')-1
        i_end = t.searchsorted(6000.0,'right')-1

        rate = numpy.ones_like(t)*fmean #11.346 #*20.0
        #rate[i_start:i_end] = 2*fmean #92.25 #40.0

        psth = zeros_like(bins)
        stg = stgen.StGen()        

        # POISSON
        a = numpy.ones_like(t)*1 
        b = 1/a/rate   
        
        print "Running %d trials of %.2f milliseconds" % (trials, tsim)
        for i in xrange(trials):
            if i%100==0:
                print "%d" % i,
                sys.stdout.flush()
            stg.seed(i)
            
            t0 = time.time()
            #st = stg.inh_gamma_generator(a,b,t,tsim,array=True)
            st = stg.inh_poisson_generator(rate, t, tsim, array=True)
            
            tsolve = time.time() - t0
            
            s1 = signals.SpikeTrain(st)
            
            isi = s1.isi()
            fano_factor_isi = numpy.var(isi)/numpy.mean(isi)
            cv_isi = numpy.std(isi)/numpy.mean(isi)
            
            figure(3)
            values, xaxis = numpy.histogram(s1.isi(), bins=50)
            values = values/float(values.sum())
            xaxis = xaxis[:-1]
            plt.plot(xaxis,values, color="k")
            
            values, xaxis = numpy.histogram(s1.spike_times, bins=50)
            fano = numpy.var(values)/numpy.mean(values)
            
            print "POISSON, CV: ", cv_isi, "MEAN: ", s1.mean_rate(), "FANO:", fano, "tsolve:", tsolve, "s"
            

            psth[1:]+=numpy.histogram(st,bins)[0]
            figure(2)
            plt.plot(st, 0*numpy.ones_like(st),"|k")
        
        print "\n"
        
        # normalize
        psth = psth.astype(float)
        psth/= dt*float(trials)/1000.0
        # this is for correct 'steps' plotting only
        psth[0] = psth[1]
        figure(1)
        plt.plot(bins,psth,linestyle='steps', color="k")        
        
        
        psth = zeros_like(bins)
        stg = stgen.StGen()        

        # LOW NOISE IF
        noise = 0.01
        noise_tau = -1
        
        print "Running %d trials of %.2f milliseconds" % (trials, tsim)
        for i in xrange(trials):
            if i%100==0:
                print "%d" % i,
                sys.stdout.flush()
            #stg.seed(i)
            #st = stg.inh_gamma_generator(a,b,t,1000.0,array=True)
            
            t0 = time.time()
            st, u = mod_spike_train(modulation=[t*ms, rate], noise = noise, seed=i, noise_tau = noise_tau)
            tsolve = time.time() - t0
            
            s1 = signals.SpikeTrain(st)
            
            figure(3)
            values, xaxis = numpy.histogram(s1.isi(), bins=50)
            values = values/float(values.sum())
            xaxis = xaxis[:-1]
            plt.plot(xaxis,values, color="b")
            
            values, xaxis = numpy.histogram(s1.spike_times, bins=50)
            fano = numpy.var(values)/numpy.mean(values)
            
            print "Low Noise IF, NO TAU, CV: ", s1.cv_isi(), "MEAN: ", s1.mean_rate(), "FANO:", fano, "tsolve:", tsolve, "s"
                        
            psth[1:]+=numpy.histogram(st,bins)[0]
            figure(2)
            plt.plot(st, 1*numpy.ones_like(st),"|b")
            
            figure(4)
            plt.plot(u, "b")
            #plt.plot(n, "b--")
            
        
        print "\n"
        
        # normalize
        psth = psth.astype(float)
        psth/= dt*float(trials)/1000.0
        # this is for correct 'steps' plotting only
        psth[0] = psth[1]
        figure(1)
        plt.plot(bins,psth,linestyle='steps', color="b")
        
        
        
        psth = zeros_like(bins)
        stg = stgen.StGen()  
        
        # HIGH NOISE IF, NO TAU
        noise = 10
        noise_tau = -1
        
        print "Running %d trials of %.2f milliseconds" % (trials, tsim)
        for i in xrange(trials):
            if i%100==0:
                print "%d" % i,
                sys.stdout.flush()
                        
            t0 = time.time()
            st, u = mod_spike_train(modulation=[t*ms, rate], noise = noise, seed=i, noise_tau = noise_tau)
            tsolve = time.time() - t0
            
            s1 = signals.SpikeTrain(st)
            
            figure(3)
            values, xaxis = numpy.histogram(s1.isi(), bins=50)
            values = values/float(values.sum())
            xaxis = xaxis[:-1]
            plt.plot(xaxis,values, color="r")
            
            values, xaxis = numpy.histogram(s1.spike_times, bins=50)
            fano = numpy.var(values)/numpy.mean(values)
            
            print "High Noise IF, NO TAU, CV: ", s1.cv_isi(), "MEAN: ", s1.mean_rate(), "FANO:", fano, "tsolve:", tsolve, "s"
            
            psth[1:]+=numpy.histogram(st,bins)[0]
            figure(2)
            plt.plot(st, 2*numpy.ones_like(st),"|r")
            
            figure(4)
            plt.plot(u, "r")
            #plt.plot(n, "r--")
        
        print "\n"
        
        # normalize
        psth = psth.astype(float)
        psth/= dt*float(trials)/1000.0
        # this is for correct 'steps' plotting only
        psth[0] = psth[1]
        figure(1)
        plt.plot(bins,psth,linestyle='steps', color="r")
        
        
        psth = zeros_like(bins)
        stg = stgen.StGen()  
        
        # MEDIUM NOISE IF, NO TAU
        noise = 0.5
        noise_tau = -1
        
        print "Running %d trials of %.2f milliseconds" % (trials, tsim)
        for i in xrange(trials):
            if i%100==0:
                print "%d" % i,
                sys.stdout.flush()
                        
            t0 = time.time()
            st, u = mod_spike_train(modulation=[t*ms, rate], noise = noise, seed=i, noise_tau = noise_tau)
            tsolve = time.time() - t0
            
            s1 = signals.SpikeTrain(st)
            
            figure(3)
            values, xaxis = numpy.histogram(s1.isi(), bins=50)
            values = values/float(values.sum())
            xaxis = xaxis[:-1]
            plt.plot(xaxis,values, color="m")
            
            values, xaxis = numpy.histogram(s1.spike_times, bins=50)
            fano = numpy.var(values)/numpy.mean(values)
            
            print "Medium Noise IF, NO TAU, CV: ", s1.cv_isi(), "MEAN: ", s1.mean_rate(), "FANO:", fano, "tsolve:", tsolve, "s"
            
            psth[1:]+=numpy.histogram(st,bins)[0]
            figure(2)
            plt.plot(st, 3*numpy.ones_like(st),"|m")
            
            figure(4)
            plt.plot(u, "g")
            #plt.plot(n, "r--")
        
        print "\n"
        
        # normalize
        psth = psth.astype(float)
        psth/= dt*float(trials)/1000.0
        # this is for correct 'steps' plotting only
        psth[0] = psth[1]
        figure(1)
        plt.plot(bins,psth,linestyle='steps', color="m")


        
        psth = zeros_like(bins)
        stg = stgen.StGen()        

#        # GAMMA
#        a = numpy.ones_like(t)*100 
#        b = 1/a/rate   
#        
#        print "Running %d trials of %.2f milliseconds" % (trials, tsim)
#        for i in xrange(trials):
#            if i%100==0:
#                print "%d" % i,
#                sys.stdout.flush()
#            stg.seed(i)
#            
#            t0 = time.time()
#            
#            st = stg.inh_gamma_generator(a,b,t,tsim,array=True)
#            
#            tsolve = time.time() - t0
#            
#            s1 = signals.SpikeTrain(st)
#            
#            isi = s1.isi()
#            fano_factor_isi = numpy.var(isi)/numpy.mean(isi)
#            cv_isi = numpy.std(isi)/numpy.mean(isi)
#            
#            figure(3)
#            values, xaxis = numpy.histogram(s1.isi(), bins=50)
#            values = values/float(values.sum())
#            xaxis = xaxis[:-1]
#            plt.plot(xaxis,values, color="g")
#            
#            values, xaxis = numpy.histogram(s1.spike_times, bins=50)
#            fano = numpy.var(values)/numpy.mean(values)
#            
#            print "GAMMA, CV: ", cv_isi, "MEAN: ", s1.mean_rate(), "FANO:", fano, "tsolve:", tsolve, "s"
#            
#
#            psth[1:]+=numpy.histogram(st,bins)[0]
#            figure(2)
#            plt.plot(st, 4*numpy.ones_like(st),"|g")
#        
#        print "\n"
#        
#        # normalize
#        psth = psth.astype(float)
#        psth/= dt*float(trials)/1000.0
#        # this is for correct 'steps' plotting only
#        psth[0] = psth[1]
#        figure(1)
#        plt.plot(bins,psth,linestyle='steps', color="g")        
        
        
        
#        psth = zeros_like(t)
#        
#        a = numpy.ones_like(t)*40 # 11.346
#        bq = numpy.ones_like(t)*0.1231*14.48
#        a[i_start:i_end] = 40# 92.25
#        bq[i_start:i_end] = 0.09793*14.48
#        
#        tau_s = 110.0
#        tau_r = 1.97
#        qrqs = 221.96
#        t_stop = 1000.0
#        
#        #a is in units of Hz.  Typical values are available 
#        #in Fig. 1 of Muller et al 2007, a~5-80Hz (low to high stimulus)
#    
#        #bq here is taken to be the quantity b*q_s in Muller et al 2007, is thus
#        #dimensionless, and has typical values bq~3.0-1.0 (low to high stimulus)
#    
#        #qrqs is the quantity q_r/q_s in Muller et al 2007, 
#        #where a value of qrqs = 3124.0nS/14.48nS = 221.96 was used.
#    
#        #tau_s has typical values on the order of 100 ms
#        #tau_r has typical values on the order of 2 ms
#
#        
#        print "Running %d trials of %.2f milliseconds" % (trials, tsim)
#        for i in xrange(trials):
#            if i%100==0:
#                print "%d" % i,
#                sys.stdout.flush()
#            #stg.seed(i)
#            st = stg.inh_2Dadaptingmarkov_generator(a,bq,tau_s,tau_r,qrqs,t,t_stop,array=True)
#            
#            psth[1:]+=numpy.histogram(st,t)[0]
#            figure(3)
#            plt.plot(st, i*numpy.ones_like(st),"|r")
#        
#        print "\n"
#        
#        # normalize
#        
#        psth = psth.astype(float)
#        psth/= dt*float(trials)/1000.0 #psth/= dt*10000.0/1000.0
#        
#        # this is for correct 'steps' plotting only
#        psth[0] = psth[1]
#        
#        figure(1)
#        plt.plot(t,psth,linestyle='steps', color="r")
        
        plt.show()
        
        
    if "plot_train" in test:
        
        t = arange(0, 1, 0.001) 
        noise = 40*create_colnoise(t, -1, 20, seed=4)+40
        noise = ones(len(t))*6
        
        modulation = (t, noise)
        train, _ = mod_spike_train(modulation) 
        
        print train
        
        figure()
        subplot(1,2,1)
        plot(t, noise, 'b')
        subplot(1,2,2)
        plot(train*1e-3, ones(len(train)), 'b|')
        
        
    if "plot_train2" in test:
        
        t = arange(0, 1, 0.001) 
        noise = create_colnoise(t, -1, 20, seed=4)
        
        addnoise = create_colnoise(t, -1, 100, seed=4)
        
        noise1 = 10*40*noise+40
        noise2 = -10*40*noise+40
        
        noise1 = noise1.clip(min=0)
        noise2 = noise2.clip(min=0)
        #noise = ones(len(t))*6
        
        noise_a = 1; noise_syn = 0; noise_syn_tau = 0 # poisson
        noise_a = 1e9; noise_syn = 0.5; noise_syn_tau = -1 # medium noise

        modulation1 = (t, noise1)
        train1, _ = mod_spike_train(modulation1, noise = noise_syn, seed = 1, noise_tau = noise_syn_tau, noise_a = noise_a)
        
        modulation2 = (t, noise2)
        train2, _ = mod_spike_train(modulation2, noise = noise_syn, seed = 1, noise_tau = noise_syn_tau, noise_a = noise_a)
        
        tau = 0.1
        t_kernel = np.arange(0, tau*6, 0.001)
        
        kernel1 = syn_kernel(t_kernel, 0, tau)
        kernel = np.concatenate((np.zeros(len(kernel1)-1),kernel1))
         
        print "- Basis convolution"
        noise0 = concatenate([noise,noise,noise,noise,noise])
        
        noise_conv0 = np.convolve(noise0, kernel, mode='same')
        
        noise_conv = noise_conv0[3000:4000]
                
        figure()
        subplot(4,2,1)
        plot(t, noise, 'b')
        plt.axis([0.25, 0.8, -1.5, 1.5]) 
        
        subplot(4,2,3)
        plot(t, noise1, 'b')
        plt.axis([0.25, 0.8, -10, 650]) 
                   
        subplot(4,2,5)
        plot(train1*1e-3, ones(len(train1)), 'b|')
        plt.axis([0.25, 0.8, 0.94, 1.06]) 
                
        subplot(4,2,4)
        plot(t, noise2, 'b')
        plt.axis([0.25, 0.8, -10, 650])
        
        subplot(4,2,6)
        plot(train2*1e-3, ones(len(train2)), 'b|')
        plt.axis([0.25, 0.8, 0.94, 1.06])

        subplot(4,2,7)
        
        noisy=concatenate([noise_conv[0:600]+6*addnoise[0:600],noise_conv[600:1000]+20*addnoise[600:1000]])
        plot(t, noise_conv, 'b', t, noisy, 'r')
        plt.axis([0.25, 0.8, -60, 60])   
        
        subplot(4,2,8)
        plot(t_kernel, kernel1, 'b')
        plt.axis([0, 0.55, 0, 1.1])    

                  
        
    if "est_quality" in test:  # Shows that SNR depends on noise!!!
        
        
        tstart = 0; dt = 1*ms; tstop = 100 # s 
        fs = 1/dt # sampling rate
        fmax = fs/2 # maximum frequency (nyquist)
        t = arange(tstart,tstop,dt) # create stimulus time vector
        fu = 10 # [Hz]
        amp = 0.2
        ihold = 0
        
        n_in = 0.2*create_colnoise(t, sexp = 4, cutf = None)          
        addmean = 0 #40
        ampmult = 10
        
        # SINGLE SINE, NOISE N from fit 
        input_signal = amp*sin(2 * pi * t * fu) + ihold 
        output_signal = n_in+(ampmult * amp * sin(2 * pi * t * fu + 0*(pi / 2)) + ihold + addmean) 
        
        amp_mean, mag_mean, phase_mean, fmean = get_magphase(input_signal, t, output_signal, t, method = "fft", f = fu)   
        
        print "mean amplitude output signal " + str(amp_mean)
        print "mean magnitude " + str(mag_mean) + " ,mean phase = " + str(phase_mean)
        
        results = est_quality(t, fu, output_signal, amp*mag_mean, phase_mean/ (180 / pi), fmean)  
                            
        NI, VAF, R_est, N = results.get('NI'), results.get('VAF'), results.get('R_est'), results.get('N')
        
        #amp_mean, mag_mean, phase_mean, fmean = get_magphase(input_signal, t, N, t, method = "fft", f = fu)
        
        SNR1 = mean(input_signal**2) / std(N) 
        SNR2 = mean(input_signal**2) / mean(N**2)
        
        print "- NI: " + str(NI) + ", VAF: " + str(VAF) + ", SNR2: " + str(SNR2) + ", SNR1: " + str(SNR1)         
        
        figure(11)
        Ps, f = psd(input_signal, Fs=1/dt)
        Pn, f = psd(N, Fs=1/dt)
        SNRf = Ps/Pn

        nfft = 2 ** int(ceil(log2(2 * fs))) 
        P_nn, freq = csd(N, N, Fs = fs, NFFT = nfft, noverlap = nfft / 4, sides = 'twosided')
        P_ss, freq = csd(input_signal, input_signal, Fs = fs, NFFT = nfft, noverlap = nfft / 4, sides = 'twosided')
        SNRx = ifftshift(P_ss / P_nn)[0:len(freq)/2] # 10*log10()
        freq =  ifftshift(freq)[0:len(freq)/2]  # only positive frequencies

        figure(12)        
        semilogx(f, SNRf, 'r', freq, SNRx, 'g')
        print mean(SNRf)
        print mean(real(SNRx))
        
        
        figure(1)  
        plot(t, input_signal ,'g--', t, output_signal, 'k-', t, R_est, 'b-', t, N, 'r-')
        
        
        # NOISE, NOISE N from reconstruction  
        method_interpol = array(["linear"]) 
        
        input_signal = amp*create_colnoise(t, sexp = 4, cutf = None)  
        output_signal = n_in + (ampmult * input_signal + addmean)        
        noise_data_points = len(input_signal)
        
        w_length = 5*s
        results = compute_Transfer(output_signal, t, input_signal, t, noise_data_points, gsyn = output_signal, freq_wp = None, method_interpol = method_interpol, do_csd = 1, nc_delay = 0, w_length = w_length, t_kernel = 0, t_qual = 990)
        freq, tc, stim_re_mat, stim, resp_mat, noise_mat, CF_mat, SNR_mat = results.get('freq_used'), results.get('tc'), results.get('stim_re_mat'), results.get('stim'), results.get('resp_mat'), results.get('noise_mat'), results.get('CF_mat'), results.get('SNR_mat')
        tk, K_mat = results.get('tk'), results.get('K_mat')
        
        SNR =  SNR_mat[1][0,:]
        figure(2)        
        semilogx(freq, SNR, 'b')
        print "- SNR(f=" + str(freq[(freq>=9.8) & (freq<=10)]) + "): " + str(real(SNR[(freq>=9.8) & (freq<=10)]))  
        print mean(SNR)


        w_length = 0.5*s
        results = compute_Transfer(output_signal, t, input_signal, t, noise_data_points, gsyn = output_signal, freq_wp = None, method_interpol = method_interpol, do_csd = 1, nc_delay = 0, w_length = w_length, t_kernel = 0, t_qual = 990)
        freq, tc, stim_re_mat, stim, resp_mat, noise_mat, CF_mat, SNR_mat = results.get('freq_used'), results.get('tc'), results.get('stim_re_mat'), results.get('stim'), results.get('resp_mat'), results.get('noise_mat'), results.get('CF_mat'), results.get('SNR_mat')
        tk, K_mat = results.get('tk'), results.get('K_mat')
        
        figure(4)
        plot(tk, K_mat[0,:])
        
        SNR =  SNR_mat[1][0,:]
        figure(2)        
        semilogx(freq, SNR, 'g')
        print "- SNR(f=" + str(freq[(freq>=9.5) & (freq<=10.5)]) + "): " + str(real(SNR[(freq>=9.5) & (freq<=10)]))  
        print mean(SNR)        
        
        figure(3) 
        tc, resp_cc, stim_cc, stim_re_cc, noise_cc, CF, VAF = reconstruct_Stimulus(K_mat[0,:], resp_mat[0,:], stim, t)
        plot(tc, stim_cc, 'k')
        plot(tc, stim_re_cc, 'b') 
        plot(tc, noise_cc, 'r')
        
        print "- CF=" + str(CF) + " VAF=" + str(VAF)
        
        #show()
        
        #SNR CHECK!
        
        figure(31) 
        nfft = 2 ** int(ceil(log2(w_length * fs))) 
        P_nn, freq = csd(noise_cc, noise_cc, Fs = fs, NFFT = nfft, noverlap = nfft / 4, sides = 'twosided')
        P_ss, freq = csd(stim_cc, stim_cc, Fs = fs, NFFT = nfft, noverlap = nfft / 4, sides = 'twosided')
        SNR = ifftshift(P_ss / P_nn)[0:len(freq)/2] # 10*log10()
        freq =  ifftshift(freq)[0:len(freq)/2]  # only positive frequencies
        figure(2)        
        semilogx(freq, SNR, 'r')
        figure(32)        
        semilogx(freq, SNR, 'r')
        
        figure(33)
        pxx, f = psd(input_signal, Fs=1/dt)
        clf()
        semilogx(f, pxx / pxx[1])
               
    
        show()
        
    
    if "hdf5" in test:
        
        filename = 'text.hdf5'
        y = []        
        x = np.arange(0, 1, 0.1) 
        y.append(x)
        y.append(x)
        
        d = {}
        d['y'] = y
        d['b'] = 'array'
        
        print d
        
        rw_hdf5(filename, d)
        
        d = rw_hdf5(filename)
        
        print d
        
        
        
        
                
        
        