TITLE Fluctuating current

COMMENT
-----------------------------------------------------------------------------

    Fluctuating current model for synaptic bombardment
    ==================================================

 The present version implements and generate a realization of an Ornstein-Uhlenbeck (OU) process
 (see Cox & Miller, 1969; see Tuckwell) to mimick the somatic impact of linearly adding EPSPs and
 IPSPs. Thus, it generates and injects in the specified neuronal compartment a fluctuating current
 waveform, characterized by a gauss-distributed amplitude, where neighboring amplitude samples are
 by definition linearly correlated on a time scale set by the correlation time-length "tau" of the
 process.
 
 The numerical scheme for integration of OU processes takes advantage of the fact that these
 processes are gaussian, which led to an exact update rule independent of the time step dt
 (see Gillespie DT, Am J Phys 64: 225, 1996):

 x(t+dt) = x(t) + (1. - exp(-dt/tau)) * (m - x) + sqrt(1.-exp(-2.*dt/tau)) * s * N(0,1)  

 where N(0,1) is a normal random number (avg=0, sigma=1)..

 Please note that only fixed integration time-step methods makes sense, since the stochastic current
 synthesized by the present mechanism is produced randomly and on-line. In other words, it is wrong to
 assume that neglecting the present integration step, reducing it and resynthesizing the current,
 lead to the same overall trajectory in the compartment output voltage.

IMPLEMENTATION

  This mechanism is implemented as a nonspecific current defined as a point process.

PARAMETERS

  The mechanism takes as input the following parameters:

     m   = 0. (nA)      : steady-state expected value of the current amplitude.
     s   = 0. (nA)      : square root of the steady-state variance of the current amplitude (standard deviation).
     tau = 2. (ms)      : steady-state correlation time length of the current.

REFERENCES

Rauch, A., La Camera, G., Luescher, H.-R., Senn, W., and Fusi, S. (2003). Neocortical pyramidal cells respond as integrate-and-fire neurons to in vivo-like input currents.  J. Neurophysiol. 90(3), 1598-612.
Giugliano, M., Darbon, P., Arsiero, M., Luescher, H.-R., and Streit, J. (2004). Single-neuron discharge properties and network activity in dissociated cultures of neocortex.  J. Neurophysiol. 92(2), 977-96. 
La Camera, G., Rauch, A., Thurbon, D., Luescher, H.-R., Senn, W., and Fusi, S. (2006). Multiple time scales of temporal response in pyramidal and fast spiking cortical neurons, J. Neurophysiol. 96(6):3448-64.
Arsiero, M., Luescher, H.-R., Lundstrom, B.N., and Giugliano, M. (2007). The Impact of Input Fluctuations on the Frequency-Current Relationships of Layer 5 Pyramidal Neurons in the Rat Medial Prefrontal Cortex. sumbitted.

The present mechanism is inspired by "Gfluct.mod", by A. Destexhe (1999), as taken from ModelDB.
Destexhe, A., Rudolph, M., Fellous, J-M. and Sejnowski, T.J. (2001). Fluctuating synaptic conductances recreate in-vivo-like activity in  neocortical neurons. Neuroscience 107: 13-24.

AUTHOR
M. Giugliano, Brain Mind Institute, EPFL Lausanne, May 2006 - mgiugliano@gmail.com

MODIFIED BY
Christian Roessert, November 2011 - christian.a@roessert.de

-----------------------------------------------------------------------------
ENDCOMMENT


INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}


NEURON {
    POINT_PROCESS Ifluct2
    RANGE m, s, tau, x
    NONSPECIFIC_CURRENT i
    THREADSAFE : only true if every instance has its own distinct Random
    POINTER donotuse
}


UNITS {
    (nA) = (nanoamp) 
    (mV) = (millivolt)
}


PARAMETER {
    m   = 0. (nA)      : steady-state expected value of the current amplitude
    s   = 0. (nA)      : square root of the steady-state variance of the current amplitude
    tau = 2. (ms)      : steady-state correlation time length of the current
}


ASSIGNED {
    noise
    i     (nA)          : fluctuating current
    x                   : state variable
    donotuse
}


INITIAL {
    x = m               : to reduce the transient, the state is set to its (expected) steady-state
}


BEFORE BREAKPOINT {
    noise = erand()
}
VERBATIM
double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);
ENDVERBATIM


FUNCTION erand() {
VERBATIM
	if (_p_donotuse) {
		/*
		:Supports separate independent but reproducible streams for
		: each instance. However, the corresponding hoc Random
		: distribution MUST be set to Random.negexp(1)
		*/
		_lerand = nrn_random_pick(_p_donotuse);
	}else{
		/* only can be used in main thread */
		if (_nt != nrn_threads) {
hoc_execerror("multithread random in NetStim"," only via hoc Random");
		}
ENDVERBATIM
		: the old standby. Cannot use if reproducible parallel sim
		: independent of nhost or which host this instance is on
		: is desired, since each instance on this cpu draws from
		: the same stream
		erand = exprand(1)
VERBATIM
	}
ENDVERBATIM
}


PROCEDURE noiseFromRandom() {
VERBATIM
 {
	void** pv = (void**)(&_p_donotuse);
	if (ifarg(1)) {
		*pv = nrn_random_arg(1);
	}else{
		*pv = (void*)0;
	}
 }
ENDVERBATIM
}

BREAKPOINT {
    SOLVE oup
    if (tau <= 0) {  x = m + s  * noise }  : white-noise is impossible to generate anyway..
    i = - x
}


PROCEDURE oup() {       
    if (tau > 0) {  x = x + (1. - exp(-dt/tau)) * (m - x) + sqrt(1.-exp(-2.*dt/tau)) * s  * noise }
}

