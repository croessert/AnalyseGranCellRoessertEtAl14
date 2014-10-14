NEURON {
	POINT_PROCESS SpikeOutRandreset
	RANGE mtau, thresh, refrac, vrefrac, grefrac
	NONSPECIFIC_CURRENT i
     THREADSAFE : only true if every instance has its own distinct Random
     POINTER donotuse
}

PARAMETER {
      mtau = 1 (ms) 
	thresh = 1 (millivolt)
	refrac = 5 (ms)
	vrefrac = 0 (millivolt)
	grefrac = 100 (microsiemens) :clamp to vrefrac
}

ASSIGNED {
      noise
	i (nanoamp)
	v (millivolt)
	g (microsiemens)
      vrefrac_rand (millivolt)
      donotuse   
}

INITIAL {
	net_send(0, 3)
	g = 0
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
     	i = g*(v - vrefrac_rand)
}

NET_RECEIVE(w) {
	if (flag == 1) {
		net_event(t)
		net_send(refrac, 2)
           vrefrac_rand = vrefrac+noise
		v = vrefrac_rand
		g = vrefrac_rand
            
	}else if (flag == 2) {
		g = 0
	}else if (flag == 3) {
		WATCH (v > thresh) 1
	}	
}
