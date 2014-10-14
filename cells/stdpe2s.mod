COMMENT
Two state kinetic scheme synapse described by rise time tau1,
and decay time constant tau2. The normalized peak condunductance is 1.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
	where tau1 < tau2

If tau2-tau1 -> 0 then we have a alphasynapse.
and if tau1 -> 0 then we have just single exponential decay.

The factor is evaluated in the
initial block such that an event of weight 1 generates a
peak conductance of 1.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

ENDCOMMENT

NEURON {
    POINT_PROCESS stdpE2S
    RANGE tau1, tau2, e, i, thresh
    RANGE interval, tpre, tpost, M, P, deltaw, w, wmax, apre, apost, taupre, taupost, on, tend
    NONSPECIFIC_CURRENT i
    RANGE g
}


UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    tau1=.1 (ms) <1e-9,1e9>
    tau2 = 10 (ms) <1e-9,1e9>
    e=0	(mV)

    thresh = -20 (mV)	: postsynaptic voltage threshold

    taupre  = 20	(ms) : decay time for presynaptic part ( values from           )
    taupost  = 20 (ms) : decay time for postsynaptic part ( Song and Abbott, 2001 )
    w = 0.001
    wmax = 0 : max values of synaptic weight

    : apre+ apost- = Hebbian, apre- apost+ = anti-Hebbian
    apre    = 0.0001		: amplitude of step when a presynaptic spike arrives, LTP when positive
    apost   = -0.000106	: amplitude of step when a postsynaptic spike arrives, LTD when negative
    
    on	= 0		: allows learning to be turned on and off globally
    tend = 1e9
}

ASSIGNED {
    v (mV)
    i (nA)
    g (uS)
    factor

    interval	(ms)	: since last spike of the other kind
    tpre	(ms)	: time of last presynaptic spike
    tpost	(ms)	: time of last postsynaptic spike
    M			: LTD function
    P			: LTP function
    deltaw			: change in weight
    
}

STATE {
	A (uS)
	B (uS)
}

INITIAL {
    LOCAL tp
    if (tau1/tau2 > .9999) {
        tau1 = .9999*tau2
    }
    A = 0
    B = 0
    tp = (tau1*tau2)/(tau2 - tau1) * log(tau2/tau1)
    factor = -exp(-tp/tau1) + exp(-tp/tau2)
    factor = 1/factor

    interval = 0
    tpre = 0
    tpost = 0
    M = 0
    P = 0
    deltaw = 0

    net_send(0, 1)
    net_send(tend, 3)

}

BREAKPOINT {
    SOLVE state METHOD cnexp
    g = B - A
    i = g*(v - e)
}

DERIVATIVE state {
    A' = -A/tau1
    B' = -B/tau2
}

NET_RECEIVE(wex (uS)) {

    if (flag == 0) { : presynaptic spike

        :printf("entry flag=%g t=%g w=%g tpre=%g tpost=%g\n", flag, t, w, tpre, tpost)

        :FOR_NETCONS(wex) {: also can hide NET_RECEIVE args
        :    wex = w
        :    printf("entry FOR_NETCONS wex=%g w=%g\n", wex, w)
        :}

        A = A + w*factor
        B = B + w*factor

        if (on == 1) {

            P = P*exp((tpre-t)/taupre) + apre : modified by each presynaptic spike, decay with taupre

            interval = tpost - t : interval is negative
            deltaw = wmax * M * exp(interval/taupost)

            :printf("P=%g deltaw=%g interval=%g\n", P, deltaw, interval)
                                
        }
        tpre = t

    } else if (flag == 2) { : postsynaptic spike
        
        :printf("entry flag=%g t=%g w=%g tpre=%g tpost=%g\n", flag, t, w, tpre, tpost)

        if (on == 1) {

            M = M*exp((tpost-t)/taupost) + apost : modified by each postsynaptic spike, decay with taupost

            interval = t - tpre	: interval is positive    
            deltaw = wmax * P * exp(-interval/taupre)	

            :printf("M=%g deltaw=%g interval=%g\n", M, deltaw, interval)

        }
        tpost = t

    } else if (flag == 1) { : flag == 1 from INITIAL block

        :printf("entry flag=%g t=%g thresh=%g\n", flag, t, thresh)

        WATCH (v > thresh) 2 : needs only init at the beginning!

    } else if (flag == 3) { : stop learning
        on = 0
    }

    if (on == 1 && (flag == 2 || flag == 0) ) {

        w = w + deltaw
        if (w > wmax) {
            w = wmax
        }
        if (w < 0) {
            w = 0
        }
        :printf("w=%g\n", w)
    }
}
