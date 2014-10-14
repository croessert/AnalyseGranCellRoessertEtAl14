NEURON {
        POINT_PROCESS ExpSynGmax
        RANGE tau, e, i, gmax
        NONSPECIFIC_CURRENT i
}

UNITS {
        (nA) = (nanoamp)
        (mV) = (millivolt)
        (uS) = (microsiemens)
}

PARAMETER {
        tau = 0.1 (ms) <1e-9,1e9>
        e = 0   (mV)
        gmax = 10 (nS)
}

ASSIGNED {
        v (mV)
        i (nA)
}

STATE {
        g (uS)
}

INITIAL {
        g=0
}

BREAKPOINT {
        SOLVE state METHOD cnexp
        i = g*(v - e)*gmax
}

DERIVATIVE state {
        g' = -g/tau
}

NET_RECEIVE(weight (uS)) {
        g = g + weight
}