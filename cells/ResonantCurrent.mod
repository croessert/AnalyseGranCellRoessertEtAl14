TITLE simple resonant current, modeled like extra current for GIF model (Brunel et al. 2003)

UNITS {
    (nA) = (milliamp)
    (mV) = (millivolt)
}

NEURON {
    POINT_PROCESS ResonantCurrent
    RANGE g, gbar, tau, vrest
    NONSPECIFIC_CURRENT i
}

PARAMETER {
    g = 0 (uS) <0,1e9>   
    :gbar = 0 (mS/cm2) <0,1e9>
    tau =  5   (ms)
    vrest = -70 (mV)     
}

STATE {
    w
}

ASSIGNED {
    i       (nA)
    v       (mv)
    area    (um2) : area of current segment (automatically available within NMODL, like v)
    :g      (uS) : NOT ANY MORE: set to dgbar * segment area at initialization
}

BREAKPOINT {
    SOLVE state METHOD cnexp 
    i = g*w 
}

INITIAL {
    :g = gbar*area*1e-5 : because area will be in um2, but dg is in uS and dgbar in mS/cm2
    w = 0    
}

DERIVATIVE state {     : exact when v held constant; integrates over dt step
    w' = ((v-vrest)-w)/tau
}

