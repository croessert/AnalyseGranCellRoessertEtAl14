TITLE conductance-based integrate and fire with KCa style adaptation current

UNITS {
    (nA) = (milliamp)
    (mV) = (millivolt)
}

NEURON {
    POINT_PROCESS SpikeOutAdapt2
    RANGE thresh, refrac, vrefrac, grefrac, dgkbar, egk, ctau, idrive, dgkbar2, dtau
    NONSPECIFIC_CURRENT i
}

PARAMETER {
    thresh = -50 (mv)
    refrac = 5 (ms)
    vrefrac = -65 (mv)
    grefrac = 100 (microsiemens) :clamp to vrefrac

    dgkbar = 0 (mS/cm2) <0,1e9>
    egk   = -80 (mV)   
    ctau =  5   (ms)
    
    dgkbar2 = 0 (mS/cm2) <0,1e9>
    dtau = 1  (ms)

    idrive = 0 (nA)
}

STATE {
    c
    d
}

ASSIGNED {
    i       (nA)
    v       (mv)
    g       (microsiemens)
    area    (um2) : area of current segment (automatically available within NMODL, like v)
    dgk     (uS) : set to dgkbar * segment area at initialization
    dgk2     (uS) : set to dgkbar * segment area at initialization
    gk      (uS) : will be 0 or dgk*c depending on recent spiking history
    gk2      (uS) : will be 0 or dgk*c depending on recent spiking history
}

BREAKPOINT {
    SOLVE state METHOD cnexp 
    gk = dgk*c  
    gk2 = dgk2*d
    i = g*(v - vrefrac) +  gk*(v - egk) +  gk2*(v - egk) - idrive
}

INITIAL {
    dgk = dgkbar*area*1e-5 : because area will be in um2, but dgk is in uS and dgkbar in mS/cm2
    dgk2 = dgkbar2*area*1e-5 : because area will be in um2, but dgk is in uS and dgkbar in mS/cm2
    gk = 0 : because at t = 0 we assume that the cell has not yet spiked
    c = 0
    d = 0
    g = 0 : HAST TO BE 0!!!
    net_send(0, 3)
}

DERIVATIVE state {     : exact when v held constant; integrates over dt step
    c' = -c / ctau
    d' = -d / dtau
}

NET_RECEIVE(w) {

    if (flag == 1) {    : spike has occured            

        net_event(t)
	  net_send(refrac, 2)
	  v = vrefrac
	  g = grefrac

    }else if (flag == 2) {
        
        g = 0
        c = c+1 : increase c and wait for next spike
        d = d+1 : increase d and wait for next spike
        at_time(t)
        net_send(0, 3)

    }else if (flag == 3) {  : no spike has occured

        WATCH (v > thresh) 1    : detect spike

    }	
}
