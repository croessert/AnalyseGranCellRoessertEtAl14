# -*- coding: utf-8 -*-
from neuron import h

class Synapse:
    def __init__(self,source,target,section,nrel = 0,syntype = 'ANK',record_all = 0,weight = 1, weight_gmax = 1, mglufac = 1):
        self.record_all = record_all
        # if record_all:
        #     print "Recording all in Synapse"

        self.input = h.NetStim(0.5)
        self.input.start = -10
        self.input.number = 1
        self.input.interval = 1e9
        self.weight = weight
        self.weight_gmax = weight_gmax
        self.mglufac = mglufac

        self.nrel = nrel
        self.syntype = syntype

        self.postsyns = {}
        self.noises = []

        if (type(source) == type('s')):
            sourcetype = source
        else:
            sourcetype = source.whatami

        # This is set to a value not 0 to avoid problems with the pc.set_maxstep()
        nc_delay = 0.01
        # total accumultate delay = nc_delay* (1+1) = 0.02 ms
        # this should be removed from the gid_connect delay

        if self.record_all:
            self.SpikeTrain_input = [h.Vector(),h.Vector()]
            self.netcon_in = h.NetCon(self.input,None, 0, nc_delay, 1)
            self.netcon_in.record(self.SpikeTrain_input[1], self.SpikeTrain_input[0], 1)

        # Decide what kind of connection to create based on the source and target types
        if sourcetype == 'glom':
            if target.whatami == 'pf':
                ## No input to the fake grcs
                print 'Pf!!!!'
            else:
                # Make a mf synapse
                if self.nrel>0 :
                    if target.whatami == 'grc':
                        self.whatami = "syn_glom2grc_stoch"

                    # Use stochastic synapses
                    self.release_sites = [h.Release_site(0.5, sec=section) for i in range(self.nrel)]
                    # Set Prob release
                    for i, site in enumerate(self.release_sites):
                        site.U = 0.42 * self.weight

                        lowindex = (target.gid+1) * (len(target.MF_L)+1) * (i+1)
                        noiseRandObj = h.Random()  # provides NOISE with random stream
                        self.noises.append(noiseRandObj)  # has to be set here not inside the nmodl function!!
                        site.noiseFromRandom(self.noises[-1])  # connect random generator!
                        self.noises[-1].MCellRan4(1, lowindex)  # set lowindex to gid+1, set highindex to > 0
                        self.noises[-1].uniform(0,1)


                    # Set netcon for spikes
                    self.netcon_out = [h.NetCon(release_site,None, 0, nc_delay, 1) for release_site in self.release_sites]
                    # Connect input
                    self.nc_rel = [h.NetCon(self.input,release_site, 0, nc_delay, 1) for release_site in self.release_sites]
                    # Add poststnaptic sites
                    if 'A'  in self.syntype:
                        # AMPA
                        self.postsyns['AMPA'] = [h.GRANULE_Ampa_stoch_vi(.5, sec=section) for r in self.release_sites]

                    if 'N'  in self.syntype:
                        # NMDA
                        self.postsyns['NMDA'] = [h.GRANULE_Nmda_stoch_vi(.5, sec=section) for r in self.release_sites]

                    for rec_site in self.postsyns['AMPA']:
                        rec_site.gmax_factor = self.weight_gmax*(1./0.5)/self.nrel
                    for rec_site in self.postsyns['NMDA']:
                        rec_site.gmax_factor = self.weight_gmax*(1./0.6)/self.nrel

                    # Connect pre to post synapse
                    self.nc_syn = [[h.NetCon(release_site, receptor[k], 0, nc_delay, 1) for k, release_site in enumerate(self.release_sites)] for receptor in self.postsyns.itervalues()]

                    # If required recod the timing of each vescicle release
                    if self.record_all:
                        self.SpikeTrain_output = [[h.Vector(),h.Vector()] for n in self.netcon_out]
                        for i,n in enumerate(self.netcon_out):
                            n.record(self.SpikeTrain_output[i][1],self.SpikeTrain_output[i][0],i+2)

                else:
                    if target.whatami == 'grc':
                        self.whatami = "syn_glom2grc_det"

                    # Use deterministic synapses
                    if 'A'  in self.syntype:
                        # AMPA
                        if 'wopre' in self.syntype:
                            self.postsyns['AMPA'] = [h.GRANULE_Ampa_stoch_vi(0.5, sec=section)]
                        else:
                            self.postsyns['AMPA'] = [h.GRANULE_Ampa_det_vi(0.5, sec=section)]

                        self.postsyns['AMPA'][0].U = self.postsyns['AMPA'][0].U * self.weight
                        self.postsyns['AMPA'][0].gmax_factor = self.weight_gmax


                    if 'N'  in self.syntype:
                        # NMDA
                        if 'wopre' in self.syntype:
                            self.postsyns['NMDA'] = [h.GRANULE_Nmda_stoch_vi(0.5, sec=section)]
                        else:
                            self.postsyns['NMDA'] = [h.GRANULE_Nmda_det_vi(0.5, sec=section)]

                        self.postsyns['NMDA'][0].U = self.postsyns['NMDA'][0].U * self.weight
                        self.postsyns['NMDA'][0].gmax_factor = self.weight_gmax
                        

                    # Connect input to the receptors
                    self.nc_syn = [h.NetCon(self.input,receptor[0], 0, nc_delay, 1) for receptor in self.postsyns.itervalues()]



        elif sourcetype == 'goc':
            if target.whatami == 'grc':
                # Make a Golgi (GABAergic) synapse onto a granule cell

                if self.nrel>0 :

                    # Use stochastic synapses
                    self.whatami = "syn_goc2grc_stoch"
                    self.nc_syn = [h.NetCon(self.input,receptor[0], 0, nc_delay, 1) for receptor in self.postsyns.itervalues()]

                    self.release_sites = [h.Release_site(0.5, sec=section) for i in range(self.nrel)]
                    # Set Prob release
                    for i, site in enumerate(self.release_sites):
                        site.U = 0.35 * self.weight
                        site.tau_1 	= 0.1
                        site.tau_rec = 36.169
                        site.tau_facil = 58.517

                        lowindex = (target.gid+1) * (len(target.GOC_L)+1) * (i+1)
                        noiseRandObj = h.Random()  # provides NOISE with random stream
                        self.noises.append(noiseRandObj)  # has to be set here not inside the nmodl function!!
                        site.noiseFromRandom(self.noises[-1])  # connect random generator!
                        self.noises[-1].MCellRan4(1, lowindex)  # set lowindex to gid+1, set highindex to > 0
                        self.noises[-1].uniform(0,1)

                    # Set netcon for spikes
                    self.netcon_out = [h.NetCon(release_site,None, 0, nc_delay, 1) for release_site in self.release_sites]
                    # Connect input
                    self.nc_rel = [h.NetCon(self.input,release_site, 0, nc_delay, 1) for release_site in self.release_sites]
                    # Add poststnaptic sites

                    self.postsyns['GABA'] = [h.GRANULE_Gaba_stoch_vi(0.5, sec=section) for r in self.release_sites]
                    for rec_site in self.postsyns['GABA']:
                        rec_site.gmax_factor = self.weight_gmax*(1./0.5)/self.nrel

                    # Connect pre to post synapse
                    self.nc_syn = [[h.NetCon(release_site, receptor[k], 0, nc_delay, 1) for k, release_site in enumerate(self.release_sites)] for receptor in self.postsyns.itervalues()]

                    # If required recod the timing of each vescicle release
                    if self.record_all:
                        self.SpikeTrain_output = [[h.Vector(),h.Vector()] for n in self.netcon_out]
                        for i,n in enumerate(self.netcon_out):
                            n.record(self.SpikeTrain_output[i][1],self.SpikeTrain_output[i][0],i+2)

                else:

                    # Use deterministic synapses
                    self.whatami = "syn_goc2grc_det"
                    if 'wopre' in self.syntype:
                        self.postsyns['GABA'] = [h.GRANULE_Gaba_stoch_vi(0.5, sec=section)]
                    else:
                        self.postsyns['GABA'] = [h.GRANULE_Gaba_det_vi(0.5, sec=section)]

                    self.postsyns['GABA'][0].U = self.postsyns['GABA'][0].U * self.weight
                    self.postsyns['GABA'][0].gmax_factor = self.weight_gmax
                    #print "GABA:", self.weight ,self.weight_gmax

                    self.nc_syn = [h.NetCon(self.input,receptor[0], 0, nc_delay, 1) for receptor in self.postsyns.itervalues()]



        else:
            print 'SOURCE TYPE DOES NOT EXIST SOMETHING WRONG!!!!!!!!!'

        if self.record_all and len(self.postsyns) > 0 and 'R' not in self.syntype:
            self.i = {}
            for (post_type,post) in self.postsyns.iteritems():
                self.i[post_type] = []
                for p in post:
                    self.i[post_type].append(h.Vector())
                    self.i[post_type][-1].record(p._ref_i)
            self.g = {}
            for (post_type,post) in self.postsyns.iteritems():
                self.g[post_type] = []
                for p in post:
                    self.g[post_type].append(h.Vector())
                    self.g[post_type][-1].record(p._ref_g)

            if (self.whatami == "syn_glom2grc_det") or (self.whatami == "syn_grcaa2goc_det") or (self.whatami == "syn_glom2goc_det") or (self.whatami == "syn_grc2goc_det"):
                if 'N'  in self.syntype:
                    self.mgblock = []
                    for p in self.postsyns['NMDA']:
                        self.mgblock.append(h.Vector())
                        self.mgblock[-1].record(p._ref_MgBlock)

    def prel(self,prel):
        for r in self.release_sites:
            r.U = 0.42

    def destroy(self):
        if self.record_all:
            del self.netcon_in
        if self.nrel > 0:
            for r in self.netcon_out:
                del r
            for r in self.nc_rel:
                del r
        if self.whatami == "syn_grc2stl_relay" or self.whatami == "syn_pf2stl_relay":
            del self.relay
        for r in self.nc_syn:
            del r
        del self
