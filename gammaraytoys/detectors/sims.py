from histpy import Histogram, Axes
from astropy import units as u
import numpy as np
from .reco import RecoCompton
from .source import Source

class Simulator:

    def __init__(self, detector, sources, reconstructor,
                 duration = None, nsim = None, ntrig = None,
                 doppler_broadening = True):

        self.detector = detector

        if isinstance(sources, Source): 
            self.sources = [sources]
        else:
            # Multiple sources
            self.sources = sources

        fluxes = u.Quantity([s.flux for s in self.sources])

        self.total_flux = np.sum(fluxes)
        self._relative_flux = (fluxes/self.total_flux).to_value('')
            
        self.reconstructor = reconstructor

        if np.sum([duration is not None,
                   nsim is not None,
                   ntrig is not None]) != 1:
            raise ValueError("Specify one and only one finishing condition")

        if duration is not None:
            self.duration = duration
            self.nsim = int(np.round(self.total_flux*duration*detector.throwing_plane_size))
            # TBD after sims
            self.ntrig = np.inf

        if nsim is not None:
            self.nsim = nsim
            self.duration = nsim/self.total_flux/detector.throwing_plane_size

            # TBD after sims
            self.ntrig = np.inf
            
        if ntrig is not None:
            self.ntrig = ntrig
            # TBD after sims
            self.nsim = np.inf
            self.duration = np.inf

        # Default, can be changed
        self._axes_compton = Axes([np.geomspace(.1,10,200)*u.MeV,
                                   np.linspace(0,180, 180)*u.deg,
                                   np.linspace(-180,180, 360)*u.deg],
                                  labels = ["Em", "Phi", "Psi"],
                                axis_scale = ['log','linear','linear'])

        # Other
        self.doppler_broadening = doppler_broadening
        
    @property
    def nsources(self):
        return len(self.sources)
        
    @property
    def compton_data_axes(self):
        return self._axes_compton

    @compton_data_axes.setter
    def compton_data_axes(self, new):
        self._axes_compton = new
    
    def run_binned(self):

        h_data = Histogram(self.compton_data_axes)

        for sim_event, reco_event in self.run_events():

            if reco_event.triggered:

                h_data.fill(reco_event.energy,
                            reco_event.phi,
                            reco_event.psi)

        return h_data
        
    def run_events(self):

        nsim = 0
        ntrig = 0

        terminate = False
        
        while True:

            if terminate:
                self.nsim = nsim
                self.ntrig = ntrig
                self.duration = (nsim/self.total_flux/self.detector.throwing_plane_size).to(u.s)

                break
                
            nsim += 1

            source = self.sources[np.random.choice(range(self.nsources),
                                                   p = self._relative_flux)]
            
            primary = source.random_photon(self.detector)
            
            sim_event = self.detector.simulate_event(primary,
                                                     doppler_broadening = self.doppler_broadening)

            reco_event = self.reconstructor.reconstruct(sim_event)

            if nsim >= self.nsim:
                terminate = True
                
            if reco_event.triggered:
                ntrig += 1

                if ntrig >= self.ntrig:
                    terminate = True
                
            yield sim_event, reco_event
            
        
