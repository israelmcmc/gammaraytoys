from abc import ABC, abstractmethod
from gammaraytoys.coordinates import Cartesian2D
import numpy as np
import astropy.units as u
from .event import Photon
from copy import copy, deepcopy

class Spectrum(ABC):

    @property
    @abstractmethod
    def min_energy(self):
        pass

    @property
    @abstractmethod
    def max_energy(self):
        pass
    
    @abstractmethod
    def energy_pdf(self, energy):
        # Normalized to 1
        pass

    @abstractmethod
    def random_energy(self):
        pass

class MonoenergeticSpectrum(Spectrum):

    def __init__(self, energy):
        self.energy = energy

    @property
    def min_energy(self):
        return 0*u.keV

    @property
    def max_energy(self):
        return np.inf*u.keV

    def energy_pdf(self, energy):
        if energy == energy:
            return 1
        else:
            return 0
        
    def random_energy(self):

        return self.energy
    
class Source(ABC):

    @property
    @abstractmethod
    def flux(self):
        # Total
        pass
    
    @abstractmethod
    def random_photon(self, detector):
        pass

class PointSource(Source):

    def __init__(self, offaxis_angle, spectrum, flux = None, chirality = None, chirality_degree = 1):
        """
        chirality_degree [0,1]
        flux needed for normalization
        """

        self.spectrum = spectrum
        self.chirality = chirality
        self.chirality_degree = chirality_degree

        self.offaxis_angle = offaxis_angle
        
        self._flux = flux

        #cache
        self._plane_origin = None
        self._detector = None
        
    @property
    def flux(self):
        return self._flux
    
    def random_injection_position(self, detector):

        if detector is not self._detector:
            self._detector = detector
            self._plane_origin, self._throw_parallel = detector.throwing_plane(self.offaxis_angle) 

        perp_norm_dist = np.random.uniform(-1,1)
        return  Cartesian2D(self._plane_origin.x + self._throw_parallel.x * perp_norm_dist,
                            self._plane_origin.y + self._throw_parallel.y * perp_norm_dist)
        
    def random_photon(self, detector):
        
        chirality = copy(self.chirality)
        if chirality is not None:
            if np.random.uniform() > 0.5 + self.chirality_degree/2:
                # Flip to non-dominant chirality
                chirality *= -1

        return Photon(position = self.random_injection_position(detector),
                      direction = 270*u.deg - self.offaxis_angle,
                      energy = self.spectrum.random_energy(),
                      chirality = chirality)
    
class IsotropicSource(Source):

    def __init__(self, spectrum, flux = None, chirality = None, chirality_degree = 0):

        self.detector = detector
        self.spectrum = spectrum
        self.chirality = chirality
        self.chirality_degree = chirality_degree
        self._flux = flux
        
    def random_photon(self, detector):

        direction = np.random.uniform(0,360)*u.deg
        
        ps = PointSource(offaxis_angle = direction,
                         spectrum = self.spectrum,
                         chirality = self.chirality,
                         chirality_degree = self.chirality_degree)

        return ps.random_photon(detector = self.detector)
        


        
