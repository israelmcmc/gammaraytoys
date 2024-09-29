from abc import ABC, abstractmethod
from gammaraytoys.coordinates import Cartesian2D
import numpy as np
import astropy.units as u
from .event import Photon

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
    def relative_flux(self, energy):
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

    def relative_flux(self, energy):
        if energy == energy:
            return 1
        else:
            return 0
        
    def random_energy(self):

        return self.energy
    
class Source(ABC):

    @abstractmethod
    def random_photon(self):
        pass

class PointSource(Source):

    def __init__(self, detector, offaxis_angle, spectrum, chirality = None, chirality_degree = 0):
        """
        chirality_degree [0,1]
        """

        self.spectrum = spectrum
        self.chirality = chirality
        self.chirality_degree = chirality_degree

        self._surr_center, self._surr_radius = detector.surrounding_circle()

        self.offaxis_angle = offaxis_angle

        self._cart_angle = 90*u.deg - self.offaxis_angle
        
        self._norm_surr = Cartesian2D(self._surr_radius*np.cos(self._cart_angle),
                                      self._surr_radius*np.sin(self._cart_angle))

        self._perp_origin = Cartesian2D(self._surr_center.x + self._norm_surr.x,
                                        self._surr_center.y + self._norm_surr.y)
        
        self._perp_surr = Cartesian2D(-self._norm_surr.y,
                                        self._norm_surr.x)
        
    def random_injection_position(self):

        perp_norm_dist = np.random.uniform(-1,1)
        return  Cartesian2D(self._perp_origin.x + self._perp_surr.x * perp_norm_dist,
                            self._perp_origin.y + self._perp_surr.y * perp_norm_dist)
        
    def random_photon(self):

        chirality = self.chirality
        if chirality is not None:
            if np.random.uniform() > 0.5 + self.chirality_degree/2:
                # Flip to non-dominant chirality
                chirality *= -1
        
        return Photon(position = self.random_injection_position(),
                      direction = 270*u.deg - self.offaxis_angle,
                      energy = self.spectrum.random_energy(),
                      chirality = chirality)
    
class IsotropicSource(Source):

    def __init__(self, detector, spectrum, chirality = None, chirality_degree = 0):

        self.detector = detector
        self.spectrum = spectrum
        self.chirality = chirality
        self.chirality_degree = chirality_degree
        
    def random_photon(self):

        direction = np.random.uniform(0,360)*u.deg
        
        ps = PointSource(detector = self.detector,
                         offaxis_angle = direction,
                         spectrum = self.spectrum,
                         chirality = self.chirality,
                         chirality_degree = self.chirality_degree)

        return ps.random_photon()
        


        
