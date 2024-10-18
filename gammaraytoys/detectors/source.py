from abc import ABC, abstractmethod
from gammaraytoys.coordinates import Cartesian2D
import numpy as np
import astropy.units as u
from .event import Photon
from copy import copy, deepcopy
import matplotlib.pyplot as plt
from scipy.stats.sampling import SimpleRatioUniforms, NumericalInverseHermite
from histpy import Histogram, Axis

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
    def pdf(self, energy):
        # Normalized to 1
        pass

    @abstractmethod
    def cdf(self, energy):
        # Normalized to 1
        pass

    def integrate(self, lo_energy, hi_energy):
        return self.cdf(hi_energy) - self.cdf(lo_energy)
    
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

    def pdf(self, energy):
        raise ValueError("Do not use PDF for Mono, only CDF")
        
    def cdf(self, energy):
        return (np.array(energy >= self.energy, dtype = int)+1)/2

    def random_energy(self):

        return self.energy

class PowerLawSpectrum(Spectrum):

    def __init__(self, index, min_energy, max_energy):
        self.index = index
        self._min_energy = min_energy
        self._eunit = min_energy.unit
        self._max_energy = max_energy.to(self._eunit)

        if self.index == -1:
            # Special case
            self._norm = 1/min_energy/np.log(max_energy/min_energy)
        else:
            self._norm = ((1+index)/(max_energy*np.power(max_energy/min_energy, index)-min_energy)).to(1/self._eunit)
        
        class AuxEnergyPDF:
            pdf = lambda energy: self._pdf(energy)
            cdf = lambda energy: self._cdf(energy)

        self._rvs = NumericalInverseHermite(AuxEnergyPDF,
                                            domain = (self.min_energy.value,
                                                      self.max_energy.value))
        
    @property
    def min_energy(self):
        return self._min_energy

    @property
    def max_energy(self):
        return self._max_energy

    def _log_pdf(self, log_energy):
        return (self.index * (log_energy - np.log(self.min_energy.value)) + np.log(self._norm.value))/(self.index * (np.log(self.max_energy.value) - np.log(self.min_energy.value)) + 2*np.log(self._norm.value))
    
    def _pdf(self, energy):
        # in min_energy units
        values = self._norm.value*np.power(energy/self.min_energy.value, self.index)

        if np.ndim(values) == 0:
            if energy > self.max_energy.value or energy < self.min_energy.value:
                values = 0
        else:
            values[energy < self.min_energy.value] = 0
            values[energy > self.max_energy.value] = 0
             
        return values
    
    def random_energy(self, size = None):

        return self._rvs.rvs(size) * self.min_energy.unit

    def pdf(self, energy):

        return self._pdf(energy.to_value(self._eunit)) * self._norm.unit

    def _cdf(self, energy):
        if self.index == -1:
            # Special case
            cumm = self._norm*self.min_energy*np.log(energy/self.min_energy.value)
            cumm = cumm.to_value('')
        else:
            cumm = self._norm.value*(energy*np.power(energy/self.min_energy.value, self.index)-self.min_energy.value)/(1+self.index)
            
        if np.ndim(cumm) == 0:
            if energy < self.min_energy.value:
                cumm = 0
            elif energy > self.max_energy.value:
                cumm = 1
        else:
            cumm[energy < self.min_energy.value] = 0
            cumm[energy > self.max_energy.value] = 1
        
        return cumm
            
    def cdf(self, energy):

        return self._cdf(energy.to_value(self._eunit))        


class MultiComponentSpectrum(Spectrum):
    
    def __init__(self, *components, weights = None):
        
        if weights is None:
            self.weights = np.ones(len(components))
        else:
            self.weights = np.array(weights, dtype = float)
            
        self.weights /= np.sum(self.weights)
            
        self.components = components
        
        self._min_energy = np.min(u.Quantity([c.min_energy for c in components]))
        self._max_energy = np.max(u.Quantity([c.max_energy for c in components]))
        
    def ncomponents(self):
        return len(self.components)
        
    @property
    def min_energy(self):
        return self._min_energy

    @property
    def max_energy(self):
        return self._max_energy

    def random_energy(self, size = None):

        component_idx = np.random.choice(self.ncomponents, size = size, p = self.weights)
        
        energies = []
        
        for ncomponent in range(self.ncomponents):
            
            nsamples = np.sum(component_idx == ncomponent)
            
            energies.append(self.components[component_idx].random_energy(size = nsamples))
            
        energies = u.Quantity(np.shuffle(energies))
        
        return energies

    def pdf(self, energy):

        prob = u.Quantity([w*c.pdf(energy) for c,w in zip(self.components, self.weights)])
        
        prob = np.sum(prob, axis = None if np.ndim(energy) == 0 else 0)
        
        return prob

    def cdf(self, energy):

        cdf = [w*c.cdf(energy) for c,w in zip(self.components, self.weights)]
        
        cdf = np.sum(cdf, axis = None if np.ndim(energy) == 0 else 0)
        
        return cdf
    
    
class Source(ABC):

    @property
    @abstractmethod
    def flux(self):
        # Total
        pass
    
    @property
    @abstractmethod
    def spectrum(self):
        pass
    
    def diff_flux(self, energy):
        return self._flux * self.spectrum.pdf(energy)

    def integrate_flux(self, lo_energy, hi_energy):
        return self._flux * self.spectrum.integrate(lo_energy, hi_energy)

    def discretize_spectrum(self, axis):

        binned_spec = Histogram(axis,
                                unit = self.flux.unit,
                                contents = self.integrate_flux(axis.lower_bounds,
                                                               axis.upper_bounds)
                                )
        
        return binned_spec
    
    def plot_spectrum(self, ax = None, e2 = False,
                      energy_units = None, y_units = None,
                      discretize_axis = None, 
                      **kwargs):

        if self.flux is None:
            raise RuntimeError("Set a flux before plotting the spectrum")

        if ax is None:
            fig,ax = plt.subplots()

        if isinstance(self.spectrum, MonoenergeticSpectrum):
            raise RuntimeError("Can't plot monoenergetic spectrum")

        if energy_units is None:
            energy_units = u.MeV
        else:
            energy_units = u.Unit(energy_units)

        if discretize_axis is None:
            energy = np.geomspace(self.spectrum.min_energy, self.spectrum.max_energy, 1000).to(energy_units)
            y = self.diff_flux(energy)
        else:
            discretize_axis = Axis(discretize_axis)
            energy = discretize_axis.centers
            binned_spec = self.discretize_spectrum(discretize_axis)
            y = binned_spec / binned_spec.axis.widths
        
        if e2:
            if y_units is None:
                y_units = u.Unit(u.erg/u.cm/u.s)
            else:
                y_units = u.Unit(y_units)
                
            y *= energy**2
            y_label = f'$E^2 dN/dE$ [{y_units}]'
        else:
            if y_units is None:
                y_units = u.Unit(1/u.erg/u.cm/u.s)
            else:
                y_units = u.Unit(y_units)

            y_label = f'$dN/dE$ [{y_units}]'

        y = y.to(y_units)

        if discretize_axis is None:
            ax.plot(energy.value, y.value, **kwargs)
        else:
            y.plot(ax, **kwargs)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel(f'Energy [{energy_units}]')

        ax.set_ylabel(y_label)

        return ax
        
    @abstractmethod
    def random_photon(self, detector):
        pass

class PointSource(Source):

    def __init__(self, offaxis_angle, spectrum,
                 flux = None, flux_pivot = None, pivot_energy = None,
                 chirality = None, chirality_degree = 1):
        """
        chirality_degree [0,1]
        flux needed for normalization
        """

        self._spectrum = spectrum
        self.chirality = chirality
        self.chirality_degree = chirality_degree

        self.offaxis_angle = offaxis_angle

        if flux is not None:
            self._flux = flux
        else:
            if flux_pivot is None or pivot_energy is None:
                self._flux = None
            else:
                self._flux = (flux_pivot/spectrum.pdf(pivot_energy)).to(1/u.cm/u.s)
                
        #cache
        self._plane_origin = None
        self._detector = None
        
    @property
    def flux(self):
        return self._flux

    @property
    def spectrum(self):
        return self._spectrum

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

        self._spectrum = spectrum
        self.chirality = chirality
        self.chirality_degree = chirality_degree
        self._flux = flux
        
    @property
    def flux(self):
        return self._flux

    @property
    def spectrum(self):
        return self._spectrum

    def random_photon(self, detector):

        direction = np.random.uniform(0,360)*u.deg
        
        ps = PointSource(offaxis_angle = direction,
                         spectrum = self.spectrum,
                         chirality = self.chirality,
                         chirality_degree = self.chirality_degree)

        return ps.random_photon(detector = detector)
        


        
