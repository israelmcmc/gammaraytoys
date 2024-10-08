import astropy.units as u
import numpy as np
from scipy.stats.sampling import SimpleRatioUniforms
from histpy import Histogram,Axis
import matplotlib.pyplot as plt

class ComptonPhysics2D:
    """
    Klein Nishina. Normalized in 2D, that is, using the differential dtheta, not sin(theta) dtheta dphi as in the 3D case

    
    """

    def __init__(self, energy):

        self.energy = energy
        self.epsilon = (energy/(0.51099895069*u.MeV)).to_value('')

        class AuxPDFScatteringAngle:
            pdf = lambda phi: self._scattering_angle_pdf(phi)

        self._rvs = SimpleRatioUniforms(AuxPDFScatteringAngle,
                                        mode = 0, domain = (-np.pi, np.pi))


    def scattering_angle_pdf(self, phi):

        phi = phi.to_value(u.rad)

        return self._scattering_angle_pdf(phi)
        
    def _scattering_angle_pdf(self, phi):

        A = self.epsilon**2 * (1 + 2*self.epsilon)**(5/2)
        epsilonCos = 1 + self.epsilon - self.epsilon * np.cos(phi)
        B = (-2 - 10*self.epsilon - 12*self.epsilon**2 + 4*self.epsilon**3 + 11*self.epsilon**4 + 2*(1 + 2*self.epsilon)**(5/2))
        
        return ((A * (epsilonCos + 1/epsilonCos - np.sin(phi)**2)) /
                (np.pi * B * epsilonCos**2))

    def random_scattering_angle(self, size = None):

        return self._rvs.rvs(size) * u.rad
        
    def energy_out(self, phi):

        return self.energy / (1 + self.epsilon * (1-np.cos(phi)))
    
    def scattering_angle(self, energy_out):

        return np.arccos(1 - (self.energy/energy_out - 1)/self.epsilon)
        
    def plot_scattering_angle_pdf(self, ax = None, **kwargs):

        theta_plot = Axis(np.linspace(-180,180, 3600)*u.deg)

        if ax is None:
            fig,ax = plt.subplots(subplot_kw={'projection': 'polar'})

        fun = self.scattering_angle_pdf(theta_plot.centers)

        ax.plot(theta_plot.centers.to_value(u.rad), fun/np.max(fun),
                **kwargs)

        return ax

    def modulation_factor(self, phi):

        phi = phi.to_value(u.rad)

        energy_out = self.energy_out(phi)
        
        return 2 * np.pow(np.sin(phi), 2) / (energy_out/self.energy + self.energy/energy_out)

    def plot_modulation_factor(self, ax = None, **kwargs):

        if ax is None:
            fig,ax = plt.subplots()

        phi = np.linspace(0, 180.001, 100)*u.deg
        ax.plot(phi, self.modulation_factor(phi), **kwargs)

        ax.set_xlabel("Polar scattering angle [deg]")
        ax.set_ylabel("Modulation factor")
        
        return ax
