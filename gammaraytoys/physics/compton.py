import astropy.units as u
import numpy as np
from scipy.stats.sampling import SimpleRatioUniforms

class KleinNishina2D:
    """
    Klein Nishina. Normalized in 2D, that is, using the differential dtheta, not sin(theta) dtheta dphi as in the 3D case
    """
    
    @classmethod
    def _epsilon(cls, energy):

        return (energy/(0.51099895069*u.MeV)).value

    @classmethod
    def angle_pdf(cls, energy, angle):

        epsilon = cls._epsilon(energy)
        angle = angle.to_value(u.rad)

        return cls._angle_pdf(epsilon, angle)

    @classmethod
    def _angle_pdf(cls, epsilon, angle):

        A = epsilon**2 * (1 + 2*epsilon)**(5/2)
        epsilonCos = 1 + epsilon - epsilon * np.cos(angle)
        B = (-2 - 10*epsilon - 12*epsilon**2 + 4*epsilon**3 + 11*epsilon**4 + 2*(1 + 2*epsilon)**(5/2))
        
        return (A * (epsilonCos + 1/epsilonCos - np.sin(angle)**2)) / (np.pi * B * epsilonCos**2)

    @classmethod
    def _rvs(cls, epsilon, size = None):

        class PdfAux:
            pdf = lambda angle: cls._angle_pdf(epsilon, angle)
        
        return SimpleRatioUniforms(PdfAux, mode = 0, domain = (-np.pi, np.pi)).rvs(size)

    @classmethod
    def angle_rvs(cls, energy, size = 1):

        epsilon = cls._epsilon(energy)

        if np.isscalar(epsilon):
            return cls._rvs(epsilon, size) * u.rad
        else:
            return np.array([cls._rvs(e, size) for e in epsilon]) * u.rad
