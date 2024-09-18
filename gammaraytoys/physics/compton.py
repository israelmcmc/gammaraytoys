import astropy.units as u
import numpy as np
from scipy.stats.sampling import SimpleRatioUniforms

class ComptonPhysics2D:
    """
    Klein Nishina. Normalized in 2D, that is, using the differential dtheta, not sin(theta) dtheta dphi as in the 3D case

    
    """

    def __init__(self, energy):

        self.energy = energy
        self.epsilon = (energy/(0.51099895069*u.MeV)).value

        class AuxPDFScatteringAngle:
            pdf = lambda angle: self._scattering_angle_pdf(angle)

        self._rvs = SimpleRatioUniforms(AuxPDFScatteringAngle,
                                        mode = 0, domain = (-np.pi, np.pi))


    def scattering_angle_pdf(self, angle):

        angle = angle.to_value(u.rad)

        return self._scattering_angle_pdf(angle)
        
    def _scattering_angle_pdf(self, angle):

        A = self.epsilon**2 * (1 + 2*self.epsilon)**(5/2)
        epsilonCos = 1 + self.epsilon - self.epsilon * np.cos(angle)
        B = (-2 - 10*self.epsilon - 12*self.epsilon**2 + 4*self.epsilon**3 + 11*self.epsilon**4 + 2*(1 + 2*self.epsilon)**(5/2))
        
        return ((A * (epsilonCos + 1/epsilonCos - np.sin(angle)**2)) /
                (np.pi * B * epsilonCos**2))

    def random_scattering_angle(self, size = None):

        return self._rvs.rvs(size) * u.rad
        
    def energy_out(self, angle):

        return self.energy / (1 + self.epsilon * (1-np.cos(angle)))
    
    def scattering_angle(self, energy_out):

        return np.arccos(1 - (self.energy/energy_out - 1)/self.epsilon)
        
