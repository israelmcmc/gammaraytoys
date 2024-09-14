import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import astropy.units as u
import numpy as np
from matplotlib import pyplot as plt

class Material:

    def __init__(self, attenuation,
                 energy_unit = u.MeV, coeff_unit = u.cm*u.cm/u.g):
        
        self._att_coeff = attenuation
        self._energy_unit = energy_unit
        self._coeff_unit = coeff_unit


        self._att_energy = self._att_coeff['energy'].to_numpy() * self._energy_unit
        
        log_energy = np.log(self._att_coeff['energy'])
        
        self._photo_loginterp = interp1d(log_energy,
                                         np.log(self._att_coeff['photo']))
        
        self._pair_loginterp = interp1d(log_energy,
                                         np.log(self._att_coeff['pair']))
        
        self._compton_loginterp = interp1d(log_energy,
                                           np.log(self._att_coeff['compton']))
        
    @classmethod
    def from_name(cls, name):

        if isinstance(name, cls):
            return name
        
        this_file_path = Path(__file__).parent.resolve()

        return cls(pd.read_csv(this_file_path/f"data/attenuation_coefficients/{name}.txt",
                    sep = '\s+', header = 0, comment = '#'))

    def photo_attenuation(self, energy):

        log_energy = np.log(energy.to_value(self._energy_unit))

        return np.exp(self._photo_loginterp(log_energy))*self._coeff_unit
    
    def pair_attenuation(self, energy):

        log_energy = np.log(energy.to_value(self._energy_unit))

        return np.exp(self._pair_loginterp(log_energy))*self._coeff_unit

    def compton_attenuation(self, energy):

        log_energy = np.log(energy.to_value(self._energy_unit))

        return np.exp(self._compton_loginterp(log_energy))*self._coeff_unit

    def plot_attenuation(self, ax = None):

        if ax is None:
            fig,ax = plt.subplots()

        ax.plot(self._att_energy.to_value(self._energy_unit),
                self.photo_attenuation(self._att_energy).to_value(self._coeff_unit),
                label = "Photoelectric absorption")

        ax.plot(self._att_energy.to_value(self._energy_unit),
                self.pair_attenuation(self._att_energy).to_value(self._coeff_unit),
                label = "Pair production")

        ax.plot(self._att_energy.to_value(self._energy_unit),
                self.compton_attenuation(self._att_energy).to_value(self._coeff_unit),
                label = "Compton")

        ax.legend()

        ax.set_xlabel(f"Energy [{self._energy_unit}]")
        ax.set_ylabel(f"Mass attenuation coefficient [{self._coeff_unit}]")

        ax.set_xscale('log')
        ax.set_yscale('log')

        return ax