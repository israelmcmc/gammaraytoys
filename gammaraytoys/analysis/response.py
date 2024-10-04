from histpy import Histogram
import numpy as np

class SpectralResponse(Histogram):

    @classmethod
    def open(cls, *args, **kwargs):

        h = Histogram.open(*args, **kwargs)

        if h.ndim != 2:
            raise RuntimeError("The spectral response has only 2 axes")
        if 'Em' not in h.axes.labels:
            raise RuntimeError("Missing Em axis")
        if 'Ei' not in h.axes.labels:
            raise RuntimeError("Missing Ei axis")
        
        return cls(h.axes, h.contents)
        
    @property
    def measured_energy_axis(self):
        return self.axes['Em']
        
    @property
    def photon_energy_axis(self):
        return self.axes['Ei']
    
    def effective_area(self):
        return self.project('Ei')

    def energy_dispersion(self):
        return self/self.project('Ei').contents[:,None]

    def expected_counts(self, source, duration):

        binned_spec = source.discretize_spectrum(self.photon_energy_axis)

        expected_counts = np.dot(self.contents.transpose(),
                                 duration*binned_spec.contents)
        expected_counts = expected_counts.to_value('')

        return Histogram(self.measured_energy_axis, expected_counts)

    
