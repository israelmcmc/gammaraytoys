import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.units import Quantity
from scipy.stats import poisson
from histpy import Histogram, Axis

class ToyCodedMaskDetector2D:

    def __init__(self, detector_axis, mask, mask_separation, detector_efficiency):
        
        self._mask_sep = mask_separation
        self._det_axis = detector_axis
        self._mask = mask
        self._det_eff = detector_efficiency

    @property
    def mask(self):
        return self._mask
    
    @classmethod
    def create_random_mask(cls, mask_size, mask_npix, mask_separation, open_fraction, detector_size, detector_npix, detector_efficiency):

        return cls(detector_axis = Axis(np.linspace(-detector_size/2, detector_size/2, detector_npix+1)), 
                   mask = Histogram(np.linspace(-mask_size/2, mask_size/2, mask_npix+1),
                                    (np.random.uniform(size = mask_npix) < open_fraction).astype(int)), 
                   mask_separation = mask_separation, 
                   detector_efficiency = detector_efficiency)

    def plot(self, ax = None, expectation = None, angle = None):

        if ax is None:
            fig,ax = plt.subplots()

        ax.errorbar(self._det_axis.centers, -.05*np.ones(self._det_axis.nbins), 
                    xerr = self._det_axis.widths/2, 
                    capsize = 2,
                    color = 'red', alpha = .2)

        # Mask
        for lo,hi,open in zip(self._mask.axis.lower_bounds, self._mask.axis.upper_bounds, self._mask.contents):
            if not open:
                ax.plot([lo, hi],
                        [self._mask_sep, self._mask_sep],
                       color = 'black')

        # Tube
        ax.plot([self._mask.axis.lo_lim, self._mask.axis.lo_lim],[-.2, self._mask_sep],
                color = 'black')
        ax.plot([self._mask.axis.hi_lim, self._mask.axis.hi_lim],[-.2, self._mask_sep],
                color = 'black')
        ax.plot([self._mask.axis.lo_lim,self._mask.axis.hi_lim],[-.2,-.2], color = 'black')

        if expectation is not None:
            axr = ax.twinx()

            expectation.plot(axr)
            
            axr.set_ylim(-2*np.max(expectation)*.5/(self._mask_sep + .5), 2*np.max(expectation))

            axr.set_ylabel("Expected counts")
        
        if angle is not None:
            ax.plot(self._mask.axis.lo_lim - np.array([self._mask_sep * np.tan(angle), 0]),
                                                      [0, self._mask_sep], 
                    color = 'grey', ls = ':')
            ax.plot(self._mask.axis.hi_lim - np.array([self._mask_sep * np.tan(angle), 0]),
                                                      [0, self._mask_sep],
                    color = 'grey', ls = ':')

            ax.plot(self._det_axis.lo_lim + np.array([0, self._mask_sep * np.tan(angle)]),
                                                      [0, self._mask_sep], 
                    color = 'red', ls = ':', alpha = .3)
            ax.plot(self._det_axis.hi_lim + np.array([0, self._mask_sep * np.tan(angle)]),
                                                      [0, self._mask_sep],
                    color = 'red', ls = ':', alpha = .3)

        ax.set_ylim(-.5, self._mask_sep + .5)
        ax.set_xlim(self._mask.axis.lo_lim*1.1, self._mask.axis.hi_lim*1.1)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        return ax

    def point_source_response(self, flux, angle, fluctuate = False):

        if isinstance(angle, (Quantity, Angle)):
            angle = angle.to_value(u.rad)
        
        mask_proj_edges = self._mask.axis.edges - self._mask_sep * np.tan(angle)

        mask_proj_idx = self._det_axis.find_bin(mask_proj_edges)

        expectation = Histogram(self._det_axis)

        for mask_bin,(det_bin_i,det_bin_f) in enumerate(zip(mask_proj_idx[:-1], mask_proj_idx[1:])):

            if det_bin_f < 0 or det_bin_i >= self._det_axis.nbins or self.mask[mask_bin] == 0: 
                continue
            
            # Middle bins. Full
            expectation[det_bin_i+1: det_bin_f] += self.mask[mask_bin] * self._det_axis.widths[det_bin_i+1: det_bin_f]

            # Lower edge
            if det_bin_i == -1:
                upper_bound = self._det_axis.lo_lim
            else:
                upper_bound = self._det_axis.upper_bounds[det_bin_i]
                
            expectation[det_bin_i] += self.mask[mask_bin] * (upper_bound - mask_proj_edges[mask_bin])

            # Upper edge
            if det_bin_f == self._det_axis.nbins:
                lower_bound = self._det_axis.lo_lim
            else:
                lower_bound = self._det_axis.lower_bounds[det_bin_f]
                
            expectation[det_bin_f] += self.mask[mask_bin] * (mask_proj_edges[mask_bin+1] - lower_bound)

        expectation *= flux * np.cos(angle) * self._det_eff

        expectation.clear_underflow_and_overflow()

        if fluctuate:
            expectation[:] = poisson.rvs(mu = expectation.contents)
        
        return expectation

    def convolve_model(self, model, fluctuate = True):

        expectation = Histogram(self._det_axis)

        for angle, flux in zip(model.axis.centers, model.contents):

            expectation += self.point_source_response(flux, angle, fluctuate)

        return expectation
            
