import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
from astropy.units import Quantity
from scipy.stats import poisson
from histpy import Histogram, Axis
from scipy.stats import poisson, norm

class ToyCodedMaskDetector2D:

    def __init__(self, detector_axis, mask, mask_separation, detector_efficiency, shielding = 1):
        """
        Shielding = shield efficiency
        """
        
        self._mask_sep = mask_separation
        self._det_axis = detector_axis
        self._mask = mask
        self._det_eff = detector_efficiency
        self._sky_axis = None
        self._response = None
        self.shielding = shielding

    @property
    def sky_axis(self):

        if self._response is None:
            # Compute and cache

            fov = self.partially_coded_fov

            self._sky_axis =  Axis(np.arange(-fov.to_value(u.degree),
                                             fov.to_value(u.degree),
                                             self.angular_resolution.to_value(u.degree)/5) * u.degree,
                                   label = 'off_axis_angle')

        return self._sky_axis
            
    @property
    def response(self):
        if self._response is None:
            # Compute and cache

            flux = 1/u.cm/u.s
            duration = 1*u.s
            
            response = Histogram.concatenate(self.sky_axis,
                                             [self.point_source_response(flux = flux,
                                                                         angle = a,
                                                                         duration = duration,
                                                                         fluctuate = False)
                                              for a in self.sky_axis.centers])
            
            response = response.project(1,0) # Transpose

            # Give correct area units
            response = Histogram(response.axes, response.contents/flux/duration)

            self._response = response
                            
        return self._response

    def effective_area(self, angle):

        if np.abs(angle > self.fully_coded_fov):
            return 0*u.cm
        
        return np.sum(self.response[:, self.sky_axis.find_bin(angle)])
    
    @property
    def mask(self):
        return self._mask

    @property
    def detector_axis(self):
        return self._det_axis

    @property
    def detector_efficiency(self):
        return self._det_eff

    @property
    def mask_separation(self):
        return self._mask_sep

    @property
    def angular_resolution(self):
        return np.arctan(np.min(self.mask.axis.widths)/self.mask_separation)

    @property
    def mask_size(self):
        return self.mask.axis.hi_lim - self.mask.axis.lo_lim

    @property
    def detector_size(self):
        return self.detector_axis.hi_lim - self.detector_axis.lo_lim
    
    @property
    def fully_coded_fov(self):
        return np.arctan((self.mask_size/2-self.detector_size/2)/self.mask_separation)
    
    @property
    def partially_coded_fov(self):
        return np.arctan((self.mask_size/2+self.detector_size/2)/self.mask_separation)

    @classmethod
    @u.quantity_input(mask_size = u.m, mask_separation = u.m, detector_size = u.m)
    def create_random_mask(cls, mask_size, mask_npix, mask_separation, open_fraction, detector_size, detector_npix, detector_efficiency, shielding = 1):

        return cls(detector_axis = Axis(np.linspace(-detector_size/2, detector_size/2, detector_npix+1),
                                        label = 'detector_axis'), 
                   mask = Histogram(np.linspace(-mask_size/2, mask_size/2, mask_npix+1),
                                    (np.random.uniform(size = mask_npix) < open_fraction).astype(int)), 
                   mask_separation = mask_separation, 
                   detector_efficiency = detector_efficiency,
                   shielding = shielding)

    def plot(self, ax = None, data = None, angle = None):

        if ax is None:
            fig,ax = plt.subplots()

        length_unit = self.detector_axis.unit

        ax.errorbar(self._det_axis.centers.to_value(),
                    -.05*np.ones(self._det_axis.nbins), 
                    xerr = self._det_axis.widths.to_value(length_unit)/2, 
                    capsize = 2,
                    color = 'red', alpha = .2)

        # Mask
        for lo,hi,open in zip(self._mask.axis.lower_bounds.to_value(length_unit),
                              self._mask.axis.upper_bounds.to_value(length_unit),
                              self._mask.contents):
            if not open:
                ax.plot([lo, hi],
                        [self.mask_separation.to_value(length_unit), self.mask_separation.to_value(length_unit)],
                       color = 'black')

        # Shield
        if self.shielding != 1:
            line_style = '--'
        else:
            line_style = '-'
        ax.plot([self.detector_axis.lo_lim.to_value(length_unit),
                 self._mask.axis.lo_lim.to_value(length_unit)],
                [-.2, self.mask_separation.to_value(length_unit)],
                color = 'black', ls = line_style)
        ax.plot([self.detector_axis.hi_lim.to_value(length_unit),
                 self._mask.axis.hi_lim.to_value(length_unit)],
                [-.2, self.mask_separation.to_value(length_unit)],
                color = 'black', ls = line_style)
        ax.plot([self.detector_axis.lo_lim.to_value(length_unit),
                 self.detector_axis.hi_lim.to_value(length_unit)],[-.2,-.2], color = 'black')

        if data is not None:
            axr = ax.twinx()

            data.plot(axr)

            axr.set_ylim(-2*np.maximum(1, np.max(data))*.5/(self.mask_separation.to_value(length_unit) + .5), 2*np.maximum(1,np.max(data)))

            axr.set_ylabel("Expected counts")
        
        if angle is not None:
            ax.plot(self._mask.axis.lo_lim.to_value(length_unit) - np.array([self.mask_separation.to_value(length_unit) * np.tan(angle), 0]),
                                                      [0, self.mask_separation.to_value(length_unit)], 
                    color = 'grey', ls = ':')
            ax.plot(self._mask.axis.hi_lim.to_value(length_unit) - np.array([self.mask_separation.to_value(length_unit) * np.tan(angle), 0]),
                                                      [0, self.mask_separation.to_value(length_unit)],
                    color = 'grey', ls = ':')

            ax.plot(self._det_axis.lo_lim.to_value(length_unit) + np.array([0, self.mask_separation.to_value(length_unit) * np.tan(angle)]),
                                                      [0, self.mask_separation.to_value(length_unit)], 
                    color = 'red', ls = ':', alpha = .3)
            ax.plot(self._det_axis.hi_lim.to_value(length_unit) + np.array([0, self.mask_separation.to_value(length_unit) * np.tan(angle)]),
                                                      [0, self.mask_separation.to_value(length_unit)],
                    color = 'red', ls = ':', alpha = .3)

        ax.set_ylim(-.5, self.mask_separation.to_value(length_unit) + .5)
        ax.set_xlim(self._mask.axis.lo_lim.to_value(length_unit)*1.1, self._mask.axis.hi_lim.to_value(length_unit)*1.1)

        ax.set_xlabel(f"x [{length_unit}]")
        ax.set_ylabel(f"y [{length_unit}]")
        
        return ax

    @u.quantity_input(flux = u.Unit()/u.m/u.s, duration = u.s, angle = u.rad)
    def point_source_response(self, flux, duration, angle, fluctuate = False):

        if isinstance(angle, (Quantity, Angle)):
            angle = angle.to_value(u.rad)
        
        mask_proj_edges = self._mask.axis.edges - self._mask_sep * np.tan(angle)

        mask_proj_idx = self._det_axis.find_bin(mask_proj_edges)

        # Will remove units when multiplying by flux and duration
        expectation = Histogram(self._det_axis, unit = 1/flux.unit/duration.unit)

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

        # Outside the coded mask
        right_edge = self.mask.axis.hi_lim - np.tan(angle)*self.mask_separation
        right_edge_bin = self.detector_axis.find_bin(right_edge)
        if right_edge_bin < self.detector_axis.nbins:
            expectation[right_edge_bin] = (1-self.shielding)*(self.detector_axis.upper_bounds[right_edge_bin] - right_edge)
            expectation[right_edge_bin+1:] = (1-self.shielding)*self.detector_axis.widths[right_edge_bin+1:]

        left_edge = self.mask.axis.lo_lim - np.tan(angle)*self.mask_separation
        left_edge_bin = self.detector_axis.find_bin(left_edge)
        if left_edge_bin >= 0:
            expectation[left_edge_bin] = (1-self.shielding)*(left_edge - self.detector_axis.lower_bounds[left_edge_bin])
            if left_edge_bin > 0:
                expectation[:left_edge_bin-1] = (1-self.shielding)*self.detector_axis.widths[:left_edge_bin-1]

        # Weight by exposure
        expectation *= flux * duration * np.cos(angle) * self._det_eff

        expectation.clear_underflow_and_overflow()

        # Convert Quantity to np array  (it should be already unitless)
        expectation = Histogram(expectation.axis, expectation.contents.to('').value)
        
        if fluctuate:
            expectation[:] = poisson.rvs(mu = expectation.contents)
        
        return expectation

    @u.quantity_input(model = u.Unit()/u.m/u.s, duration = u.s)
    def convolve_model(self, model, duration, fluctuate = True):

        expectation = duration*np.dot(self.response.contents, model.contents)
        
        expectation = Histogram(self._det_axis,
                                contents = expectation.to('').value)

        return expectation

    @u.quantity_input(flux = u.Unit()/u.m/u.s, angle = u.rad, width = u.rad)
    def gaussian_model(self, flux, angle, width):

        #Prevent numerical error from point sources
        width = np.maximum(self.angular_resolution/1e6, width) 
        
        # Factor 5 ang res, somewhat arbitrary
        model = Histogram(self.sky_axis, unit = flux.unit)

        norm_cdf = norm.cdf(model.axis.edges.to_value(u.rad),
                            loc = angle.to_value(u.rad),
                            scale = width.to_value(u.rad))
        model[:] = flux * (norm_cdf[1:] - norm_cdf[:-1])

        return model

    @u.quantity_input(rate = u.Hz, duration = u.s)
    def uniform_bkg(self, rate, duration):

        bkg = Histogram(self.detector_axis)

        bkg[:] = bkg.axis.widths.value
        
        bkg *= (rate*duration).to('').value / np.sum(bkg)

        return bkg
