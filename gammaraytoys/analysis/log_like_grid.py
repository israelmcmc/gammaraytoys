from histpy import Histogram
import tqdm
import numpy as np
import itertools
from scipy.stats import chi2
import astropy.units as u

class LogLikeGrid(Histogram):
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.ndof = self.ndim
    
    def compute(self, fun):
        
        for prod in tqdm(itertools.product(*[enumerate(a.centers) for a in self.axes]), total = np.prod(self.nbins)):
            
            idx,values = zip(*prod)
            
            self[idx] = fun(*values)
    
    def maximum(self, axis = None):
        """
        Returns either float (axis=None) or Histogram
        """
        
        if axis is None:
            return np.max(self)
        else:
            axis = tuple(np.array(self.axes.label_to_index(axis), ndmin = 1))
            
            kept_axis = tuple([a for a in range(self.ndim) if a not in axis])
        
            profile_like = LogLikeGrid([self.axes[a] for a in kept_axis],
                                       np.max(self.contents, axis = axis))
            
            # Same as the original
            profile_like.ndof = self.ndof
            
            return profile_like
        
    def max_ts(self):
        return 2*self.maximum()
    
    def get_ts_delta(self, cont = .9):
        return chi2.ppf(cont, df = self.ndof)
    
    def get_ts_thresh(self, cont = .9):
        return self.max_ts() - chi2.ppf(cont, df = self.ndof)
        
    def optimal_parameters(self):
        """
        """
        
        argmax = np.unravel_index(np.argmax(self), self.nbins)

        popt = [self.axes[i].centers[argmax[i]] for i in range(self.ndim)]
              
        if self.ndim == 1:
            return popt[0]
        else:
            return popt
        
    def contour_indices(self, cont = .9):
        
        return np.nonzero(self >= self.get_ts_thresh(cont)/2)
        
    def contour_indices_values(self, cont= .9):
        
        contour_idx = self.contour_indices(cont)
        
        return contour_idx,[([self.axes[j].centers[i] for j,i in enumerate(idx)]) for idx in zip(*contour_idx)]
        
    def parameter_bounds(self, cont = .9):
        """
        """
        
        contour_idx = self.contour_indices(cont)
        
        upper_bounds = [self.axes[i].upper_bounds[np.max(contour_idx[i])] for i in range(self.ndim)]
        lower_bounds = [self.axes[i].lower_bounds[np.min(contour_idx[i])] for i in range(self.ndim)]

        bounds = list(zip(lower_bounds, upper_bounds))
        
        if self.ndim == 1:
            return bounds[0]
        else:
            return bounds
            
    def plot_ts(self, ax = None, cont = .9, *args, **kwargs):
        
        if self.ndim > 2:
            raise RuntimeError("Can only plot 1D or 2D. Slice or profile likelihood")
        
        popt = self.optimal_parameters()
        
        max_ts = 2*self.maximum()
        
        contour_ts_delta = self.get_ts_delta(cont)
        contour_ts_thresh = max_ts - contour_ts_delta
        
        if self.ndim == 1:
            ax,_ = (2*self).plot(ax, *args, **kwargs)
            
            ax.axvline(u.Quantity(popt).value, color = 'red', label = 'Best estimate')
            
            pbounds = self.parameter_bounds(cont)
        
            ax.axvline(u.Quantity(pbounds[0]).value, color = 'red', ls = ":", label = 'Uncertainty bounds')
            ax.axvline(u.Quantity(pbounds[1]).value, color = 'red', ls = ":")
            
            ax.set_ylim(max_ts - 10*contour_ts_delta, max_ts + 3*contour_ts_delta)
        else: 
            #self.ndim == 2
            ax,_ = (2*self).plot(ax, *args, vmin = contour_ts_thresh, **kwargs)
            ax.scatter(*popt, color = 'red', label = 'Best estimate')
            ax.contour(self.axes[0].centers,
                       self.axes[1].centers,
                       2*self.contents.transpose(),
                       levels = [contour_ts_thresh],
                       colors = 'red', ls = ':')
            
        ax.legend()
        
