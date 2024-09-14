import matplotlib.pyplot as plt
import numpy as np
from gammaraytoys import Material
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, Angle

class ToyLayeredTracker2D:

    def __init__(self, size, layer_positions, mass_thickness, energy_resolution,  position_resolution, material = 'Ge'):
        """
        
        """

        self._size = size
        self._area = size*size

        self._layer_pos = layer_positions
        self._mthick = np.broadcast_to(mass_thickness, self.nlayers)
        self._energy_res = np.broadcast_to(energy_resolution, self.nlayers)
        self._pos_res = np.broadcast_to(position_resolution, self.nlayers)

        #Sort layers
        asort = np.argsort(self._layer_pos)
        self._layer_pos = self._layer_pos[asort]
        self._mthick = self._mthick[asort]
        self._energy_res = self._energy_res[asort]
        self._pos_res = self._pos_res[asort]
        
        self._material = Material.from_name(material)
        
    @property
    def nlayers(self):
        return self._layer_pos.size

    @property
    def size(self):
        return self._size
    
    @property
    def instrumented_area(self):
        return self.size*self.size

    @property
    def material(self):
        return self._material

    @property
    def layer_positions(self):
        return self._layer_pos

    @property
    def mass_thickness(self):
        return self._mthick

    @property
    def position_resolution(self):
        return self._pos_res

    @property
    def energy_resolution(self):
        return self._energy_res

    @property
    def left_bound(self):
        return -self.size/2

    @property
    def right_bound(self):
        return self.size/2

    @property
    def top_bound(self):
        return self.layer_positions[-1]
    
    @property
    def bottom_bound(self):
        return self.layer_positions[0]
    
    def plot(self, ax = None):

        if ax is None:
            fig,ax = plt.subplots()

        plot_unit = u.cm
            
        for pos in self._layer_pos:
            ax.plot([-self._size.to_value(plot_unit)/2, self._size.to_value(plot_unit)/2],
                    [pos.to_value(plot_unit), pos.to_value(plot_unit)],
                    color = 'black')

        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
            
        return ax

    def sim_event(self, position, direction, energy):

        # Standarize inputs format
        if not isinstance(position, CartesianRepresentation):

            position = u.Quantity(position, ndmin = 2)

            position_list = CartesianRepresentation(x = position[:,0],
                                                    y = position[:,1],
                                                    z = 0*u.cm)

        nevents = position_list.size
        position_list = np.broadcast_to(position_list, nevents)    
        direction_list = np.broadcast_to(Angle(direction).wrap_at(360*u.deg),
                                         nevents, subok=True)
        energy_list = np.broadcast_to(energy, nevents, subok=True) 

        # Sims
        for position, direction, energy in zip(position_list, direction_list, energy_list):
            """
            Direction is photon direction. Wrt x axis, counter-clockwise
            """

            layer_idx = None # First iteration
            
            while True:

                flying_up = direction < 180*u.deg
                flying_right = direction < 90*u.deg or direction > 270*u.deg
                
                # Terminate horizontal events
                if direction == 0*u.deg or direction == 180*u.deg:
                    break

                # Terminate events flying out of boundaries
                if ((position.x >= self.right_bound and flying_right)
                    or
                    (position.x <= self.left_bound  and not flying_right)
                    or
                    (position.y >= self.top_bound and flying_up)
                    or
                    (position.y <= self.bottom_bound and not flying_up)):
                    break

                # Determine interaction
                if layer_idx is None:
                    # First iteration
                    new_pos_x = position.x + (self.layer_positions - position.y)/np.tan(direction)

                    crossed_tracker_idx = np.where((new_pos_x < self.right_bound) & (new_pos_x > self.left_bound))[0]
                    y_dist_to_crosses = self.layer_positions[crossed_tracker_idx] - position.y
                    
                    if flying_up:
                        layer_idx_crossed = np.argmin(y_dist_to_crosses)
                    else:
                        layer_idx_crossed = np.argmax(y_dist_to_crosses)

                    layer_idx = crossed_tracker_idx[layer_idx_crossed]

                    print(layer_idx)
                else:
                    if flying_up:
                        layer_idx += 1
                    else:
                        layer_idx -= 1

                # Temp. Remove
                break






        
  
