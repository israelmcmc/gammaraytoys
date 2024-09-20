import matplotlib.pyplot as plt
import numpy as np
from gammaraytoys import Material
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, Angle
from .event import Interaction, Particle, Photon, Compton, Absorption, EventList    
from gammaraytoys.physics import ComptonPhysics2D
from gammaraytoys.coordinates import Cartesian2D
from scipy.stats import norm
from copy import copy, deepcopy

class ToyLayeredTracker2D:

    def __init__(self, size, layer_positions, mass_thickness, energy_resolution,  position_resolution, material = 'Ge'):
        """
        
        """

        self._size = size
        self._area = size*size

        self._layer_pos = layer_positions
        self._mthick = np.broadcast_to(mass_thickness, self.nlayers, subok=True)
        self._energy_res = np.broadcast_to(energy_resolution, self.nlayers)
        self._pos_res = np.broadcast_to(position_resolution, self.nlayers, subok=True)

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
    
    def plot(self, ax = None, event = None):

        if ax is None:
            fig,ax = plt.subplots()

        length_unit = u.cm
            
        for pos in self._layer_pos:
            ax.plot([-self._size.to_value(length_unit)/2, self._size.to_value(length_unit)/2],
                    [pos.to_value(length_unit), pos.to_value(length_unit)],
                    color = 'black')

        if event is not None:
            event.plot(ax, length_unit)
            
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
            
        return ax

    def sim_event(self, particle):

        # We need to copy position since we need to keep track where it is, but we don't want to change the
        # initial injection position
        position = copy(particle.position)

        nhits = 0
        
        while True:
            
            flying_up = particle.direction < 180*u.deg
            flying_down = not flying_up
            flying_right = particle.direction < 90*u.deg or particle.direction > 270*u.deg
            flying_left = not flying_right

            # Terminate events flying out of boundaries
            if ((position.x >= self.right_bound and flying_right)
                or
                (position.x <= self.left_bound  and flying_left)
                or
                (position.y >= self.top_bound and flying_up)
                or
                (position.y <= self.bottom_bound and flying_down)):
                break

            # Determine interaction location
            new_pos_x = position.x + (self.layer_positions - position.y)/np.tan(particle.direction)

            # Check only the crosses within the detector, along the flying direction,
            # and excluding the current layer (if the particle starts exactly at a layer)
            y_dist_to_layers = self.layer_positions - position.y

            crossed_tracker_idx = np.where((new_pos_x < self.right_bound) &
                                           (new_pos_x > self.left_bound) &
                                           (y_dist_to_layers > 0 if flying_up else y_dist_to_layers < 0)
                                           )[0]
            
            y_dist_to_crosses = y_dist_to_layers[crossed_tracker_idx] * (-1 if flying_down else 1)

            if y_dist_to_crosses.size == 0:
                # No interactions, flew in between layers
                break
            
            layer_idx_crossed = np.argmin(y_dist_to_crosses)
            
            layer_idx = crossed_tracker_idx[layer_idx_crossed].item()

            new_pos_x = new_pos_x[layer_idx]
            new_pos_y = self.layer_positions[layer_idx]

            new_pos = Cartesian2D(new_pos_x, new_pos_y)

            # Determine if it interacted based on the total attenuation coefficient
            total_attenuation_coeff = self.material.total_attenuation(particle.energy)
            interaction_prob = np.exp(self.mass_thickness[layer_idx] * total_attenuation_coeff)

            if np.random.uniform() > interaction_prob:
                # Didn't interact. Continues flying
                position = new_pos
                continue

            nhits += 1
            
            # Add measurement error to position
            pos_res = self.position_resolution[layer_idx]
            measured_x = norm.rvs(new_pos_x.to_value(pos_res.unit),
                                      scale = pos_res.value)
            measured_y = norm.rvs(new_pos_y.to_value(pos_res.unit),
                                      scale = pos_res.value)
            measured_pos = Cartesian2D(measured_x * pos_res.unit,
                                       measured_y * pos_res.unit)

            # Determined which interaction type we have. Only Compton or total absorption for now.
            # If pair and compton, we assume that the e- and e+ are fully absorbed.
            compton_attenuation_coeff = self.material.compton_attenuation(particle.energy)

            if np.random.uniform() < compton_attenuation_coeff / total_attenuation_coeff:
                # Compton.
                compton_physics = ComptonPhysics2D(particle.energy)
                
                # Get random direction
                scattering_angle = compton_physics.random_scattering_angle()
                new_direction = particle.direction + scattering_angle
                
                # Derive the deposited energy from kinematics
                energy_out = compton_physics.energy_out(scattering_angle)
                deposited_energy = particle.energy - energy_out

                # Add measurement errors energy
                measured_energy = norm.rvs(deposited_energy,
                                           scale = self.energy_resolution[layer_idx] * deposited_energy)
                
                # Add interaction to tree
                compton = Compton(position = new_pos,
                                  energy = deposited_energy)

                compton.set_measurement(layer = layer_idx,
                                        position = measured_pos,
                                        energy = measured_energy)
                
                compton.add_parent(particle)

                # Add child particles (no electron, assumed fully absorbed for now)
                photon = Photon(position = new_pos,
                                direction = new_direction,
                                energy = energy_out)

                photon.add_parent(compton)

                # Continue simulation, iterative
                child, nhits_child = self.sim_event(photon)

                nhits += nhits_child

            else:
                # Full absorption

                # Add interaction to tree
                absorption = Absorption(position = new_pos,
                                        energy = particle.energy)

                absorption.add_parent(particle)

                # Add measurement errors energy
                measured_energy = norm.rvs(particle.energy,
                                           scale = self.energy_resolution[layer_idx] * particle.energy)
            
                absorption.set_measurement(layer = layer_idx,
                                           position = measured_pos,
                                           energy = measured_energy)

            # Terminate
            break

        return particle, nhits






