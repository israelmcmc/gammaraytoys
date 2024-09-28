import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from gammaraytoys import Material
from astropy import units as u
from astropy.coordinates import CartesianRepresentation, Angle
from .event import Interaction, Particle, Photon, Compton, Absorption, EventList    
from gammaraytoys.physics import ComptonPhysics2D
from gammaraytoys.coordinates import Cartesian2D
from scipy.stats import norm
from copy import copy, deepcopy

class ToyTracker2D:

    def __init__(self, material, layer_length, layer_positions, layer_thickness, energy_resolution, energy_threshold):
        """
        
        """
        
        self._size = layer_length

        self._layer_pos = layer_positions

        self._material = Material.from_name(material)

        self._layer_thickness = np.broadcast_to(layer_thickness, self.nlayers, subok=True)

        self._mthick = self._layer_thickness * self.material.density
        self._energy_res = np.broadcast_to(energy_resolution, self.nlayers)
        self._energy_thresh = np.broadcast_to(energy_threshold, self.nlayers, subok = True)
        
        self._npix = (layer_length/self._layer_thickness).to_value('').astype(int)
        self._pix_size = layer_length/self._npix

        # Checks

        # Overlaps
        argsort_pos = np.argsort(layer_positions)
        sort_layer_pos = layer_positions[argsort_pos]
        sort_layer_thickness = self._layer_thickness[argsort_pos]

        gaps  = ((sort_layer_pos[1:]  - sort_layer_thickness[1:]/2) -
                 (sort_layer_pos[:-1] + sort_layer_thickness[:-1]/2))

        if np.any(gaps < 0):
            raise ValueError("Overlaps detected. Increase the space between layers or make them thinner.")
        
    @property
    def nlayers(self):
        return self._layer_pos.size

    @property
    def size(self):
        return self._size
    
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
        return self._pix_size

    @property
    def energy_resolution(self):
        return self._energy_res

    @property
    def energy_threshold(self):
        return self._energy_thresh

    @property
    def left_bound(self):
        return -self.size/2

    @property
    def right_bound(self):
        return self.size/2

    @property
    def top_bound(self):
        return np.max(self.layer_positions)
    
    @property
    def bottom_bound(self):
        return np.min(self.layer_positions)

    @property
    def height(self):
        return self.top_bound - self.bottom_bound
    
    def plot(self, ax = None, event = None):

        if ax is None:
            fig,ax = plt.subplots()

        length_unit = u.cm

        voxels = []
        for pos,layer_thickness,npix, pix_size in zip(self._layer_pos, self._layer_thickness, self._npix, self._pix_size):
            pos = pos.to_value(length_unit)
            layer_thickness = layer_thickness.to_value(length_unit)
            pix_size = pix_size.to_value(length_unit)
            for i in range(npix): 
                voxels.append(mpl.patches.Rectangle((-self.size.to_value(length_unit)/2 + i*pix_size, pos - layer_thickness/2),
                                                      pix_size, layer_thickness,
                                                     edgecolor = '.5',
                                                    facecolor = '.9', lw = 1)
                              )

        ax.add_collection(mpl.collections.PatchCollection(voxels, match_original=True))
                
        if event is not None:
            ax.text(event.position.x.to_value(length_unit),
                    event.position.y.to_value(length_unit),
                    f"$\gamma(E = {event.energy:.1f}, k = {event.chirality})$")
            hits = event.hits
            ax.text(.03,.03,f"Nhits = {hits.nhits}\nMeasured energy = {np.sum(hits.energy):.2f}",
                    transform=ax.transAxes)
            event.plot(ax, length_unit)
            
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")

        ax.set_xlim(2*self.left_bound.to_value(length_unit),
                    2*self.right_bound.to_value(length_unit))
        ax.set_ylim((self.bottom_bound - self.height/2).to_value(length_unit),
                    (self.top_bound + self.height/2).to_value(length_unit))

        ax.set_aspect('equal')
        
        return ax

    def simulate_event(self, particle):

        # We need to copy position since we need to keep track where it is, but we don't want to change the
        # initial injection position
        position = copy(particle.position)

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

            # Add measurement error to position
            pix_size = self._pix_size[layer_idx]
            measured_x = (np.floor(new_pos_x/pix_size) + 1/2)*pix_size
            measured_y = new_pos_y
            measured_pos = Cartesian2D(measured_x,
                                       measured_y)

            # Determined which interaction type we have. Only Compton or total absorption for now.
            # If pair and compton, we assume that the e- and e+ are fully absorbed.
            compton_attenuation_coeff = self.material.compton_attenuation(particle.energy)

            if np.random.uniform() < compton_attenuation_coeff / total_attenuation_coeff:
                # Compton.
                compton_physics = ComptonPhysics2D(particle.energy)
                
                # Get random direction
                scattering_angle = compton_physics.random_scattering_angle(chirality = particle.chirality)
                new_direction = particle.direction + scattering_angle
                
                # Derive the deposited energy from kinematics
                energy_out = compton_physics.energy_out(scattering_angle)
                deposited_energy = particle.energy - energy_out

                # Add measurement errors energy
                energy_res = self.energy_resolution[layer_idx] * deposited_energy.value
                measured_energy = norm.rvs(deposited_energy.value,
                                           scale = energy_res)
                measured_energy = np.maximum(0, measured_energy)
                measured_energy *= deposited_energy.unit
                
                # Add interaction to tree
                compton = Compton(position = new_pos,
                                  energy = deposited_energy)

                compton.add_parent(particle)

                if deposited_energy > self.energy_threshold[layer_idx]:
                    compton.set_measurement(layer = layer_idx,
                                            position = measured_pos,
                                            energy = measured_energy)
                
                # Add child particles (no electron, assumed fully absorbed for now)
                photon = Photon(position = new_pos,
                                direction = new_direction,
                                energy = energy_out,
                                chirality = particle.chirality)

                photon.add_parent(compton)

                # Continue simulation, iterative
                child = self.simulate_event(photon)

            else:
                # Full absorption

                # Add interaction to tree
                absorption = Absorption(position = new_pos,
                                        energy = particle.energy)

                absorption.add_parent(particle)

                # Add measurement errors energy
                measured_energy = norm.rvs(particle.energy.value,
                                           scale = self.energy_resolution[layer_idx] * particle.energy.value)
                measured_energy *= particle.energy.unit
            
                absorption.set_measurement(layer = layer_idx,
                                           position = measured_pos,
                                           energy = measured_energy)

            # Terminate
            break

        return particle






