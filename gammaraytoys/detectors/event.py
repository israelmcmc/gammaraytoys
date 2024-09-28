import numpy as np
from astropy.coordinates import CartesianRepresentation, Angle
from astropy import units as u
import yaml
from gammaraytoys.coordinates import Cartesian2D
from copy import copy, deepcopy

class EventList:

    def __init__(self):
        
        self._events = []
        self.nsim = None
        self.sim_time = None

    def __getitem__(self, key):
        return self._events[key]
    
    def append(self, event):
        self._events.append(event)

    def write(self, filename):
        """
        
        """

        with open(filename, 'w') as f:
            yaml.dump(dict(nsim = self.nsim,
                           sim_time = self.sim_time,
                           events = [{'nevent':n} | e.to_dict() for n,e in enumerate(self._events)]),
                      f, sort_keys=False)

class Hits:

    def __init__(self, layer = None, position = None, energy = None):

        if layer is None:

            if position is not None or energy is not None:
                RuntimeError("Layer, position and energy should have the same size")
            
            layer = np.array([], dtype = int)
            position = Cartesian2D([]*u.mm, []*u.mm)
            energy = []*u.MeV
        
        self.layer = layer
        self.position = position
        self.energy = energy

    @classmethod
    def from_list(cls, hits):

        layer = []
        position = []
        energy = []
        
        for hit in hits:

            layer += [hit.layer]
            position += [hit.position]
            energy += [hit.energy]

        if len(layer) == 0:
            return cls()
            
        layer = np.array(layer)

        pos_unit = position[0].x.unit

        x = [p.x.to_value(pos_unit) for p in position]
        y = [p.y.to_value(pos_unit) for p in position]
        
        position = Cartesian2D(x*pos_unit, y*pos_unit)

        energy = u.Quantity(energy)

        return cls(layer, position, energy)
        
    @property
    def nhits(self):
        return self.layer.size

    def __repr__(self):
        return f"Hits({self.layer}, {self.position}, {self.energy})"
    
class Particle:

    def __init__(self, particle_type, position, direction, energy):

        self.particle_type = particle_type
        self.position = position
        self.direction = u.Quantity(Angle(direction).wrap_at(360*u.deg))
        self.energy = energy
        self.interaction = None
        self.reco = None

    @property
    def hits_iter(self):

        if self.interaction is not None:
            for hit in self.interaction.hits_iter:
                yield hit

    @property
    def hits(self):
        """
        Computed on the fly fron interaction tree. Cache if possible
        """
        
        return Hits.from_list(self.hits_iter)
                
    def to_dict(self):
        
        output = dict(particle_type = self.particle_type,
                      pos_x = str(self.position.x),
                      pos_y = str(self.position.y),
                      direction = str(self.direction),
                      energy = str(self.energy))

        if self.reco is not None:
            output['reco'] = self.reco.to_dict()

        if self.interaction is not None:
            output['interaction'] = self.interaction.to_dict()

        return output
        
    class Reconstruction:

        def __init__(self, event_type, direction, energy):

            self.event_type = event_type
            self.direction = direction
            self.energy = energy

        def to_dict(self):

            output = dict(event_type = self.event_type,
                          direction = str(self.direction),
                          energy = str(self.energy))
            
            return output
            
        def __repr__(self):
            return yaml.dump(self.to_dict(), sort_keys=False)

    def set_interaction(self, interaction):
        self.interaction = interaction
        return self

    def add_parent(self, interaction):
        interaction.add_child(self)
        return self
        
    def set_reco(self, event_type, direction, energy):

        self.reco = self.Reconstruction(event_type = event_type,
                                        direction = direction,
                                        energy = energy)

    def __repr__(self):
        return yaml.dump(self.to_dict(), sort_keys=False)
        
    def plot(self, ax, length_unit, primary_energy = None, **kwargs):

        if primary_energy is None:
            primary_energy = self.energy
        
        if self.interaction is not None:
            end_pos = self.interaction.position
        else:
            end_pos = Cartesian2D(self.position.x + 1*u.km*np.cos(self.direction),
                                  self.position.y + 1*u.km*np.sin(self.direction))
                                  
        ax.plot([self.position.x.to_value(length_unit),
                 end_pos.x.to_value(length_unit)],
                [self.position.y.to_value(length_unit),
                 end_pos.y.to_value(length_unit)],
                color = 'red',
                alpha = (self.energy/primary_energy).to_value(''))

        if self.interaction is not None:
            self.interaction.plot(ax, length_unit, primary_energy)

        return ax

class Interaction:

    def __init__(self, interaction_type, position, energy):

        self.interaction_type = interaction_type
        self.position = position
        self.energy = energy
        self.measurement = None
        self.children = []

    class Measurement:

        def __init__(self, layer, position, energy):

            self.layer = layer
            self.position = position
            self.energy = energy

        def to_dict(self):

            output = dict(layer = self.layer,
                          pos_x = str(self.position.x),
                          pos_y = str(self.position.y),
                          energy = str(self.energy))
            
            return output
            
        def __repr__(self):
            return yaml.dump(self.to_dict(), sort_keys=False)

    @property
    def hits_iter(self):

        if self.measurement is not None:
            yield Hits(self.measurement.layer, self.measurement.position, self.measurement.energy)

        for child in self.children:
            for hit in child.hits_iter:
                yield hit
        
    def set_measurement(self, layer, position, energy):

        self.measurement = self.Measurement(layer = layer,
                                            position = position,
                                            energy = energy)
        
    def add_child(self, particle):
        self.children.append(particle)
        return self

    def add_parent(self, particle):
        particle.set_interaction(self)
        return self
        
    def to_dict(self):
        
        output = dict(interaction_type = self.interaction_type,
                      pos_x = str(self.position.x),
                      pos_y = str(self.position.y),
                      energy = str(self.energy))

        if self.measurement is not None:
            output['measurement'] = self.measurement.to_dict()

        if self.children:
            output['children'] = [c.to_dict() for c in self.children]

        return output
    
    def __repr__(self):
        return yaml.dump(self.to_dict(), sort_keys=False)

    def plot(self, ax, length_unit, primary_energy = None, **kwargs):

        if self.measurement is not None:
            ax.scatter([self.position.x.to_value(length_unit)],
                       [self.position.y.to_value(length_unit)],
                       color = 'red',
                       s = 50*np.sqrt((self.energy/primary_energy).to_value('')))

        for child in self.children:
            child.plot(ax, length_unit, primary_energy)

        return ax
    
class Absorption(Interaction):

    def __init__(self, position, energy, **data):

        super().__init__(interaction_type = 'absorption',
                         position = position,
                         energy = energy,
                         **data)

class Compton(Interaction):

    def __init__(self, position, energy, **data):

        super().__init__(interaction_type = 'compton',
                         position = position,
                         energy = energy,
                         **data)
    
class Photon(Particle):

    def __init__(self, position, direction, energy, chirality = None, **data):

        super().__init__(particle_type = 'photon',
                         position = position,
                         direction = direction,
                         energy = energy,
                         **data)

        if chirality is None:
            self.chirality = np.random.choice([1,-1])
        else:
            self.chirality = chirality

    def to_dict(self):

        output = dict(particle_type = self.particle_type,
                      pos_x = str(self.position.x),
                      pos_y = str(self.position.y),
                      direction = str(self.direction),
                      energy = str(self.energy),
                      chirality = self.chirality)

        if self.reco is not None:
            output['reco'] = self.reco.to_dict()

        if self.interaction is not None:
            output['interaction'] = self.interaction.to_dict()

        return output    
