import numpy as np
from astropy.coordinates import CartesianRepresentation, Angle
from astropy import units as u
import yaml

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
        
class Particle:

    def __init__(self, particle_type, position, direction, energy):

        self.particle_type = particle_type
        self.position = position
        self.direction = u.Quantity(Angle(direction).wrap_at(360*u.deg))
        self.energy = energy
        self.interaction = None
        self.reco = None

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
        
class Interaction:

    def __init__(self, interaction_type, layer, position, energy):

        self.interaction_type = interaction_type
        self.layer = layer
        self.position = position
        self.energy = energy
        self.children = []
        
    def add_child(self, particle):
        self.children.append(particle)
        return self

    def add_parent(self, particle):
        particle.set_interaction(self)
        return self
        
    def to_dict(self):
        
        output = dict(interaction_type = self.interaction_type,
                      layer = self.layer,
                      pos_x = str(self.position.x),
                      pos_y = str(self.position.y),
                      energy = str(self.energy))

        if self.children:
            output['children'] = [c.to_dict() for c in self.children]

        return output
    
    def __repr__(self):
        return yaml.dump(self.to_dict(), sort_keys=False)

class Absorption(Interaction):

    def __init__(self, layer, position, energy, **data):

        super().__init__(interaction_type = 'absorption',
                         layer = layer,
                         position = position,
                         energy = energy,
                         **data)

class Compton(Interaction):

    def __init__(self, layer, position, energy, **data):

        super().__init__(interaction_type = 'compton',
                         layer = layer,
                         position = position,
                         energy = energy,
                         **data)
        
        
class Photon(Particle):

    def __init__(self, position, direction, energy, chirality = .5, **data):

        super().__init__(particle_type = 'photon',
                         position = position,
                         direction = direction,
                         energy = energy,
                         **data)
        
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
