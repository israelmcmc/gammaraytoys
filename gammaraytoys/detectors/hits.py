import numpy as np

class EventList:

    def __init__(self):
        
        self._events = []
        self._nsim = 0

    @property
    def nsim(self):
        return self._nsim

    def appedd(self, event):
        self._events.append(event)

    def write(self, filename):

        """
        """

class Event:

    def __init__(self, event_type, **data):

        self._data = {'event_type': event_type, 'data':data, 'children': []}

    @property
    def event_type(self):
        return self._data['event_type']
        
    @property
    def children(self):
        return self._data['children']

    @property
    def data(self):
        return self._data['data']

    def __getitem__(self, key):
        if isinstance(key, (int, np.int)):
            return self.children[key]
        else:
            return self._data.data[key]
    
    def add_child(self, child):
        
        self.children.append(child)

    def add_parent(self, parent):

        parent.add_child(self)

        return self

    def as_yaml(self, index = 0, indent = 0, in_list = True):
        
        if in_list:
            string = ' '*(indent) + "- "
            indent += 2
        else:
            string = ' '*indent 

        string += f"event_type: {self.event_type}\n"
        
        string += ' '*(indent) + f"nevent: {index}\n"
        string += ' '*(indent) + f"data:\n"
        
        for key,value in self.data.items():
            string += ' '*(indent+2) + f"{key}: {value}\n"

        string += ' '*(indent) + "children:\n"    

        for n,child in enumerate(self.children):
            string += child.as_yaml(index = n, indent = indent + 4, in_list = True)

        return string
    
    def __repr__(self):
        return self.as_yaml()

    
class Particle(Event):

    def __init__(self, particle_type, pos_x, pos_y, direction, energy, **data):

        super().__init__(event_type = particle_type,
                         pos_x = pos_x,
                         pos_y = pos_y,
                         direction = direction,
                         energy = energy,
                         **data)

    def add_interaction(self, interaction):

        super().add_child(interaction)

class Interaction(Event):

    def __init__(self, interation_type, layer, pos_x, pos_y, energy):

        super().__init__(event_type = interation_type,
                         layer = layer,
                         pos_x = pos_x,
                         pos_y = pos_y,
                         energy = energy)

    def add_child_particle(self, particle):

        super().add_child(particle)        

class Absorption(Interaction):

    def __init__(self, layer, pos_x, pos_y, energy):

        super().__init__(interation_type = 'absorption',
                         layer = layer,
                         pos_x = pos_x,
                         pos_y = pos_y,
                         energy = energy)

class Compton(Interaction):

    def __init__(self, layer, pos_x, pos_y, energy):

        super().__init__(interation_type = 'compton',
                         layer = layer,
                         pos_x = pos_x,
                         pos_y = pos_y,
                         energy = energy)
        
        
class Photon(Particle):

    def __init__(self, pos_x, pos_y, direction, energy, chirality = .5):

        super().__init__(particle_type = 'photon',
                         pos_x = pos_x,
                         pos_y = pos_y,
                         direction = direction,
                         energy = energy,
                         chirality = chirality)

    
    
