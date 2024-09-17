from astropy.coordinates import CartesianRepresentation
from collections import OrderedDict

class Cartesian2D(CartesianRepresentation):

    def __init__(self, x, y, copy = True):
         super().__init__(x = x, y = y, z = 0*x, copy = copy)

    @classmethod
    def from_cartesian(cls, c):
        return cls(c.x, c.y)

    def to_cartesian(self):
        return CartesianRepresentation(x = c.x, y = c.y, z = 0*x)

    
