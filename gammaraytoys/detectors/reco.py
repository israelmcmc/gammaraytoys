from abc import ABC, abstractmethod

class Reconstructor(ABC):

    @property
    @abstractmethod
    def reconstruct(self) -> bool:
        pass


    
class RecoEvent(ABC):
    """
    """

class RecoCompton(RecoEvent):

    def __init__(self, energy, phi, psi):
        pass


    
        
        
