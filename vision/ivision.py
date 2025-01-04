from abc import ABC, abstractmethod

class IVision(ABC):
    @abstractmethod
    def __init__(self):
        """Costruttore per inizializzare la camera."""
        pass

    @abstractmethod
    def get_frames(self):
        """Legge i frame di profondità e colore dalla camera."""
        pass

    @abstractmethod
    def __del__(self):
        """Distruttore per arrestare la camera e rilasciare le risorse."""
        pass