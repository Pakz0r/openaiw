from abc import ABC, abstractmethod

class IVision(ABC):
    def __init__(self, wrapper):
        """Costruttore per inizializzare la camera."""
        self.wrapper = wrapper

    @abstractmethod
    def get_frames(self):
        """Legge i frame di profondità e colore dalla camera."""
        pass

    @abstractmethod
    def get_device_info(self):
        """Ritorna le informazioni sul device collegato"""
        pass

    @abstractmethod
    def convert_point_to_world(self, u: float, v: float):
        """Converte un punto da 2D a 3D"""
        pass

    @abstractmethod
    def __del__(self):
        """Distruttore per arrestare la camera e rilasciare le risorse."""
        pass