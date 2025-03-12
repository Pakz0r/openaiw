if __name__ == 'HPE':
    from .hopenet import Hopenet, ResNet, AlexNet
    from .HPEModel import HPEModel
    from .HPEnet import HPEnet
    from .utils import *
else:
    from hopenet import Hopenet, ResNet, AlexNet
    from HPEModel import HPEModel
    from HPEnet import HPEnet
    from utils import *


__all__ = [
    "Hopenet",
    "ResNet",
    "AlexNet"
    "HPEModel"
    "HPEnet"
]