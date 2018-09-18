import torch.nn as nn
import functools

class NetworksFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(network_name, *args, **kwargs):

        if network_name == 'cascade_lmk_detector':
            from .cascade_lmk_detector import CascadeLmkDectector
            network = CascadeLmkDectector(*args, **kwargs)
        elif network_name == 'began_discriminator':
            from .began_discriminator import BeganDiscriminator
            network = BeganDiscriminator(*args, **kwargs)
        elif network_name == 'casia_generator':
            from .casia_generator import CasiaGenerator
            network = CasiaGenerator(*args, **kwargs)
        else:
            raise ValueError("Network %s not recognized." % network_name)

        print ("Network %s was created" % network_name)

        return network