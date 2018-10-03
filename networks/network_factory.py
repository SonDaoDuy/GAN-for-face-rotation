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
        elif network_name == 'cascade_lmk_detector_v2':
            from .cascade_lmk_detector_v2 import CascadeLmkDectectorv2
            network = CascadeLmkDectectorv2(*args, **kwargs)
        elif network_name == 'began_discriminator':
            from .began_discriminator import BeganDiscriminator
            network = BeganDiscriminator(*args, **kwargs)
        elif network_name == 'casia_generator':
            from .casia_generator import CasiaGenerator
            network = CasiaGenerator(*args, **kwargs)
        elif network_name == 'generator_wgan':
            from .generator_wasserstein_gan import Generator
            network = Generator(*args, **kwargs)
        elif network_name == 'unet_generator':
            from .unet import UNet
            network = UNet(*args, **kwargs)
        elif network_name == 'casia_generator_v2':
            from .casia_generator_v2 import CasiaGeneratorv2
            network = CasiaGeneratorv2(*args, **kwargs)
        elif network_name == 'casia_generator_v3':
            from .casia_generator_v3 import CasiaGeneratorv3
            network = CasiaGeneratorv3(*args, **kwargs)
        elif network_name == 'began_discriminator_v2':
            from .began_discriminator_v2 import BeganDiscriminatorv2
            network = BeganDiscriminatorv2(*args, **kwargs)
        elif network_name == 'began_discriminator_v3':
            from .began_discriminator_v3 import BeganDiscriminatorv3
            network = BeganDiscriminatorv3(*args, **kwargs)
        elif network_name == 'began_discriminator_v4':
            from .began_discriminator_v4 import BeganDiscriminatorv4
            network = BeganDiscriminatorv4(*args, **kwargs)
        elif network_name == 'discriminator_wgan':
            from .discriminator_wasserstein_gan import DiscriminatorWGAN
            network = DiscriminatorWGAN(*args, **kwargs)
        elif network_name == 'LightCNN_9':
            from .light_cnn import LightCNN_9Layers
            network = LightCNN_9Layers(*args, **kwargs)
        elif network_name == 'LightCNN_29v2':
            from .light_cnn import LightCNN_29Layers_v2
            network = LightCNN_29Layers_v2(*args, **kwargs)
        elif network_name == 'openface_net':
            from .loadOpenFace import netOpenFace
            network = netOpenFace(*args, **kwargs)
        else:
            raise ValueError("Network %s not recognized." % network_name)

        print ("Network %s was created" % network_name)

        return network