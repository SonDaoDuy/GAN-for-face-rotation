import os
import torch

class ModelsFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(model_name, *args, **kwargs):
        model = None

        if model_name == 'ganormalization':
            from .ganormalization import GANormalization
            model = GANormalization(*args, **kwargs)
        elif model_name == 'ganormalization_v2':
            from .ganormalization_v2 import GANormalizationv2
            model = GANormalizationv2(*args, **kwargs)
        else:
            raise ValueError("Model %s not recognized." % model_name)

        print("Model %s was created" % model.name)
        return model