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
        else:
            raise ValueError("Model %s not recognized." % model_name)

        print("Model %s was created" % model.name)
        return model