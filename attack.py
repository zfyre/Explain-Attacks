import torch 
from collections import OrderedDict

class Attack():
    r"""
    Base class for all attacks.

    .. note::
        It automatically set device to the device where given model is.
        It basically changes training mode to eval during attack process.
        To change this, please see `set_model_training_mode`.
    """
    def __init__(self, name, model) -> None:
        self.attack = name
        self._attacks = OrderedDict()
        self.set_model(model)

        try:
            self.device = next(model.parameters()).device
        except Exception:
            self.device = None
            print("Failed to set device automatically, please try set_device() manual.")

        # Define the supported attacks mode
        self.attack_mode = "default"
        self.supported_modes = ["default"]
        self.targeted = False

        

    def forward(self, inputs, labels=None, *args, **kwargs):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

def set_model(self, model):
    self.model = model
    self.model_name = model.__class__.__name__

def set_device(self, device):
    self.device = device

def get_target_labels(inputs, labels):
    raise NotImplementedError

def get_projection(delta, eps, norm):
    with torch.no_grad():
        if norm == 'L2':
            if torch.norm(delta) > eps:
                delta = delta / torch.norm(delta) * eps # project to L2 ball    

        if norm == 'Linf':
            delta = torch.clamp(delta, -eps, eps)
    return delta

def get_logits(self, inputs, labels=None, *args, **kwargs):
    if self._normalization_applied is False:
        inputs = self.normalize(inputs)
    logits = self.model(inputs)
    return logits