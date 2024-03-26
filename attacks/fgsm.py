import torch
import torch.nn as nn
from ..attack import Attack

class FGSM(Attack):
    """
    Norm: L_inf
    Criterion: Cross Entropy
    """
    def __init__(self, model, eps=0.5, iter=1, random_start=False):
        super().__init__("FGSM", model)
        self.eps = eps
        self.iter = iter
        self.random_start = random_start
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None, bounds=None):
        """
        Images : shape = [B, C, H, W]

        Return : list of Successfully attacked Images
        """
        images = inputs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Intializing the mask
        if bounds != None:
            (x1, y1), (x2, y2) = bounds
            mask = torch.zeros_like(images, requires_grad=False)
            mask[:, :, x1:x2, y1:y2] = 1
        else:
            mask = torch.ones_like(images, requires_grad=False)

        # Get the labels
        if self.targeted:
            labels = self.get_target_labels(images, labels)

        # Attack Loop
        deltas = []
        for itr in range(iter):
            delta = attack(images=images, labels=labels, mask=mask)
            # Checking the success...
            outputs = self.get_logits(images + delta)
            new_labels = torch.argmax(outputs, dim=1)
            if (labels != new_labels and ~self.targeted) or (labels == new_labels and self.targeted):
                deltas.append(delta)
        
        return deltas


def attack(self, images, labels, mask):
    
    # Initializing the initial noise
    if self.random_start:
        delta = torch.empty_like(images, device=self.device, requires_grad=True).uniform_(-self.eps, self.eps)
        delta = self.get_projection(delta, self.eps, 'Linf')
    else:
        delta = torch.zeros_like(images, device=self.device, requires_grad=True)

    # mask the initial noise
    delta *= mask

    # Getting the outputs with added noise
    outputs = self.get_logits(images + delta)

    # Calculating the loss w.r.t the targeted | w/o target
    if self.targeted:
        loss = - self.criterion(outputs, labels)
    else:
        loss = self.criterion(outputs, labels)
    
    # Calculating the gradient w.r.t the noise
    grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

    # masking the gradients
    grad *= mask

    # Updating the noise 
    delta = self.eps * torch.sign(grad)

    # Projecting to the desired norm
    delta = self.get_projection(delta, self.eps, 'Linf') # No change here but just for completeness of other methods!!

    # masking the new noise
    delta *= mask

    return delta


