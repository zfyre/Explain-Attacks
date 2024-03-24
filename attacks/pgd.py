import torch
import torch.nn as nn

from ..attack import Attack

class pgd(Attack):
    r""" delta := P`(delta + alphs*grad__eps(L(f(x+delta),Y)))
        images : [B, C, W, H]
        labels : [B]
        eps : norm budget
        alpha : step /  learning rate in attacks.
        steps : number of steps to update a single delta.
        iter : Number of iteration to search for delta.
    """

    def __init__(self, model, eps=0.5, norm='L2',alpha=1/255, steps=100,  iter=1, random_start=True):
        super().__init__("PGD", model)                     # What to pass in the __init__ of it's super class
        self.eps = eps
        self.norm = norm
        self.alpha = alpha
        self.iter = iter
        self.steps = steps
        self.random_start = random_start
        self.criterion = nn.CrossEntropyLoss()
        self.supported_modes = ["default", 'targeted']
    def forward(self, images, labels):
        r"""
            Overridden over base class forward method.
        """

        images  = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)   # detach is used to detach the gradients.

        #NOTE: Implemented for non-targeted attacks.
        if self.targeted:
            labels = self.get_target_labels(images, labels)

        # Attack Loop:
        deltas = []
        for itr in range(self.iter):
            delta = attack(images=images, _labels=labels, itr=itr)
            deltas.append(delta)

        return torch.tensor(deltas, device=self.device, requires_grad=False)

def attack(self, images, _labels, itr):
    delta = torch.zeros_like(images, device=self.device, requires_grad=True)
    if self.random_start:
        delta = torch.empty_like(images, device=self.device, requires_grad=True).uniform_(-self.eps, self.eps)
        delta = self.get_projection(delta, self.eps, self.norm)
    
    for step in range(self.steps):
        delta = delta.requires_grad(True)

        # getting logits:
        outputs = self.get_logits(images + delta)

        # calculating loss:
        if self.target_labels:
            loss = - self.criterion(outputs, _labels, device = self.device)
        else:
            loss = self.criterion(outputs, _labels, device = self.device)

        loss.backward()
        # print("Gradients: ", delta.grad.abs().mean().item())

        # calculating grad w.r.t. delta
        direc = torch.autograd.grad(
                loss, delta, retain_graph=False, create_graph=False
            )[0]

        # Normalizing the gradients for better attacks -> as per the blog
        if self.norm == 'L2':
            direc = direc / torch.norm(direc)
        elif self.norm == 'inf':
            direc = torch.sign(direc)
        
        delta = delta.detach() + direc * self.alpha

        # Projecting the delat back to Lp norm
        delta = self.get_projection(delta, self.eps, self.norm)

        print(f'#try {itr} #step {step} {self.norm} norm = {torch.norm(delta):.8f}', end='\r', flush=True)
    
    return delta

