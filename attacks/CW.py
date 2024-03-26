import torch
import torch.nn as nn
from ..attack import Attack


# TODO: Test the affect of eps on the adversarial attacks

class CW(Attack):
    def __init__(self, model, norm='L2', iter=1, lr=0.01, steps=50, c=1, kappa=0, eps = 1e-5 , random_start = True) -> None:
        super().__init__("CW", model)
        self.norm = norm
        self.iter = iter
        self.lr = lr
        self.steps = steps
        self.c = c
        self.kappa = kappa
        self.random_start = random_start
        self.eps = eps

    def forward(self, inputs, labels=None, bounds=None):
        
        images = inputs.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Get the labels
        if self.targeted:
            labels = self.get_target_labels(images, labels)

        # Attack Loop
        deltas = []
        for itr in range(iter):
            delta = l2_attack(images=images, labels=labels, lr=self.lr, steps = self.steps, c=self.c, kappa=self.kappa)
            deltas.append(delta)

        return deltas

def l2_attack(self, images, labels):
    """
    Objective:
        minimize  ||1/2 * (tanh(w) + 1) - x|| + c * f(1/2 * (tanh(w) + 1))

    Change of variable is used to get the gradient descent working,
        delta_i = 1/2 * (tanh(w_i)+1) - x_i

    this is box in the value of 0<= x + delta_i <=1.
    """

    if self.random_start:
        delta = torch.empty_like(images, device=self.device, requires_grad=False).uniform_(-self.eps, self.eps)
        delta = self.get_projection(delta, self.eps, 'L2')
    else:
        delta = torch.zeros_like(images, device=self.device, requires_grad=False)

    w = self.delta_to_w(images, delta).detach()
    w = nn.parameter.Parameter(w)

    optimizer = torch.optim.Adam(w,lr=self.lr)
    l2_norm = nn.MSELoss(reduction='none')


    for step in range(self.steps):

        adv_images = images + self.w_to_delta(images, w)
        outputs = self.get_logits(adv_images)
        loss = l2_norm(w, torch.zeros_like(w)) + self.c * self.f(outputs, labels)

 

    def delta_to_w(self, images, delta):
        # atanh() is defined within [-1, 1]
        w = torch.atanh(torch.clamp(2*(images+delta)-1, min=-1, max=1))
        return w

    def w_to_delta(self, images, w):
        delta = 0.5 * (torch.tanh(w) + 1) - images
        return delta
    
    def f(self, outputs, labels):
        """
            f: objective function defined as per the paper,

                f(x') = max(max{Z(x')_i : i != t} - Z(x')_t, -K).

            f is based on the best objective function found in the paper by authors.
        """
        one_hot_labels = torch.eye(outputs.shape[1]).to(self.device)[labels]

        # find the max logit other than the target class
        other = torch.max((1 - one_hot_labels) * outputs, dim=1)[0]
        # get the target class's logit
        real = torch.max(one_hot_labels * outputs, dim=1)[0]


