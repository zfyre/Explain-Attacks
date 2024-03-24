# FGSM : Fast gradient Sign Method

Task at Hand ***Inner Maximization***.
$$\underset{||\delta||\; \leq \;\epsilon}{\max} \; l(h_{\theta}(x+\delta), y)$$
One method to find the noise level $\delta$, is by computing the **gradient** of the **Loss function** w.r.t the **noise $\delta$**. 
Let the gradient $g$ be defined as follows,
$$g = \nabla_{\delta}\:(h_{\theta}(x + \delta), y)$$
the gradient descent step can be defined as 
$$\delta^* = \delta + \alpha g$$
where, $\alpha$ is the *step size*.
>To keep the noise $\delta$ withing the specified norm, we project the $\delta^*$ to the specified $L_p$ ball.

### How big $\alpha$ can be?
Let's consider the $L_{\inf}$ norm, i.e $L_{\inf} \leq \epsilon$. Let initially $\delta = 0$, then applying the above process to the noise results in:
$$\delta^* = clip(\alpha g, [-\epsilon, \epsilon])$$
Now, to get an adversarial we try to make $\alpha$ sufficiently large. This lead to most of the gradient values getting clipped to the $\epsilon$ level.
Hence, for large $\alpha$,
$$\delta^* = \epsilon \; sign(g)$$
This is known as the **FGSM**, one of the first attacks.

**Note:** Even if initial $\delta \neq 0$, we can still define the following method since the whole method is based of sufficiently large $\alpha$.
### Things to Remember
- This method works better for fully connected Neural Networks
- CNNs are slightly more robust against these attacks
- Since we are taking only one step in FGSM, this method is better at finding an adversary for *binary classification models*.
- Only defined for $L_{\inf}$ norm.