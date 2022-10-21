# Uncertainty Loss 
*Loss functions for uncertainty quantification in deep learning.*

This package implements loss functions from the following papers

* Evidential Deep Learning to Quantify Classification Uncertainty
    * `ul.evidential_loss`
* Information Aware Max-Norm Dirichlet Networks for Predictive Uncertainty Estimation
    * `ul.maxnorm_loss`



These loss functions can be used as drop in replacements for 
`torch.nn.functional.cross_entropy`.  See QuickStart and Examples below.

## Quickstart 
Install the package with pip
```bash
pip install uncertainty-loss
```
Then use the loss in a training pipeline. For example:
```python
import uncertainty_loss as ul
import torch 

def fit_step(model, x, targets, reg_factor=0):
    """Runs a single training step and retuns the loss for the batch.

    Note the inputs to the uncertainty loss function need to be 
    non-negative.  Any transformation will work (exp, relu, softplus,
    etc) but we have found that exp works best (in agreement with the 
    original papers).  For convenience we provide a clamped exp function
    to avoid overflow.
    """
    logits = model(x)
    evidence = ul.clamped_exp(logits) # non-negative transform
    loss = ul.maxnorm_loss(evidence, targets, reg_factor)
    return loss
```


### Examples
Replace 
```python
from torch.nn import functional as F

loss = F.cross_entropy(x,y)
```
With
```python
import uncertainy_loss as ul

loss = ul.evidential_loss(x,y)
# or 
loss = ul.maxnorm_loss(x,y)
```

The loss functions also accept a reduction parameter with the same
properties as the `cross_entropy` loss.

#### Important
For each loss function is a regularization term that is shown to be 
beneficial for learning to quantify uncertainty.  In practice, 
to ensure that the regularization term does not dominate early 
in training, we ramp up the regularization term from 0 to a max factor
e.g. 0->1.  It is up to the user to ensure this happens.  Each loss 
function takes an additional parameter `reg_factor`.  During training 
one can increment `reg_factor` to accomplish this ramp up.  By 
default `reg_factor==0` so there is no regularization unless 
explicitly "turned on"

### Example with Regularization Annealing
```python
import uncertainty_loss as ul

reg_steps = 1000
reg_step_size = 1/reg_steps
reg_factor = 0
for epoch in range(epochs):

    for x,y in dataloader:
        logits = model(x)
        evidence = ul.clamped_exp(logits)
        loss = ul.maxnorm_loss(evidence, y, reg_factor=reg_factor)
        reg_factor = min(reg_factor+reg_step_size, 1)
```


## Motivation
Uncertainty quantification has important applications in AI Safety and active learning.  Neural networks trained with a traditional cross entropy loss are often over-confident in unfamiliar situations.  It's easy to see why this can be disastrous: An AI surgeon making a confident but wrong incision in an unfamilar situation, a self-driving car making a confident but wrong turn, an AI investor making a confident but wrong buy/sell decision.

There have been several methods proposed for uncertainty quantification.  Many of the popular methods require specific network architectures (e.g. Monte Carlo Dropout requires dropout layers) or require expensive inference (Monte Carlo dropout requires multiple runs through the same model, ensemble methods require multiple models). 

Recently methods for uncertainty quantification have been proposed that do not require any changes to the network architecture and have no inference overhead.  Instead they propose to learn parameters of a "higher order distribution" and use this distribution to quantify the uncertainty in the prediction.  They have been shown to be effective.

Unfortunately, these methods haven't been integrated into any of the main deep learning packages and the heavy math makes the implementation a bit tricky.  

For these reasons we have created the `uncertainty-loss` package.