#%%
from torchdiffeq import odeint

import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from pyhessian import hessian

def loss_fn(x, x0):
    return torch.mean((x - x0) ** 2)

data = (torch.randn(20, 10).to(torch.float64).cuda(), 
        torch.randn(20, 10).to(torch.float64).cuda())

#%%

# y0: initial params of the model as one tensor
# t: time
# f: function from params to gradient of loss wrt params
# odeint: solves the ODE

def loss_flat(y):
    # Reconstruct the state dict from the flat tensor y
    state_dict = model.state_dict()
    start = 0
    for name, param in model.named_parameters():
        param_shape = param.shape
        param_size = param.numel()
        state_dict[name] = y[start:start+param_size].view(param_shape)
        start += param_size

    # Load the reconstructed state dict
    model.load_state_dict(state_dict)

    loss = loss_fn(model(data[0]), data[1])
    return loss

def f(t, y):
    # compute the loss
    loss = loss_flat(y)
    # compute the gradient of the loss wrt the params
    grads = torch.autograd.grad(loss, model.parameters())
    # return the gradient
    return -torch.cat([g.view(-1) for g in grads])
#%%

model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10)
).to(torch.float64)
model.cuda()
#%%
y0 = torch.cat([p.view(-1) for p in model.parameters()])
t = torch.arange(0, 20., .2)
#%%
%%time
sol = odeint(f, y0, t)
#%%

losses = [loss_flat(y).cpu().item() for y in sol]
plt.plot(t, losses)
plt.show()

#%%
sol.shape
#%%
yf = sol[-1]

# run backwards!
f_back = lambda t, y: -f(t, y)
sol_back = odeint(f_back, yf, t)

#%%

losses_back = [loss_flat(y).cpu().item() for y in sol_back]
plt.plot(t, losses, label='forward')
plt.plot(torch.flip(t, [0]), losses_back, label='backward')
plt.legend()
plt.xlabel('time')
plt.ylabel('loss')
plt.title('ODE on loss landscape')
plt.show()  