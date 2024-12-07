# %%

import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from pyhessian import hessian

def loss_fn(x, y):
    return torch.mean((x - y) ** 2)

def train_step(model, data, loss_fn, optimizer):
    optimizer.zero_grad()
    loss = loss_fn(model(data[0]), data[1])
    loss.backward()
    optimizer.step()
    return loss.item()

def train(model, data, loss_fn, optimizer, num_epochs=1):
    loss_curve = []
    model.train()
    for epoch in range(num_epochs):
        loss = train_step(model, data, loss_fn, optimizer)
        loss_curve.append(loss)
    return loss_curve

def train_and_trace(model, data, loss_fn, optimizer, num_epochs=1):
    loss_curve = []
    traces = []
    trace_std = []
    for epoch in range(num_epochs):
        model.train()
        loss = train_step(model, data, loss_fn, optimizer)
        loss_curve.append(loss)
        model.eval()
        hessian_comp = hessian(model, loss_fn, data, cuda=True)
        trace = hessian_comp.trace(tol=1e-6)
        traces.append(np.mean(trace))
        trace_std.append(np.std(trace) / np.sqrt(len(trace) - 1))
    return loss_curve, traces, trace_std

# %%

model = nn.Sequential(
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 10)
).to(torch.float64)

data = (torch.randn(20, 10).to(torch.float64), torch.randn(20, 10).to(torch.float64))

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

model.cuda()
data = (data[0].cuda(), data[1].cuda())

# %%
%%time
# train and compute Tr(H) after each epoch
loss_curve, traces, trace_std = train_and_trace(model, data, loss_fn, optimizer, num_epochs=200)

# %%
%%time
# untrain and compute Tr(H) after each epoch
loss_curve2, traces2, trace_std2 = train_and_trace(model, data, lambda x, y: -loss_fn(x, y), optimizer, num_epochs=200)

# %%
# plot loss curve
plt.plot(loss_curve, label='train')
plt.plot(-np.array(loss_curve2)[::-1], label='untrain')
plt.legend()
plt.xlabel('step=epoch')
plt.ylabel('loss')
plt.ylim(np.min(loss_curve) - 0.1, np.max(loss_curve) + 0.1)
plt.title('full batch GD')
plt.show()

# plot traces
# with error bars
plt.errorbar(range(len(traces)), traces, yerr=trace_std, fmt='o', label='train')
plt.errorbar(range(len(traces2)), -np.array(traces2)[::-1], yerr=trace_std2[::-1], fmt='o', label='untrain')
plt.legend()
plt.xlabel('step=epoch')
plt.ylabel('Tr(H)')
plt.ylim(np.min(traces) - 1, np.max(traces) * 1.2)
plt.title('full batch GD')
plt.show()
# moving average of traces
width = 10
traces_smoothed = np.convolve(traces, np.ones(width)/width, mode='valid')
traces_smoothed2 = np.convolve(-np.array(traces2)[::-1], np.ones(width)/width, mode='valid')
plt.plot(traces_smoothed, label='train')
plt.plot(traces_smoothed2, label='untrain')
plt.legend()
plt.xlabel('step=epoch')
plt.ylabel('Tr(H)')
plt.ylim(np.min(traces) - 1, np.max(traces) * 1.2)
plt.title('full batch GD')
plt.show()

# %%

width = 10
traces_smoothed = np.convolve(traces, np.ones(width)/width, mode='valid')
trace_error = traces[width//2:-width//2+1] - traces_smoothed
trace_z = trace_error / trace_std[width//2:-width//2+1]

plt.hist(trace_z, bins=30)
plt.show()

# normality test: plot qq plot
from scipy.stats import probplot
probplot(trace_z, dist='norm', plot=plt)
plt.show()

# Calculate mean and std of trace_z
mean_z = np.mean(trace_z)
std_z = np.std(trace_z)

# Create Q-Q plot
fig, ax = plt.subplots(figsize=(8, 6))
(osm, osr) = probplot(trace_z, dist='norm', plot=ax, fit=False)

# Plot the actual values
ax.scatter(osm, osr, label='Sample')

# Plot the ideal line
ideal_line = np.linspace(min(osm), max(osm), 100)
ax.plot(ideal_line, ideal_line, 'r-', label='Ideal Normal')
ax.set_aspect('equal')

ax.set_xlabel('Theoretical Quantiles')
ax.set_ylabel('Sample Quantiles')
ax.set_title(f'Q-Q Plot (Mean: {mean_z:.2f}, Std: {std_z:.2f})')
ax.legend()

plt.show()

# %%

cutoff = None
# dots and lines
plt.loglog(-np.array(loss_curve2)[:cutoff], -np.array(traces2)[:cutoff], '.-')
plt.xlim(1e-1, 1e5)
plt.ylim(1, 1e5)
plt.xlabel('loss')
plt.ylabel('Tr(H)')
plt.title('full batch GD: untrain')
plt.show()
# %%

estimates = hessian_comp.trace(tol=1e-6)
print(len(estimates))
print(estimates)

print(np.mean(estimates))
# bootstrap the estimates
bootstrap_estimates = []
for i in range(1000):
    bootstrap_estimates.append(np.mean(np.random.choice(estimates, len(estimates))))

# estimate the standard deviation of the estimates
print(np.std(bootstrap_estimates))
print(np.std(bootstrap_estimates) / np.mean(bootstrap_estimates))

# histogram of estimates
plt.hist(bootstrap_estimates, bins=30)

plt.show()

# %%
# count total number of model parameters
total_params = sum(p.numel() for p in model.parameters())
print(total_params)

# %%
hessian_comp = hessian(model, lambda x, y: loss_fn(x, y), data, cuda=True)
values, vectors = hessian_comp.eigenvalues(top_n=100)
print(values)
plt.plot(values)
plt.show()

# %%
print(np.sum(values))