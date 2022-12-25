import torch
import matplotlib.pyplot as plt


def histogram(p):
    hist = torch.histc(p, 255, torch.min(p), torch.max(p))
    plt.bar(range(255), hist)
    plt.show()


# data = numpy
# rates = torch.rand_like(data, device=data.device) * self.sigma
# p = torch.poisson(rates)
#
# return data + p

sigma = 1
data = torch.rand(256, 256, 3) * 255
histogram(data)
print("data - max:", torch.max(data), "min:", torch.min(data), "mean:", torch.mean(data),
      "var:", torch.var(data), "std:", torch.std(data))
g = torch.randn_like(data) * sigma
histogram(g)
print("g - max:", torch.max(g), "min:", torch.min(g), "mean:", torch.mean(g),
      "var:", torch.var(g), "std:", torch.std(g))
noisy = data + g
histogram(noisy)
print("noisy - max:", torch.max(noisy), "min:", torch.min(noisy), "mean:", torch.mean(noisy),
      "var:", torch.var(noisy), "std:", torch.std(noisy))


print()
p = torch.poisson(data)
histogram(p)
print("p - max:", torch.max(p), "min:", torch.min(p), "mean:", torch.mean(p),
      "var:", torch.var(p), "std:", torch.std(p))
p = torch.poisson(data) - data
histogram(p)
print("p - max:", torch.max(p), "min:", torch.min(p), "mean:", torch.mean(p),
      "var:", torch.var(p), "std:", torch.std(p))
p = p * sigma / torch.std(p)
histogram(p)
print("p - max:", torch.max(p), "min:", torch.min(p), "mean:", torch.mean(p),
      "var:", torch.var(p), "std:", torch.std(p))
noisy = data + p
histogram(noisy)
print("noisy - max:", torch.max(noisy), "min:", torch.min(noisy), "mean:", torch.mean(noisy),
      "var:", torch.var(noisy), "std:", torch.std(noisy))
