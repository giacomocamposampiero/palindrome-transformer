import torch

sm = torch.tensor([0.])

n = 1000

for x in range(0,n+1):
    sm -= torch.tensor([2**x/ (2**(n+1) - 1) ])

for x in range(n,-1,-1):
    sm += torch.tensor([2**x/(2**(n+1) - 1)])

print(float(sm[0]))
