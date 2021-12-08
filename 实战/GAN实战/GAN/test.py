import torch
import torch.nn as nn

m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
print(input)
target = torch.empty(3).random_(2)
print(target)
print(m(input))
output = loss(m(input), target)
output.backward()