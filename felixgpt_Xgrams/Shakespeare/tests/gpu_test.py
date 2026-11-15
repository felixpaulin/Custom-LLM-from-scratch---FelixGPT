import torch
import torch_directml

device = torch_directml.device()

x = torch.randn(1000, 1000).to(device)
y = torch.randn(1000, 1000).to(device)

z = torch.matmul(x, y)

print("Device:", z.device)
print("Result shape:", z.shape)