import torch
import torch_directml

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("DirectML device:", torch_directml.device())