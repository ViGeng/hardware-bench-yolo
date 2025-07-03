import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))


import torch
print("PyTorch version:", torch.__version__)
print("CUDA version used by PyTorch:", torch.version.cuda)
print("Is CUDA available:", torch.cuda.is_available())