import torch
ckpt = torch.load("model_weight/best_model_weight/KGNN_14semeval_laptop_78.91_75.21.pth", map_location="cpu")
print(ckpt['graph_embed.weight'].shape) 