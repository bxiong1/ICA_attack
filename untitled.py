import numpy as np
import torch
defense_embeddings = np.load(f'/root/autodl-tmp/final_defense_embeddings_without_target_v2.npy')
defense_embeddings_torch_1 = torch.tensor(defense_embeddings, device='cuda:0', requires_grad = False)

defense_embeddings = np.load(f'/root/autodl-tmp/final_defense_embeddings_without_helpful.npy')
defense_embeddings_torch_2 = torch.tensor(defense_embeddings, device='cuda:0', requires_grad = False)

defense_embeddings = np.load(f'/root/autodl-tmp/final_defense_embeddings_v3.npy')
defense_embeddings_torch_3 = torch.tensor(defense_embeddings, device='cuda:0', requires_grad = False)


print(defense_embeddings_torch_1)
print(defense_embeddings_torch_2)
print(defense_embeddings_torch_3)