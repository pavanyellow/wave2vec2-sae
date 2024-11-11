import torch
import json
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class SAEConfig:
    input_dim: int = 768
    hidden_dim: int = 768*8
    l1_penalty: float = 3
    num_epochs: int = 12000
    batch_size: int = 16384
    learning_rate: float = 1e-4
    val_split: float = 0.2
     

# Follows the architecture from https://transformer-circuits.pub/2024/april-update/index.html#training-saes
class SparseAutoencoder(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super(SparseAutoencoder, self).__init__()
        self.encoder : nn.Module = nn.Linear(cfg.input_dim, cfg.hidden_dim, bias=False)
        self.decoder : nn.Module = nn.Linear(cfg.hidden_dim, cfg.input_dim, bias=False)
        self.encoder_bias = nn.Parameter(torch.zeros(cfg.hidden_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(cfg.input_dim))
        self.input_dim = cfg.input_dim
        self.hidden_dim = cfg.hidden_dim
        self.l1_penalty = cfg.l1_penalty
        self.init_weights()
        self.to(DEVICE)
    
    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def init_weights(self):
            nn.init.uniform_(self.decoder.weight, -1, 1)  # Random directions
            with torch.no_grad():
                norms = torch.rand(self.hidden_dim) * 0.95 + 0.05  # Random norms between 0.05 and 1
                self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True))
                self.decoder.weight.mul_(norms)
            
            self.encoder.weight.data.copy_(self.decoder.weight.data.t())

    def forward(self, x):
        encoded = F.relu(self.encoder(x) + self.encoder_bias)
        decoded = self.decoder(encoded) + self.decoder_bias
        return decoded, encoded
    
    def save(self, folder_path: str , name: str):
        PATH = f"{folder_path}/{name}.pt"
        torch.save(self.state_dict(), PATH)
        print(f"Model saved at {PATH}")
            
    
    @classmethod
    def load(cls, folder_path: str, name: str):
        PATH = f"{folder_path}/{name}.pt"
        self = cls(cfg=SAEConfig())
        self.load_state_dict(torch.load(PATH, map_location=DEVICE))
        self.eval()
        return self


def sae_loss(X, reconstructed_X, encoded_X, W_d, lambda_val):
    mse_loss = ((X - reconstructed_X) ** 2).sum(dim = -1).mean(0)
    sparsity_loss = lambda_val * ((torch.norm(W_d, p=2, dim=0)*encoded_X).sum(dim=1)).mean(0) 
    return (mse_loss + sparsity_loss, mse_loss)







