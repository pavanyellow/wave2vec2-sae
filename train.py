
import torch
from model import SAEConfig, SparseAutoencoder, sae_loss
from trainer import train_autoencoder
import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

resid_stream_dimension = 768
layer_num = 5
PATH  = "activations/train-clean-100-concat-3.pt"

NUM_EPOCHS = 24000
batch_size = 16384
learning_rate = 1e-4
sae_expansion_factor = 8
l1_penalty = 3

input_dim = 768


config = SAEConfig(
        input_dim,
        input_dim * sae_expansion_factor,
        l1_penalty=l1_penalty, 
        num_epochs=NUM_EPOCHS, 
        batch_size = batch_size, 
        learning_rate=learning_rate
        )

model = SparseAutoencoder(config)
print(f"No of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

data = torch.load(PATH, map_location=DEVICE)

train_autoencoder(model, data, config)

model.eval()

model.save(folder_path= 'saved_models', name=f"final")