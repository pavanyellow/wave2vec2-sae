import torch
from model import SAEConfig, SparseAutoencoder, sae_loss
import wandb


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

wandb_log = True

def init_wandb(config: SAEConfig):
    wandb.init(project='wave-sae', config=config.__dict__)
    
def split_data(original_dataset, val_split : float):
    dataset_size = original_dataset.shape[0]
    dataset = original_dataset[torch.randperm(dataset_size)]

    train_dataset, val_dataset = dataset[int(dataset_size*val_split):], dataset[:int(dataset_size*val_split)]
    dataset_size = original_dataset.shape[0]
    train_dataset, val_dataset = original_dataset[int(dataset_size*val_split):], original_dataset[:int(dataset_size*val_split)]

    return train_dataset, val_dataset

@torch.no_grad()
def estimate_losses(model, train_dataset, val_dataset, config: SAEConfig, epoch : int):
    model.eval()
    eval_iters = 20
    losses = []
    l_zero = []
    l_one = []
    mse_losses = []

    for dataset in [train_dataset, val_dataset]:
        loss = torch.zeros(eval_iters)
        mse = torch.zeros(eval_iters)
        for i in range(eval_iters):
            batch = dataset[torch.randint(0, dataset.shape[0], (config.batch_size,))]
            reconstructed, encoded = model(batch)
            loss_,mse_loss_ = sae_loss(batch, reconstructed, encoded, model.decoder.weight, config.l1_penalty)
            loss[i] = loss_
            mse[i] = mse_loss_ / (batch ** 2).sum(dim=-1).mean(0)
            
            encoded_normalized = encoded*torch.norm(model.decoder.weight, p=2, dim=0) 
            l_zero.append(encoded_normalized.norm(0, dim = 1, keepdim=True).mean())
            l_one.append(encoded_normalized.norm(1, dim = 1, keepdim=True).mean())

        losses.append(loss.mean().item())
        mse_losses.append(mse.mean().item())
    print(f'Epoch {epoch}/{config.num_epochs} val loss: {losses[1]:.3f}  train loss: {losses[0]:.3f} mse :{mse_losses[1]:.3f} l0: {l_zero[1]:.2f} l1:{l_one[1]:.2f} \n')
    if wandb_log:
        wandb.log({"train_loss": losses[0], "val_loss": losses[1], "mse": mse_losses[1], "l0": l_zero[1], "epoch": epoch})
    model.train()
    return losses, l_zero, mse_losses # Val loss is losses[1]



def train_autoencoder(model: SparseAutoencoder, dataset: torch.Tensor, config: SAEConfig) -> None:

    print (f"Training autoencoder with config: {config} and shape: {dataset.shape} \n")
    if wandb_log:
        init_wandb(config)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 - max(0, epoch - config.num_epochs * 0.8) / (config.num_epochs * 0.2))
    current_norm = torch.norm(dataset, dim=1, keepdim=True).mean()
    dataset.mul_((config.input_dim**0.5) / current_norm)

    train_dataset, val_dataset = split_data(dataset, config.val_split)
    del dataset
    torch.cuda.empty_cache()
    estimate_losses(model, train_dataset, val_dataset, config, -1)
    
    for epoch in range(1, config.num_epochs):
        batch = train_dataset[torch.randint(0, train_dataset.shape[0], (config.batch_size,))]
        optimizer.zero_grad()
        reconstructed, encoded = model(batch)
        loss,_ = sae_loss(batch, reconstructed, encoded, model.decoder.weight, config.l1_penalty)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        #scheduler.step()

        if epoch % 20 == 0:
            if wandb_log:
                wandb.log({"epoch": epoch, "loss": loss.item()})


        if epoch % 200 == 0:
            estimate_losses(model, train_dataset, val_dataset, config, epoch)
        
        if epoch % 10000 == 0:
            model.save(folder_path= 'temp_models', name=f"epoch-{epoch}")
        

    if wandb_log:
        wandb.finish(quiet = True)

