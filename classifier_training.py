import wandb
import torch
import os 
import tqdm
import torch.nn.functional as F

def evaluate(model, dataloader, device, epoch=None, step=None):
    model.eval()
    
    total_loss = 0.0
    total_batches = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    with torch.no_grad():
        for batch in loop:
            images, labels = [x.to(device) for x in batch]

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()
            total_batches += 1
            loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_batches if total_batches > 0 else float('nan')
    if step is not None:
        wandb.log({'val/loss': avg_loss}, step=step)
    return avg_loss

def train_one_epoch(model, dataloader, optimizer, device, epoch, step_offset=0):
    model.train()
    step = step_offset

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in loop:
        images, labels = [x.to(device) for x in batch]

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({'train/loss': loss.item()}, step=step)
        loop.set_postfix(loss=loss.item())
        step += 1

    return step

def save_checkpoint(model, epoch, ts):
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_type = type(model).__name__
    descriptive_name = f'ts.{ts}.epoch.{epoch + 1}.{model_type}'
    checkpoint_name = f'{descriptive_name}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
    }, checkpoint_path)

    # Create wandb artifact and log it
    artifact = wandb.Artifact(
        name=descriptive_name,
        type='model',
        description=f'{model_type} model weights from epoch {epoch + 1}, timestamp {ts}')
    
    #actually upload the artifact!!!!
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)