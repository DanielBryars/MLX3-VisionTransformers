import wandb
import torch
import os 
from tqdm import tqdm 
import torch.nn.functional as F
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, dataloader, device, epoch=None, step=None):
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
    with torch.no_grad():
        for batch in loop:
            images, labels = [x.to(device) for x in batch]

            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            total_batches += 1
            loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / total_batches if total_batches > 0 else float('nan')
    accuracy = total_correct / total_samples if total_samples > 0 else float('nan')

    if step is not None:
        wandb.log({'val/loss': avg_loss, 'val/accuracy': accuracy}, step=step)

    return avg_loss,accuracy

def train_one_epoch(model, dataloader, optimizer, device, epoch, step_offset=0):
    model.train()
    step = step_offset

    total_correct = 0
    total_samples = 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in loop:
        images, labels = [x.to(device) for x in batch]

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        wandb.log({'train/loss': loss.item()}, step=step)
        loop.set_postfix(loss=loss.item())

        if step % 100 == 0:
            probs = F.softmax(logits, dim=-1)
            # Log the average confidence for the predicted class
            max_probs, preds = probs.max(dim=1)
            avg_confidence = max_probs.mean().item()
            wandb.log({
                'train/avg_confidence': avg_confidence,
                'train/confidence_hist': wandb.Histogram(max_probs.detach().cpu().numpy())
            }, step=step)

        step += 1

    accuracy = total_correct / total_samples if total_samples > 0 else float('nan')
    wandb.log({'train/accuracy': accuracy}, step=step - 1)

    return step

def save_checkpoint(model, hyperparameters, epoch, ts):
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_type = type(model).__name__
    descriptive_name = f'ts.{ts}.epoch.{epoch + 1}.{model_type}'
    checkpoint_name = f'{descriptive_name}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    print(f"Saving '{checkpoint_path}'")
    torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'hyperparameters': hyperparameters
    }, checkpoint_path)

    # Create wandb artifact and log it
    artifact = wandb.Artifact(
        name=descriptive_name,
        type='model',
        description=f'{model_type} model weights from epoch {epoch + 1}, timestamp {ts}')
    
    #actually upload the artifact!!!!
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)