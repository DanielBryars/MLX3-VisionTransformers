import wandb
import torch
import datetime
import model
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
import wandb
import time
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import TripletMarginLoss
import dataset
from Pipeline import *

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

hyperparameters = {
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'batch_size': 512,
        'num_epochs': 5,        
        'patch_size': 7,
        'embedding_dim': 128,
        'patience': 3
}

sweep_config = {
    'method': 'grid',  # or 'random', 'bayes'
    'metric': {
        'name': 'val/loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [12e-5]
        },
        'weight_decay': {
            'values': [0.01] #[0.01, 0.001
        },
        'patch_size':[7,14]
        
    }
}


def train_one_epoch(daModel, dataloader, optimizer, device, epoch, loss_fn, step_offset=0):
    daModel.train()
    step = step_offset

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in loop:
        image_patch = [x.to(device) for x in batch]

        #how to calculate the loss
        loss = loss_fn(query_emb, pos_doc_emb, neg_doc_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        wandb.log({'train/loss': loss.item()}, step=step)
        loop.set_postfix(loss=loss.item())
        step += 1

    return step

def train():
    with wandb.init(config=hyperparameters):
        
        config = wandb.config
        # override hyperparameters here with config values
        hyperparameters.update({
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,   
            'patch_size': config.patch_size         
        })

        train_dataset = dataset.mnist_train
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hyperparameters['batch_size'])

        val_dataset = dataset.mnist_test #use test for validation right now
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=hyperparameters['batch_size'])

        transformer_model = PipelineFactory().create(
            patch_size = hyperparameters['patch_size'], 
            embedding_size= hyperparameters['embedding_size'], 
            num_classes = hyperparameters['num_classes'])

        transformer_model.to(device)
        
        print('transformer_model:params', sum(p.numel() for p in transformer_model.parameters()))
        
        optimizer = torch.optim.Adam(
            transformer_model.parameters(), 
            lr=hyperparameters['learning_rate'], 
            weight_decay=hyperparameters['weight_decay']
        )

        step = 0
        best_val_loss = float('inf')
        epochs_no_improve = 0

        patience= hyperparameters['patience']

        for epoch in range(1, hyperparameters['num_epochs'] + 1):
            step = train_one_epoch(queryModel, docModel, train_loader, optimizer, device, epoch, loss_fn, step_offset=step)
            val_loss = evaluate(queryModel, docModel, val_loader, device, loss_fn, epoch=epoch, step=step)

            print(f"Epoch {epoch} complete | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_checkpoint(queryModel, docModel, epoch, ts)
            else:
                epochs_no_improve += 1
                print(f"No improvement. Early stop patience: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

sweep_id = wandb.sweep(sweep_config, project='mlx7-week3-visual_transformers')
wandb.agent(sweep_id, function=train)