import wandb
import torch
import datetime
import torch
import wandb
import torch
import dataset
from VisualTransformer import *
from classifier_training import *

set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

hyperparameters = {
        'patch_size':7,
        'embedding_size':64,
        'num_classes':10,
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
        'patch_size': {
            'values': [7,14]
        }
    }
}

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

        model = VisualTransformer(
            patch_size = hyperparameters['patch_size'], 
            embedding_size = hyperparameters['embedding_size'], 
            num_classes = hyperparameters['num_classes']
        )

        model.to(device)

        print('model:params', sum(p.numel() for p in model.parameters()))
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=hyperparameters['learning_rate'], 
            weight_decay=hyperparameters['weight_decay']
        )

        step = 0
        best_val_loss = float('inf')
        epochs_no_improve = 0

        patience= hyperparameters['patience']

        for epoch in range(1, hyperparameters['num_epochs'] + 1):
            step = train_one_epoch(model, train_loader, optimizer, device, epoch, step_offset=step)
            val_loss = evaluate(model, val_loader, device, epoch=epoch, step=step)

            print(f"Epoch {epoch} complete | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_checkpoint(model, epoch, ts)
            else:
                epochs_no_improve += 1
                print(f"No improvement. Early stop patience: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

sweep_id = wandb.sweep(sweep_config, project='mlx7-week3-visual_transformers')
wandb.agent(sweep_id, function=train)