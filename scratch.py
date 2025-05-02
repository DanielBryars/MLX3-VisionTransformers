import wandb
import torch
import datetime
import torch
import wandb
import torch
import dataset
from models.VisualTransformer import *
from classifier_training import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

set_seed()

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

train_dataset = dataset.mnist_train
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hyperparameters['batch_size'])

        
model = VisualTransformer(
            patch_size = hyperparameters['patch_size'], 
            embedding_size = hyperparameters['embedding_size'], 
            num_classes = hyperparameters['num_classes']
        )

optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=hyperparameters['learning_rate'], 
            weight_decay=hyperparameters['weight_decay']
        )


epoch = 1

train_one_epoch(model, train_loader, optimizer, device, epoch, step_offset=0)