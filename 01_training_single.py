import wandb
import torch
import datetime
import torch
import wandb
import torch
import dataset
from models.ModelFactory import *
from models.VisualTransformer import *
from classifier_training import *

set_seed()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
print(f"Using device:{device}")

hyperparameters = {
        'patch_size':14,        
        'num_classes':10,
        'learning_rate': 3e-4,
        'weight_decay': 0.01,
        'batch_size': 512,
        'num_epochs': 10,                
        'patience': 3,
        'num_transformer_blocks': 4,
        'transformerType':'StandardTransformerBlock',
        'embedding_size':64,
        'num_heads':4,
        'mlp_dim':64,
        'dropout':0.1,
}

wandb.init(project='MLX7-W3-VIT-SINGLE', config=hyperparameters)
config = wandb.config

train_dataset = dataset.mnist_train
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=hyperparameters['batch_size'])
val_dataset = dataset.mnist_test #use test for validation right now
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=hyperparameters['batch_size'])

model = CreateModelFromHyperParameters(hyperparameters)
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
    val_loss, accuracy  = evaluate(model, val_loader, device, epoch=epoch, step=step)

    print(f"Epoch {epoch} complete | Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.4f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        save_checkpoint(model, hyperparameters, epoch, ts)
    else:
        epochs_no_improve += 1
        print(f"No improvement. Early stop patience: {epochs_no_improve}/{patience}")
    if epochs_no_improve >= patience:
        print("Early stopping triggered.")
        break
