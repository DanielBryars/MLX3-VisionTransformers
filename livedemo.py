import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
import io
from classifier_training import evaluate
import dataset
from models.ModelFactory import CreateModelFromCheckPoint
from models.VisualTransformer import VisualTransformer

import torch

#import wandb
#run = wandb.init(project="MLX7-W3-VIT-SINGLE")
#artifact = run.use_artifact('bryars-bryars/MLX7-W3-VIT-SINGLE/ts.2025_05_02__10_31_48.epoch.6.VisualTransformer:v0', type='model')
#artifact_dir = artifact.download()

# Path to the checkpoint file inside the artifact
checkpoint_path = f"artifacts/ts.2025_05_02__10_31_48.epoch.6.VisualTransformer.pth"

# Load into your model
checkpoint = torch.load(checkpoint_path, map_location='cpu')

model = CreateModelFromCheckPoint(checkpoint)

val_dataset = dataset.mnist_test #use test for validation right now
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=checkpoint['hyperparameters']['batch_size'])

val_loss, accuracy  = evaluate(model, val_loader, 'cpu')

preprocess = T.Compose([
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])

inv_preprocess = T.Compose([
    T.Normalize(mean=[-0.1307 / 0.3081], std=[1 / 0.3081]),
    T.ToPILImage()
])

def visualise_attention(attn_maps, image_tensor):
    attn = attn_maps[-1][0, 0, :]  # Dummy: [CLS] to tokens
    grid_size = int(len(attn) ** 0.5)
    if grid_size * grid_size != len(attn):
        grid_size += 1  # pad if necessary
        attn = F.pad(attn, (0, grid_size**2 - len(attn)))

    attn_grid = attn.reshape(grid_size, grid_size).cpu()
    attn_img = F.interpolate(attn_grid.unsqueeze(0).unsqueeze(0), size=(28, 28), mode='bilinear')[0,0]

    fig, ax = plt.subplots()
    ax.imshow(image_tensor.squeeze(), cmap='gray')
    ax.imshow(attn_img, cmap='jet', alpha=0.5)

    num_patches_per_side = int(attn_grid.shape[0])

    # Set ticks at the centre of each patch
    tick_positions = [28 / num_patches_per_side * (i + 0.5) for i in range(num_patches_per_side)]
    tick_labels = list(range(num_patches_per_side))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    ax.set_xlabel("Patch column")
    ax.set_ylabel("Patch row")

    #ax.axis('off')
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def visualise_prediction_confidence(output):
    probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    # Create bar chart
    fig, ax = plt.subplots()
    ax.bar(range(10), probs)
    ax.set_xticks(range(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Confidence")
    ax.set_title("Model Confidence per Digit")

    # Convert to image
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    confidence_plot = Image.open(buf)
    return confidence_plot

def classify_and_show_attention(image_dict):
    image_bytes = image_dict["composite"]
    image = Image.fromarray(image_bytes).convert('L')
    image_tensor = preprocess(image)

    model_input_image = inv_preprocess(image_tensor.cpu())

    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0), return_attn=True)
    pred = output.argmax(dim=1).item()
    confidence_plot = visualise_prediction_confidence(output)
    attn_maps = model.attn_weights
    attention_heatmap = visualise_attention(attn_maps, image_tensor)
    return f"Predicted: {pred}",model_input_image, confidence_plot, attention_heatmap 

# --- Launch the interface with Sketchpad ---
interface = gr.Interface(
    fn=classify_and_show_attention,
    inputs=gr.Sketchpad(canvas_size=(280, 280), brush=10),
    outputs=[gr.Text(), gr.Image(label="Model Input"), gr.Image(label="Confidence Plot"), gr.Image(label="Attention")],
    title="Vision Transformer Attention Explorer",
    description=f"Draw a digit and see what the ViT model attends to. This model val_loss:{val_loss}, accuracy:{accuracy}"
)

if __name__ == '__main__':
    interface.launch(share=True)
