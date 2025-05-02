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

print(checkpoint['hyperparameters'])

#val_loss, accuracy  = evaluate(model, val_loader, 'cpu')

val_loss = 0.20688627008348703
accuracy = 0.939

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
    num_layers = len(attn_maps)
    num_heads = attn_maps[0].shape[1]  # assume consistent across layers

    print(f"num_layers:{num_layers},num_heads:{num_heads}")

    fig, axes = plt.subplots(num_layers, num_heads, figsize=(4 * num_heads, 4 * num_layers))

    for layer in range(num_layers):
        for head in range(num_heads):
            ax = axes[layer][head] if num_layers > 1 else axes[head]

            # Extract CLS → patch attention
            attn = attn_maps[layer][0, head, 0, 1:]
            grid_size = int(len(attn) ** 0.5)
            if grid_size * grid_size != len(attn):
                attn = F.pad(attn, (0, grid_size**2 - len(attn)))
            attn_grid = attn.reshape(grid_size, grid_size).cpu()
            attn_img = F.interpolate(attn_grid.unsqueeze(0).unsqueeze(0), size=(28, 28), mode='bilinear')[0, 0]

            ax.imshow(image_tensor.squeeze(), cmap='gray')
            ax.imshow(attn_img, cmap='jet', alpha=0.5)
            ax.set_title(f"Layer {layer}, Head {head}")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
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
