import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image
import io

# --- Load your trained ViT model ---
# Replace this with your own model loading
class DummyViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_weights = []
        self.classifier = torch.nn.Linear(784, 10)

    def forward(self, x):
        self.attn_weights = [torch.rand(1, 1, 50)]  # Dummy attention map
        x = x.view(x.size(0), -1)
        return self.classifier(x)

model = DummyViT()
model.eval()

# --- Preprocessing function ---
preprocess = T.Compose([
    T.Resize((28, 28)),
    T.ToTensor(),
    T.Normalize((0.1307,), (0.3081,))
])

# --- Visualisation function ---
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
    ax.axis('off')
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# --- Main interface function ---
def classify_and_show_attention(image_dict):
    image_bytes = image_dict["composite"]
    image = Image.fromarray(image_bytes).convert('L')
    image_tensor = preprocess(image)
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
    pred = output.argmax(dim=1).item()
    attn_maps = model.attn_weights
    heatmap = visualise_attention(attn_maps, image_tensor)
    return f"Predicted: {pred}", heatmap

# --- Launch the interface with Sketchpad ---
interface = gr.Interface(
    fn=classify_and_show_attention,
    inputs=gr.Sketchpad(canvas_size=(280, 280), brush=10),
    outputs=[gr.Text(), gr.Image()],
    title="Vision Transformer Attention Explorer",
    description="Draw a digit and see what the ViT model attends to."
)

if __name__ == '__main__':
    interface.launch()
