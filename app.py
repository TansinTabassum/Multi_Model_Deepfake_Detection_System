import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from facenet_pytorch import MTCNN

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- FACE DETECTOR ----------------
mtcnn = MTCNN(image_size=224, margin=20, device=device)

# ---------------- RESNET ----------------
resnet = models.resnet50(pretrained=False)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet = resnet.to(device)
resnet.eval()

# ---------------- MODEL ----------------
class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=256,
            batch_first=True
        )

        self.fc = nn.Linear(256, 2)

    def forward(self, x):
        batch, seq, c, h, w = x.shape

        x = x.view(batch * seq, c, h, w)

        with torch.no_grad():
            features = resnet(x)

        features = features.view(batch, seq, 2048)

        lstm_out, _ = self.lstm(features)

        out = self.fc(lstm_out[:, -1, :])

        return out

# ---------------- LOAD MODEL ----------------
model = DeepfakeModel().to(device)
model.load_state_dict(torch.load("deepfake_model.pt", map_location=device))
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# ---------------- PREDICT ----------------
def predict(image):
    if image is None:
        return {"Error": 1.0}

    # 🔥 Face detect
    face = mtcnn(image)

    if face is None:
        return {"No Face Detected": 1.0}

    img = face  # already tensor (3,224,224)

    # 🔥 Fake sequence বানানো (10 frames)
    sequence = torch.stack([img]*10)
    sequence = sequence.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(sequence)
        probs = torch.softmax(output, dim=1)

    fake_prob = probs[0][1].item()
    real_prob = probs[0][0].item()

    return {
        "Fake": float(fake_prob),
        "Real": float(real_prob)
    }

# ---------------- UI ----------------
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=gr.Label(num_top_classes=2),
    title="Deepfake Detector (LSTM + ResNet)",
    description="Upload a face image to detect if it's Real or Fake"
)

# ---------------- LAUNCH ----------------
if __name__ == "__main__":
    iface.launch(share=True)