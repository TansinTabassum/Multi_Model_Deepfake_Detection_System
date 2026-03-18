import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# ResNet50 feature extractor (same as used in training)
# ------------------------------
resnet = models.resnet50(weights=None)  # don't load pretrained weights
resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove FC layer
resnet = resnet.to(device)
resnet.eval()

# ------------------------------
# LSTM-based Deepfake model
# ------------------------------
class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2048, hidden_size=256, batch_first=True)
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

# Load your trained model
model = DeepfakeModel().to(device)
model.load_state_dict(torch.load("deepfake_model.pt", map_location=device))
model.eval()

# ------------------------------
# Image transform
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------------------
# Prediction function
# ------------------------------
def predict(video_file):
    if video_file is None:
        return {"Error": 1.0}

    # Open video
    cap = cv2.VideoCapture(video_file)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, 10).astype(int)  # sample 10 frames

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = transform(img)
        frames.append(img)

    cap.release()

    # If no frames found
    if len(frames) == 0:
        return {"Error": 1.0}

    # Pad if fewer than 10 frames
    while len(frames) < 10:
        frames.append(frames[-1])

    # Stack frames
    sequence = torch.stack(frames).unsqueeze(0).to(device)  # shape: (1, seq, C, H, W)

    # Add slight noise to avoid identical frames (optional)
    sequence = sequence + 0.01 * torch.randn_like(sequence)

    # Predict
    with torch.no_grad():
        output = model(sequence)
        probs = torch.softmax(output, dim=1)

    fake_prob = float(probs[0][1].item())
    real_prob = float(probs[0][0].item())

    return {"Fake": round(fake_prob, 3), "Real": round(real_prob, 3)}

# ------------------------------
# Gradio interface
# ------------------------------
iface = gr.Interface(
    fn=predict,
    inputs=gr.Video(label="Upload Video"),  # video input
    outputs=gr.Label(num_top_classes=2),
    title="Deepfake Detector",
    description="Upload a short video (1-5 seconds) to detect if it is Real or Fake."
)

if __name__ == "__main__":
    iface.launch(share=True)