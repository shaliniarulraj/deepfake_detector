import os, torch, numpy as np, streamlit as st, gdown
from PIL import Image
from torchvision import models, transforms

@st.cache_resource
def load_model(ckpt_path):
  device = torch.device("cpu")
  ck = torch.load(ckpt_path, map_location=device)
  model = models.resnet18(weights=None)
  in_f = model.fc.in_features
  model.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(in_f, 2))
  model.load_state_dict(ck["state_dict"]); model.eval().to(device)
  return model, device, ck["img_size"], ck["classes"]

def make_tfms(img_size):
  return transforms.Compose([
    transforms.Resize((img_size, img_size)),                                                                           
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
  ])

st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️")
st.title("🛡️ Deepfake Detector (CPU)")

# Google Drive file ID
file_id = "1D-G-JegVEMu0uI_R9HHvf5nLdTpGl7GJ"  # <-- Replace with your actual file ID
ckpt_path = "models/best_model.pth"
ckpt_url = f"https://drive.google.com/uc?id={file_id}"

# Download if not already present
if not os.path.exists(ckpt_path):
  st.info("Downloading model checkpoint from Google Drive...")
  os.makedirs("models", exist_ok=True)
  gdown.download(ckpt_url, ckpt_path, quiet=False)

model, device, img_size, classes = load_model(ckpt_path)
tfms = make_tfms(img_size)

file = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])
if file:
  img = Image.open(file).convert("RGB")
  x = tfms(img).unsqueeze(0).to(device)
  with torch.no_grad():
    probs = torch.softmax(model(x), dim=1).cpu().numpy().squeeze()
  p_fake = float(probs[1])
  pred = "FAKE" if p_fake>=0.5 else "REAL"
  st.subheader(f"Prediction: {pred}  |  Fake probability: {p_fake:.3f}")
  st.image(img, caption="Input", use_column_width=True)
