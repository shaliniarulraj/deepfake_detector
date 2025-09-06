import os, torch, numpy as np, streamlit as st, gdown
from PIL import Image
from torchvision import models, transforms

@st.cache_resource
def load_model(ckpt_path):
    device = torch.device("cpu")
    ck = torch.load(ckpt_path, map_location=device)

    # Validate checkpoint structure
    assert "state_dict" in ck and "img_size" in ck and "classes" in ck, "Checkpoint missing required keys"

    model = models.resnet18(weights=None)
    in_f = model.fc.in_features
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(in_f, 2))

    # Load weights with strict=False to avoid key mismatch errors
    model.load_state_dict(ck["state_dict"], strict=False)
    model.eval().to(device)

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
file_id = "1D-G-JegVEMu0uI_R9HHvf5nLdTpGl7GJ"
ckpt_path = "models/best_model.pth"
ckpt_url = f"https://drive.google.com/uc?id={file_id}"

# Download if not already present
if not os.path.exists(ckpt_path):
    st.info("Downloading model checkpoint from Google Drive...")
    os.makedirs("models", exist_ok=True)
    gdown.download(ckpt_url, ckpt_path, quiet=False)

# Load model with error handling
try:
    model, device, img_size, classes = load_model(ckpt_path)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

tfms = make_tfms(img_size)

file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
if file:
    img = Image.open(file).convert("RGB")
    x = tfms(img).unsqueeze(0).to(device)

    with st.spinner("Processing image..."):
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy().squeeze()
        p_fake = float(probs[1])
        pred = "FAKE" if p_fake >= 0.5 else "REAL"

    st.subheader(f"Prediction: {pred}  |  Fake probability: {p_fake:.3f}")
    st.image(img, caption="Input", use_column_width=True)

    # Confidence bar
    st.markdown("**Confidence (Fake Probability):**")
    st.progress(p_fake)

