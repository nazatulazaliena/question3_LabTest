import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import requests
import pandas as pd

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Real-time Webcam Image Classification (ResNet-18)",
    layout="centered",
)

st.title("Real-time Webcam Image Classification")
st.write("This web app uses **PyTorch ResNet-18 (pretrained on ImageNet)** to classify a webcam photo.")


# -----------------------------
# Step 2: Download ImageNet class labels from GitHub text file
# -----------------------------
@st.cache_data
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text.strip().split("\n")


# -----------------------------
# Step 3: Load pretrained ResNet-18 model from Torchvision
# -----------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model


labels = load_imagenet_labels()
model = load_model()

# -----------------------------
# Step 4: Define preprocessing pipeline (Resize -> CenterCrop -> Tensor -> Normalize)
# -----------------------------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# -----------------------------
# Step 5: Capture webcam image + display + convert to batch tensor
# -----------------------------
st.subheader("Step 5: Capture an Image")
st.info("Click **Take photo**. The model will classify the captured image immediately.")

img_file = st.camera_input("Take a photo")

if img_file is None:
    st.warning("No image captured yet. Please click **Take photo**.")
    st.stop()

image = Image.open(img_file).convert("RGB")
st.image(image, caption="Captured Image", use_container_width=True)

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # shape: [1, 3, 224, 224]

# Optional (good practice): move to same device as model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_batch = input_batch.to(device)

# -----------------------------
# Step 6: Run prediction, apply softmax, show Top-5 in a table
# -----------------------------
st.subheader("Step 6: Top-5 Predictions")

with torch.no_grad():
    logits = model(input_batch)              # raw outputs
    probs = F.softmax(logits, dim=1)[0]      # probabilities for 1000 classes

top5_prob, top5_idx = torch.topk(probs, 5)

# Convert to dataframe for display
df = pd.DataFrame({
    "Rank": [1, 2, 3, 4, 5],
    "Label": [labels[i] for i in top5_idx.cpu().tolist()],
    "Probability": [float(p) for p in top5_prob.cpu().tolist()],
})

st.dataframe(df, use_container_width=True)

# Also display nicely as text (optional but clear for marking)
st.markdown("### Results (Top-5)")
for r in df.itertuples(index=False):
    st.write(f"**{r.Rank}. {r.Label}** — {r.Probability:.4f}")
