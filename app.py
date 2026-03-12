import os

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from model import MNIST_CNN  # Ensure model.py is in the same folder
from data_loader import get_available_class_names

# Page Config
st.set_page_config(page_title="Health-Cure Federated AI", layout="wide")

# 1. LOAD THE PYTORCH MODEL
@st.cache_resource
def load_trained_model():
    # Path to your PyTorch weights
    model_path = "models/global_fedavg.pth"
    
    if os.path.exists(model_path):
        # map_location='cpu' ensures it works on any Windows PC (GPU or no GPU)
        try:
            checkpoint = torch.load(
                model_path,
                map_location=torch.device('cpu'),
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        num_classes = checkpoint["fc2.weight"].shape[0]
        model = MNIST_CNN(num_classes=num_classes)
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    else:
        st.error(f"Model file not found at {model_path}. Please run training first.")
        return None

model = load_trained_model()
dataset_names = get_available_class_names()

# Align class names with the loaded checkpoint output dimension.
if model is not None:
    model_num_classes = model.fc2.out_features
else:
    model_num_classes = 0

effective_dataset_names = dataset_names[:model_num_classes]


def confidence_band(confidence_pct):
    if confidence_pct >= 85:
        return "High confidence"
    if confidence_pct >= 60:
        return "Moderate confidence"
    return "Low confidence"


def indices_for_scope(names, scope):
    if scope == "All datasets":
        return list(range(len(names)))
    if scope == "Medical only":
        return [
            idx for idx, name in enumerate(names)
            if not (name.startswith("CIFAR10") or name.startswith("CIFAR100") or name == "CIFAR10" or name == "CIFAR100")
        ]
    if scope == "CIFAR10 only":
        return [idx for idx, name in enumerate(names) if name.startswith("CIFAR10") or name == "CIFAR10"]
    if scope == "CIFAR100 only":
        return [idx for idx, name in enumerate(names) if name.startswith("CIFAR100") or name == "CIFAR100"]
    return list(range(len(names)))

# 2. UI LAYOUT
st.title("🩺 Health-Cure: Clinical Federated Dashboard")
st.write("Cross-platform Diagnostic Tool (PyTorch Implementation)")

tab1, tab2 = st.tabs(["🔍 Live Prediction", "📊 Clinical Analytics"])

with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Upload CT Scan")
        prediction_scope = st.selectbox(
            "Prediction Scope",
            ["All datasets", "Medical only", "CIFAR10 only", "CIFAR100 only"],
            index=0,
            help="Restrict predictions to a selected dataset group.",
        )
        file = st.file_uploader("Upload CT Image", type=["jpg", "png", "jpeg"])
        if file:
            img = Image.open(file).convert('L')
            img_resized = cv2.resize(np.array(img), (64, 64))
            
            # Preprocessing matching your PyTorch training pipeline
            img_tensor = img_resized / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5  # Normalize
            img_tensor = torch.FloatTensor(img_tensor).unsqueeze(0).unsqueeze(0)
            
            st.image(img, caption="Original Scan", use_column_width=True)

    with col2:
        st.subheader("AI Diagnostic Result")
        if file and model:
            with torch.no_grad():
                output = model(img_tensor)
                prob = torch.softmax(output, dim=1)
                # Scope indices from names actually available in this checkpoint.
                allowed_indices = indices_for_scope(effective_dataset_names, prediction_scope)

                if not allowed_indices:
                    st.error(
                        "No classes available for selected prediction scope in this model checkpoint. "
                        "Retrain model with that dataset scope and try again."
                    )
                    st.stop()

                allowed_indices = [idx for idx in allowed_indices if idx < prob.shape[1]]
                if not allowed_indices:
                    st.error("Selected scope has no valid indices for current model output size.")
                    st.stop()

                allowed_tensor = prob[0, allowed_indices]
                allowed_sum = allowed_tensor.sum().item()
                if allowed_sum <= 0:
                    st.error("Could not compute scoped probabilities for selected dataset.")
                    st.stop()

                scoped_probs = allowed_tensor / allowed_sum
                best_local_idx = torch.argmax(scoped_probs).item()
                prediction = allowed_indices[best_local_idx]

            if prediction < len(effective_dataset_names):
                label = effective_dataset_names[prediction]
            else:
                label = f"Class {prediction}"
            confidence = scoped_probs[best_local_idx].item() * 100
            
            st.metric("Detected Anatomy", label)
            st.info(f"**Confidence Score:** {confidence:.2f}%")
            st.caption(f"Prediction scope used: {prediction_scope}")

            # Top differentials for clinician review
            top_k = min(5, len(allowed_indices))
            top_probs, top_idx_local = torch.topk(scoped_probs, k=top_k)
            top_rows = []
            for rank, (local_idx, cls_prob) in enumerate(zip(top_idx_local.tolist(), top_probs.tolist()), start=1):
                cls_idx = allowed_indices[local_idx]
                cls_name = effective_dataset_names[cls_idx] if cls_idx < len(effective_dataset_names) else f"Class {cls_idx}"
                top_rows.append(
                    {
                        "Rank": rank,
                        "Class": cls_name,
                        "Probability (%)": round(cls_prob * 100, 2),
                    }
                )

            if len(dataset_names) != model_num_classes:
                st.warning(
                    f"Model outputs {model_num_classes} classes, but {len(dataset_names)} classes are currently detected in data. "
                    "Predictions are limited to classes present in the loaded model checkpoint."
                )

            st.write("---")
            st.write("**Top Differential Predictions**")
            st.table(top_rows)

            # Confidence interpretation and suggested next steps
            band = confidence_band(confidence)
            st.write("---")
            st.write("**Clinical Decision Support (AI-assisted)**")
            st.write(f"- Confidence level: **{band}**")

            if confidence >= 85:
                st.success(
                    "Suggested action: Correlate with clinical findings and proceed with standard treatment pathway if consistent."
                )
            elif confidence >= 60:
                st.warning(
                    "Suggested action: Consider secondary review by radiologist and correlate with labs/history before treatment decision."
                )
            else:
                st.error(
                    "Suggested action: Do not rely on this prediction alone; request further imaging or specialist review."
                )

            # Clinician-friendly summary block
            differential_text = ", ".join(
                [f"{row['Class']} ({row['Probability (%)']:.2f}%)" for row in top_rows[:3]]
            )
            st.write("---")
            st.write("**Clinical Summary**")
            st.code(
                f"Primary AI label: {label}\n"
                f"Confidence: {confidence:.2f}% ({band})\n"
                f"Top differentials: {differential_text}\n"
                f"Scope: {prediction_scope}",
                language="text",
            )

            st.caption("This tool supports clinical workflow and is not a standalone diagnostic system.")
            
            # Skull Boundary Visualization (from your HEADCT logic)
            st.write("---")
            st.write("**Edge Analysis (Canny):**")
            edges = cv2.Canny(img_resized.astype(np.uint8), 100, 200)
            fig_edge, ax_edge = plt.subplots()
            ax_edge.imshow(edges, cmap='gray')
            plt.axis('off')
            st.pyplot(fig_edge)

with tab2:
    st.subheader("Global Health Insights")
    if os.path.exists('federated_medical_report.png'):
        st.image('federated_medical_report.png', caption="Global Performance Metrics")
    else:
        st.warning("Run visualize.py to generate the Global Performance Report.")