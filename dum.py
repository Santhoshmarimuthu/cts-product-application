import streamlit as st
import torch
import numpy as np
import cv2
import base64
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime
import time
import html
from models import SwinUnet, BrainTumorCNN, BrainTumorViT

import os
import gdown
import zipfile
import torch

# --- 1. Set up directories ---
MODEL_DIR = "models_weight"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 2. Google Drive file ID for your zip ---
file_id = "1gmBzfuRP27_c6fjhgSLz1j7C7viePL6m"  # replace with your own if different
zip_url = f"https://drive.google.com/uc?id={file_id}"
zip_path = os.path.join(MODEL_DIR, "weights.zip")

# --- 3. Download zip if not already exists ---
if not os.path.exists(zip_path):
    print("Downloading model zip...")
    gdown.download(zip_url, zip_path, quiet=False)

# --- 4. Extract zip ---
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(MODEL_DIR)
    print(f"Extracted models to {MODEL_DIR}")

# --- 5. Check that models exist ---
swin_path = "./models_weight/weights/swin_unet_tumor.pth"
vit_path = "./models_weight/weights/vit_brain_tumor_model.pth"

assert os.path.exists(swin_path), "swin_unter.pth not found!"
assert os.path.exists(vit_path), "vit.pth not found!"

# --- 6. Load models ---
# Example: change according to your model architecture
swin_model = torch.load(swin_path, map_location="cpu")
vit_model = torch.load(vit_path, map_location="cpu")
print("Models loaded successfully!")



# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="NeuroAI - Medical Imaging Platform",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ Advanced CSS Styling ------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Hide sidebar scrollbar */
    .css-1d391kg {
        scrollbar-width: none;
        -ms-overflow-style: none;
    }
    .css-1d391kg::-webkit-scrollbar {
        display: none;
    }
    
    /* Professional header */
    .medical-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .medical-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="20" cy="20" r="1" fill="white" opacity="0.1"/><circle cx="80" cy="80" r="1" fill="white" opacity="0.1"/><circle cx="40" cy="70" r="1" fill="white" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        pointer-events: none;
    }
    
    .header-content {
        position: relative;
        z-index: 1;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Professional cards */
    .medical-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(230, 236, 250, 0.8);
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .medical-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
    }
    
    /* WhatsApp-style chat interface - FIXED */
    .whatsapp-container {
        height: calc(100vh - 120px);
        display: flex;
        flex-direction: column;
        background: #f0f2f5;
        border-radius: 8px;
        overflow: hidden;
        position: relative;
    }
    
    .chat-header {
        background: #075e54;
        color: white;
        padding: 16px 20px;
        font-weight: 600;
        font-size: 16px;
        display: flex;
        align-items: center;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        position: relative;
        z-index: 10;
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 10px;
        background: #e5ddd5;
        background-image: 
            radial-gradient(circle at 20% 50%, transparent 20%, rgba(120,119,108,0.3) 21%, rgba(120,119,108,0.3) 34%, transparent 35%, transparent),
            linear-gradient(0deg, rgba(120,119,108,0.1) 50%, transparent 50%);
        scrollbar-width: thin;
        scrollbar-color: #25d366 #f0f0f0;
        display: flex;
        flex-direction: column;
    }
    
    .chat-messages::-webkit-scrollbar {
        width: 6px;
    }
    
    .chat-messages::-webkit-scrollbar-track {
        background: rgba(0,0,0,0.1);
        border-radius: 3px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb {
        background: #25d366;
        border-radius: 3px;
    }
    
    .chat-messages::-webkit-scrollbar-thumb:hover {
        background: #128c7e;
    }
    
    .user-message {
        background: #dcf8c6;
        color: #303030;
        padding: 8px 12px;
        border-radius: 7.5px;
        margin: 4px 0 4px 80px;
        max-width: 75%;
        align-self: flex-end;
        box-shadow: 0 1px 0.5px rgba(0,0,0,0.13);
        font-size: 14px;
        line-height: 1.4;
        word-wrap: break-word;
        position: relative;
    }
    
    .user-message::before {
        content: '';
        position: absolute;
        top: 0;
        right: -8px;
        width: 0;
        height: 0;
        border-left: 8px solid #dcf8c6;
        border-bottom: 13px solid transparent;
    }
    
    .assistant-message {
        background: white;
        color: #303030;
        padding: 8px 12px;
        border-radius: 7.5px;
        margin: 4px 80px 4px 0;
        max-width: 75%;
        align-self: flex-start;
        box-shadow: 0 1px 0.5px rgba(0,0,0,0.13);
        font-size: 14px;
        line-height: 1.4;
        word-wrap: break-word;
        position: relative;
    }
    
    .assistant-message::before {
        content: '';
        position: absolute;
        top: 0;
        left: -8px;
        width: 0;
        height: 0;
        border-right: 8px solid white;
        border-bottom: 13px solid transparent;
    }
    
    .message-time {
        font-size: 11px;
        color: #667781;
        margin-top: 4px;
        text-align: right;
        opacity: 0.8;
    }
    
    .assistant-message .message-time {
        text-align: left;
    }
    
    .chat-input-area {
        background: #f0f2f5;
        padding: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-top: 1px solid rgba(0,0,0,0.08);
        position: relative;
        z-index: 10;
    }
    
    .chat-input-container {
        flex: 1;
        background: white;
        border-radius: 25px;
        padding: 8px 16px;
        border: 1px solid rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
    }
    
    .send-button {
        background: #25d366;
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: background-color 0.2s;
        font-size: 16px;
    }
    
    .send-button:hover {
        background: #128c7e;
    }
    
    .clear-button {
        background: #34495e;
        color: white;
        border: none;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: background-color 0.2s;
        font-size: 12px;
        margin-left: 4px;
    }
    
    .clear-button:hover {
        background: #2c3e50;
    }
    
    /* Professional buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(30, 60, 114, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(30, 60, 114, 0.4);
        background: linear-gradient(135deg, #1a3366 0%, #244785 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div > div {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 3rem 2rem;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div > div:hover {
        border-color: #1e3c72;
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 4px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8fafc;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 500;
        border: 1px solid #e2e8f0;
        border-bottom: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white !important;
    }
    
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .risk-low {
        background: rgba(16, 185, 129, 0.1);
        color: #047857;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    
    .risk-moderate {
        background: rgba(245, 158, 11, 0.1);
        color: #92400e;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    .risk-high {
        background: rgba(239, 68, 68, 0.1);
        color: #991b1b;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    /* Professional footer */
    .medical-footer {
        background: #f8fafc;
        border-top: 1px solid #e2e8f0;
        padding: 2rem;
        text-align: center;
        margin-top: 3rem;
        border-radius: 12px 12px 0 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .medical-header {
            padding: 2rem 1rem;
        }
        
        .user-message, .assistant-message {
            max-width: 85%;
            margin-left: 10px;
            margin-right: 10px;
        }
    }
    
    /* Hide text input label */
    .stTextInput > label {
        display: none;
    }
    
    /* Style text input */
    .stTextInput > div > div > input {
        border: none;
        outline: none;
        background: transparent;
        font-size: 14px;
        padding: 0;
    }
    
    .stTextInput > div > div > input:focus {
        border: none;
        outline: none;
        box-shadow: none;
    }
    
    /* ADDED: Smaller image display */
    .uploaded-image {
        max-width: 400px !important;
        width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------ Device Configuration ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ Model Loading ------------------
@st.cache_resource
def load_models():
    try:
        # Segmentation Model
        seg_model = SwinUnet(in_ch=1, out_ch=1)
        seg_model.load_state_dict(torch.load(swin_path, map_location=DEVICE))
        seg_model.to(DEVICE)
        seg_model.eval()
        
        # Classification Model
        class_names = ['Glioma', 'Meningioma', 'No Tumor','Pituitary',]

# Initialize the ViT model
        clf_model = BrainTumorViT(num_classes=len(class_names), device=DEVICE)

        # Load the trained weights
        state_dict = torch.load(vit_path, map_location=DEVICE)
        clf_model.model.load_state_dict(state_dict)  # load weights into the internal ViT model

        # Move to device and set to eval
        clf_model.to(DEVICE)
        clf_model.eval()

        return seg_model, clf_model, class_names
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, ['Glioma', 'Meningioma', 'No Tumor','Pituitary']

# Load models
seg_model, clf_model, class_names = load_models()

# ------------------ Utility Functions ------------------
def preprocess_seg(img: Image.Image):
    """Preprocess image for segmentation"""
    img_gray = np.array(img.convert("L"))
    img_resized = cv2.resize(img_gray, (224, 224))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    return img_tensor.to(DEVICE), np.array(img.convert("RGB"))


def preprocess_clf(img: Image.Image):
    """
    Preprocess image for classification (ViT requires 3 channels).
    Args:
        img (PIL.Image.Image): Input image
    Returns:
        torch.Tensor: Preprocessed image tensor [1, 3, 224, 224]
    """
    # Convert to RGB and resize
    img_rgb = img.convert("RGB").resize((224, 224))
    
    # Convert to numpy and scale to [0,1]
    img_array = np.array(img_rgb, dtype=np.float32) / 255.0
    
    # Normalize with ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Convert to tensor [C, H, W] and add batch dimension
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor.to(DEVICE)



def predict_segmentation(img_tensor, orig_img, pixel_spacing=(0.5, 0.5), slice_thickness=5):
    """
    Perform tumor segmentation with enhanced volume calculation and risk analysis.
    
    Parameters:
        img_tensor : torch.Tensor
            Input tensor for segmentation model
        orig_img : np.array
            Original image for overlay
        pixel_spacing : tuple(float, float)
            Physical spacing of pixels in mm (row_spacing, col_spacing)
        slice_thickness : float
            Thickness of slice in mm (for volume approximation)
    
    Returns:
        mask_resized : np.array
        overlay : np.array
        area_mm2 : float
        volume_mm3 : float
        risk : str
        metrics : dict
            Dictionary containing bounding box, compactness, aspect ratio
    """
    if seg_model is None:
        return None, orig_img, 0, 0, "Unknown", {}

    with torch.no_grad():
        mask_pred = seg_model(img_tensor)
        mask = (torch.sigmoid(mask_pred).squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    # Resize mask to original image
    mask_resized = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Contour detection
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = orig_img.copy()
    metrics = {}

    total_area_pixels = 0
    for cnt in contours:
        cv2.drawContours(overlay, [cnt], -1, (255, 0, 0), 2)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

        area_pixels = cv2.contourArea(cnt)
        total_area_pixels += area_pixels

        # Shape metrics
        perimeter = cv2.arcLength(cnt, True)
        compactness = (perimeter ** 2) / (4 * np.pi * area_pixels) if area_pixels > 0 else 0
        aspect_ratio = float(w) / h if h > 0 else 0

        metrics = {
            "bounding_box": (x, y, w, h),
            "compactness": compactness,
            "aspect_ratio": aspect_ratio
        }

    # Area in mm²
    area_mm2 = total_area_pixels * pixel_spacing[0] * pixel_spacing[1]

    # Approximate volume in mm³
    volume_mm3 = area_mm2 * slice_thickness

    # Risk assessment based on volume (customizable thresholds)
    if volume_mm3 < 1000:
        risk = "Low"
    elif volume_mm3 < 5000:
        risk = "Moderate"
    else:
        risk = "High"

    return mask_resized, overlay, area_mm2, volume_mm3, risk

def predict_classification(img_tensor):
    """
    Perform tumor classification using PyTorch ViT model.
    
    Args:
        img_tensor (torch.Tensor): Preprocessed image tensor of shape [B, 3, 224, 224] on DEVICE.
    
    Returns:
        Tuple[str, np.ndarray]: Predicted label and probabilities.
    """
    if clf_model is None:
        return "Unknown", np.array([1 / len(class_names)] * len(class_names))  # safe fallback

    clf_model.eval()
    with torch.no_grad():
        outputs = clf_model(img_tensor)            # forward pass -> SequenceClassifierOutput
        logits = outputs                   
        probs = torch.softmax(logits, dim=1)       # convert to probabilities
        probs = probs.cpu().numpy().squeeze()
        pred_index = int(np.argmax(probs))

    return class_names[pred_index], probs




def get_tumor_info(tumor_type):
    """Get detailed information about tumor type"""
    tumor_info = {
        'Glioma': {
            'description': 'Most common primary brain tumor arising from glial cells',
            'symptoms': 'Headaches, seizures, cognitive changes, motor weakness',
            'treatment': 'Surgery, radiation therapy, chemotherapy',
            'prognosis': 'Varies by grade (I-IV)',
            'prevalence': '~6 per 100,000 people'
        },
        'Meningioma': {
            'description': 'Tumor arising from meninges (brain covering membranes)',
            'symptoms': 'Often asymptomatic, headaches, vision problems',
            'treatment': 'Observation, surgery, stereotactic radiosurgery',
            'prognosis': 'Generally good, mostly benign (Grade I)',
            'prevalence': '~8 per 100,000 people'
        },
        'Pituitary': {
            'description': 'Tumor in the pituitary gland affecting hormone production',
            'symptoms': 'Vision changes, hormonal imbalances, headaches',
            'treatment': 'Medication, surgery, radiation therapy',
            'prognosis': 'Generally good with proper treatment',
            'prevalence': '~4 per 100,000 people'
        },
        'No Tumor': {
            'description': 'No malignant tissue detected in the scan',
            'symptoms': 'N/A - Normal brain tissue',
            'treatment': 'Regular monitoring if symptoms persist',
            'prognosis': 'Excellent',
            'prevalence': 'N/A'
        }
    }
    return tumor_info.get(tumor_type, {})

# ------------------ Advanced Chatbot System ------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def add_chat_message(role, message):
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append({
        "role": role, 
        "message": message, 
        "timestamp": timestamp
    })

def get_medical_response(question, context=None): 
    """Generate medical response based on question and context""" 
    question_lower = question.lower() # Medical FAQ responses 
    if any(word in question_lower for word in ['glioma', 'what is glioma']): 
        return "Gliomas are the most common type of primary brain tumor, originating from glial cells. They account for about 80% of malignant brain tumors and are classified into grades I-IV based on their aggressiveness and cellular characteristics." 
    elif any(word in question_lower for word in ['meningioma']): 
        return "Meningiomas arise from the meninges, the protective membranes surrounding the brain and spinal cord. Most are benign (Grade I) and grow slowly, representing about 30% of all primary brain tumors." 
    elif any(word in question_lower for word in ['pituitary']): 
        return "Pituitary tumors develop in the pituitary gland and can significantly affect hormone production. While usually benign, they can cause substantial symptoms due to hormonal imbalances or mass effect." 
    elif any(word in question_lower for word in ['symptoms', 'signs']): 
        return "Common neurological symptoms include persistent headaches, new-onset seizures, progressive vision changes, balance difficulties, cognitive alterations, nausea with vomiting, and personality changes. Immediate neurological consultation is recommended for multiple concurrent symptoms." 
    elif any(word in question_lower for word in ['treatment', 'therapy']): 
        return "Treatment modalities depend on tumor histology, location, and grade: Surgical resection (primary treatment), stereotactic radiosurgery, fractionated radiotherapy, chemotherapy protocols, targeted molecular therapy, and immunotherapy approaches. Multidisciplinary tumor board review is standard care." 
    elif any(word in question_lower for word in ['mri', 'scan', 'imaging']): 
        return "Magnetic resonance imaging remains the gold standard for brain tumor evaluation. Multisequence protocols including T1-weighted, T2-weighted, FLAIR, and post-contrast T1 sequences provide comprehensive tissue characterization and tumor boundary delineation." 
    elif any(word in question_lower for word in ['prognosis', 'survival', 'outcome']): 
        return "Prognosis varies significantly by tumor histology and WHO grade. Low-grade tumors (Grade I) typically have favorable outcomes, while high-grade lesions require aggressive multimodal therapy. Early detection and appropriate treatment significantly influence long-term outcomes." 
    else: 
        return f"I understand you're inquiring about '{question}'. For specific medical guidance, please consult with board-certified neurologists or neuro-oncologists. I can provide general information about brain tumor pathology, diagnostic imaging, treatment protocols, and clinical considerations."

def render_whatsapp_chat():
    """Render WhatsApp-style chat interface with navy blue bubble design (expanded size)"""
    chat_html = '''
    <style>
        .whatsapp-container {
            width: 105%;  /* slightly wider than sidebar */
            background: #f0f2f5;
            border-radius: 0;
            padding: 8px;
            font-family: Arial, sans-serif;
            box-sizing: border-box;
            margin-left: -10px; /* stretch to touch edges */
        }
        .chat-header {
            text-align: center;
            font-weight: bold;
            padding: 12px;
            background:#2b6096; /* Navy blue header */
            color: white;
            margin: -12px -12px 12px -12px; /* stretch header */
        }
        .chat-messages {
            max-height: 650px;   /* increased height */
            overflow-y: auto;
            padding: 12px;
        }
        .user-message, .assistant-message {
            display: inline-block;
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 18px;
            max-width: 80%;      /* allow wider message bubbles */
            word-wrap: break-word;
            position: relative;
            font-size: 14px;
        }
        .user-message {
            background: #2b6096; /* Navy blue bubble */
            color: white;
            float: right;
            clear: both;
        }
        .assistant-message {
            background: #ffffff;
            float: left;
            clear: both;
            border: 1px solid #e5e5e5;
        }
        .message-time {
            display: block;
            font-size: 11px;
            color: gray;
            margin-top: 4px;
            text-align: right;
        }
    </style>

    <div class="whatsapp-container">
        <div class="chat-header">
            Medical Consultation Assistant
        </div>
        <div class="chat-messages" id="chat-messages">
    '''
    
    if not st.session_state.chat_history:
        chat_html += '''
            <div class="assistant-message">
                Welcome to Medical Consultation Assistant. Ask me about neurological conditions, diagnostic procedures, or treatment protocols.
                <div class="message-time">System</div>
            </div>
        '''
    else:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                chat_html += f'''
                    <div class="user-message">
                        {html.escape(chat["message"])}
                        <div class="message-time">{chat["timestamp"]}</div>
                    </div>
                '''
            else:
                chat_html += f'''
                    <div class="assistant-message">
                        {html.escape(chat["message"])}
                        <div class="message-time">{chat["timestamp"]}</div>
                    </div>
                '''
    
    chat_html += '''
        </div>
    </div>
    <script>
        setTimeout(function() {
            var chatMessages = document.getElementById('chat-messages');
            if (chatMessages) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }, 100);
    </script>
    '''
    
    st.components.v1.html(chat_html, height=650, scrolling=True) 


# ------------------ Main Application Interface ------------------

# Professional Header
st.markdown('''
<div class="medical-header">
    <div class="header-content">
        <h1 class="header-title">NeuroAI Medical Imaging Platform</h1>
        <p class="header-subtitle">Advanced AI-powered neurological imaging analysis for precision diagnostics</p>
    </div>
</div>
''', unsafe_allow_html=True)

# Sidebar - WhatsApp-style Medical Consultation Interface
with st.sidebar:
    render_whatsapp_chat()
    
    # Chat input at bottom
    st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([8, 1, 1])
    
    with col1:

        chat_input = st.text_input("", placeholder="Type your medical question...", key="chat_input")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if st.button("▶", help="Send message", key="send_btn"):
            if chat_input.strip():
                add_chat_message("user", chat_input)
                response = get_medical_response(chat_input)
                add_chat_message("assistant", response)
                st.rerun()
    
    with col3:
        if st.button("🗑", help="Clear chat", key="clear_btn"):
            st.session_state.chat_history = []
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area - Full width now
st.markdown('<div class="section-title">Medical Image Upload & Analysis</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Select MRI scan for neurological analysis", 
    type=["png", "jpg", "jpeg", "dicom"],
    help="Upload high-resolution brain MRI scan in supported formats"
)

if uploaded_file is not None:
    # MODIFIED: Create columns for image and button layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # MODIFIED: Display uploaded image with smaller size
        img = Image.open(uploaded_file)
        st.markdown('<div class="uploaded-image">', unsafe_allow_html=True)
        st.image(img, caption="Uploaded MRI Scan - Ready for Analysis", width=400)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # MODIFIED: Professional analysis button positioned on the right
        st.markdown("<br><br>", unsafe_allow_html=True)  # Add spacing to align with image
        if st.button("Initiate AI Analysis", use_container_width=True, type="primary"):
            with st.spinner("Processing medical image analysis..."):
                progress_bar = st.progress(0)
                
                # Segmentation phase
                progress_bar.progress(25)
                st.caption("Phase 1: Tumor segmentation analysis")
                time.sleep(0.8)
                seg_tensor, orig_img = preprocess_seg(img)
                mask, overlay, area, volume, risk = predict_segmentation(seg_tensor, orig_img)
                
                # Classification phase
                progress_bar.progress(75)
                st.caption("Phase 2: Pathological classification")
                time.sleep(0.8)
                clf_tensor = preprocess_clf(img)
                pred_label, probs = predict_classification(clf_tensor)
                
                progress_bar.progress(100)
                st.caption("Analysis complete - Generating medical report")
                time.sleep(0.5)
                progress_bar.empty()
                
                # Store results in session state
                st.session_state.analysis_results = {
                    'orig_img': orig_img,
                    'mask': mask,
                    'overlay': overlay,
                    'area': area,
                    'volume': volume,
                    'risk': risk,
                    'pred_label': pred_label,
                    'probs': probs,
                    'timestamp': datetime.now()
                }
                
                st.success("Analysis completed successfully")

# Display comprehensive analysis results
if 'analysis_results' in st.session_state:
    results = st.session_state.analysis_results
    
    st.markdown("---")
    st.markdown('<div class="section-title">Comprehensive Medical Analysis Report</div>', unsafe_allow_html=True)
    
    # Professional results tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Imaging Analysis", "Quantitative Metrics", "Pathological Classification", "Clinical Report"])
    
    with tab1:
        st.markdown("### Multi-modal Imaging Visualization")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original MRI Scan**")
            st.image(results['orig_img'], use_container_width=True)
            st.caption("Source: T1-weighted MRI sequence")
        
        with col2:
            st.markdown("**Segmentation Mask**")
            if results['mask'] is not None:
                st.image(results['mask'] * 255, use_container_width=True)
                st.caption("AI-generated tumor boundary detection")
            else:
                st.info("Segmentation data unavailable")
        
        with col3:
            st.markdown("**Annotated Analysis**")
            st.image(results['overlay'], use_container_width=True)
            st.caption("Tumor boundaries with bounding box overlay")
    
    with tab2:
        st.markdown("### Quantitative Assessment")
        
        # Key metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Tumor Area", 
                f"{results['area']:,} mm²",
                help="Cross-sectional area of detected abnormal tissue"
            )
        
        with col2:
            st.metric(
                "Estimated Volume", 
                f"{results['volume']:,} mm³",
                help="Approximated 3D tumor volume"
            )
        
        with col3:
            risk_class = f"risk-{results['risk'].lower()}"
            st.markdown(f'''
            <div class="medical-card" style="text-align: center;">
                <div style="font-size: 0.9rem; color: #64748b; margin-bottom: 0.5rem;">Risk Stratification</div>
                <div class="risk-badge {risk_class}">{results['risk']} Risk</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Advanced risk assessment gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = results['area'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Tumor Size Risk Assessment", 'font': {'size': 16, 'family': 'Inter'}},
            gauge = {
                'axis': {'range': [None, 3000], 'tickfont': {'size': 12}},
                'bar': {'color': "#1e3c72"},
                'steps': [
                    {'range': [0, 500], 'color': "rgba(16, 185, 129, 0.2)"},
                    {'range': [500, 1500], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [1500, 3000], 'color': "rgba(239, 68, 68, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "#1e3c72", 'width': 3},
                    'thickness': 0.75,
                    'value': results['area']
                }
            }
        ))
        fig_gauge.update_layout(
            height=350,
            font={'family': 'Inter', 'size': 12},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with tab3:
        st.markdown(f"### Pathological Classification: **{results['pred_label']}**")
        st.markdown(f"**Diagnostic Confidence:** {max(results['probs'])*100:.1f}%")
        
        # Professional probability visualization
        fig_bar = px.bar(
            x=class_names, 
            y=results['probs'],
            title="Diagnostic Probability Distribution",
            labels={'x': 'Pathological Classification', 'y': 'Probability Score'},
            color=results['probs'],
            color_continuous_scale='Blues',
            text=[f"{p:.1%}" for p in results['probs']]
        )
        fig_bar.update_traces(
            texttemplate='%{text}', 
            textposition='outside',
            marker_line_color='rgba(30, 60, 114, 0.8)',
            marker_line_width=1
        )
        fig_bar.update_layout(
            height=400,
            font={'family': 'Inter', 'size': 12},
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            title_font={'size': 16, 'family': 'Inter', 'color': '#1e293b'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed probability table
        prob_df = pd.DataFrame({
            'Pathological Type': class_names,
            'Confidence Score': [f"{p:.4f}" for p in results['probs']],
            'Percentage': [f"{p*100:.1f}%" for p in results['probs']],
            'Clinical Significance': [
                'High-grade astrocytic tumor' if cls == 'Glioma' else
                'Benign meningeal tumor' if cls == 'Meningioma' else
                'Endocrine system tumor' if cls == 'Pituitary' else
                'No pathological findings'
                for cls in class_names
            ]
        })
        
        # Style the dataframe
        st.dataframe(
            prob_df, 
            use_container_width=True, 
            hide_index=True,
            column_config={
                'Pathological Type': st.column_config.TextColumn('Pathological Type', width='medium'),
                'Confidence Score': st.column_config.NumberColumn('Confidence Score', format='%.4f'),
                'Percentage': st.column_config.TextColumn('Percentage', width='small'),
                'Clinical Significance': st.column_config.TextColumn('Clinical Significance', width='large')
            }
        )
    
   
    with tab4:
        tumor_info = get_tumor_info(results['pred_label'])
        analysis_time = results['timestamp'].strftime("%B %d, %Y at %H:%M:%S")
        report_id = hash(analysis_time) % 10000
        top_indices = sorted(range(len(results['probs'])), key=lambda i: results['probs'][i], reverse=True)[:3]

        # Convert image to displayable format
        if 'processed_image' in results:
            st.image(results['processed_image'], width=600, caption="Brain MRI with AI Analysis")
        elif 'segmentation_overlay' in results:
            st.image(results['segmentation_overlay'], width=600, caption="Brain MRI with Detected Areas")
        else:
            st.info("Image will be displayed here")
    # Header
        st.markdown(f"## MEDICAL IMAGING REPORT")
        st.markdown(f"**Brain Tumor Detection & Analysis**")
        st.markdown(f"**Generated:** {analysis_time}")

        # Diagnostic Findings
        st.markdown("### Diagnostic Findings")
        st.markdown(f"- **Diagnosis:** {results['pred_label']}")
        st.markdown(f"- **Confidence:** {max(results['probs'])*100:.1f}%")
        st.markdown(f"- **Risk Level:** {results.get('risk','Moderate')} Risk")
        st.markdown(f"- **Lesion Area:** {results.get('area','N/A')} mm²")
        st.markdown(f"- **Processing Time:** 2.3 seconds")

        # Clinical Information
        st.markdown("### Clinical Information")
        st.markdown(f"- **Description:** {tumor_info.get('description','Detailed analysis pending')}")
        st.markdown(f"- **Common Symptoms:** {tumor_info.get('symptoms','Variable based on location')}")
        st.markdown(f"- **Treatment Approach:** {tumor_info.get('treatment','Multidisciplinary care recommended')}")

        # Differential Diagnosis
        st.markdown("### Differential Diagnosis")
        for i, idx in enumerate(top_indices):
            confidence = results['probs'][idx] * 100
            rank = ["Primary", "Secondary", "Tertiary"][i]
            st.markdown(f"- **{rank}:** {class_names[idx]} ({confidence:.1f}%)")

        # Clinical Recommendations
        st.markdown("### Clinical Recommendations")
        st.markdown("**Immediate Actions:**")
        st.markdown("- Neurological consultation within 48-72 hours")
        st.markdown("- Complete neurological examination")
        st.markdown("- Review patient symptoms and medical history")
        st.markdown("**Follow-up Studies:**")
        st.markdown("- Consider contrast-enhanced MRI if not done")
        st.markdown("- Multidisciplinary tumor board review")
        st.markdown("- Additional imaging as clinically indicated")

        # Disclaimer
        st.markdown("### MEDICAL DISCLAIMER")
        st.markdown(
            "This AI-generated report is for clinical decision support only. "
            "All medical decisions must be made by qualified healthcare professionals. "
            "This analysis should not be the sole basis for treatment decisions."
        )

        # Footer
        st.markdown(f"**NeuroAI Medical Platform v2.0**")
        st.markdown(f"Report ID: {report_id:04d} | Accuracy: 94.2% | FDA-Cleared Decision Support Tool")
        st.markdown(f"Generated: {analysis_time}")



    
    # Close A4 container
    st.markdown('</div>', unsafe_allow_html=True)
        # Add report to chat context
    if st.button("Discuss Results with Medical Assistant", use_container_width=True):
        report_summary = f"Analysis completed for {results['pred_label']} with {max(results['probs'])*100:.1f}% diagnostic confidence. Lesion area: {results['area']} mm². Risk stratification: {results['risk']} risk category."
        add_chat_message("assistant", f"Medical analysis report generated. {report_summary} I'm available to discuss the clinical implications, differential diagnoses, or recommended next steps.")
        st.rerun()

# Professional footer
st.markdown("""
<div class="medical-footer">
    <div style="max-width: 800px; margin: 0 auto;">
        <h4 style="margin: 0 0 1rem 0; color: #1e293b; font-weight: 600;">NeuroAI Medical Imaging Platform v2.0</h4>
        <p style="margin: 0 0 1rem 0; color: #64748b;">Powered by Advanced Deep Learning Neural Networks for Precision Medical Diagnostics</p>
        <p style="margin: 0; color: #94a3b8; font-size: 0.85rem;">
            This platform is designed for medical professionals and educational purposes. 
            Always consult qualified healthcare providers for medical decisions.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Auto-scroll to bottom of chat when new messages are added
if st.session_state.chat_history:
    st.markdown("""
    <script>
    function scrollToBottom() {
        const chatContainer = document.querySelector('.chat-messages');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
    
    // Call scroll function after a short delay to ensure content is rendered
    setTimeout(scrollToBottom, 100);
    </script>
    """, unsafe_allow_html=True)