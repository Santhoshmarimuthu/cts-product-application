# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import swin_tiny_patch4_window7_224
from transformers import ViTForImageClassification, ViTConfig

# ================= Swin-Unet (Segmentation) =================
class SwinUnet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.encoder = swin_tiny_patch4_window7_224(pretrained=True, features_only=True)
        self.input_conv = nn.Conv2d(in_ch, 3, 1) if in_ch == 1 else nn.Identity()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 512, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 1)
        )

    def forward(self, x):
        x = self.input_conv(x)
        features = self.encoder(x)
        x = features[-1]
        if x.ndim == 4 and x.shape[-1] == 768:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return x

# ================= BrainTumorCNN (Custom CNN) =================
class BrainTumorCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ================= ViT-Based Classifier =================
class BrainTumorViT(nn.Module):
    def __init__(self, num_classes=4, model_name="google/vit-base-patch16-224-in21k", device="cuda"):
        super(BrainTumorViT, self).__init__()
        self.device = device
        # Load pretrained ViT with num_labels adjusted
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        ).to(self.device)

    def forward(self, x):
        # x should be normalized and resized [B, 3, 224, 224]
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        return logits
