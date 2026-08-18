"""
X-VQA model definition.

Swin-Base visual encoder + ELECTRA-Base text encoder, fused by a 6-layer
Multimodal Cross-Attention Network (MCAN), classified over a 1,833-answer
vocabulary. Also contains the wrapper and reshape transform that make
Score-CAM usable on a two-input transformer model.
"""

import torch
import torch.nn as nn
from transformers import SwinModel, ElectraModel

class MCANLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.text_to_image_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.image_to_text_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.ffn_text = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_image = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.layer_norm3 = nn.LayerNorm(hidden_dim)
        self.layer_norm4 = nn.LayerNorm(hidden_dim)

    def forward(self, text_features, image_features):
        text_attended, _ = self.text_to_image_attn(query=text_features, key=image_features, value=image_features)
        text_out = self.layer_norm1(text_features + text_attended)
        image_attended, _ = self.image_to_text_attn(query=image_features, key=text_features, value=text_features)
        image_out = self.layer_norm2(image_features + image_attended)
        text_out = self.layer_norm3(text_out + self.ffn_text(text_out))
        image_out = self.layer_norm4(image_out + self.ffn_image(image_out))
        return text_out, image_out


class MCAN(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, num_layers=6):
        super().__init__()
        self.layers = nn.ModuleList([
            MCANLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])

    def forward(self, text_features, image_features):
        for layer in self.layers:
            text_features, image_features = layer(text_features, image_features)
        return text_features, image_features


class XVQAModel(nn.Module):
    def __init__(self, num_answers, hidden_dim=512):
        super().__init__()
        self.vision_encoder = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")
        self.text_encoder = ElectraModel.from_pretrained("google/electra-base-discriminator")
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, hidden_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        self.mcan = MCAN(hidden_dim=hidden_dim, num_heads=8, num_layers=6)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_answers)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        image_features = self.vision_encoder(pixel_values=pixel_values).last_hidden_state
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        image_features = self.vision_proj(image_features)
        text_features = self.text_proj(text_features)
        text_fused, image_fused = self.mcan(text_features, image_features)
        text_pooled = text_fused.mean(dim=1)
        image_pooled = image_fused.mean(dim=1)
        combined_features = torch.cat([text_pooled, image_pooled], dim=1)
        return self.classifier(combined_features)


# Fixed wrapper — expands text batch to match ScoreCAM's internal batch size
class XVQACamWrapper(torch.nn.Module):
    def __init__(self, model, input_ids, attention_mask):
        super().__init__()
        self.model = model
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        input_ids = self.input_ids.expand(batch_size, -1)
        attention_mask = self.attention_mask.expand(batch_size, -1)
        return self.model(pixel_values, input_ids, attention_mask)


def swin_reshape_transform(tensor):
    batch, seq_len, hidden_dim = tensor.shape
    h = w = int(seq_len ** 0.5)
    assert h * w == seq_len, f"Cannot reshape seq_len={seq_len} into square grid"
    result = tensor.reshape(batch, h, w, hidden_dim)
    result = result.transpose(2, 3).transpose(1, 2)
    return result