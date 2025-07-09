import torch
import torch.nn as nn
import math
import os
import json

class SphericalEmotionEncoder(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super().__init__()
        hidden_dim = model_config["emosphere"]["hidden"]
        with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "emotions.json"), "r") as f:
            n_emotion = len(json.load(f))

        self.style_proj = nn.Linear(40, hidden_dim // 2)           # 输入 [θ_norm, φ_norm]
        self.intensity_proj = nn.Linear(10, hidden_dim)       # 输入 r
        self.class_embed = nn.Embedding(n_emotion, hidden_dim // 2)
        self.class_proj = nn.Linear(hidden_dim // 2, hidden_dim // 2)

        self.after_proj = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Softplus()
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def pos_enc(self, x, L=10):
        """
        x: [B] → 输出 [B, 2L]
        编码为: [sin(2⁰πx), cos(2⁰πx), ..., sin(2^{L-1}πx), cos(2^{L-1}πx)]
        """
        freq_bands = 2.0 ** torch.arange(L).float().to(x.device) * math.pi  # [L]
        x_expand = x.unsqueeze(-1) * freq_bands  # [B, L]
        return torch.cat([torch.sin(x_expand), torch.cos(x_expand)], dim=-1)  # [B, 2L]


    def forward(self, r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor, emotion_id: torch.Tensor):
        """
        r: [B]         - 已归一化的强度
        theta, phi: [B] - 原始角度，单位是弧度
        emotion_id: [B] - int 类型 emotion 类别编号
        return: h_emo: [B, D]
        """

        # 角度归一化 θ ∈ [0, π] → [–1, 1], φ ∈ [–π, π] → [–1, 1]
        # theta_norm = (theta / math.pi) * 2 - 1     # [–1, 1]
        # phi_norm = phi / math.pi                   # [–1, 1]

        # style_input = torch.stack([torch.sin(theta), torch.cos(theta), torch.sin(phi), torch.cos(phi)], dim=-1)  # [B, 4]
        theta_enc = self.pos_enc(theta, L=10)  # [B, 20]
        phi_enc = self.pos_enc(phi, L=10)      # [B, 20]
        style_input = torch.cat([theta_enc, phi_enc], dim=-1)  # [B, 40]
        h_sty = self.style_proj(style_input)       # [B, D]

        r_enc = self.pos_enc(r, L=5)        # [B, 10]
        h_int = self.intensity_proj(r_enc)         # [B, D]

        h_cls = self.class_embed(emotion_id)       # [B, D]
        h_cls = self.class_proj(h_cls)             # [B, D]

        # 拼接风格与类别 → 激活 → 归一化
        h = torch.cat([h_sty, h_cls], dim=-1)      # [B, 2D]
        h = self.after_proj(h)
        h = self.activation(h)
        h = self.norm(h)

        # 最后加上强度分量
        h_emo = self.output_proj(h + h_int)                          # [B, D]

        return h_emo
