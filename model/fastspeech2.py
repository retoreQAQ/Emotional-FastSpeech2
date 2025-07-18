import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from .sphericalEmotionEncoder import SphericalEmotionEncoder, SphericalEmotionEncoder_old
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config
        self.encoder = Encoder(model_config)
        self.spherical_emotion_encoder = SphericalEmotionEncoder(preprocess_config, model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()
        self.joint_linear = nn.Sequential(
                nn.Linear(model_config["transformer"]["encoder_hidden"]*2, model_config["transformer"]["encoder_hidden"]),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.LayerNorm(model_config["transformer"]["encoder_hidden"])
            )

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )
        
        self.emotion_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "emotions.json"
                ),
                "r",
            ) as f:
                n_emotion = len(json.load(f))
            self.emotion_emb = nn.Embedding(
                n_emotion,
                model_config["transformer"]["encoder_hidden"],
            )

    def forward(
        self,
        speakers,
        emotions,
        arousals,
        valences,
        dominances,
        r_norms,
        thetas,
        phis,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.encoder(texts, src_masks)
        
        # # 原版
        # if self.speaker_emb is not None:
        #     output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
        #         -1, max_src_len, -1
        #     )
        
        # # 仅emo
        # if self.emotion_emb is not None:
        #     output = output + self.emotion_emb(emotions).unsqueeze(1).expand(
        #         -1, max_src_len, -1
        #     )
            
        # output = output + self.spherical_emotion_encoder(r_norms, thetas, phis, emotions).unsqueeze(1).expand(-1, max_src_len, -1)

        # new_combine
        output = output + self.joint_linear(torch.cat([self.speaker_emb(speakers), self.spherical_emotion_encoder(r_norms, thetas, phis, emotions)], dim=-1)).unsqueeze(1).expand(-1, max_src_len, -1)

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            d_targets,
            p_control,
            e_control,
            d_control,
        )
        def plot_mel_tensor(mel, title="Mel", save_path=None):
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.imshow(mel.detach().cpu().T, aspect="auto", origin="lower", interpolation="none")
            plt.colorbar()
            plt.title(title)
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
                print(f"Saved plot to {save_path}")
            else:
                plt.show()
        output, mel_masks = self.decoder(output, mel_masks) # [B, T, hidden(256)]
        output = self.mel_linear(output) # [B, T, mel_dim(80)]
        # plot_mel_tensor(mel_base[0], "mel_base", save_path="mel_base.png")
        postnet_output = self.postnet(output) + output # [B, T, mel_dim(80)]
        # plot_mel_tensor(postnet_output[0], "mel_post", save_path="mel_post.png")
        emo_pred = None

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            emo_pred,
        )