import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet()

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
        self.use_va = model_config.get("use_va", True)
        self.use_emo = model_config.get("use_emo", True)
        self.va_continuous = model_config.get("va_continuous", True)
        if model_config["multi_emotion"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "emotions.json"
                ),
                "r",
            ) as f:
                json_raw = json.load(f)
                n_emotion = len(json_raw["emotion_dict"])
                n_arousal = len(json_raw["arousal_dict"])
                n_valence = len(json_raw["valence_dict"])
            encoder_hidden = model_config["transformer"]["encoder_hidden"]
            self.emotion_emb = nn.Embedding(
                n_emotion,
                encoder_hidden,
            )
            if self.use_va and self.use_emo:
                if self.va_continuous:
                    pass
                self.emotion_emb = nn.Embedding(
                    n_emotion,
                    encoder_hidden//2,
                )
                self.arousal_emb = nn.Embedding(
                    n_arousal,
                    encoder_hidden//4,
                )
                self.valence_emb = nn.Embedding(
                    n_valence,
                    encoder_hidden//4,
                )
            elif self.use_va:
                self.arousal_emb = nn.Embedding(
                    n_arousal,
                    encoder_hidden//2,
                )
                self.valence_emb = nn.Embedding(
                    n_valence,
                    encoder_hidden//2,
                )
            else:
                self.arousal_emb = None
                self.valence_emb = None
            self.emotion_linear = nn.Sequential(
                nn.Linear(encoder_hidden, encoder_hidden),
                nn.ReLU()
            )
            self.joint_linear = nn.Sequential(
                nn.Linear(encoder_hidden*2, encoder_hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.LayerNorm(encoder_hidden)
            )
        
            self.use_emo_classifier = model_config.get("use_emo_classifier", False)
            if self.use_emo_classifier:
                self.emo_classifier = nn.Sequential(
                    nn.Conv1d(model_config["transformer"]["decoder_hidden"], 64, kernel_size=3, padding=1),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(2),

                    nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),  # 变长支持关键

                    nn.Flatten(),  # [B, 128, 1] → [B, 128]
                    nn.Linear(128, 64),
                    nn.Dropout(0.5),
                    nn.Linear(64, n_emotion),
                )
            else:
                self.emo_classifier = None

    def forward(
        self,
        speakers,
        emotions,
        arousals,
        valences,
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
        
        # 原版
        # if self.speaker_emb is not None:
        #     output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
        #         -1, max_src_len, -1
        #     )

        # clean2
        # if self.emotion_emb is not None:
        #     emb = torch.cat((self.emotion_emb(emotions), self.arousal_emb(arousals), self.valence_emb(valences)), dim=-1) 
        #     output = output + self.emotion_linear(emb).unsqueeze(1).expand(
        #         -1, max_src_len, -1
        #     )

        # MSP_finetune_keepall_clean_only_emotion_dbPitch_front_combine
        # if self.speaker_emb is not None and self.emotion_emb is not None:
        #     if self.use_va and self.use_emo:
        #         emb = torch.cat((self.emotion_emb(emotions), self.arousal_emb(arousals), self.valence_emb(valences)), dim=-1)
        #     elif self.use_va:
        #         emb = torch.cat((self.arousal_emb(arousals), self.valence_emb(valences)), dim=-1)
        #     else:
        #         emb = self.emotion_emb(emotions)

        #     output = output + (self.speaker_emb(speakers) + self.emotion_linear(emb)).unsqueeze(1).expand(-1, max_src_len, -1)
            # output = output + self.emotion_linear(emb).unsqueeze(1).expand(-1, max_src_len, -1)
        output = output + self.joint_linear(torch.cat([self.speaker_emb(speakers), self.emotion_emb(emotions)], dim=-1)).unsqueeze(1).expand(-1, max_src_len, -1)

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

        # if self.emotion_emb is not None:
        #     if max_mel_len is None:
        #         max_mel_len = max(mel_lens)
        #     if self.use_va and self.use_emo:
        #         emb = torch.cat((self.emotion_emb(emotions), self.arousal_emb(arousals), self.valence_emb(valences)), dim=-1)
        #     elif self.use_va:
        #         emb = torch.cat((self.arousal_emb(arousals), self.valence_emb(valences)), dim=-1)
        #     else:
        #         emb = self.emotion_emb(emotions)
        #     output = output + self.emotion_linear(emb).unsqueeze(1).expand(
        #         -1, max_mel_len, -1
        #     )

        output, mel_masks = self.decoder(output, mel_masks) # [B, T, hidden(256)]

        if self.emo_classifier is not None:
            emo_pred = self.emo_classifier(output.transpose(1, 2))   # [B, n_emo]
        else:
            emo_pred = None

        output = self.mel_linear(output) # [B, T, mel_dim(80)]
        postnet_output = self.postnet(output) + output # [B, T, mel_dim(80)]

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