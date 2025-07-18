import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.pitch_loss_weight = model_config["loss"]["pitch_loss_weight"]

        self.use_emo_classifier = model_config.get("use_emo_classifier", False)
        self.emo_cls_loss = nn.CrossEntropyLoss()
        self.emo_loss_weight = model_config.get("emo_loss_weight", 0.0)

        self.use_new_loss = model_config["use_new_loss"]

    def forward(self, inputs, predictions):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
        ) = inputs[13:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            _,
            emo_predictions,
        ) = predictions
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False

        if self.pitch_feature_level == "phoneme_level":
            pitch_predictions = pitch_predictions.masked_select(src_masks)
            pitch_targets = pitch_targets.masked_select(src_masks)
        elif self.pitch_feature_level == "frame_level":
            pitch_predictions = pitch_predictions.masked_select(mel_masks)
            pitch_targets = pitch_targets.masked_select(mel_masks)

        if self.energy_feature_level == "phoneme_level":
            energy_predictions = energy_predictions.masked_select(src_masks)
            energy_targets = energy_targets.masked_select(src_masks)
        if self.energy_feature_level == "frame_level":
            energy_predictions = energy_predictions.masked_select(mel_masks)
            energy_targets = energy_targets.masked_select(mel_masks)

        log_duration_predictions = log_duration_predictions.masked_select(src_masks)
        log_duration_targets = log_duration_targets.masked_select(src_masks)

        if self.use_new_loss:
            # 缓存 mel 维度
            mel_dim = mel_targets.size(-1)

            # 执行 masked_select 并 reshape
            mel_predictions = mel_predictions.masked_fill(~mel_masks.unsqueeze(-1), 0.0)
            postnet_mel_predictions = postnet_mel_predictions.masked_fill(~mel_masks.unsqueeze(-1), 0.0)
            mel_targets = mel_targets.masked_fill(~mel_masks.unsqueeze(-1), 0.0)


            # 频谱一分为二
            split_point = mel_dim // 2

            # 高频加权 mel loss
            mel_loss = self.mae_loss(mel_predictions[:, :split_point], mel_targets[:, :split_point]) + \
                    2.0 * self.mae_loss(mel_predictions[:, split_point:], mel_targets[:, split_point:])

            postnet_mel_loss = self.mae_loss(postnet_mel_predictions[:, :split_point], mel_targets[:, :split_point]) + \
                            2.0 * self.mae_loss(postnet_mel_predictions[:, split_point:], mel_targets[:, split_point:])
            # Compute delta loss
            batch_size = mel_masks.size(0)
            time_steps = mel_masks.size(1)

            postnet_mel = postnet_mel_predictions.view(batch_size, time_steps, mel_dim)
            mel_tgt = mel_targets.view(batch_size, time_steps, mel_dim)

            delta_pred = postnet_mel[:, 1:] - postnet_mel[:, :-1]
            delta_tgt = mel_tgt[:, 1:] - mel_tgt[:, :-1]
            delta_loss = self.mae_loss(delta_pred, delta_tgt)

            delta2_pred = delta_pred[:, 1:] - delta_pred[:, :-1]
            delta2_tgt = delta_tgt[:, 1:] - delta_tgt[:, :-1]
            delta_loss += 0.5 * self.mae_loss(delta2_pred, delta2_tgt)
        else:
        # 原loss
            mel_loss = self.mae_loss(mel_predictions, mel_targets)
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
            delta_loss = torch.tensor(0.0).to(mel_targets.device)



        pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
        energy_loss = self.mse_loss(energy_predictions, energy_targets)
        duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)

        if self.use_emo_classifier:
            emo_loss = self.emo_cls_loss(emo_predictions, emotion_targets)
        else:
            emo_loss = torch.tensor(0.0).to(mel_targets.device)

        total_loss = (
            mel_loss
            + postnet_mel_loss
            + delta_loss
            + duration_loss
            + pitch_loss * self.pitch_loss_weight
            + energy_loss
            + emo_loss * self.emo_loss_weight
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            emo_loss,
            delta_loss,
        )
