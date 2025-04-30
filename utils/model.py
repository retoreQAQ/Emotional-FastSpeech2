import os
import json

import torch
import numpy as np

import hifigan
from model import FastSpeech2, ScheduledOptim


def get_model(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if args:
        if args.restore_step:
            ckpt_path = os.path.join(
                train_config["path"]["ckpt_path"],
                "{}.pth.tar".format(args.restore_step),
            )
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model"])

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config, args.restore_step
        )
        if args.restore_step:
            scheduled_optim.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def get_model_finetune(args, configs, device, train=False):
    (preprocess_config, model_config, train_config) = configs

    model = FastSpeech2(preprocess_config, model_config).to(device)

    if args and args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        print(f"🔄 正在加载检查点: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            print(f"❌ 加载检查点失败: {e}")
            return None

        # 宽松加载参数
        ckpt_model_state = ckpt["model"]
        current_model_state = model.state_dict()

        matched_state = {}
        for k, v in ckpt_model_state.items():
            if k not in current_model_state:
                print(f"⚠️ 跳过参数 {k}: 在当前模型中不存在")
                continue

            if k == "speaker_emb.weight":
                pretrained, current = v.shape[0], current_model_state[k].shape[0]
                if pretrained <= current:
                    print(f"🔧 合并speaker_emb: 预训练={pretrained}, 当前={current}")
                    new_weight = current_model_state[k].clone()
                    new_weight[:pretrained] = v  # 用旧的前部分参数替换
                    matched_state[k] = new_weight
                else:
                    print(f"⚠️ 无法加载speaker_emb: 预训练={pretrained}, 当前={current}")
                    continue
            elif v.shape == current_model_state[k].shape:
                matched_state[k] = v

        skipped = [k for k in current_model_state if k not in matched_state]
        print(f"✅ Loaded parameters: {list(matched_state.keys())}")
        print(f"⚠️ Skipped parameters (missing or shape mismatch): {skipped}")

        model.load_state_dict(matched_state, strict=False)
        print(f"✅ 成功加载 {len(matched_state)} 个参数")

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config,
            args.restore_step if args else 0
        )
        if args and args.restore_step and "optimizer" in ckpt:
            try:
                scheduled_optim.load_state_dict(ckpt["optimizer"])
                print("✅ 优化器状态加载成功")
            except Exception as e:
                print(f"⚠️ 优化器状态加载失败,将重新初始化: {e}")
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param


def get_vocoder(config, device):
    name = config["vocoder"]["model"]
    speaker = config["vocoder"]["speaker"]

    if name == "MelGAN":
        if speaker == "LJSpeech":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "linda_johnson"
            )
        elif speaker == "universal":
            vocoder = torch.hub.load(
                "descriptinc/melgan-neurips", "load_melgan", "multi_speaker"
            )
        vocoder.mel2wav.eval()
        vocoder.mel2wav.to(device)
    elif name == "HiFi-GAN":
        with open("hifigan/config.json", "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)
        vocoder = hifigan.Generator(config)
        if speaker == "LJSpeech":
            ckpt = torch.load("hifigan/generator_LJSpeech.pth.tar")
        elif speaker == "universal":
            ckpt = torch.load("hifigan/generator_universal.pth.tar")
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        vocoder.to(device)

    return vocoder


def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)

    wavs = (
        wavs.cpu().numpy()
        * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs
