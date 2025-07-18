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
            model.load_state_dict(ckpt["model"], strict=False)

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

    if args and args.restore_path:
        ckpt_path = args.restore_path
        print(f"正在加载检查点: {ckpt_path}")
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return None

        # 宽松加载参数
        ckpt_model_state = ckpt["model"]
        current_model_state = model.state_dict()

        matched_state = {}
        skipped = []
        for k, v in ckpt_model_state.items():
            if k not in current_model_state:
                print(f"跳过参数 {k}: 在当前模型中不存在")
                skipped.append(k)
                continue

            if k == "speaker_emb.weight":
                if not model_config["use_libri_speaker"]:
                    print(f"不加载旧的speaker_emb")
                    skipped.append(k)
                    continue
                else:
                    pretrained, current = v.shape[0], current_model_state[k].shape[0]
                    if pretrained <= current:
                        print(f"合并speaker_emb: 预训练={pretrained}, 当前={current}")
                        new_weight = current_model_state[k].clone()
                        new_weight[:pretrained] = v  # 用旧的前部分参数替换
                        matched_state[k] = new_weight
                    else:
                        print(f"无法加载speaker_emb: 预训练={pretrained}, 当前={current}")
                        skipped.append(k)
                        continue
            elif k == "emotion_emb.weight" or k == "arousal_emb.weight" or k == "valence_emb.weight":
                # 对于情感相关参数，如果形状不匹配，使用当前模型的参数
                print(f"使用当前模型的情感参数: {k}")
                matched_state[k] = current_model_state[k]
            elif v.shape == current_model_state[k].shape:
                matched_state[k] = v
            else:
                print(f"参数形状不匹配: {k}, 预训练={v.shape}, 当前={current_model_state[k].shape}")
                skipped.append(k)

        print(f"成功加载 {len(matched_state)} 个参数")
        print(f"加载的参数: {list(matched_state.keys())}")
        print(f"跳过的参数: {skipped}")

        model.load_state_dict(matched_state, strict=False)

    if train:
        scheduled_optim = ScheduledOptim(
            model, train_config, model_config,
            args.restore_step if args else 0
        )
        if args and args.restore_step and "optimizer" in ckpt:
            try:
                scheduled_optim.load_state_dict(ckpt["optimizer"])
                print("优化器状态加载成功")
            except Exception as e:
                print(f"优化器状态加载失败,将重新初始化: {e}")
        model.train()
        return model, scheduled_optim

    model.eval()
    model.requires_grad_ = False
    return model

def freeze_modules(model, freeze_encoder=True, freeze_decoder=True, freeze_variance=True, 
                  freeze_speaker=True, freeze_emotion=True, freeze_postnet=True):
    """冻结选定的模块
    
    Args:
        model: FastSpeech2 模型
        freeze_encoder: 是否冻结编码器参数
        freeze_decoder: 是否冻结解码器参数
        freeze_variance: 是否冻结方差适配器参数
        freeze_speaker: 是否冻结说话人嵌入参数
        freeze_emotion: 是否冻结情感相关参数
        freeze_postnet: 是否冻结后处理网络参数
    """
    # 统计参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = 0
    
    # if train_all:
    #     # 全解冻模式,所有参数都可训练
    #     for p in model.parameters():
    #         p.requires_grad = True
    #     trainable_params = total_params
    #     print("\n🔓 全解冻模式: 所有参数都可训练")
    #     print(f"   - 总参数数量: {total_params}")
    #     print(f"   - 可训练参数: {trainable_params}")
    #     return
        
    # 默认冻结所有参数
    for p in model.parameters():
        p.requires_grad = False
    
    # 根据配置解冻特定模块
    if not freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = True
        trainable_params += sum(p.numel() for p in model.encoder.parameters())
        print("编码器解冻: 编码器参数可训练")
    else:
        print("编码器冻结: 编码器参数不可训练")
        
    if not freeze_decoder:
        for p in model.decoder.parameters():
            p.requires_grad = True
        trainable_params += sum(p.numel() for p in model.decoder.parameters())
        print("解码器解冻: 解码器参数可训练")
    else:
        print("解码器冻结: 解码器参数不可训练")
        
    # 方差适配器
    if not freeze_variance:
        for p in model.variance_adaptor.parameters():
            p.requires_grad = True
        trainable_params += sum(p.numel() for p in model.variance_adaptor.parameters())
        print("方差适配器解冻: 方差适配器参数可训练")
    else:
        print("方差适配器冻结: 方差适配器参数不可训练")
    
    # 说话人嵌入
    if not freeze_speaker and model.speaker_emb is not None:
        for p in model.speaker_emb.parameters():
            p.requires_grad = True
        trainable_params += sum(p.numel() for p in model.speaker_emb.parameters())
        print("说话人嵌入解冻: 说话人嵌入参数可训练")
    else:
        print("说话人嵌入冻结: 说话人嵌入参数不可训练")
    
    # 后处理网络
    if not freeze_postnet:
        for p in model.mel_linear.parameters():
            p.requires_grad = True
        for p in model.postnet.parameters():
            p.requires_grad = True
        trainable_params += sum(p.numel() for p in model.mel_linear.parameters())
        trainable_params += sum(p.numel() for p in model.postnet.parameters())
        print("后处理网络解冻: 梅尔线性层和后处理网络参数可训练")
    else:
        print("后处理网络冻结: 梅尔线性层和后处理网络参数不可训练")
    
    # 打印参数统计
    print(f"\n参数统计:")
    print(f"   - 总参数数量: {total_params}")
    print(f"   - 可训练参数: {trainable_params}")
    print(f"   - 冻结参数: {total_params - trainable_params}")
    print(f"   - 可训练比例: {trainable_params/total_params*100:.2f}%")


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
