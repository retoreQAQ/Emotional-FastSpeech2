# finetune.py
import argparse
import os
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from distutils.version import LooseVersion

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model_finetune, get_vocoder, get_param_num, freeze_modules
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
from evaluate import evaluate

def main(args, configs):
    
    disable_tqdm = os.environ.get("DISABLE_TQDM", "false").lower() == "true"

    preprocess_config, model_config, train_config = configs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.max
    # Load dataset
    dataset = Dataset("train.txt", preprocess_config, train_config, sort=True, drop_last=True)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    loader = DataLoader(dataset, batch_size=batch_size * group_size, shuffle=True, collate_fn=dataset.collate_fn)

    # Load model and freeze
    freeze_encoder = train_config["freeze"]["encoder"]
    freeze_decoder = train_config["freeze"]["decoder"]
    freeze_variance = train_config["freeze"]["variance"]
    freeze_speaker = train_config["freeze"]["speaker"]
    freeze_emotion = train_config["freeze"]["emotion"]
    freeze_postnet = train_config["freeze"]["postnet"]
    model, optimizer = get_model_finetune(args, configs, device, train=True)
    freeze_modules(model, freeze_encoder=freeze_encoder, freeze_decoder=freeze_decoder, freeze_variance=freeze_variance, freeze_speaker=freeze_speaker, freeze_emotion=freeze_emotion, freeze_postnet=freeze_postnet)
    model = nn.DataParallel(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Logging
    train_log_path = os.path.join(train_config["path"]["log_path"], "finetune_train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "finetune_val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training setup
    step = args.restore_step + 1
    total_step = train_config["step"]["total_step"]
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    outer_bar = tqdm(total=total_step, desc="Finetuning", position=0, disable=disable_tqdm)
    outer_bar.n = args.restore_step
    outer_bar.update()
    epoch = 1

    while step <= total_step:
        inner_bar = tqdm(total=len(loader), desc=f"Epoch {epoch}", position=1, disable=disable_tqdm)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                # Forward
                output = model(
                    batch[2], batch[12], batch[13], batch[14],
                    batch[3], batch[4], batch[5],
                    batch[6], batch[7], batch[8],
                    batch[9], batch[10], batch[11]
                )
                # Loss
                losses = Loss(batch, output)
                total_loss = losses[0] / grad_acc_step
                total_loss.backward()

                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    loss_values = [l.item() for l in losses]
                    msg = f"Step {step}/{total_step}, " + \
                          f"Total: {loss_values[0]:.4f}, Mel: {loss_values[1]:.4f}, PostNet: {loss_values[2]:.4f}, " + \
                          f"Pitch: {loss_values[3]:.4f}, Energy: {loss_values[4]:.4f}, Duration: {loss_values[5]:.4f}"
                    print(msg)
                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(msg + "\n")
                    log(train_logger, step, losses=loss_values)

                if step % synth_step == 0:
                    fig, wav_rec, wav_pred, tag = synth_one_sample(batch, output, vocoder, model_config, preprocess_config)
                    log(train_logger, fig=fig, tag=f"Finetune/step_{step}_{tag}")
                    log(train_logger, audio=wav_rec, sampling_rate=22050, tag=f"Finetune/{tag}_reconstructed")
                    log(train_logger, audio=wav_pred, sampling_rate=22050, tag=f"Finetune/{tag}_synthesized")

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    model.train()

                if step % save_step == 0:
                    ckpt_path = os.path.join(train_config["path"]["ckpt_path"], f"finetune_{step}.pth.tar")
                    torch.save({
                        "model": model.module.state_dict(),
                        "optimizer": optimizer._optimizer.state_dict(),
                    }, ckpt_path)

                if step >= total_step:
                    print("Finetuning complete.")
                    return
                step += 1
                outer_bar.update(1)
            inner_bar.update(1)
        epoch += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("-p", "--preprocess_config", type=str, required=True)
    parser.add_argument("-m", "--model_config", type=str, required=True)
    parser.add_argument("-t", "--train_config", type=str, required=True)
    parser.add_argument("--restore_path", type=str, required=True)
    args = parser.parse_args()

    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)

    configs = (preprocess_config, model_config, train_config)
    main(args, configs)
