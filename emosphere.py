import os
import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import argparse
import yaml

# === CCC Loss ===
def concordance_cc(pred, target):
    pred_mean = pred.mean(dim=0)
    target_mean = target.mean(dim=0)
    cov = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)
    pred_var = ((pred - pred_mean) ** 2).mean(dim=0)
    target_var = ((target - target_mean) ** 2).mean(dim=0)
    ccc = (2 * cov) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)
    return ccc.mean()

def ccc_loss(pred, target):
    return 1 - concordance_cc(pred, target)

# === Dataset ===
class AVDDataset(Dataset):
    def __init__(self, txt_path, processor, raw_data_dir):
        self.processor = processor
        self.raw_data_dir = raw_data_dir
        self.entries = []
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                utt_id = parts[0]
                speaker = parts[1]
                avd = [float(parts[-3]), float(parts[-2]), float(parts[-1])]
                self.entries.append((utt_id, speaker, avd))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        utt_id, speaker, avd = self.entries[idx]
        wav_path = os.path.join(self.raw_data_dir, speaker, utt_id + '.wav')
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        # inputs = self.processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding="longest")
        return waveform.squeeze(), torch.tensor(avd, dtype=torch.float)
    
    @staticmethod
    def collate_fn(batch, processor=None):
        assert processor is not None, "Must provide processor to collate_fn"
        waveforms = [item[0].numpy() for item in batch]  # ← 转成 np.ndarray
        labels = torch.stack([item[1] for item in batch])  # [B, 3]
        inputs = processor(waveforms, sampling_rate=16000, return_tensors="pt", padding=True)
        return inputs.input_values, labels

# === Model ===
class Wav2Vec2AVDPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
        self.head = nn.Sequential(
            nn.Linear(self.wav2vec.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, input_values):
        features = self.wav2vec(input_values).last_hidden_state
        pooled = features.mean(dim=1)
        return self.head(pooled)

# === Validation ===
def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            with torch.cuda.amp.autocast():
                pred = model(x)
            preds.append(pred)
            labels.append(y)
    pred_all = torch.cat(preds, dim=0)
    label_all = torch.cat(labels, dim=0)
    return concordance_cc(pred_all, label_all).item()

# === Training ===
def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")

    train_set = AVDDataset(config["train_txt"], processor, config["raw_data_dir"])
    val_set = AVDDataset(config["val_txt"], processor, config["raw_data_dir"])
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, num_workers=4, collate_fn=lambda batch: AVDDataset.collate_fn(batch, processor=processor))
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False, num_workers=4, collate_fn=lambda batch: AVDDataset.collate_fn(batch, processor=processor))

    model = Wav2Vec2AVDPredictor().to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"])
    scaler = torch.cuda.amp.GradScaler()  # AMP

    best_val_ccc = -float('inf')

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = ccc_loss(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_ccc = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch+1}] Train CCC Loss: {avg_loss:.4f} | Val CCC: {val_ccc:.4f}")

        # Save best checkpoint
        os.makedirs(config["save_dir"], exist_ok=True)
        if val_ccc > best_val_ccc:
            best_val_ccc = val_ccc
            torch.save(model.state_dict(), os.path.join(config["save_dir"], "best.ckpt"))

# === Entry ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    train(config)
