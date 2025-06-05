#!/bin/bash
ckpt="finetune_1150000"
text="I am so hungry that I have to eat something."

# 定义情绪类别
emotions=("A" "C" "D" "F" "H" "N" "O" "S" "U" "X")  # 可根据你数据集修改

for emo in "${emotions[@]}"; do
  tag="${emo}"
  echo "🔊 Synthesizing: $tag"
  python synthesize.py \
    --text "$text" \
    --restore_step $ckpt \
    --mode single \
    -p config/MSP/preprocess.yaml \
    -m config/MSP/model.yaml \
    -t config/MSP/train_finetune.yaml \
    --emotion $emo \
    --valence 4.0 \
    --arousal 4.0 \
    --pitch_control 1.0 \
    --energy_control 1.0 \
    --duration_control 1.0 \
    --output_dir output/synth_$tag
done