#!/bin/bash

# Conda环境名称
CONDA_ENV_NAME=fs2

# Python脚本路径
PYTHON_SCRIPT_PATH=finetune.py

# 训练参数
# ARGS="-p config/CREMA-D/preprocess.yaml -m config/CREMA-D/model.yaml -t config/CREMA-D/train_finetune.yaml --restore_step 900000 --restore_path output/ckpt/LibriTTS/900000.pth.tar"
ARGS="-p config/CREMA-D/preprocess.yaml -m config/CREMA-D/model.yaml -t config/CREMA-D/train_finetune.yaml --restore_step 1200000 --restore_path output/ckpt/CREMA-D_only_emo_dbPitch_front_newCombine/finetune_1200000.pth.tar"

# 日志目录与文件
LOG_DIR="run_in_background_log"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/script_${TIMESTAMP}.log"

# 检查conda
if ! command -v conda &> /dev/null; then
    echo "❌ Conda 未安装，请先安装 Conda。" | tee -a "$LOG_FILE"
    exit 1
fi

# 初始化conda shell
eval "$(conda shell.bash hook)"

# 激活环境
if [ "$CONDA_DEFAULT_ENV" = "$CONDA_ENV_NAME" ]; then
    echo "✅ 已在 Conda 环境：$CONDA_ENV_NAME" | tee -a "$LOG_FILE"
else
    echo "🔁 激活 Conda 环境：$CONDA_ENV_NAME" | tee -a "$LOG_FILE"
    if ! conda activate "$CONDA_ENV_NAME" 2>> "$LOG_FILE"; then
        echo "❌ 激活失败：$CONDA_ENV_NAME" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

# 检查脚本是否存在
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "❌ Python 脚本不存在: $PYTHON_SCRIPT_PATH" | tee -a "$LOG_FILE"
    exit 1
fi

# 设置 LD_LIBRARY_PATH 以避免 GLIBCXX 错误
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "📦 设置 LD_LIBRARY_PATH=$LD_LIBRARY_PATH" | tee -a "$LOG_FILE"

# 执行 Python 脚本
echo "🚀 启动脚本: $PYTHON_SCRIPT_PATH $ARGS" | tee -a "$LOG_FILE"
DISABLE_TQDM=true nohup python3 "$PYTHON_SCRIPT_PATH" $ARGS >> "$LOG_FILE" 2>&1 &

echo "🎉 脚本已后台运行，日志路径: $LOG_FILE" | tee -a "$LOG_FILE"
