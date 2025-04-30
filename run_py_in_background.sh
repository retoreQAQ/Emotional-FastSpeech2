#!/bin/bash

# 选取conda环境
CONDA_ENV_NAME=fs2

# 选取用于运行的py脚本
PYTHON_SCRIPT_PATH=/home/you/workspace/son/FastSpeech2/finetune.py

# 训练参数
ARGS="-p config/MSP/preprocess.yaml -m config/MSP/model.yaml -t config/MSP/train_finetune.yaml --restore_step 900000"

# 指定日志目录
LOG_DIR="/home/you/workspace/son/FastSpeech2/run_in_background_log"

# 创建日志目录（如果不存在）
mkdir -p "$LOG_DIR"

# 指定日志文件名
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="${LOG_DIR}/script_${TIMESTAMP}.log"

# 检查Conda是否安装
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first." | tee -a "$LOG_FILE"
    exit 1
fi

# 初始化conda
eval "$(conda shell.bash hook)"

# 检查当前环境
if [ "$CONDA_DEFAULT_ENV" = "$CONDA_ENV_NAME" ]; then
    echo "Already in the correct Conda environment: $CONDA_ENV_NAME" | tee -a "$LOG_FILE"
else
    echo "Activating Conda environment: $CONDA_ENV_NAME" | tee -a "$LOG_FILE"
    if ! conda activate "$CONDA_ENV_NAME" 2>> "$LOG_FILE"; then
        echo "Failed to activate Conda environment: $CONDA_ENV_NAME" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

# 检查Python脚本是否存在
if [ ! -f "$PYTHON_SCRIPT_PATH" ]; then
    echo "Python script not found: $PYTHON_SCRIPT_PATH" | tee -a "$LOG_FILE"
    exit 1
fi

# 执行Python脚本（修正错别字 `eecho` -> `echo`）
echo "Executing Python script: $PYTHON_SCRIPT_PATH with arguments: $ARGS" | tee -a "$LOG_FILE"
DISABLE_TQDM=true nohup python3 "$PYTHON_SCRIPT_PATH" $ARGS >> "$LOG_FILE" 2>&1 &

echo "Script started in background. Check $LOG_FILE for output." | tee -a "$LOG_FILE"




