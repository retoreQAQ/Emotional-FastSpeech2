import os
import numpy as np
import librosa
from tqdm import tqdm

def calculate_max_sequence_length(audio_dir):
    """
    计算音频文件的最大序列长度
    """
    # FastSpeech2参数
    sampling_rate = 22050  # 采样率
    hop_length = 256      # 帧移
    
    max_length = 0
    total_files = 0
    file_lengths = []
    
    # 获取所有音频文件
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(('.wav', '.flac', '.mp3')):
                audio_files.append(os.path.join(root, file))
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 遍历所有音频文件
    for audio_file in tqdm(audio_files):
        try:
            # 加载音频文件
            y, sr = librosa.load(audio_file, sr=sampling_rate)
            
            # 计算序列长度 (帧数)
            length = len(y) // hop_length
            file_lengths.append(length)
            max_length = max(max_length, length)
            total_files += 1
            
        except Exception as e:
            print(f"处理文件 {audio_file} 时出错: {str(e)}")
    
    # 计算统计信息
    mean_length = np.mean(file_lengths)
    median_length = np.median(file_lengths)
    std_length = np.std(file_lengths)
    
    # 计算对应的秒数
    max_seconds = max_length * hop_length / sampling_rate
    mean_seconds = mean_length * hop_length / sampling_rate
    median_seconds = median_length * hop_length / sampling_rate
    
    print("\n统计信息:")
    print(f"总文件数: {total_files}")
    print(f"最大序列长度: {max_length} 帧")
    print(f"平均序列长度: {mean_length:.2f} 帧")
    print(f"中位数序列长度: {median_length:.2f} 帧")
    print(f"序列长度标准差: {std_length:.2f} 帧")
    print(f"\n对应时长:")
    print(f"最大时长: {max_seconds:.2f} 秒")
    print(f"平均时长: {mean_seconds:.2f} 秒")
    print(f"中位数时长: {median_seconds:.2f} 秒")
    
    # 保存结果到文件
    with open("sequence_length_stats.txt", "w") as f:
        f.write(f"总文件数: {total_files}\n")
        f.write(f"最大序列长度: {max_length} 帧\n")
        f.write(f"平均序列长度: {mean_length:.2f} 帧\n")
        f.write(f"中位数序列长度: {median_length:.2f} 帧\n")
        f.write(f"序列长度标准差: {std_length:.2f} 帧\n")
        f.write(f"\n对应时长:\n")
        f.write(f"最大时长: {max_seconds:.2f} 秒\n")
        f.write(f"平均时长: {mean_seconds:.2f} 秒\n")
        f.write(f"中位数时长: {median_seconds:.2f} 秒\n")
    
    return max_length, mean_length, median_length, std_length

if __name__ == "__main__":
    audio_dir = "/home/you/workspace/son/FastSpeech2/raw_data/MSP_single"
    print(f"开始计算 {audio_dir} 目录下音频文件的最大序列长度...")
    calculate_max_sequence_length(audio_dir)
    print("完成！结果已保存到 sequence_length_stats.txt") 