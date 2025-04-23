import os
import shutil
from pathlib import Path

def extract_files(source_dir, target_dir):
    """
    深度遍历源目录，将所有非文件夹文件复制到目标目录
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 用于记录复制的文件数量
    count = 0
    
    # 遍历源目录
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 获取源文件的完整路径
            source_file = os.path.join(root, file)
            # 获取目标文件的完整路径
            target_file = os.path.join(target_dir, file)
            
            # 如果目标文件已存在，添加数字后缀
            if os.path.exists(target_file):
                base, ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(target_file):
                    target_file = os.path.join(target_dir, f"{base}_{counter}{ext}")
                    counter += 1
            
            # 复制文件
            shutil.copy2(source_file, target_file)
            count += 1
            if count % 100 == 0:
                print(f"已复制 {count} 个文件...")
    
    print(f"总共复制了 {count} 个文件到 {target_dir}")

if __name__ == "__main__":
    source_dir = "/home/you/workspace/son/FastSpeech2/raw_data/MSP"
    target_dir = "/home/you/workspace/son/FastSpeech2/raw_data/MSP_single"
    
    print(f"开始从 {source_dir} 提取文件到 {target_dir}")
    extract_files(source_dir, target_dir)
    print("完成！") 