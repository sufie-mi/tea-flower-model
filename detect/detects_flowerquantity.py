# Perform batch detection (进行批量detect)
import subprocess
import os
from pathlib import Path

# Define detection function (定义检测函数)
def run_detection(source_path):
    command = [
        'python', 'detect.py',
        '--source', str(source_path)
    ]
    subprocess.run(command, check=True)

# Input directory and detection script path (输入目录和检测脚本路径)
input_dir = 'datasets/path'
# Get all subfolder paths (获取所有子文件夹路径)
subfolders = [f for f in Path(input_dir).iterdir() if f.is_dir()]
# print(subfolders)
# Iterate through each subfolder and call the detection function (遍历每个子文件夹，调用检测函数)
for subfolder in subfolders:
    print(f"Processing folder: {subfolder}")
    run_detection(subfolder)

