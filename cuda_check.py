#!/usr/bin/env python3
"""
CUDA环境检查脚本
用于诊断CUDA相关问题
"""

import torch
import subprocess
import sys
import os

def check_cuda_environment():
    print("="*60)
    print("CUDA环境检查")
    print("="*60)
    
    # 1. 基本信息
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA编译版本: {torch.version.cuda}")
    
    # 2. CUDA可用性
    print(f"\nCUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    
    # 3. 如果有CUDA设备，显示详细信息
    if torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            print(f"\n设备 {i}:")
            print(f"  名称: {torch.cuda.get_device_name(i)}")
            print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            
        # 4. 测试CUDA张量操作
        print(f"\n测试CUDA张量操作:")
        try:
            x = torch.tensor([1.0, 2.0, 3.0])
            print(f"CPU张量: {x}")
            
            x_cuda = x.cuda()
            print(f"CUDA张量: {x_cuda}")
            print("✅ CUDA张量操作成功")
            
            # 简单计算测试
            result = x_cuda * 2
            print(f"CUDA计算结果: {result}")
            print("✅ CUDA计算操作成功")
            
        except Exception as e:
            print(f"❌ CUDA操作失败: {e}")
    
    # 5. 检查NVIDIA驱动
    print(f"\n" + "="*60)
    print("NVIDIA驱动检查")
    print("="*60)
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi可用")
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"驱动版本信息: {line.strip()}")
                if 'CUDA Version' in line:
                    print(f"CUDA版本信息: {line.strip()}")
        else:
            print("❌ nvidia-smi不可用")
    except FileNotFoundError:
        print("❌ 未找到nvidia-smi命令")
    
    # 6. 环境变量检查
    print(f"\n" + "="*60)
    print("环境变量检查")
    print("="*60)
    
    cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES']
    for var in cuda_vars:
        value = os.environ.get(var, '未设置')
        print(f"{var}: {value}")
    
    # 7. 推荐解决方案
    print(f"\n" + "="*60)
    print("推荐解决方案")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ PyTorch无法使用CUDA")
        print("解决方案:")
        print("1. 检查NVIDIA驱动是否正确安装")
        print("2. 重新安装匹配的PyTorch版本:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        
    elif torch.cuda.device_count() == 0:
        print("❌ PyTorch检测到CUDA但没有找到设备")
        print("解决方案:")
        print("1. 重启计算机")
        print("2. 检查显卡是否被其他程序占用")
        print("3. 更新显卡驱动")
        print("4. 重新安装CUDA Toolkit")
        
    else:
        print("✅ CUDA环境正常")
        print("如果基准测试仍有问题，使用 --device cpu 参数")

if __name__ == "__main__":
    check_cuda_environment()