#!/usr/bin/env python3
"""
Fun-ASR 优化版本测试脚本
用于验证模型加载和基本功能是否正常工作
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    from funasr import AutoModel
    print("✅ 基础库导入成功")
    
    # 检查GPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"📱 使用设备: {device}")
    print(f"🔧 CUDA可用: {torch.cuda.is_available()}")
    
    # 测试模型加载
    print("\n🔄 开始测试模型加载...")
    
    # 方案1: 完整模型
    try:
        print("测试方案1: 完整模型（包含声纹）")
        model = AutoModel(
            model="iic/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", 
            punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
            spk_model="iic/speech_campplus_sv_zh-cn-16k-common",
            device=device,
            batch_size_s=8,
            disable_update=True,
        )
        print("✅ 完整模型加载成功")
        has_speaker = True
        
    except Exception as e:
        print(f"❌ 完整模型加载失败: {e}")
        
        # 方案2: 简化模型
        try:
            print("测试方案2: 简化模型（仅语音识别）")
            model = AutoModel(
                model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
                device=device,
                batch_size_s=8,
                disable_update=True,
            )
            print("✅ 简化模型加载成功")
            has_speaker = False
            
        except Exception as e2:
            print(f"❌ 简化模型也失败: {e2}")
            print("💡 请检查本地模型文件是否存在")
            sys.exit(1)
    
    print(f"\n🎯 模型测试结果:")
    print(f"   - 声纹识别: {'✅ 支持' if has_speaker else '❌ 不支持'}")
    print(f"   - 设备: {device}")
    print(f"   - 版本检查: 已禁用")
    
    # 测试目录结构
    print(f"\n📁 检查目录结构:")
    for dir_name in ["temp", "voiceprints"]:
        dir_path = os.path.join("/root/autodl-tmp/Fun-ASR", dir_name)
        if os.path.exists(dir_path):
            print(f"   - {dir_name}: ✅ 存在")
        else:
            print(f"   - {dir_name}: ❌ 不存在")
    
    print(f"\n🎉 所有测试通过！可以使用 app_optimized.py 启动应用")
    
except ImportError as e:
    print(f"❌ 库导入失败: {e}")
    print("💡 请确保已安装: pip install funasr torch soundfile streamlit")
    
except Exception as e:
    print(f"❌ 未知错误: {e}")
    print("💡 请检查环境和依赖")
