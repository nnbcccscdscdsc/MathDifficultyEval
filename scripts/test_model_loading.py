#!/usr/bin/env python3
"""
模型加载测试脚本

用于测试不同模型的加载和GPU配置是否正确
"""

import os
import argparse
import logging
from pathlib import Path
import torch
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from scripts.model_evaluation import ModelEvaluator
from scripts.utils import setup_logging

def test_model_loading(model_name: str, quantization: str = "4bit", num_gpus: int = None):
    """测试模型加载"""
    print(f"\n🧪 测试模型加载: {model_name}")
    print("="*60)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"✅ 检测到 {gpu_count} 个GPU")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("❌ 未检测到CUDA GPU")
        return False
    
    try:
        # 创建评估器
        evaluator = ModelEvaluator()
        
        # 确定GPU数量
        if num_gpus is None:
            model_gpu_config = {
                "mistral-community/Mistral-7B-v0.2": 1,
                "lmsys/longchat-7b-16k": 1,
                "Yukang/LongAlpaca-13B-16k": 2,
                "Yhyu13/oasst-rlhf-2-llama-30b-7k-steps-hf": 4,
                "Yukang/LongAlpaca-70B-16k": 4
            }
            num_gpus = model_gpu_config.get(model_name, 1)
        
        print(f"📊 配置信息:")
        print(f"   模型: {model_name}")
        print(f"   量化: {quantization}")
        print(f"   GPU数量: {num_gpus}")
        
        # 检查GPU内存是否足够
        if num_gpus > gpu_count:
            print(f"❌ 请求的GPU数量({num_gpus})超过可用数量({gpu_count})")
            return False
        
        # 加载模型
        print(f"\n🔄 开始加载模型...")
        evaluator.load_model(model_name, quantization, num_gpus)
        
        print(f"✅ 模型加载成功！")
        
        # 测试简单推理
        print(f"\n🧠 测试推理...")
        test_prompt = "What is 2 + 2?"
        try:
            response = evaluator.generate_answer(test_prompt, "{problem}")
            print(f"✅ 推理测试成功")
            print(f"   输入: {test_prompt}")
            print(f"   输出: {response[:100]}...")
        except Exception as e:
            print(f"❌ 推理测试失败: {e}")
            return False
        
        # 清理内存
        del evaluator
        torch.cuda.empty_cache()
        
        print(f"\n🎉 模型 {model_name} 测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="模型加载测试脚本")
    parser.add_argument("--model", type=str, required=True,
                       choices=[
                           "mistral-community/Mistral-7B-v0.2",
                           "lmsys/longchat-7b-16k", 
                           "Yukang/LongAlpaca-13B-16k",
                           "Yhyu13/oasst-rlhf-2-llama-30b-7k-steps-hf",
                           "Yukang/LongAlpaca-70B-16k"
                       ],
                       help="要测试的模型")
    parser.add_argument("--quantization", type=str, default="4bit",
                       choices=["none", "4bit", "8bit"],
                       help="量化方式")
    parser.add_argument("--num-gpus", type=int, default=None,
                       help="GPU数量（默认根据模型自动设置）")
    
    args = parser.parse_args()
    
    print("🚀 模型加载测试工具")
    print("="*60)
    
    success = test_model_loading(
        model_name=args.model,
        quantization=args.quantization,
        num_gpus=args.num_gpus
    )
    
    if success:
        print(f"\n✅ 测试通过！模型 {args.model} 可以正常使用")
        return 0
    else:
        print(f"\n❌ 测试失败！请检查配置和GPU资源")
        return 1

if __name__ == "__main__":
    exit(main()) 