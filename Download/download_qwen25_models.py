#!/usr/bin/env python3
"""
Qwen2.5系列模型下载脚本 (安全版本 - 无删除操作)
支持下载不同大小的Qwen2.5模型
Safe Version - 不会删除任何现有文件，支持断点续传
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os
import argparse
from huggingface_hub import snapshot_download

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 支持的Qwen2.5模型列表
SUPPORTED_MODELS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "1.5b": "Qwen/Qwen2.5-1.5B-Instruct", 
    "3b": "Qwen/Qwen2.5-3B-Instruct",
    "7b": "Qwen/Qwen2.5-7B-Instruct",
    "14b": "Qwen/Qwen2.5-14B-Instruct",
    "32b": "Qwen/Qwen2.5-32B-Instruct",
    "72b": "Qwen/Qwen2.5-72B-Instruct"
}

def check_existing_model(model_cache_dir):
    """
    检查现有模型文件的状态
    参数:
        model_cache_dir: 模型缓存目录路径
    返回:
        (has_files, total_size): (是否有文件, 总文件大小)
    """
    # 检查目录是否存在
    if not os.path.exists(model_cache_dir):
        logger.info("📂 未找到现有模型缓存目录")
        return False, 0
    
    logger.info(f"📂 发现现有模型缓存目录: {model_cache_dir}")
    
    # 搜索所有.safetensors模型文件
    safetensors_files = []
    total_size = 0
    
    # 遍历目录中的所有文件
    for root, dirs, files in os.walk(model_cache_dir):
        for file in files:
            if file.endswith('.safetensors'):
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    total_size += file_size
                    safetensors_files.append((file, file_size))
    
    # 显示找到的文件信息
    if safetensors_files:
        logger.info(f"📊 发现 {len(safetensors_files)} 个现有模型文件:")
        for file, size in safetensors_files:
            logger.info(f"   📄 {file}: {size / (1024**3):.2f} GB")
        logger.info(f"📊 现有文件总大小: {total_size / (1024**3):.2f} GB")
        return True, total_size
    else:
        logger.info("📊 未找到模型文件，将从零开始下载")
        return False, 0

def download_qwen25_model_safe(model_size):
    """
    安全下载指定大小的Qwen2.5模型
    参数:
        model_size: 模型大小 (如 "7b", "3b" 等)
    """
    if model_size not in SUPPORTED_MODELS:
        logger.error(f"❌ 不支持的模型大小: {model_size}")
        logger.info(f"✅ 支持的模型大小: {', '.join(SUPPORTED_MODELS.keys())}")
        return "不支持的模型大小"
    
    model_name = SUPPORTED_MODELS[model_size]
    
    logger.info(f"📥 开始安全下载Qwen2.5-{model_size.upper()}-Instruct模型: {model_name}")
    logger.info("🛡️  安全模式：将保留现有文件并支持断点续传")
    
    try:
        # 步骤1: 检查现有模型文件
        logger.info("🔍 步骤1: 检查现有模型文件...")
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        
        # 构建模型缓存目录名
        repo_name = model_name.replace("/", "--")
        model_cache_dir = os.path.join(cache_dir, f"models--{repo_name}")
        
        has_existing, existing_size = check_existing_model(model_cache_dir)
        
        if has_existing:
            logger.info("✅ 发现现有模型文件 - 将进行断点续传")
            logger.info("💡 如果下载失败，现有文件将被保留")
        else:
            logger.info("📥 未发现现有文件 - 开始全新下载")
        
        # 步骤2: 使用snapshot_download下载模型文件（安全模式）
        logger.info("📥 步骤2: 使用snapshot_download下载模型文件...")
        logger.info("🔄 如果文件已存在，将从上次中断的地方继续")
        
        # 使用snapshot_download进行下载，支持断点续传
        local_dir = snapshot_download(
            repo_id=model_name,           # 模型仓库ID
            cache_dir=cache_dir,          # 缓存目录
            local_dir=model_cache_dir,    # 本地存储目录
            resume_download=True          # 启用断点续传
        )
        
        logger.info(f"✅ 模型文件下载完成，保存到: {local_dir}")
        
        # 步骤3: 验证下载完成情况
        logger.info("🔍 步骤3: 验证下载完成情况...")
        final_has_files, final_size = check_existing_model(model_cache_dir)
        
        if not final_has_files:
            logger.error("❌ 下载后未找到模型文件 - 可能出现了问题")
            return "下载失败 - 未找到模型文件"
        
        logger.info(f"✅ 下载验证完成: {final_size / (1024**3):.2f} GB")
        
        # 步骤4: 测试加载tokenizer
        logger.info("📥 步骤4: 测试加载tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                local_dir,
                trust_remote_code=True    # 信任远程代码
            )
            tokenizer.pad_token = tokenizer.eos_token  # 设置填充token
            logger.info("✅ Tokenizer加载成功")
        except Exception as e:
            logger.error(f"❌ Tokenizer加载失败: {e}")
            logger.info("💡 模型文件可能不完整 - 您可以稍后尝试加载")
            return "下载完成但tokenizer测试失败"
        
        # 步骤5: 测试加载模型（可选步骤 - 如果只想获取文件可以跳过）
        logger.info("📥 步骤5: 测试加载模型（4bit量化）...")
        logger.info("⚠️  如果您只想获取文件，可以跳过此步骤")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                local_dir,
                torch_dtype=torch.float16,              # 使用半精度浮点数
                device_map="auto",                      # 自动设备映射
                trust_remote_code=True,                 # 信任远程代码
                load_in_4bit=True,                      # 4bit量化加载
                bnb_4bit_compute_dtype=torch.float16    # 4bit计算数据类型
            )
            logger.info(f"✅ Qwen2.5-{model_size.upper()}-Instruct模型加载成功！")
            logger.info("🎉 所有测试通过 - 模型已准备就绪！")
        except Exception as e:
            logger.warning(f"⚠️  模型加载测试失败: {e}")
            logger.info("💡 模型文件已下载但可能需要GPU内存或其他设置")
            logger.info("💡 您可以在需要时稍后尝试加载模型")
        
        # 显示最终信息
        logger.info("📁 模型已保存到缓存目录")
        logger.info(f"📂 模型缓存目录: {model_cache_dir}")
        
        return "下载成功完成"
        
    except Exception as e:
        logger.error(f"❌ Qwen2.5-{model_size.upper()}-Instruct模型下载失败: {e}")
        logger.info("💡 现有文件（如果有的话）已被保留")
        import traceback
        traceback.print_exc()
        return ""

def list_supported_models():
    """列出所有支持的模型"""
    logger.info("📋 支持的Qwen2.5模型列表:")
    for size, model_name in SUPPORTED_MODELS.items():
        logger.info(f"   {size.upper():>4}: {model_name}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="下载Qwen2.5系列模型")
    parser.add_argument("--model", "-m", type=str, default="7b", 
                       help="模型大小 (0.5b, 1.5b, 3b, 7b, 14b, 32b, 72b)")
    parser.add_argument("--list", "-l", action="store_true", 
                       help="列出所有支持的模型")
    
    args = parser.parse_args()
    
    if args.list:
        list_supported_models()
        return
    
    # 下载指定模型
    result = download_qwen25_model_safe(args.model.lower())
    if result:
        logger.info(f"✅ 下载结果: {result}")

if __name__ == "__main__":
    main() 