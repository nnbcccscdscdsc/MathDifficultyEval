#!/usr/bin/env python3
"""
Qwen2.5-7B-Instruct模型下载脚本 (安全版本 - 无删除操作)
Safe Version - 不会删除任何现有文件，支持断点续传
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import os
from huggingface_hub import snapshot_download

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def download_qwen25_7b_instruct_safe():
    """
    安全下载Qwen2.5-7B-Instruct模型
    特点:
    - 不会删除任何现有文件
    - 支持断点续传
    - 自动检查文件完整性
    - 提供详细的下载状态信息
    """
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    logger.info(f"📥 开始安全下载Qwen2.5-7B-Instruct模型: {model_name}")
    logger.info("⚠️  注意：7B模型较大，下载可能需要一些时间")
    logger.info("🛡️  安全模式：将保留现有文件并支持断点续传")
    
    try:
        # 步骤1: 检查现有模型文件
        logger.info("🔍 步骤1: 检查现有模型文件...")
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_cache_dir = os.path.join(cache_dir, "models--Qwen--Qwen2.5-7B-Instruct")
        
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
            logger.info("✅ Qwen2.5-7B-Instruct模型加载成功！")
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
        logger.error(f"❌ Qwen2.5-7B-Instruct模型下载失败: {e}")
        logger.info("💡 现有文件（如果有的话）已被保留")
        import traceback
        traceback.print_exc()
        return ""

if __name__ == "__main__":
    # 主程序入口
    download_qwen25_7b_instruct_safe() 