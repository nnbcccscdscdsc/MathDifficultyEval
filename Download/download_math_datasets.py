#!/usr/bin/env python3
"""
数学数据集下载脚本
下载Hendrycks MATH和MATH-500数据集到本地
"""

import os
import sys
from datasets import load_dataset
import pandas as pd
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_hendrycks_math():
    """
    下载Hendrycks MATH数据集
    来源: https://huggingface.co/datasets/EleutherAI/hendrycks_math
    """
    logger.info("🚀 开始下载Hendrycks MATH数据集...")
    
    try:
        # 获取所有可用的子集
        configs = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
        logger.info(f"📋 发现 {len(configs)} 个子集: {', '.join(configs)}")
        
        # 创建保存目录
        save_dir = "data/hendrycks_math"
        os.makedirs(save_dir, exist_ok=True)
        
        all_train_data = []
        all_test_data = []
        
        # 下载每个子集
        for config in configs:
            try:
                logger.info(f"📥 下载子集: {config}")
                dataset = load_dataset("EleutherAI/hendrycks_math", config)
                
                # 合并训练集
                if 'train' in dataset:
                    train_df = dataset['train'].to_pandas()
                    train_df['subset'] = config
                    all_train_data.append(train_df)
                    logger.info(f"   - 训练集: {len(train_df)} 个样本")
                
                # 合并测试集
                if 'test' in dataset:
                    test_df = dataset['test'].to_pandas()
                    test_df['subset'] = config
                    all_test_data.append(test_df)
                    logger.info(f"   - 测试集: {len(test_df)} 个样本")
                    
            except Exception as e:
                logger.warning(f"⚠️ 下载子集 {config} 失败: {e}")
                continue
        
        # 合并所有数据
        if all_train_data:
            combined_train = pd.concat(all_train_data, ignore_index=True)
            train_path = f"{save_dir}/train.csv"
            combined_train.to_csv(train_path, index=False, encoding='utf-8')
            logger.info(f"💾 合并训练集已保存: {train_path} ({len(combined_train)} 个样本)")
        
        if all_test_data:
            combined_test = pd.concat(all_test_data, ignore_index=True)
            test_path = f"{save_dir}/test.csv"
            combined_test.to_csv(test_path, index=False, encoding='utf-8')
            logger.info(f"💾 合并测试集已保存: {test_path} ({len(combined_test)} 个样本)")
        
        # 保存数据集信息
        info = {
            "dataset_name": "Hendrycks MATH",
            "source": "https://huggingface.co/datasets/EleutherAI/hendrycks_math",
            "download_time": datetime.now().isoformat(),
            "configs": configs,
            "train_samples": len(combined_train) if all_train_data else 0,
            "test_samples": len(combined_test) if all_test_data else 0,
            "total_samples": (len(combined_train) if all_train_data else 0) + (len(combined_test) if all_test_data else 0),
            "columns": list(combined_train.columns) if all_train_data else [],
            "subset_distribution": combined_train['subset'].value_counts().to_dict() if all_train_data else {},
            "level_distribution": combined_train['level'].value_counts().to_dict() if all_train_data else {},
            "type_distribution": combined_train['type'].value_counts().to_dict() if all_train_data else {}
        }
        
        import json
        info_path = f"{save_dir}/dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        logger.info(f"📋 数据集信息已保存: {info_path}")
        
        return True
        logger.info(f"✅ 成功加载Hendrycks MATH数据集")
        logger.info(f"📊 数据集信息:")
        logger.info(f"   - 训练集: {len(dataset['train'])} 个样本")
        logger.info(f"   - 测试集: {len(dataset['test'])} 个样本")
        
        # 创建保存目录
        save_dir = "data/hendrycks_math"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存训练集
        train_df = dataset['train'].to_pandas()
        train_path = f"{save_dir}/train.csv"
        train_df.to_csv(train_path, index=False, encoding='utf-8')
        logger.info(f"💾 训练集已保存: {train_path}")
        
        # 保存测试集
        test_df = dataset['test'].to_pandas()
        test_path = f"{save_dir}/test.csv"
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        logger.info(f"💾 测试集已保存: {test_path}")
        
        # 保存数据集信息
        info = {
            "dataset_name": "Hendrycks MATH",
            "source": "https://huggingface.co/datasets/EleutherAI/hendrycks_math",
            "download_time": datetime.now().isoformat(),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "total_samples": len(train_df) + len(test_df),
            "columns": list(train_df.columns),
            "level_distribution": train_df['level'].value_counts().to_dict(),
            "type_distribution": train_df['type'].value_counts().to_dict()
        }
        
        import json
        info_path = f"{save_dir}/dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        logger.info(f"📋 数据集信息已保存: {info_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 下载Hendrycks MATH数据集失败: {e}")
        return False

def download_math_500():
    """
    下载MATH-500数据集
    来源: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
    """
    logger.info("🚀 开始下载MATH-500数据集...")
    
    try:
        # 加载数据集
        dataset = load_dataset("HuggingFaceH4/MATH-500")
        logger.info(f"✅ 成功加载MATH-500数据集")
        logger.info(f"📊 数据集信息:")
        logger.info(f"   - 测试集: {len(dataset['test'])} 个样本")
        
        # 创建保存目录
        save_dir = "data/math_500"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存测试集
        test_df = dataset['test'].to_pandas()
        test_path = f"{save_dir}/test.csv"
        test_df.to_csv(test_path, index=False, encoding='utf-8')
        logger.info(f"💾 测试集已保存: {test_path}")
        
        # 保存数据集信息
        info = {
            "dataset_name": "MATH-500",
            "source": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500",
            "download_time": datetime.now().isoformat(),
            "test_samples": len(test_df),
            "total_samples": len(test_df),
            "columns": list(test_df.columns) if len(test_df) > 0 else []
        }
        
        import json
        info_path = f"{save_dir}/dataset_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        logger.info(f"📋 数据集信息已保存: {info_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 下载MATH-500数据集失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("🎯 开始下载数学数据集")
    logger.info("=" * 50)
    
    # 检查网络连接
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        logger.info("✅ 网络连接正常")
    except Exception as e:
        logger.error(f"❌ 网络连接失败: {e}")
        logger.info("请检查网络连接或代理设置")
        return
    
    # 下载Hendrycks MATH数据集
    success1 = download_hendrycks_math()
    
    logger.info("-" * 30)
    
    # 下载MATH-500数据集
    success2 = download_math_500()
    
    logger.info("=" * 50)
    logger.info("📋 下载总结:")
    logger.info(f"   Hendrycks MATH: {'✅ 成功' if success1 else '❌ 失败'}")
    logger.info(f"   MATH-500: {'✅ 成功' if success2 else '❌ 失败'}")
    
    if success1 and success2:
        logger.info("🎉 所有数据集下载完成！")
    else:
        logger.info("⚠️ 部分数据集下载失败，请检查错误信息")

if __name__ == "__main__":
    main() 