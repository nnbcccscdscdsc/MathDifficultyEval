#!/usr/bin/env python3
"""
快速启动脚本：一键运行数学难度评估
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_command(command: str, description: str, logger: logging.Logger):
    """运行命令并处理错误"""
    logger.info(f"开始执行: {description}")
    logger.info(f"命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        logger.info(f"✅ {description} 完成")
        if result.stdout:
            logger.info(f"输出: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} 失败")
        logger.error(f"错误输出: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="数学难度评估快速启动工具")
    parser.add_argument("--model", type=str, default="llama-7b",
                       choices=["llama-7b", "llama-13b", "llama-70b"],
                       help="要评估的模型")
    parser.add_argument("--quantization", type=str, default="4bit",
                       choices=["none", "4bit", "8bit"],
                       help="量化方式")
    parser.add_argument("--dataset", type=str, default="sample",
                       choices=["sample", "combined", "math", "gsm8k", "mathqa"],
                       help="数据集名称")
    parser.add_argument("--max-samples", type=int, default=50,
                       help="最大样本数量")
    parser.add_argument("--skip-data", action="store_true",
                       help="跳过数据处理步骤")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    
    print("="*60)
    print("🚀 数学难度评估项目 - 快速启动")
    print("="*60)
    print(f"模型: {args.model}")
    print(f"量化: {args.quantization}")
    print(f"数据集: {args.dataset}")
    print(f"最大样本数: {args.max_samples}")
    print("="*60)
    
    # 检查项目结构
    project_root = Path(__file__).parent
    required_dirs = ["scripts", "configs", "data", "results"]
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            logger.error(f"缺少必要目录: {dir_name}")
            return False
    
    # 步骤1: 数据处理
    if not args.skip_data:
        logger.info("📊 步骤1: 数据处理")
        
        # 创建小样本数据集
        if not run_command(
            f"cd {project_root} && python scripts/data_processing.py --sample",
            "创建小样本数据集",
            logger
        ):
            return False
    
    print("\n" + "="*60)
    print("🎉 数据处理完成！")
    print("="*60)
    
    # 显示结果文件位置
    data_dir = project_root / "data" / "processed"
    if data_dir.exists():
        print(f"📁 数据文件位置: {data_dir}")
        
        # 列出数据文件
        data_files = list(data_dir.glob("*.csv"))
        if data_files:
            print("📄 生成的文件:")
            for file in data_files:
                print(f"  - {file.name}")
    
    print("\n📖 查看README.md了解更多信息")
    print("🔧 如需自定义配置，请编辑configs/config.yaml")
    print("\n💡 下一步：运行模型评估")
    print("   python scripts/model_evaluation.py --model llama-7b --quantization 4bit --dataset sample")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 