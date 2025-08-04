#!/usr/bin/env python3
"""
统一模型评估调度脚本

使用方法：
python scripts/evaluate_all_models.py --models mistral-community/Mistral-7B-v0.2 lmsys/longchat-7b-16k --dataset deepmath_evaluation_dataset --max-samples 100
"""

import os
import json
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import time
from datetime import datetime
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import ConfigLoader, setup_logging

class MultiModelEvaluator:
    """多模型评估调度器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化调度器"""
        self.config = ConfigLoader.load_config(config_path)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 支持的模型列表（更新为实际要测试的模型）
        self.supported_models = [
            "mistral-community/Mistral-7B-v0.2",
            "lmsys/longchat-7b-16k", 
            "Yukang/LongAlpaca-13B-16k",
            "Yhyu13/oasst-rlhf-2-llama-30b-7k-steps-hf",
            "Yukang/LongAlpaca-70B-16k"
        ]
        
        # 模型GPU配置
        self.model_gpu_config = {
            "mistral-community/Mistral-7B-v0.2": 1,
            "lmsys/longchat-7b-16k": 1,
            "Yukang/LongAlpaca-13B-16k": 2,
            "Yhyu13/oasst-rlhf-2-llama-30b-7k-steps-hf": 4,
            "Yukang/LongAlpaca-70B-16k": 4
        }
    
    def evaluate_models_sequential(self, models: List[str], dataset: str, 
                                 quantization: str = "4bit", max_samples: Optional[int] = None):
        """串行评估多个模型"""
        self.logger.info(f"开始串行评估模型: {models}")
        
        results = []
        failed_models = []
        
        for i, model_name in enumerate(models, 1):
            self.logger.info(f"评估进度: {i}/{len(models)} - {model_name}")
            
            try:
                # 调用单个模型评估脚本
                result = self.run_single_model_evaluation(
                    model_name, dataset, quantization, max_samples
                )
                
                if result:
                    results.append(result)
                    self.logger.info(f"✅ 模型 {model_name} 评估成功")
                else:
                    failed_models.append(model_name)
                    self.logger.error(f"❌ 模型 {model_name} 评估失败")
                
                # 等待一下，确保GPU资源释放
                time.sleep(10)  # 增加等待时间，确保大模型完全释放
                
            except Exception as e:
                self.logger.error(f"评估模型 {model_name} 时发生错误: {e}")
                failed_models.append(model_name)
                continue
        
        return results, failed_models
    
    def evaluate_models_parallel(self, models: List[str], dataset: str,
                               quantization: str = "4bit", max_samples: Optional[int] = None):
        """并行评估多个模型（需要多个GPU）"""
        self.logger.info(f"开始并行评估模型: {models}")
        
        # 检查GPU数量
        import torch
        gpu_count = torch.cuda.device_count()
        
        # 计算需要的总GPU数量
        total_gpus_needed = sum(self.model_gpu_config.get(model, 1) for model in models)
        
        if gpu_count < total_gpus_needed:
            self.logger.warning(f"GPU数量({gpu_count})少于所需数量({total_gpus_needed})，将串行评估")
            return self.evaluate_models_sequential(models, dataset, quantization, max_samples)
        
        # 启动并行进程
        processes = []
        current_gpu = 0
        
        for model_name in models:
            num_gpus = self.model_gpu_config.get(model_name, 1)
            
            cmd = [
                sys.executable, "scripts/evaluate_single_model.py",
                "--model", model_name,
                "--dataset", dataset,
                "--quantization", quantization,
                "--num-gpus", str(num_gpus)
            ]
            
            if max_samples:
                cmd.extend(["--max-samples", str(max_samples)])
            
            # 设置环境变量指定GPU
            env = os.environ.copy()
            gpu_list = ",".join(str(i) for i in range(current_gpu, current_gpu + num_gpus))
            env["CUDA_VISIBLE_DEVICES"] = gpu_list
            
            process = subprocess.Popen(cmd, env=env)
            processes.append((model_name, process))
            
            current_gpu += num_gpus
        
        # 等待所有进程完成
        results = []
        failed_models = []
        
        for model_name, process in processes:
            try:
                process.wait()
                if process.returncode == 0:
                    self.logger.info(f"✅ 模型 {model_name} 评估成功")
                    # 这里可以读取结果文件
                else:
                    self.logger.error(f"❌ 模型 {model_name} 评估失败")
                    failed_models.append(model_name)
            except Exception as e:
                self.logger.error(f"等待模型 {model_name} 时发生错误: {e}")
                failed_models.append(model_name)
        
        return results, failed_models
    
    def run_single_model_evaluation(self, model_name: str, dataset: str,
                                  quantization: str, max_samples: Optional[int] = None):
        """运行单个模型评估"""
        num_gpus = self.model_gpu_config.get(model_name, 1)
        
        cmd = [
            sys.executable, "scripts/evaluate_single_model.py",
            "--model", model_name,
            "--dataset", dataset,
            "--quantization", quantization,
            "--num-gpus", str(num_gpus)
        ]
        
        if max_samples:
            cmd.extend(["--max-samples", str(max_samples)])
        
        try:
            # 运行子进程
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
            
            if result.returncode == 0:
                self.logger.info(f"模型 {model_name} 评估完成")
                return True
            else:
                self.logger.error(f"模型 {model_name} 评估失败: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"模型 {model_name} 评估超时")
            return False
        except Exception as e:
            self.logger.error(f"运行模型 {model_name} 时发生错误: {e}")
            return False
    
    def generate_summary_report(self, results: List[Dict], failed_models: List[str]):
        """生成汇总报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 收集所有结果文件
        all_results = []
        for result in results:
            if result:
                all_results.append(result)
        
        # 生成汇总报告
        report = f"""
# 多模型评估汇总报告

## 评估概览
- 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 成功评估模型: {len(all_results)} 个
- 失败模型: {len(failed_models)} 个
- 失败模型列表: {', '.join(failed_models) if failed_models else '无'}

## 各模型性能对比
"""
        
        if all_results:
            # 按OpenAI评分排序
            all_results.sort(key=lambda x: x.get('avg_openai_score', 0), reverse=True)
            
            for i, result in enumerate(all_results, 1):
                report += f"""
### {i}. {result['model_name']}
- 样本数: {result['total_samples']}
- GPU数量: {result['num_gpus']}
- 平均OpenAI评分: {result['avg_openai_score']:.2f}
- 平均准确率: {result['avg_accuracy']:.4f}
- 评估时间: {result['timestamp']}
- 结果文件: {result['results_file']}
"""
        
        # 保存报告
        report_file = self.results_dir / f"multi_model_summary_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"汇总报告已保存: {report_file}")
        return str(report_file)
    
    def run_evaluation(self, models: List[str], dataset: str, 
                      quantization: str = "4bit", max_samples: Optional[int] = None,
                      parallel: bool = False):
        """运行完整的评估流程"""
        self.logger.info("开始多模型评估流程")
        
        # 检查模型是否支持
        unsupported_models = [m for m in models if m not in self.supported_models]
        if unsupported_models:
            self.logger.error(f"不支持的模型: {unsupported_models}")
            return
        
        # 选择评估方式
        if parallel:
            results, failed_models = self.evaluate_models_parallel(
                models, dataset, quantization, max_samples
            )
        else:
            results, failed_models = self.evaluate_models_sequential(
                models, dataset, quantization, max_samples
            )
        
        # 生成汇总报告
        report_file = self.generate_summary_report(results, failed_models)
        
        # 打印最终摘要
        print("\n" + "="*60)
        print("🎉 多模型评估完成！")
        print("="*60)
        print(f"成功评估: {len(results)} 个模型")
        print(f"失败模型: {len(failed_models)} 个")
        
        if failed_models:
            print(f"失败模型: {', '.join(failed_models)}")
        
        print(f"汇总报告: {report_file}")
        print("="*60)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多模型评估调度脚本")
    parser.add_argument("--models", nargs="+", required=True,
                       choices=[
                           "mistral-community/Mistral-7B-v0.2",
                           "lmsys/longchat-7b-16k", 
                           "Yukang/LongAlpaca-13B-16k",
                           "Yhyu13/oasst-rlhf-2-llama-30b-7k-steps-hf",
                           "Yukang/LongAlpaca-70B-16k"
                       ],
                       help="要评估的模型列表")
    parser.add_argument("--dataset", type=str, default="deepmath_evaluation_dataset",
                       help="数据集名称")
    parser.add_argument("--quantization", type=str, default="4bit",
                       choices=["none", "4bit", "8bit"],
                       help="量化方式")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="每个模型的最大样本数量")
    parser.add_argument("--parallel", action="store_true",
                       help="并行评估（需要多个GPU）")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 创建调度器
    evaluator = MultiModelEvaluator(args.config)
    
    try:
        # 运行评估
        evaluator.run_evaluation(
            models=args.models,
            dataset=args.dataset,
            quantization=args.quantization,
            max_samples=args.max_samples,
            parallel=args.parallel
        )
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 