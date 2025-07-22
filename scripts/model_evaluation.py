#!/usr/bin/env python3
"""
模型评估脚本：评估不同参数的Llama模型在数学题上的表现
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
import yaml
from tqdm import tqdm
import sys
import time
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils import ConfigLoader, setup_logging, calculate_metrics
from scripts.openai_scorer import OpenAIScorer

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化评估器"""
        self.config = ConfigLoader.load_config(config_path)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 设备配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger.info(f"使用设备: {self.device}")
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        # 初始化OpenAI评分器
        self.openai_scorer = None
        if self.config.get('openai_scoring', {}).get('enabled', False):
            try:
                self.openai_scorer = OpenAIScorer(config_path)
                self.logger.info("OpenAI评分器初始化成功")
            except Exception as e:
                self.logger.warning(f"OpenAI评分器初始化失败: {e}")
                self.openai_scorer = None
    
    def load_model(self, model_name: str, quantization: str = "4bit"):
        """加载模型"""
        self.logger.info(f"加载模型: {model_name}, 量化: {quantization}")
        
        model_config = self.config['models'][model_name]
        quant_config = self.config['quantization'][quantization]
        
        # 配置量化参数
        quantization_config = None
        if quantization != "none":
            quantization_config = BitsAndBytesConfig(**quant_config)
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_config['model_name'],
                trust_remote_code=True
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_config['model_name'],
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
            
            # 创建pipeline
            if quantization != "none":
                # 使用量化时，不指定device，让accelerate自动处理
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    **{k: v for k, v in model_config.items() if k != 'model_name'}
                )
            else:
                # 不使用量化时，可以指定device
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    **{k: v for k, v in model_config.items() if k != 'model_name'}
                )
            
            self.logger.info(f"模型 {model_name} 加载成功")
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            raise
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        """加载数据集"""
        self.logger.info(f"加载数据集: {dataset_name}")
        
        data_path = Path("data/processed") / f"{dataset_name}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {data_path}")
        
        df = pd.read_csv(data_path)
        self.logger.info(f"数据集加载完成，共 {len(df)} 个样本")
        return df
    
    def generate_answer(self, problem: str, prompt_template: str) -> str:
        """生成答案"""
        try:
            # 构建提示
            prompt = prompt_template.format(problem=problem)
            
            # 生成回答 - 使用更安全的参数
            response = self.pipeline(
                prompt,
                max_new_tokens=128,  # 进一步减少生成长度
                do_sample=True,
                temperature=0.8,     # 提高温度，避免数值问题
                top_p=0.9,           # 降低top_p
                top_k=40,            # 降低top_k
                repetition_penalty=1.0,  # 移除重复惩罚
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False  # 只返回新生成的文本
            )
            
            # 提取生成的文本
            if isinstance(response, list) and len(response) > 0:
                generated_text = response[0].get('generated_text', '')
                if generated_text:
                    answer = generated_text.strip()
                else:
                    answer = "生成失败：无输出"
            else:
                answer = "生成失败：响应格式错误"
            
            return answer
            
        except Exception as e:
            self.logger.error(f"生成答案失败: {e}")
            return f"生成失败: {str(e)}"
    
    def evaluate_dataset(self, dataset_name: str, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """评估数据集"""
        self.logger.info(f"开始评估数据集: {dataset_name}")
        
        # 加载数据集
        df = self.load_dataset(dataset_name)
        
        # 限制样本数量
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
            self.logger.info(f"限制样本数量为: {max_samples}")
        
        # 获取数据集配置
        dataset_config = self.config['datasets'].get(dataset_name, {})
        prompt_template = dataset_config.get('prompt_template', "Problem: {problem}\n\nSolution:")
        
        results = []
        
        # 按难度分组评估
        for difficulty in ['elementary', 'middle', 'college']:
            difficulty_df = df[df['difficulty'] == difficulty]
            
            if len(difficulty_df) == 0:
                self.logger.warning(f"难度等级 {difficulty} 没有数据")
                continue
            
            self.logger.info(f"评估难度等级: {difficulty}, 样本数: {len(difficulty_df)}")
            
            difficulty_results = []
            
            for idx, row in tqdm(difficulty_df.iterrows(), total=len(difficulty_df), desc=f"评估 {difficulty}"):
                problem = row['problem']
                expected_answer = row.get('solution', '')
                
                # 生成答案
                start_time = time.time()
                generated_answer = self.generate_answer(problem, prompt_template)
                generation_time = time.time() - start_time
                
                # 计算基础指标
                metrics = calculate_metrics(generated_answer, expected_answer)
                
                # 添加OpenAI评分
                if self.openai_scorer:
                    try:
                        openai_result = self.openai_scorer.score_answer(
                            problem=problem,
                            reference_answer=expected_answer,
                            student_answer=generated_answer
                        )
                        metrics['openai_score'] = openai_result['openai_score']
                        metrics['openai_score_text'] = openai_result.get('score_text', '')
                    except Exception as e:
                        self.logger.warning(f"OpenAI评分失败: {e}")
                        metrics['openai_score'] = 50.0
                        metrics['openai_score_text'] = f"评分失败: {str(e)}"
                else:
                    metrics['openai_score'] = 50.0
                    metrics['openai_score_text'] = "OpenAI评分未启用"
                
                result = {
                    'id': row.get('id', idx),
                    'problem': problem,
                    'expected_answer': expected_answer,
                    'generated_answer': generated_answer,
                    'difficulty': difficulty,
                    'generation_time': generation_time,
                    **metrics
                }
                
                difficulty_results.append(result)
            
            # 计算该难度等级的平均指标
            if difficulty_results:
                avg_metrics = {}
                for key in ['accuracy', 'exact_match', 'rouge_score', 'bleu_score']:
                    if key in difficulty_results[0]:
                        values = [r[key] for r in difficulty_results if key in r]
                        avg_metrics[f'avg_{key}'] = sum(values) / len(values) if values else 0
                
                self.logger.info(f"难度 {difficulty} 平均指标: {avg_metrics}")
                results.extend(difficulty_results)
        
        return results
    
    def save_results(self, results: List[Dict], model_name: str, dataset_name: str):
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_file = self.results_dir / f"{model_name}_{dataset_name}_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存CSV格式
        df_results = pd.DataFrame(results)
        csv_file = self.results_dir / f"{model_name}_{dataset_name}_{timestamp}.csv"
        df_results.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 生成摘要报告
        summary = self.generate_summary(results, model_name, dataset_name)
        summary_file = self.results_dir / f"{model_name}_{dataset_name}_{timestamp}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存到: {self.results_dir}")
        return summary
    
    def generate_summary(self, results: List[Dict], model_name: str, dataset_name: str) -> Dict[str, Any]:
        """生成摘要报告"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        summary = {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'total_samples': len(results),
            'evaluation_time': datetime.now().isoformat(),
            'overall_metrics': {},
            'difficulty_metrics': {}
        }
        
        # 总体指标
        for metric in ['accuracy', 'exact_match', 'rouge_score', 'bleu_score']:
            if metric in df.columns:
                summary['overall_metrics'][metric] = df[metric].mean()
        
        # 按难度分组的指标
        for difficulty in df['difficulty'].unique():
            difficulty_df = df[df['difficulty'] == difficulty]
            summary['difficulty_metrics'][difficulty] = {
                'sample_count': len(difficulty_df),
                'avg_generation_time': difficulty_df['generation_time'].mean()
            }
            
            for metric in ['accuracy', 'exact_match', 'rouge_score', 'bleu_score']:
                if metric in difficulty_df.columns:
                    summary['difficulty_metrics'][difficulty][f'avg_{metric}'] = difficulty_df[metric].mean()
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="模型评估脚本")
    parser.add_argument("--model", type=str, default="mistral-7b",
                       choices=["mistral-7b", "longalpaca-7b"],
                       help="要评估的模型（快速测试版本）")
    parser.add_argument("--quantization", type=str, default="4bit",
                       choices=["none", "4bit", "8bit"],
                       help="量化方式")
    parser.add_argument("--dataset", type=str, default="sample",
                       help="数据集名称")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="最大样本数量")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = ModelEvaluator(args.config)
    
    try:
        # 加载模型
        evaluator.load_model(args.model, args.quantization)
        
        # 评估数据集
        results = evaluator.evaluate_dataset(args.dataset, args.max_samples)
        
        # 保存结果
        summary = evaluator.save_results(results, args.model, args.dataset)
        
        # 打印摘要
        print("\n" + "="*60)
        print("📊 评估结果摘要")
        print("="*60)
        print(f"模型: {summary['model_name']}")
        print(f"数据集: {summary['dataset_name']}")
        print(f"总样本数: {summary['total_samples']}")
        
        print("\n📈 总体指标:")
        for metric, value in summary['overall_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        print("\n🎯 各难度等级指标:")
        for difficulty, metrics in summary['difficulty_metrics'].items():
            print(f"\n  {difficulty.upper()}:")
            print(f"    样本数: {metrics['sample_count']}")
            print(f"    平均生成时间: {metrics['avg_generation_time']:.2f}秒")
            for key, value in metrics.items():
                if key.startswith('avg_') and key != 'avg_generation_time':
                    print(f"    {key}: {value:.4f}")
        
        print("\n✅ 评估完成！")
        
    except Exception as e:
        print(f"❌ 评估失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 