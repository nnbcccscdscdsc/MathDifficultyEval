#!/usr/bin/env python3
"""
模拟评估脚本：使用模拟的模型输出来测试OpenAI评分功能
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import time
from datetime import datetime
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from scripts.openai_scorer import OpenAIScorer
from scripts.results_analysis import ResultsAnalyzer
from scripts.utils import ConfigLoader, setup_logging

class MockModelEvaluator:
    """模拟模型评估器"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """初始化评估器"""
        self.config = ConfigLoader.load_config(config_path)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # 设置日志
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 初始化OpenAI评分器
        self.openai_scorer = None
        try:
            self.openai_scorer = OpenAIScorer(config_path)
            self.logger.info("OpenAI评分器初始化成功")
        except Exception as e:
            self.logger.error(f"OpenAI评分器初始化失败: {e}")
            raise
    
    def generate_mock_answers(self, problems: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
        """生成模拟的模型答案"""
        self.logger.info(f"为模型 {model_name} 生成模拟答案")
        
        # 根据模型大小调整答案质量
        model_quality = {
            'llama-7b': 0.7,
            'llama-13b': 0.8,
            'llama-70b': 0.9
        }
        
        quality = model_quality.get(model_name, 0.7)
        
        mock_answers = []
        
        for i, problem in enumerate(problems):
            problem_text = problem['problem']
            expected_answer = problem['solution']
            difficulty = problem['difficulty']
            
            # 根据难度和模型质量生成不同质量的答案
            if '2 + 3' in problem_text:
                if quality > 0.8:
                    generated_answer = "The answer is 5. This is a simple addition problem."
                elif quality > 0.7:
                    generated_answer = "5"
                else:
                    generated_answer = "I think it might be 6, but I'm not sure."
            
            elif '2x + 5 = 13' in problem_text:
                if quality > 0.8:
                    generated_answer = "Let's solve this step by step:\n1) 2x + 5 = 13\n2) 2x = 13 - 5\n3) 2x = 8\n4) x = 8/2\n5) x = 4\n\nThe answer is x = 4."
                elif quality > 0.7:
                    generated_answer = "2x + 5 = 13\n2x = 8\nx = 4"
                else:
                    generated_answer = "I think x might be 4, but I'm not confident about the steps."
            
            elif 'circle with radius 5' in problem_text:
                if quality > 0.8:
                    generated_answer = "The area of a circle is A = πr².\nGiven radius r = 5:\nA = π × 5² = π × 25 = 25π ≈ 78.54 square units."
                elif quality > 0.7:
                    generated_answer = "Area = πr² = π × 5² = 25π"
                else:
                    generated_answer = "I think it's something with π and 25, but I'm not sure of the exact formula."
            
            elif 'sin(30°)' in problem_text:
                if quality > 0.8:
                    generated_answer = "sin(30°) = 1/2 = 0.5\nThis is a standard trigonometric value."
                elif quality > 0.7:
                    generated_answer = "sin(30°) = 1/2"
                else:
                    generated_answer = "I think it's 0.5, but I'm not certain."
            
            else:
                # 通用答案生成
                if quality > 0.8:
                    generated_answer = f"I would solve this by following the standard mathematical procedures. The answer should be {expected_answer}."
                elif quality > 0.7:
                    generated_answer = f"The answer is {expected_answer}."
                else:
                    generated_answer = f"I'm not entirely sure, but I think it might be related to {expected_answer}."
            
            # 添加一些随机性
            import random
            if random.random() < (1 - quality):
                generated_answer += " However, I'm not completely confident about this answer."
            
            mock_answers.append({
                'id': problem.get('id', f'mock_{i}'),
                'problem': problem_text,
                'expected_answer': expected_answer,
                'generated_answer': generated_answer,
                'difficulty': difficulty,
                'model_name': model_name,
                'generation_time': random.uniform(1.0, 3.0)  # 模拟生成时间
            })
        
        return mock_answers
    
    def evaluate_with_openai(self, mock_answers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用OpenAI对模拟答案进行评分"""
        self.logger.info("开始OpenAI评分")
        
        results = []
        
        for i, answer in enumerate(mock_answers):
            self.logger.info(f"评分进度: {i+1}/{len(mock_answers)}")
            
            try:
                # 使用OpenAI评分
                openai_result = self.openai_scorer.score_answer(
                    problem=answer['problem'],
                    reference_answer=answer['expected_answer'],
                    student_answer=answer['generated_answer']
                )
                
                # 计算其他指标
                from scripts.utils import calculate_metrics
                metrics = calculate_metrics(answer['generated_answer'], answer['expected_answer'])
                
                # 合并结果
                result = {
                    **answer,
                    'openai_score': openai_result['openai_score'],
                    'openai_score_text': openai_result.get('score_text', ''),
                    **metrics
                }
                
                results.append(result)
                
                # 添加延迟避免API限制
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"评分失败: {e}")
                # 使用默认分数
                result = {
                    **answer,
                    'openai_score': 50.0,
                    'openai_score_text': f"评分失败: {str(e)}",
                    'accuracy': 0.5,
                    'exact_match': 0.0,
                    'rouge_score': 0.3,
                    'bleu_score': 0.3,
                    'step_accuracy': 0.5
                }
                results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], model_name: str) -> str:
        """保存结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存CSV
        df = pd.DataFrame(results)
        csv_file = self.results_dir / f"{model_name}_mock_{timestamp}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 保存JSON
        json_file = self.results_dir / f"{model_name}_mock_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"结果已保存: {csv_file}")
        return str(csv_file)
    
    def run_mock_evaluation(self, model_name: str, dataset_name: str = "sample", max_samples: Optional[int] = None):
        """运行模拟评估"""
        self.logger.info(f"开始模拟评估: {model_name}")
        
        # 加载数据集
        data_path = Path("data/processed") / f"{dataset_name}.csv"
        if not data_path.exists():
            self.logger.error(f"数据集不存在: {data_path}")
            return None
        
        df = pd.read_csv(data_path)
        
        # 限制样本数量
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        
        # 转换为字典列表
        problems = df.to_dict('records')
        
        # 生成模拟答案
        mock_answers = self.generate_mock_answers(problems, model_name)
        
        # OpenAI评分
        results = self.evaluate_with_openai(mock_answers)
        
        # 保存结果
        result_file = self.save_results(results, model_name)
        
        # 打印摘要
        avg_openai_score = sum(r['openai_score'] for r in results) / len(results)
        avg_generation_time = sum(r['generation_time'] for r in results) / len(results)
        
        print(f"\n📊 {model_name} 模拟评估结果:")
        print(f"  样本数: {len(results)}")
        print(f"  平均OpenAI评分: {avg_openai_score:.2f}")
        print(f"  平均生成时间: {avg_generation_time:.2f}秒")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="模拟模型评估脚本")
    parser.add_argument("--model", type=str, default="llama-7b",
                       choices=["llama-7b", "llama-13b", "llama-70b"],
                       help="要评估的模型")
    parser.add_argument("--dataset", type=str, default="sample",
                       help="数据集名称")
    parser.add_argument("--max-samples", type=int, default=10,
                       help="最大样本数量")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="配置文件路径")
    
    args = parser.parse_args()
    
    try:
        # 初始化评估器
        evaluator = MockModelEvaluator(args.config)
        
        # 运行模拟评估
        results = evaluator.run_mock_evaluation(
            model_name=args.model,
            dataset_name=args.dataset,
            max_samples=args.max_samples
        )
        
        if results:
            print(f"\n✅ 模拟评估完成！")
            print(f"📁 结果文件: results/{args.model}_mock_*.csv")
            print(f"🔍 可以运行结果分析: python scripts/results_analysis.py --results-file results/{args.model}_mock_*.csv --model-name {args.model}")
        
    except Exception as e:
        print(f"❌ 模拟评估失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 