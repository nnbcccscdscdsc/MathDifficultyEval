#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MATH-500数据集数学评估框架
专门适配MATH-500数据集的字段结构

数据集字段：
- id: 问题ID
- problem: 数学问题
- solution: 解决方案
- answer: 答案
- difficulty: 难度等级
- topic: 主题分类
- difficulty_score: 难度分数
- source_dataset: 数据来源
"""

import os
import sys
import json
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from tqdm import tqdm
import openai
import re

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from math_evaluation_framework import MathEvaluationFramework

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('math500_evaluation.log'),
        logging.StreamHandler()
    ]
)

class Math500EvaluationFramework(MathEvaluationFramework):
    """
    MATH-500数据集专用评估框架
    继承自基础评估框架，适配MATH-500数据集的特殊字段
    """
    
    def __init__(self, model_name: str, dataset_path: str, **kwargs):
        """
        初始化MATH-500评估框架
        
        Args:
            model_name: 模型名称
            dataset_path: MATH-500数据集路径
            **kwargs: 其他参数
        """
        # 构建模型配置字典，使用正确的模型映射
        model_config = self._get_model_config(model_name)
        
        # 调用父类初始化
        super().__init__(model_config, **kwargs)
        
        # 保存数据集路径
        self.dataset_path = dataset_path
        
        # 设置logger属性
        self.logger = logging.getLogger(__name__)
        
        # 生成唯一的运行ID
        self.run_id = f"math500_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"🎯 MATH-500评估框架初始化完成，运行ID: {self.run_id}")
        
        # 验证数据集格式
        self._validate_math500_dataset()
    
    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        根据模型名称获取完整的模型配置
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型配置字典
        """
        # 参考 math_evaluation_framework.py 中的 MODEL_CONFIGS
        MODEL_CONFIGS = {
            # DeepSeek R1 系列模型
            "deepseek_r1_1.5b": {
                "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                "type": "1.5b_quantized",
                "max_new_tokens": 500,
                "description": "DeepSeek-R1 1.5B 模型"
            },
            "deepseek_r1_7b": {
                "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", 
                "type": "7b_quantized",
                "max_new_tokens": 600,
                "description": "DeepSeek-R1 7B 模型（4bit量化）"
            },
            "deepseek_r1_14b": {
                "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                "type": "14b_quantized",
                "max_new_tokens": 700,
                "description": "DeepSeek-R1 14B 模型（4bit量化）"
            },
            "deepseek_r1_32b": {
                "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "type": "32b_quantized",
                "max_new_tokens": 800,
                "description": "DeepSeek-R1 32B 模型（4bit量化）"
            },
            "deepseek_r1_70b": {
                "name": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                "type": "70b_quantized",
                "max_new_tokens": 1000,
                "description": "DeepSeek-R1 70B 模型（4bit量化）"
            },
            
            # Qwen2.5 系列模型
            "qwen25_0.5b": {
                "name": "Qwen/Qwen2.5-0.5B-Instruct",
                "type": "0.5b",
                "max_new_tokens": 400,
                "description": "Qwen2.5 0.5B Instruct模型"
            },
            "qwen25_1.5b": {
                "name": "Qwen/Qwen2.5-1.5B-Instruct",
                "type": "1.5b",
                "max_new_tokens": 500,
                "description": "Qwen2.5 1.5B Instruct模型"
            },
            "qwen25_3b": {
                "name": "Qwen/Qwen2.5-3B-Instruct",
                "type": "3b",
                "max_new_tokens": 600,
                "description": "Qwen2.5 3B Instruct模型"
            },
            "qwen25_7b": {
                "name": "Qwen/Qwen2.5-7B-Instruct",
                "type": "7b",
                "max_new_tokens": 700,
                "description": "Qwen2.5 7B Instruct模型"
            },
            "qwen25_14b": {
                "name": "Qwen/Qwen2.5-14B-Instruct",
                "type": "14b",
                "max_new_tokens": 800,
                "description": "Qwen2.5 14B Instruct模型"
            },
            "qwen25_32b": {
                "name": "Qwen/Qwen2.5-32B-Instruct",
                "type": "32b",
                "max_new_tokens": 900,
                "description": "Qwen2.5 32B Instruct模型"
            },
            "qwen25_72b": {
                "name": "Qwen/Qwen2.5-72B-Instruct",
                "type": "72b",
                "max_new_tokens": 1000,
                "description": "Qwen2.5 72B Instruct模型"
            }
        }
        
        if model_name in MODEL_CONFIGS:
            return MODEL_CONFIGS[model_name]
        else:
            raise ValueError(f"不支持的模型: {model_name}。支持的模型: {list(MODEL_CONFIGS.keys())}")
    
    def _validate_math500_dataset(self):
        """
        验证MATH-500数据集格式
        """
        try:
            df = pd.read_csv(self.dataset_path)
            required_columns = ['problem', 'solution', 'answer', 'subject', 'level', 'unique_id']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"❌ MATH-500数据集缺少必需列: {missing_columns}")
            
            self.logger.info(f"✅ MATH-500数据集验证通过，包含 {len(df)} 个样本")
            self.logger.info(f"📊 数据集列: {list(df.columns)}")
            
        except Exception as e:
            self.logger.error(f"❌ MATH-500数据集验证失败: {e}")
            raise
    
    def load_dataset(self) -> pd.DataFrame:
        """
        加载MATH-500数据集
        
        Returns:
            包含MATH-500数据的DataFrame
        """
        try:
            df = pd.read_csv(self.dataset_path)
            self.logger.info(f"📂 成功加载MATH-500数据集: {len(df)} 个样本")
            
            # 显示数据集统计信息
            self.logger.info(f"📊 难度分布: {df['level'].value_counts().to_dict()}")
            self.logger.info(f"📊 主题分布: {df['subject'].value_counts().to_dict()}")
            self.logger.info(f"📊 难度范围: {df['level'].min()} - {df['level'].max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ 加载MATH-500数据集失败: {e}")
            raise
    
    def format_question_for_model(self, row: pd.Series) -> str:
        """
        格式化MATH-500问题供模型使用
        
        Args:
            row: 数据集中的一行数据
            
        Returns:
            格式化后的问题文本
        """
        problem = row['problem']
        subject = row['subject']
        level = row['level']
        
        # 构建MATH-500专用提示
        prompt = f"""请解决以下数学问题：

主题: {subject}
难度等级: {level}

问题: {problem}

请提供详细的解题步骤和最终答案。确保你的答案准确且完整。"""
        
        return prompt
    
    def evaluate_with_openai(self, question: str, model_answer: str, correct_answer: str, 
                           subject: str, level: int) -> Dict[str, Any]:
        """
        使用OpenAI评估MATH-500答案
        
        Args:
            question: 原始问题
            model_answer: 模型生成的答案
            correct_answer: 正确答案
            subject: 主题分类 (MATH-500字段)
            level: 难度等级 (MATH-500字段)
            
        Returns:
            评估结果字典
        """
        try:
            # 构建MATH-500专用的评估提示，参考统一评估框架的格式
            prompt = f"""You are a professional mathematical education evaluator. Please evaluate the quality of the answer to the following mathematical problem.

Problem: {question}

Subject: {subject}
Difficulty Level: {level}

Correct Answer: {correct_answer}

Model Response: {model_answer}

Please evaluate from the following aspects and give a score from 1 to 10:

1. Answer Correctness (1-10 points): Whether the final answer is correct
2. Reasoning Logic (1-10 points): Whether the reasoning process is clear and logical
3. Step Completeness (1-10 points): Whether all necessary solution steps are shown
4. Mathematical Accuracy (1-10 points): Whether mathematical calculations and formulas are accurate
5. Expression Clarity (1-10 points): Whether the expression is clear and easy to understand

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any other text, explanations, or markdown formatting.

Required JSON format:
{{
    "answer_correctness": <score>,
    "reasoning_logic": <score>,
    "step_completeness": <score>,
    "mathematical_accuracy": <score>,
    "expression_clarity": <score>,
    "overall_score": <average_of_all_scores>,
    "comments": "<brief_evaluation_comments>"
}}

Example:
{{
    "answer_correctness": 8,
    "reasoning_logic": 7,
    "step_completeness": 9,
    "mathematical_accuracy": 8,
    "expression_clarity": 7,
    "overall_score": 7.8,
    "comments": "Good reasoning but could be more detailed in steps"
}}"""

            openai.api_key = os.getenv('OPENAI_API_KEY')
            
            # 检查OpenAI API密钥是否存在
            if not openai.api_key:
                self.logger.warning("⚠️ 未设置OpenAI API密钥，跳过评估")
                return {
                    "answer_correctness": 0,
                    "reasoning_logic": 0,
                    "step_completeness": 0,
                    "mathematical_accuracy": 0,
                    "expression_clarity": 0,
                    "overall_score": 0,
                    "comments": "未设置OpenAI API密钥，跳过评估",
                    "error": True,
                    "error_type": "no_openai_key"
                }
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional mathematical education evaluator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            response_content = response.choices[0].message.content
            
            # 解析JSON响应，参考统一评估框架的错误处理
            import json
            import re
            
            try:
                evaluation = json.loads(response_content)
                return evaluation
            except json.JSONDecodeError as e:
                self.logger.warning(f"⚠️ JSON解析失败: {e}")
                self.logger.warning(f"原始响应: {response_content[:200]}...")  # 只显示前200字符
                
                # 尝试提取JSON部分
                try:
                    # 查找JSON对象
                    json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        evaluation = json.loads(json_str)
                        self.logger.info(f"✅ 成功提取JSON部分")
                        return evaluation
                except:
                    pass
                
                # 尝试修复常见的JSON格式问题
                try:
                    # 移除可能的markdown代码块标记
                    cleaned_response = response_content.replace('```json', '').replace('```', '').strip()
                    evaluation = json.loads(cleaned_response)
                    self.logger.info(f"✅ 成功修复JSON格式")
                    return evaluation
                except:
                    pass
                
                return {
                    "raw_response": response_content,
                    "error": "JSON parsing failed",
                    "parse_error": str(e),
                    "error_type": "json_decode_error"
                }
            
        except Exception as e:
            self.logger.error(f"❌ OpenAI evaluation failed: {e}")
            return {
                "error": f"Evaluation failed: {e}",
                "error_type": "openai_api_error",
                "exception": str(e)
            }
    
    def _create_error_evaluation(self, error_message: str) -> Dict[str, Any]:
        """
        创建错误评估结果
        
        Args:
            error_message: 错误信息
            
        Returns:
            错误评估字典
        """
        return {
            'answer_correctness': 0,
            'reasoning_logic': 0,
            'step_completeness': 0,
            'mathematical_accuracy': 0,
            'expression_clarity': 0,
            'overall_score': 0,
            'comments': f"评估失败: {error_message}",
            'error': True,
            'error_type': 'evaluation_failure'
        }
    
    def save_intermediate_results(self, results: List[Dict], sample_count: int):
        """
        保存MATH-500中间结果
        
        Args:
            results: 评估结果列表
            sample_count: 已处理的样本数量
        """
        try:
            # 创建MATH-500专用目录结构
            save_dir = Path(f"data/math500_results/{self.model_name}/{self.run_id}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存中间结果
            intermediate_file = save_dir / f"intermediate_results_{sample_count}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 MATH-500中间结果已保存: {intermediate_file}")
            
        except Exception as e:
            self.logger.error(f"❌ 保存MATH-500中间结果失败: {e}")
    
    def run_evaluation(self, max_samples: Optional[int] = None) -> List[Dict]:
        """
        运行MATH-500评估
        
        Args:
            max_samples: 最大样本数量
            
        Returns:
            评估结果列表
        """
        try:
            # 加载模型 - 参考统一评估框架的方式
            self.load_model()
            
            # 加载数据集
            df = self.load_dataset()
            
            if max_samples:
                df = df.head(max_samples)
            
            self.logger.info(f"🚀 开始MATH-500评估，共 {len(df)} 个样本")
            
            results = []
            
            # 检查是否有现有结果可以恢复
            existing_count = self._check_existing_results()
            if existing_count > 0:
                self.logger.info(f"🔄 发现现有结果，从第 {existing_count + 1} 个样本开始")
                df = df.iloc[existing_count:]
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="MATH-500评估进度"):
                try:
                    # 格式化问题
                    question = self.format_question_for_model(row)
                    
                    # 生成答案
                    model_answer = self.generate_response(question)
                    
                    if not model_answer or model_answer.strip() == "":
                        self.logger.warning(f"⚠️ 样本 {idx} 生成失败")
                        evaluation = self._create_error_evaluation("生成失败：空答案")
                    else:
                        # 评估答案
                        evaluation = self.evaluate_with_openai(
                            question=row['problem'],
                            model_answer=model_answer,
                            correct_answer=row['answer'],
                            subject=row['subject'],
                            level=row['level']
                        )
                        
                        # 检查评估结果，参考统一评估框架的处理方式
                        if isinstance(evaluation, dict) and "error" not in evaluation:
                            self.logger.info(f"✅ 评估完成，总分: {evaluation.get('overall_score', 0):.2f}")
                        else:
                            # 为评估失败的样本创建特殊评估
                            failed_evaluation = {
                                "answer_correctness": 0,
                                "reasoning_logic": 0,
                                "step_completeness": 0,
                                "mathematical_accuracy": 0,
                                "expression_clarity": 0,
                                "overall_score": 0,
                                "comments": f"评估失败: {str(evaluation)}",
                                "error": "evaluation_failed"
                            }
                            evaluation = failed_evaluation
                    
                    # 构建结果
                    result = {
                        'sample_id': idx,
                        'unique_id': row['unique_id'],
                        'problem': row['problem'],
                        'correct_answer': row['answer'],
                        'model_answer': model_answer,
                        'subject': row['subject'],
                        'level': row['level'],
                        'evaluation': evaluation,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    results.append(result)
                    
                    # 每10个样本保存一次中间结果
                    if len(results) % 10 == 0:
                        self.save_intermediate_results(results, len(results))
                    
                except Exception as e:
                    self.logger.error(f"❌ 样本 {idx} 处理失败: {e}")
                    error_result = {
                        'sample_id': idx,
                        'unique_id': row.get('unique_id', f'error_{idx}'),
                        'problem': row.get('problem', ''),
                        'correct_answer': row.get('answer', ''),
                        'model_answer': '',
                        'subject': row.get('subject', 'Unknown'),
                        'level': row.get('level', 0),
                        'evaluation': self._create_error_evaluation(f"处理失败: {str(e)}"),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(error_result)
            
            # 保存最终结果
            self.save_final_results(results)
            
            self.logger.info(f"✅ MATH-500评估完成，共处理 {len(results)} 个样本")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ MATH-500评估失败: {e}")
            raise
    
    def save_final_results(self, results: List[Dict]):
        """
        保存MATH-500最终结果
        
        Args:
            results: 评估结果列表
        """
        try:
            # 创建保存目录
            save_dir = Path(f"data/math500_results/{self.model_name}/{self.run_id}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存JSON格式
            json_file = save_dir / "final_results.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 保存CSV格式
            csv_file = save_dir / "final_results.csv"
            df_results = pd.DataFrame(results)
            df_results.to_csv(csv_file, index=False, encoding='utf-8')
            
            self.logger.info(f"💾 MATH-500最终结果已保存:")
            self.logger.info(f"   JSON: {json_file}")
            self.logger.info(f"   CSV: {csv_file}")
            
        except Exception as e:
            self.logger.error(f"❌ 保存MATH-500最终结果失败: {e}")
    
    def _check_existing_results(self) -> int:
        """
        检查现有结果，返回已处理的样本数量
        
        Returns:
            已处理的样本数量
        """
        try:
            save_dir = Path(f"data/math500_results/{self.model_name}/{self.run_id}")
            if not save_dir.exists():
                return 0
            
            # 查找最新的中间结果文件
            intermediate_files = list(save_dir.glob("intermediate_results_*.json"))
            if not intermediate_files:
                return 0
            
            # 按文件名中的数字排序
            latest_file = max(intermediate_files, key=lambda x: int(x.stem.split('_')[-1]))
            
            with open(latest_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            return len(results)
            
        except Exception as e:
            self.logger.warning(f"⚠️ 检查现有结果失败: {e}")
            return 0


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='MATH-500数据集数学评估框架')
    parser.add_argument('-m', '--model', type=str, required=True,
                       help='模型名称 (deepseek_r1_1.5b, deepseek_r1_7b, deepseek_r1_14b, deepseek_r1_32b, deepseek_r1_70b, qwen25_0.5b, qwen25_1.5b, qwen25_3b, qwen25_7b, qwen25_14b, qwen25_32b, qwen25_72b)')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                       help='MATH-500数据集路径')
    parser.add_argument('-s', '--samples', type=int, default=None,
                       help='最大样本数量 (默认: 全部)')
    parser.add_argument('--hf_token', type=str, default=None,
                       help='Hugging Face API Token (用于远程模型)')
    
    args = parser.parse_args()
    
    # 设置环境变量
    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token
    
    # 验证数据集路径
    if not os.path.exists(args.dataset):
        print(f"❌ 数据集文件不存在: {args.dataset}")
        sys.exit(1)
    
    try:
        # 创建评估框架
        framework = Math500EvaluationFramework(
            model_name=args.model,
            dataset_path=args.dataset
        )
        
        # 运行评估
        results = framework.run_evaluation(max_samples=args.samples)
        
        # 显示统计信息
        print(f"\n📊 MATH-500评估统计:")
        print(f"   总样本数: {len(results)}")
        
        # 计算成功率
        successful = sum(1 for r in results if not r.get('evaluation', {}).get('error', False))
        success_rate = successful / len(results) * 100
        print(f"   成功率: {success_rate:.1f}% ({successful}/{len(results)})")
        
        # 计算平均分数
        scores = [r.get('evaluation', {}).get('overall_score', 0) for r in results 
                 if not r.get('evaluation', {}).get('error', False)]
        if scores:
            avg_score = np.mean(scores)
            print(f"   平均分数: {avg_score:.2f}/10")
        
        print(f"✅ MATH-500评估完成！")
        
    except Exception as e:
        print(f"❌ MATH-500评估失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 