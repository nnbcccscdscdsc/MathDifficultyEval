#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hendrycks Math数据集数学评估框架
专门适配Hendrycks Math数据集的字段结构

数据集字段：
- problem: 数学问题
- level: 难度等级 (Level 1-5)
- type: 问题类型 (Algebra, Geometry, Precalculus等)
- solution: 标准解答
- subset: 子集分类
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
        logging.FileHandler('hendrycks_math_evaluation.log'),
        logging.StreamHandler()
    ]
)

class HendrycksMathEvaluationFramework(MathEvaluationFramework):
    """
    Hendrycks Math数据集专用评估框架
    继承自基础评估框架，适配Hendrycks Math数据集的特殊字段
    """
    
    def __init__(self, model_name: str, dataset_path: str, **kwargs):
        """
        初始化Hendrycks Math评估框架
        
        Args:
            model_name: 模型名称
            dataset_path: Hendrycks Math数据集路径
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
        self.run_id = f"hendrycks_math_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger.info(f"🎯 Hendrycks Math评估框架初始化完成，运行ID: {self.run_id}")
        
        # 验证数据集格式
        self._validate_hendrycks_math_dataset()
    
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
    
    def _validate_hendrycks_math_dataset(self):
        """
        验证Hendrycks Math数据集格式
        """
        try:
            df = pd.read_csv(self.dataset_path)
            required_columns = ['problem', 'level', 'type', 'solution', 'subset']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"❌ Hendrycks Math数据集缺少必需列: {missing_columns}")
            
            self.logger.info(f"✅ Hendrycks Math数据集验证通过，包含 {len(df)} 个样本")
            self.logger.info(f"📊 数据集列: {list(df.columns)}")
            
        except Exception as e:
            self.logger.error(f"❌ Hendrycks Math数据集验证失败: {e}")
            raise
    
    def load_dataset(self, problem_type: str = None, samples_per_level: int = None, use_train: bool = False) -> pd.DataFrame:
        """
        加载Hendrycks Math数据集
        
        Args:
            problem_type: 问题类型筛选（如"Counting & Probability"）
            samples_per_level: 每个难度等级的样本数量
            use_train: 是否使用train.csv而不是test.csv
            
        Returns:
            包含Hendrycks Math数据的DataFrame
        """
        try:
            # 选择数据文件
            if use_train:
                data_file = self.dataset_path.replace('test.csv', 'train.csv')
                self.logger.info(f"📂 使用训练集: {data_file}")
            else:
                data_file = self.dataset_path
                self.logger.info(f"📂 使用测试集: {data_file}")
            
            df = pd.read_csv(data_file)
            self.logger.info(f"📂 成功加载Hendrycks Math数据集: {len(df)} 个样本")
            
            # 按问题类型筛选
            if problem_type:
                original_count = len(df)
                df = df[df['type'] == problem_type]
                self.logger.info(f"🔍 按类型 '{problem_type}' 筛选后: {len(df)} 个样本 (减少 {original_count - len(df)} 个)")
            
            # 按难度等级采样
            if samples_per_level:
                sampled_dfs = []
                for level in sorted(df['level'].unique()):
                    level_df = df[df['level'] == level]
                    if len(level_df) >= samples_per_level:
                        # 随机采样指定数量
                        sampled_level_df = level_df.sample(n=samples_per_level, random_state=42)
                    else:
                        # 如果该难度等级的样本不足，则使用全部
                        sampled_level_df = level_df
                        self.logger.warning(f"⚠️ 难度等级 {level} 样本不足 {samples_per_level}，使用全部 {len(level_df)} 个样本")
                    
                    sampled_dfs.append(sampled_level_df)
                
                df = pd.concat(sampled_dfs, ignore_index=True)
                self.logger.info(f"📊 按难度等级采样后: {len(df)} 个样本")
            
            # 显示数据集统计信息
            self.logger.info(f"📊 难度分布: {df['level'].value_counts().to_dict()}")
            self.logger.info(f"📊 类型分布: {df['type'].value_counts().to_dict()}")
            self.logger.info(f"📊 子集分布: {df['subset'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ 加载Hendrycks Math数据集失败: {e}")
            raise
    
    def format_question_for_model(self, row: pd.Series) -> str:
        """
        格式化Hendrycks Math问题供模型使用
        
        Args:
            row: 数据集中的一行数据
            
        Returns:
            格式化后的问题文本
        """
        problem = row['problem']
        level = row['level']
        problem_type = row['type']
        
        # 构建Hendrycks Math专用提示
        prompt = f"""请解决以下数学问题：

难度等级: {level}
问题类型: {problem_type}

问题: {problem}

请提供详细的解题步骤和最终答案。确保你的答案准确且完整。"""
        
        return prompt
    
    def evaluate_with_openai(self, question: str, model_answer: str, correct_answer: str, 
                           level: str, problem_type: str) -> Dict[str, Any]:
        """
        使用OpenAI评估Hendrycks Math答案
        
        Args:
            question: 原始问题
            model_answer: 模型生成的答案
            correct_answer: 正确答案
            level: 难度等级 (Hendrycks Math字段)
            problem_type: 问题类型 (Hendrycks Math字段)
            
        Returns:
            评估结果字典
        """
        try:
            # 构建Hendrycks Math专用的评估提示，参考统一评估框架的格式
            prompt = f"""You are a professional mathematical education evaluator. Please evaluate the quality of the answer to the following mathematical problem.

Problem: {question}

Difficulty Level: {level}
Problem Type: {problem_type}

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
        保存Hendrycks Math中间结果
        
        Args:
            results: 评估结果列表
            sample_count: 已处理的样本数量
        """
        try:
            # 创建Hendrycks Math专用目录结构
            save_dir = Path(f"data/hendrycks_math_results/{self.model_name}/{self.run_id}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存中间结果
            intermediate_file = save_dir / f"intermediate_results_{sample_count}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"💾 Hendrycks Math中间结果已保存: {intermediate_file}")
            
        except Exception as e:
            self.logger.error(f"❌ 保存Hendrycks Math中间结果失败: {e}")
    
    def run_evaluation(self, max_samples: Optional[int] = None, 
                      problem_type: str = None, samples_per_level: int = None, use_train: bool = False) -> List[Dict]:
        """
        运行Hendrycks Math评估
        
        Args:
            max_samples: 最大样本数量（已废弃，使用samples_per_level替代）
            problem_type: 问题类型筛选（如"Counting & Probability"）
            samples_per_level: 每个难度等级的样本数量
            use_train: 是否使用train.csv而不是test.csv
            
        Returns:
            评估结果列表
        """
        try:
            # 加载模型 - 参考统一评估框架的方式
            self.load_model()
            
            # 加载数据集
            df = self.load_dataset(problem_type=problem_type, samples_per_level=samples_per_level, use_train=use_train)
            
            # 兼容旧的max_samples参数
            if max_samples and not samples_per_level:
                df = df.head(max_samples)
            
            self.logger.info(f"🚀 开始Hendrycks Math评估，共 {len(df)} 个样本")
            
            results = []
            
            # 检查是否有现有结果可以恢复
            existing_count = self._check_existing_results()
            if existing_count > 0:
                self.logger.info(f"🔄 发现现有结果，从第 {existing_count + 1} 个样本开始")
                df = df.iloc[existing_count:]
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Hendrycks Math评估进度"):
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
                            correct_answer=row['solution'],
                            level=row['level'],
                            problem_type=row['type']
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
                        'problem': row['problem'],
                        'correct_answer': row['solution'],
                        'model_answer': model_answer,
                        'level': row['level'],
                        'type': row['type'],
                        'subset': row['subset'],
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
                        'problem': row.get('problem', ''),
                        'correct_answer': row.get('solution', ''),
                        'model_answer': '',
                        'level': row.get('level', 'Unknown'),
                        'type': row.get('type', 'Unknown'),
                        'subset': row.get('subset', 'Unknown'),
                        'evaluation': self._create_error_evaluation(f"处理失败: {str(e)}"),
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(error_result)
            
            # 保存最终结果
            self.save_final_results(results)
            
            self.logger.info(f"✅ Hendrycks Math评估完成，共处理 {len(results)} 个样本")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Hendrycks Math评估失败: {e}")
            raise
    
    def save_final_results(self, results: List[Dict]):
        """
        保存Hendrycks Math最终结果
        
        Args:
            results: 评估结果列表
        """
        try:
            # 创建保存目录
            save_dir = Path(f"data/hendrycks_math_results/{self.model_name}/{self.run_id}")
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # 保存JSON格式
            json_file = save_dir / "final_results.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # 保存CSV格式
            csv_file = save_dir / "final_results.csv"
            df_results = pd.DataFrame(results)
            df_results.to_csv(csv_file, index=False, encoding='utf-8')
            
            self.logger.info(f"💾 Hendrycks Math最终结果已保存:")
            self.logger.info(f"   JSON: {json_file}")
            self.logger.info(f"   CSV: {csv_file}")
            
        except Exception as e:
            self.logger.error(f"❌ 保存Hendrycks Math最终结果失败: {e}")
    
    def _check_existing_results(self) -> int:
        """
        检查现有结果，返回已处理的样本数量
        
        Returns:
            已处理的样本数量
        """
        try:
            save_dir = Path(f"data/hendrycks_math_results/{self.model_name}/{self.run_id}")
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
    parser = argparse.ArgumentParser(description='Hendrycks Math数据集数学评估框架')
    parser.add_argument('-m', '--model', type=str, required=True,
                       help='模型名称 (deepseek_r1_1.5b, deepseek_r1_7b, deepseek_r1_14b, deepseek_r1_32b, deepseek_r1_70b, qwen25_0.5b, qwen25_1.5b, qwen25_3b, qwen25_7b, qwen25_14b, qwen25_32b, qwen25_72b)')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                       help='Hendrycks Math数据集路径')
    parser.add_argument('-s', '--samples', type=int, default=None,
                       help='最大样本数量 (默认: 全部)')
    parser.add_argument('-t', '--type', type=str, default=None,
                       help='问题类型筛选 (如: "Counting & Probability")')
    parser.add_argument('-l', '--samples_per_level', type=int, default=None,
                       help='每个难度等级的样本数量')
    parser.add_argument('--use_train', action='store_true',
                       help='使用train.csv而不是test.csv')
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
        framework = HendrycksMathEvaluationFramework(
            model_name=args.model,
            dataset_path=args.dataset
        )
        
        # 运行评估
        results = framework.run_evaluation(
            max_samples=args.samples,
            problem_type=args.type,
            samples_per_level=args.samples_per_level,
            use_train=args.use_train
        )
        
        # 显示统计信息
        print(f"\n📊 Hendrycks Math评估统计:")
        print(f"   总样本数: {len(results)}")
        
        if args.type:
            print(f"   问题类型: {args.type}")
        
        if args.samples_per_level:
            print(f"   每难度等级样本数: {args.samples_per_level}")
        
        if args.use_train:
            print(f"   数据集: train.csv")
        else:
            print(f"   数据集: test.csv")
        
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
        
        print(f"✅ Hendrycks Math评估完成！")
        
    except Exception as e:
        print(f"❌ Hendrycks Math评估失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 