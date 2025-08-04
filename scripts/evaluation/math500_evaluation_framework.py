#!/usr/bin/env python3
"""
MATH-500数据集专用评估框架
专门处理MATH-500数据集的数学评估，包含本地和远程模型支持
"""

import os
import sys
import json
import logging
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import openai

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Math500EvaluationFramework:
    """MATH-500数据集专用评估框架"""
    
    def __init__(self, model_config: Dict[str, Any], openai_api_key: str = None, max_samples: int = 200):
        """
        初始化MATH-500评估框架
        
        Args:
            model_config: 模型配置字典
            openai_api_key: OpenAI API密钥
            max_samples: 最大样本数量
        """
        self.model_config = model_config
        self.model_name = model_config['name']
        self.model_type = model_config.get('type', 'unknown')
        self.max_samples = max_samples
        self.openai_api_key = openai_api_key
        
        # 生成运行标识符
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        import random
        random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
        self.run_id = f"{timestamp}_{random_suffix}"
        logger.info(f"🆔 本次运行ID: {self.run_id}")
        
        # 初始化模型和tokenizer
        self.model = None
        self.tokenizer = None
        
        # 设置OpenAI客户端
        if openai_api_key:
            openai.api_key = openai_api_key
    
    def load_model(self):
        """加载模型和tokenizer"""
        logger.info(f"🧮 加载模型: {self.model_name}")
        
        try:
            # 检查模型缓存状态
            if self._check_model_cache():
                logger.info("📦 从本地缓存加载模型...")
                cache_path = self._load_model_from_cache()
                if cache_path:
                    self._load_model_from_path(cache_path, use_cache=True)
                else:
                    raise ValueError("无法获取缓存路径")
            else:
                logger.info("📥 从Hugging Face下载模型...")
                self._load_model_from_path(self.model_name, use_cache=False)
            
            logger.info("✅ 模型加载完成！")
            self._log_gpu_memory()
                
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def _load_model_from_path(self, model_path: str, use_cache: bool = False):
        """从指定路径加载模型"""
        # 加载tokenizer
        logger.info("加载tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=use_cache
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型 - 根据模型类型使用不同配置
        logger.info("加载模型权重...")
        
        # 计算GPU内存分配策略
        total_gpus = torch.cuda.device_count()
        logger.info(f"🖥️ 检测到 {total_gpus} 个GPU")
        
        # 为不同大小的模型使用最优的并行策略
        if self.model_type in ["32b_quantized", "70b_quantized"]:
            # 大模型：使用balanced_low_0策略实现真正的多GPU并行
            device_map = "balanced_low_0"
            logger.info(f"📊 大模型使用balanced_low_0策略实现多GPU并行")
        else:
            # 小模型：使用auto策略
            device_map = "auto"
            logger.info("📊 小模型使用auto策略")
        
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": device_map,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "local_files_only": use_cache
        }
        
        # 根据模型类型添加量化配置
        if self.model_type in ["7b_quantized", "14b_quantized", "32b_quantized", "70b_quantized"]:
            # 使用4bit量化
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model_kwargs["quantization_config"] = quantization_config
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    def _check_model_cache(self) -> bool:
        """检查模型是否已缓存"""
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache_path = os.path.join(cache_dir, f"models--{self.model_name.replace('/', '--')}")
            
            if os.path.exists(model_cache_path):
                # 首先检查主模型目录（模型文件通常在这里）
                if os.path.exists(os.path.join(model_cache_path, "model-00001-of-000017.safetensors")):
                    # 检查基本文件
                    basic_files = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
                    missing_basic = []
                    for f in basic_files:
                        file_path = os.path.join(model_cache_path, f)
                        if not os.path.exists(file_path):
                            missing_basic.append(f)
                    
                    if not missing_basic:
                        logger.info(f"✅ 模型已缓存: {self.model_name}")
                        return True
                    else:
                        logger.info(f"⚠️ 模型缓存不完整，缺少文件: {missing_basic}")
                        return False
                
                # 如果主目录没有，再检查snapshots目录
                snapshots_dir = os.path.join(model_cache_path, "snapshots")
                if os.path.exists(snapshots_dir):
                    for snapshot in os.listdir(snapshots_dir):
                        snapshot_path = os.path.join(snapshots_dir, snapshot)
                        if os.path.isdir(snapshot_path):
                            # 检查基本文件（包括符号链接）
                            basic_files = ["config.json", "tokenizer.json"]
                            missing_basic = []
                            for f in basic_files:
                                file_path = os.path.join(snapshot_path, f)
                                # 检查文件是否存在（包括符号链接）
                                if not (os.path.exists(file_path) or os.path.lexists(file_path)):
                                    missing_basic.append(f)
                            
                            # 检查模型文件（支持多种格式）
                            model_files = ["model.safetensors", "model.safetensors.index.json"]
                            has_model_file = False
                            for f in model_files:
                                file_path = os.path.join(snapshot_path, f)
                                if os.path.exists(file_path) or os.path.lexists(file_path):
                                    has_model_file = True
                                    break
                            
                            if not missing_basic and has_model_file:
                                logger.info(f"✅ 模型已缓存: {self.model_name}")
                                return True
                            else:
                                missing_files = missing_basic + ([] if has_model_file else ["model files"])
                                logger.info(f"⚠️ 模型缓存不完整，缺少文件: {missing_files}")
                                return False
            
            logger.info(f"❌ 模型未缓存: {self.model_name}")
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ 检查缓存时出错: {e}")
            return False
    
    def _load_model_from_cache(self):
        """直接从缓存加载模型，不进行网络连接"""
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache_path = os.path.join(cache_dir, f"models--{self.model_name.replace('/', '--')}")
            
            if not os.path.exists(model_cache_path):
                raise ValueError(f"模型缓存不存在: {model_cache_path}")
            
            # 首先尝试从主模型目录加载（模型文件通常在这里）
            if os.path.exists(os.path.join(model_cache_path, "model-00001-of-000017.safetensors")):
                logger.info(f"📦 从主模型目录加载: {model_cache_path}")
                return model_cache_path
            
            # 如果主目录没有，再尝试snapshots目录
            snapshots_dir = os.path.join(model_cache_path, "snapshots")
            if not os.path.exists(snapshots_dir):
                raise ValueError(f"模型快照目录不存在: {snapshots_dir}")
            
            snapshots = [d for d in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, d))]
            if not snapshots:
                raise ValueError(f"没有找到模型快照")
            
            latest_snapshot = snapshots[-1]
            snapshot_path = os.path.join(snapshots_dir, latest_snapshot)
            
            logger.info(f"📦 从缓存路径加载: {snapshot_path}")
            return snapshot_path
            
        except Exception as e:
            logger.error(f"❌ 获取缓存路径失败: {e}")
            return None
    
    def _log_gpu_memory(self):
        """显示GPU内存使用情况"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                logger.info(f"GPU {i}: 已分配 {memory_allocated:.2f}GB, 已保留 {memory_reserved:.2f}GB")
        else:
            logger.info("未检测到GPU，使用CPU运行")
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """加载MATH-500数据集"""
        logger.info(f"📊 加载MATH-500数据集: {dataset_path}")
        
        try:
            df = pd.read_csv(dataset_path)
            
            # 验证MATH-500数据集格式
            required_columns = ['problem', 'solution', 'answer', 'subject', 'level', 'unique_id']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"❌ 缺少必需列: {missing_columns}")
            
            # 检查难度范围 (MATH-500只有1-5级)
            level_range = df['level'].dropna()
            if not all(level_range.isin([1, 2, 3, 4, 5])):
                invalid_levels = level_range[~level_range.isin([1, 2, 3, 4, 5])].unique()
                raise ValueError(f"❌ 发现无效难度等级: {invalid_levels}，MATH-500只支持1-5级")
            
            # 转换为标准格式
            dataset = []
            for _, row in df.iterrows():
                sample = {
                    'id': row['unique_id'],  # 使用unique_id作为id
                    'problem': row['problem'],
                    'solution': row['solution'],
                    'answer': row['answer'],
                    'difficulty': row['level'],  # 使用level作为difficulty
                    'topic': row['subject'],     # 使用subject作为topic
                    'difficulty_score': float(row['level'])  # 使用level作为difficulty_score
                }
                dataset.append(sample)
            
            # 限制样本数量
            dataset = dataset[:self.max_samples]
            logger.info(f"✅ 从MATH-500数据集加载 {len(dataset)} 个样本")
            return dataset
            
        except Exception as e:
            logger.error(f"❌ 数据集加载失败: {e}")
            raise
    
    def generate_response(self, problem: str) -> str:
        """使用本地模型生成回答"""
        try:
            # DeepSeek-R1推荐的提示格式
            prompt = f"<think>\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n</think>\n\n{problem}\n\n<think>\n"
            
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # 移动到GPU
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 生成回答
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.model_config.get('max_new_tokens', 500),
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的部分（去掉输入提示）
            generated_text = response.replace(prompt, "").strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"❌ 生成失败: {e}")
            return f"生成失败: {str(e)}"
    
    def evaluate_with_openai(self, problem: str, model_response: str, correct_answer: str, standard_solution: str = "") -> Dict[str, Any]:
        """使用OpenAI评估模型回答"""
        if not self.openai_api_key:
            return {"error": "OpenAI API密钥未设置"}
        
        try:
            evaluation_prompt = f"""
You are a professional mathematical education evaluator. Please evaluate the quality of the answer to the following mathematical problem.

Problem: {problem}

Correct Answer: {correct_answer}

Standard Solution: {standard_solution}

Model Response: {model_response}

Please evaluate from the following aspects and give a score from 1 to 10:

1. Answer Correctness (1-10 points): Whether the final answer is correct
2. Reasoning Logic (1-10 points): Whether the reasoning process is clear and logical, compared to the standard solution
3. Step Completeness (1-10 points): Whether all necessary solution steps are shown, considering what the standard solution covers
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
"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional mathematical education evaluator."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.3
            )
            
            response_content = response.choices[0].message.content
            
            # 解析JSON响应
            import re
            
            try:
                evaluation = json.loads(response_content)
                return evaluation
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ JSON解析失败: {e}")
                
                # 尝试提取JSON部分
                try:
                    json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group()
                        evaluation = json.loads(json_str)
                        logger.info(f"✅ 成功提取JSON部分")
                        return evaluation
                except:
                    pass
                
                # 尝试修复常见的JSON格式问题
                try:
                    cleaned_response = response_content.replace('```json', '').replace('```', '').strip()
                    evaluation = json.loads(cleaned_response)
                    logger.info(f"✅ 成功修复JSON格式")
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
            logger.error(f"❌ OpenAI evaluation failed: {e}")
            return {
                "error": f"Evaluation failed: {e}",
                "error_type": "openai_api_error",
                "exception": str(e)
            }
    
    def run_evaluation(self, dataset_path: str) -> Dict[str, Any]:
        """运行完整MATH-500评估流程"""
        logger.info("🚀 开始MATH-500评估流程")
        
        # 加载模型
        self.load_model()
        
        # 加载数据集
        dataset = self.load_dataset(dataset_path)
        
        # 检查已有结果，实现断点续传
        start_index = self._check_existing_results()
        if start_index > 0:
            logger.info(f"🔄 发现已有结果，从第 {start_index + 1} 个样本开始继续评估")
            dataset = dataset[start_index:]
        else:
            logger.info(f"🆕 开始全新评估")
        
        results = []
        successful_generations = 0
        successful_evaluations = 0
        
        logger.info(f"📝 开始评估 {len(dataset)} 个样本...")
        
        from tqdm import tqdm
        for i, sample in enumerate(tqdm(dataset, desc="MATH-500评估进度")):
            global_index = start_index + i + 1
            logger.info(f"\n--- 样本 {global_index}/{len(dataset)}: {sample['id']} ---")
            
            # 生成模型回答
            model_response = self.generate_response(sample['problem'])
            
            if model_response and not model_response.startswith("生成失败"):
                successful_generations += 1
                
                # OpenAI评估
                evaluation = self.evaluate_with_openai(
                    sample['problem'], 
                    model_response, 
                    sample['answer'],
                    sample['solution']
                )
                
                if isinstance(evaluation, dict) and "error" not in evaluation:
                    successful_evaluations += 1
                    logger.info(f"✅ 评估完成，总分: {evaluation.get('overall_score', 0):.2f}")
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
                
                # 保存结果
                result = {
                    "id": sample['id'],
                    "problem": sample['problem'],
                    "correct_answer": sample['answer'],
                    "standard_solution": sample['solution'],
                    "model_response": model_response,
                    "difficulty": sample['difficulty'],
                    "topic": sample['topic'],
                    "evaluation": evaluation,
                    "generation_status": "success"
                }
                results.append(result)
            else:
                logger.warning(f"❌ 生成失败: {model_response}")
                
                # 为生成失败的样本创建特殊评估
                failed_evaluation = {
                    "answer_correctness": 0,
                    "reasoning_logic": 0,
                    "step_completeness": 0,
                    "mathematical_accuracy": 0,
                    "expression_clarity": 0,
                    "overall_score": 0,
                    "comments": f"生成失败: {model_response}",
                    "error": "generation_failed"
                }
                
                result = {
                    "id": sample['id'],
                    "problem": sample['problem'],
                    "correct_answer": sample['answer'],
                    "standard_solution": sample['solution'],
                    "model_response": model_response,
                    "difficulty": sample['difficulty'],
                    "topic": sample['topic'],
                    "evaluation": failed_evaluation,
                    "generation_status": "failed"
                }
                results.append(result)
            
            # 每10个样本保存一次中间结果
            if (i + 1) % 10 == 0:
                global_count = start_index + i + 1
                self.save_intermediate_results(results, global_count)
                logger.info(f"💾 已处理 {global_count} 个样本，中间结果已保存")
        
        # 计算最终统计信息
        if results:
            scores = [r['evaluation'].get('overall_score', 0) for r in results if r.get('generation_status') == 'success' and 'overall_score' in r['evaluation']]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            final_results = {
                "results": results,
                "summary": {
                    "total_evaluated": len(results),
                    "successful_generations": successful_generations,
                    "failed_generations": len(results) - successful_generations,
                    "generation_success_rate": successful_generations / len(results) if results else 0,
                    "evaluation_success_rate": successful_evaluations / len(results) if results else 0,
                    "average_overall_score": avg_score
                }
            }
            
            logger.info(f"\n📋 MATH-500评估摘要:")
            logger.info(f"总评估样本: {len(results)}")
            logger.info(f"生成成功率: {successful_generations / len(results) * 100:.1f}%")
            logger.info(f"平均总分: {avg_score:.2f}")
            
            # 保存最终结果
            self.save_final_results(final_results)
            
            return final_results
        else:
            return {"error": "没有有效结果"}
    
    def save_intermediate_results(self, results: List[Dict], count: int):
        """保存MATH-500中间结果"""
        # 创建MATH-500专用目录
        model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
        model_dir = f"data/math500_results/{model_safe_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        # 使用运行ID创建子目录
        run_dir = f"{model_dir}/{self.run_id}"
        os.makedirs(run_dir, exist_ok=True)
        
        filename = f"{run_dir}/intermediate_results_{count}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 MATH-500中间结果已保存: {filename}")
    
    def save_final_results(self, final_results: Dict[str, Any]):
        """保存MATH-500最终结果"""
        # 确保目录存在
        os.makedirs("data/math500_results", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
        filename = f"data/math500_results/final_math500_evaluation_{model_safe_name}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 MATH-500最终结果已保存: {filename}")
        
        # 打印摘要
        summary = final_results.get("summary", {})
        logger.info(f"\n📋 MATH-500评估摘要:")
        logger.info(f"总评估样本: {summary.get('total_evaluated', 0)}")
        logger.info(f"生成成功率: {summary.get('generation_success_rate', 0):.2%}")
        logger.info(f"平均总分: {summary.get('average_overall_score', 0):.2f}")
    
    def _check_existing_results(self):
        """检查已有结果，返回应该开始的样本索引"""
        # 创建MATH-500专用目录
        model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
        model_dir = f"data/math500_results/{model_safe_name}"
        
        if not os.path.exists(model_dir):
            return 0
        
        # 查找所有运行目录
        import glob
        run_dirs = glob.glob(f"{model_dir}/*")
        if not run_dirs:
            return 0
        
        # 找到最新的运行目录
        latest_run_dir = max(run_dirs, key=os.path.getctime)
        
        # 查找该运行目录下的所有中间结果文件
        result_files = glob.glob(f"{latest_run_dir}/intermediate_results_*.json")
        if not result_files:
            return 0
        
        # 找到最大的样本数量
        max_count = 0
        for file_path in result_files:
            try:
                filename = os.path.basename(file_path)
                # 提取文件名中的数字，如 intermediate_results_30.json -> 30
                count_str = filename.replace('intermediate_results_', '').replace('.json', '')
                count = int(count_str)
                max_count = max(max_count, count)
            except:
                continue
        
        logger.info(f"📁 发现已有结果目录: {latest_run_dir}")
        logger.info(f"📊 已处理样本数量: {max_count}")
        
        return max_count

# 预定义的模型配置
MATH500_MODEL_CONFIGS = {
    "deepseek_r1_1.5b": {
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "type": "1.5b",
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
    }
} 