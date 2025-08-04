#!/usr/bin/env python3
"""
通用数学评估框架 - 支持多个模型
包括：数据集处理 -> 模型推理 -> OpenAI打分 -> 结果分析
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import json
import pandas as pd
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import openai
import matplotlib.pyplot as plt
import matplotlib
# 使用英文标签，避免中文显示问题
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 关闭OpenAI库的调试日志
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

class MathEvaluationFramework:
    """通用数学评估框架"""
    
    def __init__(self, model_config: Dict[str, Any], openai_api_key: str = None, max_samples: int = 200):
        """
        初始化评估框架
        
        Args:
            model_config: 模型配置字典
            openai_api_key: OpenAI API密钥
            max_samples: 最大测试样本数
        """
        self.model_config = model_config
        self.model_name = model_config['name']
        self.model_type = model_config.get('type', 'default')
        self.max_samples = max_samples
        self.model = None
        self.tokenizer = None
        
        # 生成运行标识符
        import random
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
        self.run_id = f"{timestamp}_{random_suffix}"
        logger.info(f"🆔 本次运行ID: {self.run_id}")
        
        # 设置OpenAI
        if openai_api_key:
            openai.api_key = openai_api_key
            # 使用旧版本OpenAI库 (0.28.0)
            self.openai_client = openai
        else:
            self.openai_client = None
            logger.warning("未提供OpenAI API密钥，将跳过OpenAI打分步骤")
    
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
        elif self.model_type in ["3b", "7b"]:
            # 3B、7B模型：跳过GPU 0，使用其他GPU
            device_map = "balanced_low_0"
            logger.info(f"📊 {self.model_type}模型使用balanced_low_0策略，跳过GPU 0")
        elif self.model_type in ["14b", "32b", "72b"]:
            # 14B、32B、72B模型：使用GPU并行策略
            device_map = "balanced_low_0"
            logger.info(f"📊 {self.model_type}模型使用balanced_low_0策略实现多GPU并行")
        else:
            # 小模型：使用auto策略
            device_map = "auto"
            logger.info("📊 小模型使用auto策略")
        
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": device_map,  # 关键：使用最优的device_map策略
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "local_files_only": use_cache
        }
        
        # 根据模型类型添加特殊配置
        if self.model_type == "7b_quantized":
            # 7B模型使用4bit量化
            model_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16
            })
        elif self.model_type == "14b_quantized":
            # 14B模型使用4bit量化
            model_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16
            })
        elif self.model_type == "32b_quantized":
            # 32B模型使用4bit量化
            model_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16
            })
        elif self.model_type == "70b_quantized":
            # 70B模型使用4bit量化
            model_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16
            })
        elif self.model_type in ["1.5b", "3b", "7b"]:
            # 1.5B、3B、7B模型使用标准配置
            pass
        elif self.model_type in ["14b", "32b", "72b"]:
            # 14B、32B、72B模型使用标准配置，支持多GPU并行
            pass
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    def _check_model_cache(self) -> bool:
        """检查模型是否已缓存"""
        try:
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_cache_path = os.path.join(cache_dir, f"models--{self.model_name.replace('/', '--')}")
            
            if os.path.exists(model_cache_path):
                # 检查主模型目录（模型文件通常在这里）
                # 动态检查模型文件，支持分片文件和单个文件
                model_files = [f for f in os.listdir(model_cache_path) if f.startswith("model-") and f.endswith(".safetensors")]
                single_model_file = os.path.join(model_cache_path, "model.safetensors")
                
                if model_files or os.path.exists(single_model_file):
                    # 检查基本文件
                    basic_files = ["config.json", "tokenizer.json"]
                    missing_basic = []
                    for f in basic_files:
                        file_path = os.path.join(model_cache_path, f)
                        if not os.path.exists(file_path):
                            missing_basic.append(f)
                    
                    # 检查模型索引文件（可选，小模型可能没有）
                    index_file = os.path.join(model_cache_path, "model.safetensors.index.json")
                    if not os.path.exists(index_file):
                        logger.debug(f"⚠️ 模型索引文件不存在（小模型可能不需要）: {index_file}")
                    
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
                            # 支持分片文件和单个文件
                            model_files = [f for f in os.listdir(snapshot_path) if f.startswith("model-") and f.endswith(".safetensors")]
                            single_model_file = os.path.join(snapshot_path, "model.safetensors")
                            
                            has_model_file = bool(model_files) or os.path.exists(single_model_file)
                            
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
            # 动态检查模型文件，支持分片文件和单个文件
            model_files = [f for f in os.listdir(model_cache_path) if f.startswith("model-") and f.endswith(".safetensors")]
            single_model_file = os.path.join(model_cache_path, "model.safetensors")
            
            if model_files or os.path.exists(single_model_file):
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
        """加载数据集"""
        logger.info(f"📊 加载数据集: {dataset_path}")
        
        # 优先使用指定的数据集路径
        if os.path.exists(dataset_path):
            logger.info(f"📁 使用指定数据集: {dataset_path}")
            try:
                df = pd.read_csv(dataset_path)
                logger.info(f"✅ 从指定文件加载 {len(df)} 个样本")
                return self._convert_df_to_dataset(df)
            except Exception as e:
                logger.error(f"❌ 指定数据集加载失败: {e}")
        
        # 如果没有指定数据集，检查是否有固定样本文件
        model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
        fixed_samples_path = f"data/processed/fixed_{self.max_samples}_samples_{model_safe_name}.csv"
        
        if os.path.exists(fixed_samples_path):
            logger.info(f"📁 发现固定样本文件: {fixed_samples_path}")
            try:
                df = pd.read_csv(fixed_samples_path)
                logger.info(f"✅ 从固定文件加载 {len(df)} 个样本")
                return self._convert_df_to_dataset(df)
            except Exception as e:
                logger.error(f"❌ 固定样本文件加载失败: {e}")
                logger.info("🔄 重新生成固定样本...")
        
        # 如果都没有，从原始数据集创建分层采样的样本
        logger.info("🔄 创建分层采样的固定样本...")
        try:
            original_dataset = "data/processed/deepmath_evaluation_dataset.csv"
            df = pd.read_csv(original_dataset)
            logger.info(f"原始数据集包含 {len(df)} 个样本")
            
            # 分层采样：确保包含所有难度等级
            stratified_samples = self._create_stratified_samples(df, self.max_samples)
            
            # 保存到固定位置
            stratified_samples.to_csv(fixed_samples_path, index=False)
            logger.info(f"💾 固定样本已保存到: {fixed_samples_path}")
            
            return self._convert_df_to_dataset(stratified_samples)
            
        except Exception as e:
            logger.error(f"❌ 数据集加载失败: {e}")
            raise
    
    def _convert_df_to_dataset(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """将DataFrame转换为数据集格式"""
        dataset = []
        for _, row in df.iterrows():
            sample = {
                'id': row['id'],
                'problem': row['problem'],
                'solution': row['solution'],
                'answer': row['answer'],
                'difficulty': row['difficulty'],
                'topic': row['topic'],
                'difficulty_score': row['difficulty_score']
            }
            dataset.append(sample)
        return dataset
    
    def _create_stratified_samples(self, df: pd.DataFrame, target_samples: int) -> pd.DataFrame:
        """创建分层采样的样本"""
        logger.info("📊 开始分层采样...")
        
        # 获取所有难度等级
        difficulty_levels = sorted(df['difficulty'].unique())
        logger.info(f"发现难度等级: {difficulty_levels}")
        
        # 计算每个难度等级的样本数
        total_levels = len(difficulty_levels)
        base_samples_per_level = target_samples // total_levels
        remaining_samples = target_samples % total_levels
        
        logger.info(f"每个难度等级基础样本数: {base_samples_per_level}")
        logger.info(f"剩余样本数: {remaining_samples}")
        
        stratified_samples = []
        
        for i, difficulty in enumerate(difficulty_levels):
            # 获取当前难度等级的所有样本
            level_df = df[df['difficulty'] == difficulty]
            level_count = len(level_df)
            
            # 计算当前等级应采样的数量
            if i < remaining_samples:
                samples_needed = base_samples_per_level + 1
            else:
                samples_needed = base_samples_per_level
            
            # 如果当前等级的样本数不足，全部使用
            if level_count <= samples_needed:
                samples_needed = level_count
                logger.info(f"难度 {difficulty}: 使用全部 {level_count} 个样本")
            else:
                logger.info(f"难度 {difficulty}: 随机选择 {samples_needed} 个样本（共 {level_count} 个）")
            
            # 随机采样
            if level_count > 0:
                sampled = level_df.sample(n=samples_needed, random_state=42)
                stratified_samples.append(sampled)
        
        # 合并所有采样的样本
        if stratified_samples:
            result_df = pd.concat(stratified_samples, ignore_index=True)
            logger.info(f"✅ 分层采样完成，共 {len(result_df)} 个样本")
            
            # 显示每个难度等级的样本数
            difficulty_counts = result_df['difficulty'].value_counts().sort_index()
            logger.info("📊 各难度等级样本分布:")
            for difficulty, count in difficulty_counts.items():
                logger.info(f"  难度 {difficulty}: {count} 个样本")
            
            return result_df
        else:
            raise ValueError("分层采样失败，没有生成任何样本")
    
    def generate_response(self, problem: str) -> str:
        """使用模型生成回答"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # DeepSeek-R1推荐的提示格式
                prompt = f"<think>\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n</think>\n\n{problem}\n\n<think>\n"
                
                logger.debug(f"输入提示长度: {len(prompt)} 字符")
            
                # 编码输入
                inputs = self.tokenizer(prompt, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.model.device)
                attention_mask = inputs.attention_mask.to(self.model.device)
            
                logger.debug(f"输入token数量: {input_ids.shape[1]}")
                
                # 根据模型类型调整生成参数 - 减少token数量以节省内存
                max_new_tokens = min(self.model_config.get('max_new_tokens', 500), 300)
                
                # 记录生成前的内存状态
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gpu_memory_before = torch.cuda.memory_allocated() / 1024**3
                    logger.debug(f"生成前GPU内存: {gpu_memory_before:.2f} GB")
            
                # 生成回答 - 针对32B模型优化参数
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,  # 改为False避免概率分布问题
                        temperature=1.0,  # 使用默认温度
                        top_p=1.0,        # 使用默认top_p
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.0,  # 降低重复惩罚
                        use_cache=True,   # 启用缓存
                        return_dict_in_generate=False  # 避免复杂返回格式
                    )
            
                logger.debug(f"生成token数量: {outputs.shape[1] - input_ids.shape[1]}")
                
                # 解码完整输出
                try:
                    full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    logger.debug(f"完整响应长度: {len(full_response)} 字符")
                except Exception as decode_error:
                    logger.error(f"❌ Token解码失败: {decode_error}")
                    # 尝试不跳过特殊token
                    try:
                        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                        logger.warning("使用skip_special_tokens=False重新解码")
                    except Exception as decode_error2:
                        logger.error(f"❌ 重新解码也失败: {decode_error2}")
                        return f"生成失败: token解码错误 - {decode_error}"
            
                # 提取模型的回答部分（移除提示）
                model_response = full_response.replace(prompt, "").strip()
            
                logger.debug(f"模型回答长度: {len(model_response)} 字符")
                logger.debug(f"模型回答前50字符: {model_response[:50]}")
                
                # 检查输出是否为空或太短
                if not model_response or len(model_response) < 10:
                    logger.warning(f"⚠️ 生成输出过短或为空，尝试重试 ({attempt + 1}/{max_retries})")
                    logger.warning(f"原始完整响应: {full_response}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # 等待1秒后重试
                        continue
                    else:
                        return f"生成失败: 输出为空或过短"
                
                # 记录生成后的内存状态
                if torch.cuda.is_available():
                    gpu_memory_after = torch.cuda.memory_allocated() / 1024**3
                    logger.debug(f"生成后GPU内存: {gpu_memory_after:.2f} GB")
                
                return model_response
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"❌ GPU内存不足 (尝试 {attempt + 1}/{max_retries}): {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if attempt < max_retries - 1:
                    time.sleep(2)  # 等待2秒后重试
                    continue
                else:
                    return f"生成失败: GPU内存不足"
            
            except Exception as e:
                logger.error(f"❌ 生成回答失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                logger.error(f"错误类型: {type(e).__name__}")
                logger.error(f"错误详情: {str(e)}")
                
                # 记录更多调试信息
                if hasattr(e, '__traceback__'):
                    import traceback
                    logger.error(f"错误堆栈: {traceback.format_exc()}")
                
                if attempt < max_retries - 1:
                    time.sleep(1)  # 等待1秒后重试
                    continue
                else:
                    return f"生成失败: {type(e).__name__} - {str(e)}"
        
        return f"生成失败: 重试{max_retries}次后仍然失败"
    
    def evaluate_with_openai(self, problem: str, model_response: str, correct_answer: str, standard_solution: str = "") -> Dict[str, Any]:
        """使用OpenAI评估模型回答，包含标准解法参考"""
        if not self.openai_client:
            return {"error": "OpenAI客户端未初始化"}
        
        try:
            # 构建评估提示
            if standard_solution:
                evaluation_prompt = f"""
Please evaluate the quality of the answer to the following mathematical problem. You have access to the standard solution for reference.

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

CRITICAL: In the "comments" field, avoid using backslashes (\) or special characters that could break JSON parsing. Use simple text only.

Please return the evaluation result in JSON format:
{{
    "answer_correctness": score,
    "reasoning_logic": score,
    "step_completeness": score,
    "mathematical_accuracy": score,
    "expression_clarity": score,
    "overall_score": total_score/5,
    "comments": "Detailed evaluation with reference to standard solution"
}}
"""
            else:
                evaluation_prompt = f"""
Please evaluate the quality of the answer to the following mathematical problem.

Problem: {problem}

Correct Answer: {correct_answer}

Model Response: {model_response}

Please evaluate from the following aspects and give a score from 1 to 10:

1. Answer Correctness (1-10 points): Whether the final answer is correct
2. Reasoning Logic (1-10 points): Whether the reasoning process is clear and logical
3. Step Completeness (1-10 points): Whether all solution steps are shown
4. Mathematical Accuracy (1-10 points): Whether mathematical calculations and formulas are accurate
5. Expression Clarity (1-10 points): Whether the expression is clear and easy to understand

IMPORTANT: You must respond with ONLY a valid JSON object. Do not include any other text, explanations, or markdown formatting.

CRITICAL: In the "comments" field, avoid using backslashes (\) or special characters that could break JSON parsing. Use simple text only.

Please return the evaluation result in JSON format:
{{
    "answer_correctness": score,
    "reasoning_logic": score,
    "step_completeness": score,
    "mathematical_accuracy": score,
    "expression_clarity": score,
    "overall_score": total_score/5,
    "comments": "Detailed evaluation"
}}
"""
            
            # 使用旧版本OpenAI库 (0.28.0)
            response = self.openai_client.ChatCompletion.create(
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
                logger.warning(f"原始响应: {response_content[:200]}...")  # 只显示前200字符
                
                # 尝试提取JSON部分
                try:
                    # 查找JSON对象
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
                    # 移除可能的markdown代码块标记
                    cleaned_response = response_content.replace('```json', '').replace('```', '').strip()
                    evaluation = json.loads(cleaned_response)
                    logger.info(f"✅ 成功修复JSON格式")
                    return evaluation
                except:
                    pass
                
                # 尝试修复转义字符问题
                try:
                    # 修复常见的转义字符问题
                    fixed_response = response_content
                    # 修复 \boxed{} 格式
                    fixed_response = re.sub(r'\\boxed\{([^}]*)\}', r'\\boxed{\1}', fixed_response)
                    # 修复其他可能的转义问题
                    fixed_response = fixed_response.replace('\\n', '\\\\n')
                    fixed_response = fixed_response.replace('\\t', '\\\\t')
                    evaluation = json.loads(fixed_response)
                    logger.info(f"✅ 成功修复转义字符问题")
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
        """运行完整评估流程"""
        logger.info("🚀 开始完整评估流程")
        
        # 1. 加载模型
        self.load_model()
        
        # 2. 加载数据集
        dataset = self.load_dataset(dataset_path)
        
        # 3. 运行评估
        results = []
        evaluation_stats = {
            "total_samples": len(dataset),
            "successful_generations": 0,
            "successful_evaluations": 0,
            "average_scores": {},
            "difficulty_analysis": {}
        }
        
        logger.info(f"📝 开始评估 {len(dataset)} 个样本...")
        logger.info(f"⏱️ 预计需要时间: {len(dataset) * 30 / 60:.1f} 分钟（假设每个样本30秒）")
        
        start_time = time.time()
        for i, sample in enumerate(tqdm(dataset, desc="评估进度")):
            logger.info(f"\n--- 样本 {i+1}/{len(dataset)}: {sample['id']} ---")
            
            # 生成模型回答
            model_response = self.generate_response(sample['problem'])
            
            # 记录所有样本，包括生成失败的
            if model_response and not model_response.startswith("生成失败"):
                evaluation_stats["successful_generations"] += 1
                
                # OpenAI评估
                evaluation = self.evaluate_with_openai(
                    sample['problem'], 
                    model_response, 
                    sample['answer'],
                    sample.get('solution', '')  # 传入标准解法，如果没有则为空字符串
                )
                
                if isinstance(evaluation, dict) and "error" not in evaluation:
                    evaluation_stats["successful_evaluations"] += 1
                    logger.info(f"✅ 评估完成，总分: {evaluation.get('overall_score', 0):.2f}")
                else:
                    # 详细错误信息处理
                    if isinstance(evaluation, dict):
                        error_msg = evaluation.get('error', '未知错误')
                        parse_error = evaluation.get('parse_error', '')
                        raw_response = evaluation.get('raw_response', '')[:200]  # 只显示前200字符
                        
                        if parse_error:
                            logger.warning(f"⚠️ 评估失败: {error_msg}")
                            logger.warning(f"解析错误: {parse_error}")
                            if raw_response:
                                logger.warning(f"原始响应: {raw_response}...")
                        else:
                            logger.warning(f"⚠️ 评估失败: {error_msg}")
                    else:
                        logger.warning(f"⚠️ 评估失败: {str(evaluation)}")
                    
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
                    "standard_solution": sample.get('solution', ''),  # 添加原始解法
                    "model_response": model_response,
                    "difficulty": sample['difficulty'],
                    "topic": sample['topic'],
                    "evaluation": evaluation,
                    "generation_status": "success"
                }
                results.append(result)
            
            else:
                # 生成失败 - 记录失败信息
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
                
                # 保存生成失败的结果
                result = {
                    "id": sample['id'],
                    "problem": sample['problem'],
                    "correct_answer": sample['answer'],
                    "standard_solution": sample.get('solution', ''),  # 添加原始解法
                    "model_response": model_response,
                    "difficulty": sample['difficulty'],
                    "topic": sample['topic'],
                    "evaluation": failed_evaluation,
                    "generation_status": "failed"
                }
                results.append(result)
            
            # 每10个样本保存一次中间结果
            if (i + 1) % 10 == 0:
                self.save_intermediate_results(results, i + 1)
                logger.info(f"💾 已处理 {i + 1}/{len(dataset)} 个样本，中间结果已保存")
        
        # 4. 分析结果
        elapsed_time = time.time() - start_time
        logger.info(f"⏱️ 总耗时: {elapsed_time / 60:.1f} 分钟")
        logger.info(f"📊 平均每个样本: {elapsed_time / len(dataset):.1f} 秒")
        
        final_results = self.analyze_results(results, evaluation_stats)
        
        # 5. 生成难度-评分曲线图
        if results:
            try:
                plot_path = self.generate_difficulty_score_plot(results)
                final_results["plot_path"] = plot_path
                logger.info(f"📈 难度-评分曲线图已生成: {plot_path}")
            except Exception as e:
                logger.error(f"❌ 生成曲线图失败: {e}")
        
        # 6. 保存最终结果
        self.save_final_results(final_results)
        
        return final_results
    
    def analyze_results(self, results: List[Dict], stats: Dict) -> Dict[str, Any]:
        """分析评估结果"""
        logger.info("📊 分析评估结果...")
        
        if not results:
            return {"error": "没有有效结果可分析"}
        
        # 统计生成状态
        successful_generations = sum(1 for r in results if r.get('generation_status') == 'success')
        failed_generations = sum(1 for r in results if r.get('generation_status') == 'failed')
        total_generations = len(results)
        
        logger.info(f"📈 生成统计:")
        logger.info(f"  - 成功生成: {successful_generations}/{total_generations} ({successful_generations/total_generations*100:.1f}%)")
        logger.info(f"  - 生成失败: {failed_generations}/{total_generations} ({failed_generations/total_generations*100:.1f}%)")
        
        # 计算平均分数（只考虑成功生成的样本）
        scores = []
        difficulty_scores = {}
        difficulty_failure_rates = {}
        
        for result in results:
            evaluation = result.get('evaluation', {})
            difficulty = result.get('difficulty', 'unknown')
            generation_status = result.get('generation_status', 'unknown')
            
            # 统计每个难度的失败率
            if difficulty not in difficulty_failure_rates:
                difficulty_failure_rates[difficulty] = {'total': 0, 'failed': 0}
            difficulty_failure_rates[difficulty]['total'] += 1
            if generation_status == 'failed':
                difficulty_failure_rates[difficulty]['failed'] += 1
            
            # 只计算成功生成的样本分数
            if generation_status == 'success' and 'overall_score' in evaluation:
                score = evaluation['overall_score']
                scores.append(score)
                
                # 按难度分组
                if difficulty not in difficulty_scores:
                    difficulty_scores[difficulty] = []
                difficulty_scores[difficulty].append(score)
        
        if scores:
            stats["average_scores"] = {
                "overall": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "std": (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5
            }
        
        # 难度分析
        for difficulty, diff_scores in difficulty_scores.items():
            if diff_scores:
                stats["difficulty_analysis"][difficulty] = {
                    "count": len(diff_scores),
                    "average": sum(diff_scores) / len(diff_scores),
                    "min": min(diff_scores),
                    "max": max(diff_scores)
                }
        
        # 添加失败率统计
        stats["generation_failure_analysis"] = {
            "overall_failure_rate": failed_generations / total_generations if total_generations > 0 else 0,
            "difficulty_failure_rates": {
                diff: {
                    "failure_rate": info['failed'] / info['total'] if info['total'] > 0 else 0,
                    "total_samples": info['total'],
                    "failed_samples": info['failed']
                }
                for diff, info in difficulty_failure_rates.items()
            }
                }
        
        return {
            "results": results,
            "statistics": stats,
            "summary": {
                "total_evaluated": len(results),
                "successful_generations": successful_generations,
                "failed_generations": failed_generations,
                "generation_success_rate": successful_generations / total_generations if total_generations > 0 else 0,
                "evaluation_success_rate": stats["successful_evaluations"] / stats["total_samples"] if stats["total_samples"] > 0 else 0,
                "average_overall_score": stats["average_scores"].get("overall", 0) if "average_scores" in stats else 0
            }
        }
    
    def generate_difficulty_score_plot(self, results: List[Dict], save_path: str = None, show_plot: bool = True):
        """Generate difficulty-score curve plot"""
        if not results:
            logger.warning("No result data available, cannot generate curve plot")
            return None
        
        # 收集数据
        difficulty_scores = {}
        for result in results:
            evaluation = result.get('evaluation', {})
            if 'overall_score' in evaluation:
                difficulty = result.get('difficulty', 'unknown')
                score = evaluation['overall_score']
                
                if difficulty not in difficulty_scores:
                    difficulty_scores[difficulty] = []
                difficulty_scores[difficulty].append(score)
        
        if not difficulty_scores:
            logger.warning("No valid score data available, cannot generate curve plot")
            return None
        
        # 计算每个难度的平均分数
        difficulties = []
        avg_scores = []
        score_counts = []
        
        for difficulty in sorted(difficulty_scores.keys(), key=lambda x: float(x) if x != 'unknown' else -999):
            scores = difficulty_scores[difficulty]
            if scores:
                difficulties.append(float(difficulty) if difficulty != 'unknown' else -1)
                avg_scores.append(sum(scores) / len(scores))
                score_counts.append(len(scores))
        
        if not difficulties:
            logger.warning("No valid difficulty data available, cannot generate curve plot")
            return None
        
        # 创建图表
        plt.figure(figsize=(12, 8))
        
        # 主曲线图
        plt.subplot(2, 1, 1)
        plt.plot(difficulties, avg_scores, 'bo-', linewidth=2, markersize=8, label='Average Score')
        plt.fill_between(difficulties, avg_scores, alpha=0.3)
        
        plt.xlabel('Problem Difficulty', fontsize=12)
        plt.ylabel('Average Score', fontsize=12)
        plt.title(f'{self.model_name} - Difficulty-Score Curve (Evaluated: {len(results)} samples)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加数据点标注
        for i, (diff, score, count) in enumerate(zip(difficulties, avg_scores, score_counts)):
            plt.annotate(f'n={count}', (diff, score), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        
        # 样本数量柱状图
        plt.subplot(2, 1, 2)
        plt.bar(difficulties, score_counts, alpha=0.7, color='skyblue', edgecolor='navy')
        plt.xlabel('Problem Difficulty', fontsize=12)
        plt.ylabel('Sample Count', fontsize=12)
        plt.title('Sample Distribution by Difficulty Level', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标注
        for i, (diff, count) in enumerate(zip(difficulties, score_counts)):
            plt.text(diff, count + 0.1, str(count), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            # 确保目录存在
            os.makedirs("data/plots", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
            save_path = f"data/plots/difficulty_score_plot_{model_safe_name}_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"📈 曲线图已保存: {save_path}")
        
        # 显示图表
        if show_plot:
            plt.show()
        else:
            plt.close()  # 不显示时关闭图表以节省内存
        
        return save_path
    

    
    def save_intermediate_results(self, results: List[Dict], count: int):
        """保存中间结果"""
        # 创建模型专用目录
        model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
        model_dir = f"data/intermediate/{model_safe_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        # 使用运行ID创建子目录
        run_dir = f"{model_dir}/{self.run_id}"
        os.makedirs(run_dir, exist_ok=True)
        
        filename = f"{run_dir}/intermediate_results_{count}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 中间结果已保存: {filename}")
    
    def save_final_results(self, final_results: Dict[str, Any]):
        """保存最终结果"""
        # 确保目录存在
        os.makedirs("data/results", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe_name = self.model_name.replace('/', '_').replace('-', '_')
        filename = f"data/results/final_evaluation_{model_safe_name}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 最终结果已保存: {filename}")
        
        # 打印摘要
        summary = final_results.get("summary", {})
        logger.info(f"\n📋 评估摘要:")
        logger.info(f"总评估样本: {summary.get('total_evaluated', 0)}")
        logger.info(f"成功率: {summary.get('success_rate', 0):.2%}")
        logger.info(f"平均总分: {summary.get('average_overall_score', 0):.2f}")
        
        stats = final_results.get("statistics", {})
        if "average_scores" in stats:
            avg_scores = stats["average_scores"]
            logger.info(f"分数范围: {avg_scores.get('min', 0):.2f} - {avg_scores.get('max', 0):.2f}")
            logger.info(f"标准差: {avg_scores.get('std', 0):.2f}")

# 预定义的模型配置
MODEL_CONFIGS = {
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

def run_math_evaluation(model_key: str = "deepseek_r1_1.5b", max_samples: int = 200):
    """运行数学评估流程"""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型: {model_key}。支持的模型: {list(MODEL_CONFIGS.keys())}")
    
    model_config = MODEL_CONFIGS[model_key]
    DATASET_PATH = "data/processed/deepmath_evaluation_dataset.csv"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    logger.info("🚀 开始数学评估流程")
    logger.info(f"模型: {model_config['description']}")
    logger.info(f"模型名称: {model_config['name']}")
    logger.info(f"数据集: {DATASET_PATH}")
    logger.info(f"最大样本数: {max_samples}")
    
    if not OPENAI_API_KEY:
        logger.warning("⚠️ 未设置OPENAI_API_KEY环境变量，将跳过OpenAI打分")
        logger.info("请设置环境变量: export OPENAI_API_KEY='your-api-key'")
    
    # 创建评估框架
    framework = MathEvaluationFramework(
        model_config=model_config,
        openai_api_key=OPENAI_API_KEY,
        max_samples=max_samples
    )
    
    # 运行评估
    try:
        results = framework.run_evaluation(DATASET_PATH)
        logger.info("🎉 数学评估完成！")
        return results
        
    except Exception as e:
        logger.error(f"❌ 数学评估失败: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 默认运行1.5B模型评估
    run_math_evaluation("deepseek_r1_1.5b", max_samples=200) 