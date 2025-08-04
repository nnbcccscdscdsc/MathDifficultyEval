#!/usr/bin/env python3
"""
工具函数模块
包含数据处理、评估指标、可视化等通用函数
"""

import os
import json
import yaml
import logging
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigLoader:
    """配置加载器"""
    
    @staticmethod
    def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

class MathEvaluator:
    """数学评估工具类"""
    
    def __init__(self):
        """初始化评估器"""
        pass
        
    def extract_number_from_text(self, text: str) -> Optional[float]:
        """从文本中提取数字"""
        numbers = re.findall(r'-?\d*\.?\d+', text)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None
        return None
    
    def is_numeric_answer(self, text: str) -> bool:
        """判断答案是否为数字"""
        return self.extract_number_from_text(text) is not None
    
    def calculate_exact_match(self, prediction: str, reference: str) -> bool:
        """计算精确匹配"""
        pred_clean = self.clean_text(prediction)
        ref_clean = self.clean_text(reference)
        return pred_clean == ref_clean
    
    def clean_text(self, text: str) -> str:
        """清理文本"""
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()
    
    def calculate_rouge_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """计算ROUGE分数（简化版）"""
        # 这里使用简单的文本相似度计算
        pred_words = set(self.clean_text(prediction).split())
        ref_words = set(self.clean_text(reference).split())
        
        if not ref_words:
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
        intersection = pred_words.intersection(ref_words)
        rouge1 = len(intersection) / len(ref_words) if ref_words else 0.0
        
        return {
            'rouge1': rouge1,
            'rouge2': rouge1,  # 简化处理
            'rougeL': rouge1   # 简化处理
        }
    
    def calculate_step_accuracy(self, prediction: str, reference: str) -> float:
        """计算推理步骤准确性（简化版）"""
        pred_steps = self.extract_steps(prediction)
        ref_steps = self.extract_steps(reference)
        
        if not ref_steps:
            return 0.0
        
        correct_steps = 0
        for pred_step in pred_steps:
            if any(self.similar_steps(pred_step, ref_step) for ref_step in ref_steps):
                correct_steps += 1
        
        return correct_steps / len(ref_steps) if ref_steps else 0.0
    
    def extract_steps(self, text: str) -> List[str]:
        """提取推理步骤"""
        # 简单的步骤提取，按数字编号或换行分割
        steps = re.split(r'\n\d+\.|\n\d+\)|\n•|\n-', text)
        return [step.strip() for step in steps if step.strip()]
    
    def similar_steps(self, step1: str, step2: str) -> bool:
        """判断两个步骤是否相似"""
        # 简单的相似度计算
        words1 = set(self.clean_text(step1).split())
        words2 = set(self.clean_text(step2).split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        similarity = len(intersection) / max(len(words1), len(words2))
        return similarity > 0.5

class ModelLoader:
    """模型加载器"""
    
    @staticmethod
    def load_model_and_tokenizer(
        model_name: str,
        quantization_config: Dict[str, Any],
        device: str = "auto"
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """加载模型和分词器"""
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading model {model_name} on {device}")
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device if device == "cuda" else None,
            **quantization_config
        )
        
        return model, tokenizer

class DataProcessor:
    """数据处理器"""
    
    @staticmethod
    def load_processed_data(data_path: str) -> pd.DataFrame:
        """加载处理后的数据"""
        return pd.read_csv(data_path)
    
    @staticmethod
    def filter_by_difficulty(df: pd.DataFrame, difficulty: str) -> pd.DataFrame:
        """按难度过滤数据"""
        return df[df['difficulty'] == difficulty]
    
    @staticmethod
    def create_prompt(problem: str, prompt_template: str) -> str:
        """创建提示词"""
        return prompt_template.format(problem=problem)
    
    @staticmethod
    def save_results(results: Dict[str, Any], output_path: str):
        """保存结果"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

class Visualizer:
    """可视化工具类"""
    
    def __init__(self, output_dir: str = "results"):
        """初始化可视化器"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_accuracy_by_difficulty(self, results: Dict[str, Any], save_path: str = None):
        """绘制不同难度等级的准确率"""
        difficulties = ['elementary', 'middle', 'college']
        accuracies = []
        
        for diff in difficulties:
            key = f'{diff}_accuracy'
            accuracies.append(results.get(key, 0))
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(difficulties, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Model Accuracy by Difficulty Level', fontsize=16, fontweight='bold')
        plt.xlabel('Difficulty Level', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def check_gpu_memory():
    """检查GPU内存"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU Memory: {gpu_memory:.2f} GB")
        return gpu_memory
    else:
        logger.info("No GPU available")
        return 0

def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def calculate_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """计算评估指标"""
    evaluator = MathEvaluator()
    
    # 清理文本
    pred_clean = evaluator.clean_text(prediction)
    ref_clean = evaluator.clean_text(reference)
    
    # 计算精确匹配
    exact_match = evaluator.calculate_exact_match(prediction, reference)
    
    # 计算ROUGE分数
    rouge_scores = evaluator.calculate_rouge_scores(prediction, reference)
    
    # 计算步骤准确性
    step_accuracy = evaluator.calculate_step_accuracy(prediction, reference)
    
    # 计算BLEU分数（简化版，使用ROUGE分数代替）
    bleu_score = rouge_scores['rouge1']
    
    # 计算整体准确率（综合多个指标）
    accuracy = (exact_match + rouge_scores['rouge1'] + step_accuracy) / 3
    
    return {
        'accuracy': accuracy,
        'exact_match': float(exact_match),
        'rouge_score': rouge_scores['rouge1'],
        'bleu_score': bleu_score,
        'step_accuracy': step_accuracy
    } 