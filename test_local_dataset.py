#!/usr/bin/env python3
"""
测试本地数学数据集
使用DialoGPT-medium模型测试local_math.csv数据集
"""

import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from scripts.openai_scorer import OpenAIScorer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_and_tokenizer():
    """加载模型和tokenizer"""
    model_name = "microsoft/DialoGPT-medium"
    
    logger.info(f"加载模型: {model_name}")
    
    # 加载tokenizer（强制使用本地缓存）
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载模型（强制使用本地缓存）
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    model = model.to('cuda')
    
    return model, tokenizer

def load_local_dataset():
    """加载本地数学数据集"""
    dataset_path = "data/processed/local_math.csv"
    
    if not os.path.exists(dataset_path):
        logger.error(f"数据集不存在: {dataset_path}")
        return None
    
    logger.info(f"加载数据集: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # 按难度分组，每个难度取3个样本
    test_samples = []
    for difficulty in ['elementary', 'middle', 'college']:
        difficulty_samples = df[df['difficulty'] == difficulty].head(3)
        test_samples.append(difficulty_samples)
    
    test_df = pd.concat(test_samples, ignore_index=True)
    logger.info(f"测试样本数: {len(test_df)}")
    logger.info(f"难度分布: {test_df['difficulty'].value_counts().to_dict()}")
    
    return test_df

def generate_answer(model, tokenizer, question, max_new_tokens=30):
    """生成答案"""
    try:
        # 构建提示
        prompt = f"Question: {question}\nAnswer:"
        
        # 编码输入
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取答案部分
        if prompt in full_response:
            answer = full_response[len(prompt):].strip()
        else:
            answer = full_response.strip()
        
        return answer
        
    except Exception as e:
        logger.error(f"生成答案失败: {e}")
        return ""

def evaluate_local_dataset():
    """评估本地数据集"""
    logger.info("开始本地数学数据集评估")
    
    # 1. 加载模型
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. 加载数据集
    test_df = load_local_dataset()
    if test_df is None:
        return
    
    # 3. 初始化OpenAI评分器
    try:
        scorer = OpenAIScorer()
        logger.info("OpenAI评分器初始化成功")
    except Exception as e:
        logger.error(f"OpenAI评分器初始化失败: {e}")
        scorer = None
    
    # 4. 评估每个样本
    results = []
    
    for idx, row in test_df.iterrows():
        logger.info(f"处理样本 {idx+1}/{len(test_df)} - 难度: {row['difficulty']}")
        
        question = row['problem']
        correct_answer = row['answer']
        difficulty = row['difficulty']
        
        logger.info(f"问题: {question}")
        logger.info(f"正确答案: {correct_answer}")
        
        # 生成答案
        generated_answer = generate_answer(model, tokenizer, question)
        logger.info(f"生成答案: {generated_answer}")
        
        # OpenAI评分
        openai_score = None
        if scorer:
            try:
                score_result = scorer.score_answer(question, correct_answer, generated_answer)
                openai_score = score_result['openai_score']
                logger.info(f"OpenAI评分: {score_result}")
            except Exception as e:
                logger.error(f"OpenAI评分失败: {e}")
        
        # 保存结果
        result = {
            'id': row['id'],
            'difficulty': difficulty,
            'question': question,
            'correct_answer': correct_answer,
            'generated_answer': generated_answer,
            'openai_score': openai_score
        }
        results.append(result)
        
        logger.info("-" * 50)
    
    # 5. 保存结果
    results_df = pd.DataFrame(results)
    output_path = "results/local_math_evaluation_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"结果已保存到: {output_path}")
    
    # 6. 统计结果
    logger.info("评估结果统计:")
    logger.info(f"总样本数: {len(results)}")
    
    # 按难度统计
    for difficulty in ['elementary', 'middle', 'college']:
        difficulty_results = [r for r in results if r['difficulty'] == difficulty]
        if difficulty_results:
            valid_scores = [r['openai_score'] for r in difficulty_results if r['openai_score'] is not None]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                logger.info(f"{difficulty}难度 - 样本数: {len(difficulty_results)}, 平均分数: {avg_score:.2f}")
    
    # 总体平均分
    if scorer:
        valid_scores = [r['openai_score'] for r in results if r['openai_score'] is not None]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            logger.info(f"总体平均OpenAI评分: {avg_score:.2f}")
    
    return results_df

if __name__ == "__main__":
    evaluate_local_dataset() 