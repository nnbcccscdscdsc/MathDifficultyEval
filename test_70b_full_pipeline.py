#!/usr/bin/env python3
"""
70B模型完整评估流程
使用LongAlpaca-70B-16k模型测试数据集评估和OpenAI评分
"""

import os
import sys
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scripts.openai_scorer import OpenAIScorer
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_70b_model_and_tokenizer():
    """加载70B模型和tokenizer"""
    model_name = "Yukang/LongAlpaca-70B-16k"
    
    logger.info(f"加载70B模型: {model_name}")
    
    # 加载tokenizer（强制使用本地缓存）
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 配置量化
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # 加载模型（强制使用本地缓存）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    
    return model, tokenizer

def load_test_datasets():
    """加载测试数据集"""
    datasets = {}
    
    # 1. 本地数学数据集
    local_math_path = "data/processed/local_math.csv"
    if os.path.exists(local_math_path):
        logger.info(f"加载本地数学数据集: {local_math_path}")
        df = pd.read_csv(local_math_path)
        
        # 按难度分组，每个难度取2个样本
        test_samples = []
        for difficulty in ['elementary', 'middle', 'college']:
            difficulty_samples = df[df['difficulty'] == difficulty].head(2)
            test_samples.append(difficulty_samples)
        
        datasets['local_math'] = pd.concat(test_samples, ignore_index=True)
        logger.info(f"本地数学数据集样本数: {len(datasets['local_math'])}")
    
    # 2. DeepMath数据集
    deepmath_path = "data/processed/deepmath_evaluation_dataset.csv"
    if os.path.exists(deepmath_path):
        logger.info(f"加载DeepMath数据集: {deepmath_path}")
        df = pd.read_csv(deepmath_path)
        datasets['deepmath'] = df.head(3)  # 取前3个样本
        logger.info(f"DeepMath数据集样本数: {len(datasets['deepmath'])}")
    
    return datasets

def generate_answer_70b(model, tokenizer, question, max_new_tokens=50):
    """使用70B模型生成答案"""
    try:
        # 构建提示 - 针对70B模型优化
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
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

def evaluate_with_70b_model():
    """使用70B模型评估数据集"""
    logger.info("开始70B模型完整评估流程")
    
    # 1. 加载模型
    model, tokenizer = load_70b_model_and_tokenizer()
    
    # 2. 加载数据集
    datasets = load_test_datasets()
    if not datasets:
        logger.error("没有找到可用的数据集")
        return
    
    # 3. 初始化OpenAI评分器
    try:
        scorer = OpenAIScorer()
        logger.info("OpenAI评分器初始化成功")
    except Exception as e:
        logger.error(f"OpenAI评分器初始化失败: {e}")
        scorer = None
    
    # 4. 评估每个数据集
    all_results = []
    
    for dataset_name, test_df in datasets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"评估数据集: {dataset_name}")
        logger.info(f"{'='*60}")
        
        for idx, row in test_df.iterrows():
            logger.info(f"处理样本 {idx+1}/{len(test_df)}")
            
            # 获取问题信息
            if dataset_name == 'local_math':
                question = row['problem']
                correct_answer = row['answer']
                difficulty = row['difficulty']
                sample_id = row['id']
            else:  # deepmath
                question = row['problem']
                correct_answer = row['solution']
                difficulty = "advanced"  # DeepMath都是高级问题
                sample_id = row.get('id', f"deepmath_{idx}")
            
            logger.info(f"问题: {question[:100]}...")
            logger.info(f"正确答案: {correct_answer}")
            logger.info(f"难度: {difficulty}")
            
            # 生成答案
            generated_answer = generate_answer_70b(model, tokenizer, question)
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
                'dataset': dataset_name,
                'id': sample_id,
                'difficulty': difficulty,
                'question': question,
                'correct_answer': correct_answer,
                'generated_answer': generated_answer,
                'openai_score': openai_score
            }
            all_results.append(result)
            
            logger.info("-" * 50)
    
    # 5. 保存结果
    results_df = pd.DataFrame(all_results)
    output_path = "results/70b_model_evaluation_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"\n结果已保存到: {output_path}")
    
    # 6. 统计结果
    logger.info("\n" + "="*60)
    logger.info("70B模型评估结果统计")
    logger.info("="*60)
    logger.info(f"总样本数: {len(all_results)}")
    
    # 按数据集统计
    for dataset_name in datasets.keys():
        dataset_results = [r for r in all_results if r['dataset'] == dataset_name]
        if dataset_results:
            valid_scores = [r['openai_score'] for r in dataset_results if r['openai_score'] is not None]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                logger.info(f"{dataset_name}数据集 - 样本数: {len(dataset_results)}, 平均分数: {avg_score:.2f}")
    
    # 按难度统计
    for difficulty in ['elementary', 'middle', 'college', 'advanced']:
        difficulty_results = [r for r in all_results if r['difficulty'] == difficulty]
        if difficulty_results:
            valid_scores = [r['openai_score'] for r in difficulty_results if r['openai_score'] is not None]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                logger.info(f"{difficulty}难度 - 样本数: {len(difficulty_results)}, 平均分数: {avg_score:.2f}")
    
    # 总体平均分
    if scorer:
        valid_scores = [r['openai_score'] for r in all_results if r['openai_score'] is not None]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            logger.info(f"总体平均OpenAI评分: {avg_score:.2f}")
    
    return results_df

if __name__ == "__main__":
    evaluate_with_70b_model() 