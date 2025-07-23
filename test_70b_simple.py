#!/usr/bin/env python3
"""
简单70B模型测试
只测试基本推理功能，不进行完整评估
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_70b_simple():
    """简单测试70B模型"""
    model_name = "Yukang/LongAlpaca-70B-16k"
    
    logger.info(f"开始简单测试70B模型: {model_name}")
    
    # 1. 加载tokenizer
    logger.info("加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载模型
    logger.info("加载模型...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True
    )
    
    logger.info("模型加载完成！")
    
    # 3. 简单测试
    test_questions = [
        "What is 2 + 2?",
        "What is 5 + 3?",
        "Hello, how are you?"
    ]
    
    for i, question in enumerate(test_questions):
        logger.info(f"\n测试问题 {i+1}: {question}")
        
        try:
            # 构建提示
            prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            
            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            
            logger.info(f"输入长度: {input_ids.shape[1]}")
            logger.info("开始生成...")
            
            # 生成（使用较短的生成长度）
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=20,  # 减少生成长度
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 解码
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取答案
            if prompt in full_response:
                answer = full_response[len(prompt):].strip()
            else:
                answer = full_response.strip()
            
            logger.info(f"生成答案: {answer}")
            logger.info(f"完整输出: {full_response}")
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
        
        logger.info("-" * 50)
    
    logger.info("测试完成！")

if __name__ == "__main__":
    test_70b_simple() 