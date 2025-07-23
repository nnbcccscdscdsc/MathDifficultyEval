#!/usr/bin/env python3
"""
对比不同的Mistral模型版本
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mistral_model(model_name):
    """测试单个Mistral模型"""
    logger.info(f"🧪 测试模型: {model_name}")
    
    try:
        # 1. 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 2. 加载模型
        logger.info("加载模型...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        
        logger.info("模型加载完成！")
        
        # 3. 测试推理
        test_question = "What is 2 + 2?"
        
        # 尝试不同的提示格式
        test_prompts = [
            {
                "name": "简单格式",
                "prompt": f"{test_question}\nAnswer:"
            },
            {
                "name": "Mistral格式",
                "prompt": f"[INST] {test_question} [/INST]"
            },
            {
                "name": "详细格式",
                "prompt": f"[INST] Please solve this math problem: {test_question} [/INST]"
            }
        ]
        
        best_answer = ""
        best_prompt = ""
        
        for test_case in test_prompts:
            prompt = test_case['prompt']
            logger.info(f"测试提示: {test_case['name']}")
            
            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(model.device)
            
            logger.info("开始生成...")
            start_time = time.time()
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=20,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # 解码
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取答案
            if prompt in full_response:
                answer = full_response[len(prompt):].strip()
            else:
                answer = full_response.strip()
            
            logger.info(f"答案: '{answer}' (耗时: {generation_time:.2f}秒)")
            
            # 检查答案质量
            if "4" in answer or "four" in answer.lower():
                best_answer = answer
                best_prompt = test_case['name']
                logger.info("✅ 找到正确答案！")
                break
            elif answer.strip() and not best_answer:
                best_answer = answer
                best_prompt = test_case['name']
        
        logger.info(f"最佳答案: '{best_answer}' (使用提示: {best_prompt})")
        return best_answer.strip()
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        return ""

def main():
    """主函数"""
    # 测试不同的Mistral模型
    models = [
        "mistral-community/Mistral-7B-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.2"
    ]
    
    results = {}
    
    for model_name in models:
        logger.info(f"\n{'='*80}")
        answer = test_mistral_model(model_name)
        results[model_name] = answer
        logger.info(f"{'='*80}")
    
    # 总结结果
    logger.info("\n📊 对比结果总结:")
    for model_name, answer in results.items():
        if "4" in answer or "four" in answer.lower():
            status = "✅ 正确答案"
        elif answer.strip():
            status = "⚠️ 有答案但不正确"
        else:
            status = "❌ 无答案"
        logger.info(f"{model_name}: {status} - '{answer}'")

if __name__ == "__main__":
    main() 