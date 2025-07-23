#!/usr/bin/env python3
"""
简单稳定的模型测试
使用最基本的生成参数，避免卡住
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def simple_test(model_name):
    """简单测试单个模型"""
    logger.info(f"🧪 简单测试: {model_name}")
    
    try:
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
        test_question = "What is 2 + 2?"
        
        # 使用最简单的提示格式
        if "mistral" in model_name.lower():
            prompt = f"[INST] {test_question} [/INST]"
        elif "longalpaca" in model_name.lower():
            prompt = f"<|im_start|>user\n{test_question}<|im_end|>\n<|im_start|>assistant\n"
        else:
            prompt = f"{test_question}\nAnswer:"
        
        logger.info(f"提示: {prompt}")
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        
        logger.info(f"输入长度: {input_ids.shape[1]}")
        logger.info("开始生成...")
        
        start_time = time.time()
        
        # 使用最基本的生成参数
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,  # 减少生成长度
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        logger.info(f"生成完成，耗时: {generation_time:.2f}秒")
        logger.info(f"输出形状: {outputs.shape}")
        
        # 解码
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"完整输出: {full_response}")
        
        # 简单提取答案
        if prompt in full_response:
            answer = full_response[len(prompt):].strip()
        else:
            answer = full_response.strip()
        
        logger.info(f"提取的答案: '{answer}'")
        
        if answer.strip():
            logger.info("✅ 生成成功！")
        else:
            logger.warning("⚠️ 生成了空答案")
        
        return answer.strip()
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return ""

def main():
    """主函数"""
    # 先测试较小的模型
    models = [
        "microsoft/DialoGPT-medium",  # 小模型，应该很快
        "mistralai/Mistral-7B-Instruct-v0.2"  # 7B模型
    ]
    
    results = {}
    
    for model_name in models:
        logger.info(f"\n{'='*60}")
        answer = simple_test(model_name)
        results[model_name] = answer
        logger.info(f"{'='*60}")
        
        # 如果第一个模型成功，再测试70B
        if model_name == "microsoft/DialoGPT-medium" and answer.strip():
            logger.info("小模型测试成功，继续测试70B模型...")
            answer_70b = simple_test("Yukang/LongAlpaca-70B-16k")
            results["Yukang/LongAlpaca-70B-16k"] = answer_70b
            break
    
    # 总结结果
    logger.info("\n📊 测试结果总结:")
    for model_name, answer in results.items():
        status = "✅ 成功" if answer.strip() else "❌ 失败"
        logger.info(f"{model_name}: {status} - '{answer}'")

if __name__ == "__main__":
    main() 