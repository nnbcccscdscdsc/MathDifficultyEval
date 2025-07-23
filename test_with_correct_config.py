#!/usr/bin/env python3
"""
使用正确配置文件的模型测试
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import time
import yaml
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model_config(model_name):
    """加载模型配置文件"""
    config_path = f"configs/models/{model_name}.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        logger.warning(f"配置文件不存在: {config_path}")
        return None

def test_model_with_config(model_name):
    """使用配置文件测试模型"""
    logger.info(f"🧪 测试模型: {model_name}")
    
    # 加载配置
    config = load_model_config(model_name)
    if not config:
        logger.error("无法加载配置文件")
        return ""
    
    logger.info(f"使用配置文件: {config['model']['display_name']}")
    
    try:
        # 1. 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config['model']['name'], 
            trust_remote_code=True, 
            local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 2. 加载模型
        logger.info("加载模型...")
        
        # 获取量化配置
        quant_config = config['quantization']['options'][config['quantization']['default']]
        quantization_config = None
        if quant_config.get('load_in_4bit', False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=quant_config.get('bnb_4bit_quant_type', 'nf4'),
                bnb_4bit_use_double_quant=quant_config.get('bnb_4bit_use_double_quant', True)
            )
        elif quant_config.get('load_in_8bit', False):
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            config['model']['name'],
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map=config['gpu']['device_map'],
            trust_remote_code=config['model_specific']['trust_remote_code'],
            low_cpu_mem_usage=config['model_specific']['low_cpu_mem_usage'],
            local_files_only=True
        )
        
        logger.info("模型加载完成！")
        
        # 3. 测试推理
        test_question = "What is 2 + 2?"
        
        # 使用配置文件中的提示模板
        prompt_template = config.get('prompt_template', '{problem}')
        prompt = prompt_template.format(problem=test_question)
        
        logger.info(f"提示: {prompt}")
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        
        logger.info(f"输入长度: {input_ids.shape[1]}")
        
        logger.info("开始生成...")
        start_time = time.time()
        
        # 使用配置文件中的生成参数
        gen_config = config['generation']
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=gen_config['max_new_tokens'],
                do_sample=gen_config['do_sample'],
                temperature=gen_config['temperature'],
                top_p=gen_config['top_p'],
                top_k=gen_config['top_k'],
                num_beams=gen_config['num_beams'],
                repetition_penalty=gen_config['repetition_penalty'],
                pad_token_id=tokenizer.eos_token_id if gen_config['pad_token_id'] is None else gen_config['pad_token_id'],
                eos_token_id=tokenizer.eos_token_id if gen_config['eos_token_id'] is None else gen_config['eos_token_id']
            )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        logger.info(f"生成完成，耗时: {generation_time:.2f}秒")
        logger.info(f"输出形状: {outputs.shape}")
        
        # 解码
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"完整输出: {full_response}")
        
        # 提取答案
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
    # 测试不同的模型
    models = [
        "mistral-7b-v0.2",
        "longalpaca-70b-16k"
    ]
    
    results = {}
    
    for model_name in models:
        logger.info(f"\n{'='*80}")
        answer = test_model_with_config(model_name)
        results[model_name] = answer
        logger.info(f"{'='*80}")
    
    # 总结结果
    logger.info("\n📊 测试结果总结:")
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