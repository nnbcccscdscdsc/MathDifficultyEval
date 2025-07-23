#!/usr/bin/env python3
"""
专门测试mistral-community/Mistral-7B-v0.2模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mistral_community():
    """测试mistral-community/Mistral-7B-v0.2模型"""
    model_name = "mistral-community/Mistral-7B-v0.2"
    
    logger.info(f"🧪 测试mistral-community模型: {model_name}")
    
    try:
        # 1. 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"EOS token ID: {tokenizer.eos_token_id}")
        logger.info(f"PAD token ID: {tokenizer.pad_token_id}")
        
        # 2. 加载模型 - 先尝试不使用量化
        logger.info("加载模型（不使用量化）...")
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
                "name": "带<s>标签",
                "prompt": f"<s>[INST] {test_question} [/INST]"
            }
        ]
        
        for i, test_case in enumerate(test_prompts):
            logger.info(f"\n{'='*50}")
            logger.info(f"测试提示格式 {i+1}: {test_case['name']}")
            logger.info(f"{'='*50}")
            
            prompt = test_case['prompt']
            logger.info(f"提示: {prompt}")
            
            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(model.device)
            
            logger.info(f"输入长度: {input_ids.shape[1]}")
            logger.info(f"输入tokens: {input_ids}")
            
            logger.info("开始生成...")
            start_time = time.time()
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            logger.info(f"生成完成，耗时: {generation_time:.2f}秒")
            logger.info(f"输出形状: {outputs.shape}")
            logger.info(f"输出tokens: {outputs}")
            
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
                return answer.strip()
            else:
                logger.warning("⚠️ 生成了空答案")
            
            logger.info("-" * 30)
        
        logger.error("❌ 所有提示格式都失败了")
        return ""
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return ""

def test_mistral_community_with_quantization():
    """测试mistral-community模型（使用量化）"""
    model_name = "mistral-community/Mistral-7B-v0.2"
    
    logger.info(f"🧪 测试mistral-community模型（使用量化）: {model_name}")
    
    try:
        # 1. 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 2. 加载模型 - 使用量化
        logger.info("加载模型（使用4bit量化）...")
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
        
        # 3. 测试推理
        test_question = "What is 2 + 2?"
        prompt = f"[INST] {test_question} [/INST]"
        
        logger.info(f"提示: {prompt}")
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(model.device)
        
        logger.info(f"输入长度: {input_ids.shape[1]}")
        
        logger.info("开始生成...")
        start_time = time.time()
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=10,
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
        
        # 提取答案
        if "[/INST]" in full_response:
            answer = full_response.split("[/INST]")[-1].strip()
        else:
            answer = full_response.strip()
        
        logger.info(f"提取的答案: '{answer}'")
        
        if answer.strip():
            logger.info("✅ 量化模型生成成功！")
        else:
            logger.warning("⚠️ 量化模型生成了空答案")
        
        return answer.strip()
        
    except Exception as e:
        logger.error(f"❌ 量化模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return ""

def main():
    """主函数"""
    logger.info("🔍 测试mistral-community/Mistral-7B-v0.2模型")
    
    # 1. 先测试不使用量化
    logger.info(f"\n{'='*60}")
    answer_no_quant = test_mistral_community()
    logger.info(f"{'='*60}")
    
    # 2. 如果失败，测试使用量化
    if not answer_no_quant.strip():
        logger.info("无量化版本失败，尝试量化版本...")
        logger.info(f"\n{'='*60}")
        answer_quant = test_mistral_community_with_quantization()
        logger.info(f"{'='*60}")
        
        # 总结
        logger.info("\n📊 测试结果总结:")
        logger.info(f"无量化版本: {'✅ 成功' if answer_no_quant.strip() else '❌ 失败'} - '{answer_no_quant}'")
        logger.info(f"量化版本: {'✅ 成功' if answer_quant.strip() else '❌ 失败'} - '{answer_quant}'")
    else:
        logger.info("\n📊 测试结果总结:")
        logger.info(f"无量化版本: ✅ 成功 - '{answer_no_quant}'")

if __name__ == "__main__":
    main() 