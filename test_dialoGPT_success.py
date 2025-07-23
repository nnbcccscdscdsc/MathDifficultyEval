#!/usr/bin/env python3
"""
测试DialoGPT-medium为什么能成功
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import time

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_dialoGPT():
    """测试DialoGPT-medium"""
    model_name = "microsoft/DialoGPT-medium"
    
    logger.info(f"🧪 测试DialoGPT-medium: {model_name}")
    
    try:
        # 1. 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"EOS token ID: {tokenizer.eos_token_id}")
        logger.info(f"PAD token ID: {tokenizer.pad_token_id}")
        
        # 2. 加载模型 - 不使用量化
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
        prompt = f"{test_question}\nAnswer:"
        
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
            logger.info("✅ DialoGPT生成成功！")
        else:
            logger.warning("⚠️ DialoGPT生成了空答案")
        
        return answer.strip()
        
    except Exception as e:
        logger.error(f"❌ DialoGPT测试失败: {e}")
        import traceback
        traceback.print_exc()
        return ""

def test_70b_no_quantization():
    """测试70B模型（不使用量化）"""
    model_name = "Yukang/LongAlpaca-70B-16k"
    
    logger.info(f"🧪 测试70B模型（不使用量化）: {model_name}")
    
    try:
        # 1. 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"EOS token ID: {tokenizer.eos_token_id}")
        logger.info(f"PAD token ID: {tokenizer.pad_token_id}")
        
        # 2. 加载模型 - 不使用量化
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
        prompt = f"<|im_start|>user\n{test_question}<|im_end|>\n<|im_start|>assistant\n"
        
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
        if "<|im_start|>assistant" in full_response:
            answer = full_response.split("<|im_start|>assistant")[-1].strip()
        else:
            answer = full_response.strip()
        
        logger.info(f"提取的答案: '{answer}'")
        
        if answer.strip():
            logger.info("✅ 70B模型生成成功！")
        else:
            logger.warning("⚠️ 70B模型生成了空答案")
        
        return answer.strip()
        
    except Exception as e:
        logger.error(f"❌ 70B模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return ""

def main():
    """主函数"""
    logger.info("🔍 对比测试：为什么DialoGPT能成功，70B不行？")
    
    # 1. 测试DialoGPT
    logger.info(f"\n{'='*60}")
    dialoGPT_answer = test_dialoGPT()
    logger.info(f"{'='*60}")
    
    # 2. 如果DialoGPT成功，测试70B（不使用量化）
    if dialoGPT_answer.strip():
        logger.info("DialoGPT成功，现在测试70B模型（不使用量化）...")
        logger.info(f"\n{'='*60}")
        answer_70b = test_70b_no_quantization()
        logger.info(f"{'='*60}")
        
        # 总结
        logger.info("\n📊 对比结果:")
        logger.info(f"DialoGPT-medium: ✅ 成功 - '{dialoGPT_answer}'")
        logger.info(f"70B模型（无量化）: {'✅ 成功' if answer_70b.strip() else '❌ 失败'} - '{answer_70b}'")
    else:
        logger.error("DialoGPT也失败了，需要检查环境")

if __name__ == "__main__":
    main() 