#!/usr/bin/env python3
"""
专门测试Mistral模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_mistral():
    """测试Mistral模型"""
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    
    logger.info(f"🧪 专门测试Mistral模型: {model_name}")
    
    try:
        # 1. 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"EOS token ID: {tokenizer.eos_token_id}")
        logger.info(f"PAD token ID: {tokenizer.pad_token_id}")
        
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
            low_cpu_mem_usage=True
        )
        
        logger.info("模型加载完成！")
        
        # 3. 测试不同的问题
        test_questions = [
            "What is 2 + 2?",
            "What is 5 + 3?",
            "Hello, how are you?"
        ]
        
        for i, question in enumerate(test_questions):
            logger.info(f"\n测试问题 {i+1}: {question}")
            
            # Mistral的正确提示格式
            prompt = f"<s>[INST] {question} [/INST]"
            
            logger.info(f"提示: {prompt}")
            
            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            
            logger.info(f"输入长度: {input_ids.shape[1]}")
            logger.info(f"注意力掩码: {attention_mask}")
            
            logger.info("开始生成...")
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.1
                )
            
            logger.info(f"生成完成，输出形状: {outputs.shape}")
            
            # 解码
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"完整输出: {full_response}")
            
            # 提取答案
            if prompt in full_response:
                answer = full_response[len(prompt):].strip()
            else:
                answer = full_response.strip()
            
            logger.info(f"生成答案: {answer}")
            
            if answer.strip():
                logger.info("✅ 推理成功！")
            else:
                logger.warning("⚠️ 生成了空答案")
            
            logger.info("-" * 50)
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mistral() 