#!/usr/bin/env python3
"""
调试70B模型生成问题
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_70b_model():
    """调试70B模型"""
    model_name = "Yukang/LongAlpaca-70B-16k"
    
    logger.info(f"🔍 调试70B模型: {model_name}")
    
    try:
        # 1. 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
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
            low_cpu_mem_usage=True,
            local_files_only=True
        )
        
        logger.info("模型加载完成！")
        
        # 3. 测试不同的问题和提示格式
        test_cases = [
            {
                "name": "简单加法",
                "question": "What is 2 + 2?",
                "prompt": "What is 2 + 2?\nAnswer:"
            },
            {
                "name": "LongAlpaca格式",
                "question": "What is 2 + 2?",
                "prompt": "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n"
            },
            {
                "name": "简单提示",
                "question": "What is 2 + 2?",
                "prompt": "2 + 2 ="
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"\n{'='*60}")
            logger.info(f"测试案例 {i+1}: {test_case['name']}")
            logger.info(f"{'='*60}")
            
            prompt = test_case['prompt']
            logger.info(f"提示: {prompt}")
            
            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            
            logger.info(f"输入长度: {input_ids.shape[1]}")
            logger.info(f"输入tokens: {input_ids}")
            logger.info(f"注意力掩码: {attention_mask}")
            
            logger.info("开始生成...")
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=20,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    temperature=0.1
                )
            
            logger.info(f"生成完成，输出形状: {outputs.shape}")
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
            logger.info(f"答案长度: {len(answer)}")
            
            if answer.strip():
                logger.info("✅ 生成了答案！")
            else:
                logger.warning("⚠️ 生成了空答案")
            
            logger.info("-" * 50)
        
    except Exception as e:
        logger.error(f"❌ 调试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_70b_model() 