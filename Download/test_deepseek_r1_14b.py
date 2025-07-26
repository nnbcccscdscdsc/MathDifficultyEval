#!/usr/bin/env python3
"""
测试DeepSeek-R1-Distill-Qwen-14B数学模型
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_deepseek_r1_14b():
    """测试DeepSeek-R1-Distill-Qwen-14B模型"""
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    
    logger.info(f"🧮 测试DeepSeek-R1-Distill-Qwen-14B数学模型: {model_name}")
    
    try:
        # 1. 加载tokenizer
        logger.info("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 2. 加载模型 - 使用4bit量化节省内存
        logger.info("加载模型（4bit量化）...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        logger.info("模型加载完成！")
        
        # 3. 测试数学问题
        test_questions = [
            "What is 2 + 2?",
            "Calculate: 15 * 14 = ?",
            "What is the square root of 16?",
            "Solve for x: x + 5 = 12",
            "What is 3 to the power of 4?",
            "Find the area of a circle with radius 5",
            "What is the derivative of x^2?",
            "Solve the quadratic equation: x^2 + 5x + 6 = 0"
        ]
        
        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n--- 测试问题 {i}: {question} ---")
            
            # DeepSeek-R1推荐的提示格式
            # 注意：不要添加system prompt，所有指令都在user prompt中
            prompt = f"<think>\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n</think>\n\n{question}"
            
            logger.info(f"提示: {prompt}")
            
            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs.input_ids.to(model.device)
            attention_mask = inputs.attention_mask.to(model.device)
            
            logger.info(f"输入长度: {input_ids.shape[1]}")
            
            logger.info("开始生成...")
            
            # 生成 - 使用推荐的参数
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=400,  # 14B模型可以生成更长内容
                    do_sample=True,
                    temperature=0.6,  # 推荐温度
                    top_p=0.95,       # 推荐top_p
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            logger.info(f"生成完成，输出形状: {outputs.shape}")
            
            # 解码
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"完整输出: {full_response}")
            
            # 提取答案 - 查找boxed内容
            if "\\boxed{" in full_response:
                start = full_response.find("\\boxed{") + 14
                end = full_response.find("}", start)
                if end != -1:
                    answer = full_response[start:end]
                else:
                    answer = full_response[len(prompt):].strip()
            else:
                answer = full_response[len(prompt):].strip()
            
            logger.info(f"提取的答案: '{answer}'")
            
            if answer.strip():
                logger.info("✅ 生成成功！")
            else:
                logger.warning("⚠️ 没有生成有效答案")
        
        return "测试完成"
        
    except Exception as e:
        logger.error(f"❌ DeepSeek-R1-14B测试失败: {e}")
        import traceback
        traceback.print_exc()
        return ""

if __name__ == "__main__":
    test_deepseek_r1_14b() 