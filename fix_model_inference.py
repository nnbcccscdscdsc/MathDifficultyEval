#!/usr/bin/env python3
"""
修复模型推理问题
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_fixed(model_name):
    """测试单个模型（修复版本）"""
    logger.info(f"🧪 测试模型: {model_name}")
    
    try:
        # 1. 加载tokenizer
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
        
        # 3. 测试推理
        test_question = "What is 2 + 2?"
        
        # 根据模型类型选择正确的提示格式
        if "mistral" in model_name.lower():
            # Mistral格式 - 不需要<s>标签
            prompt = f"[INST] {test_question} [/INST]"
        elif "llama" in model_name.lower():
            # Llama格式
            prompt = f"[INST] {test_question} [/INST]"
        elif "longalpaca" in model_name.lower():
            # LongAlpaca格式
            prompt = f"<|im_start|>user\n{test_question}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # 默认格式
            prompt = f"{test_question}\nAnswer:"
        
        logger.info(f"提示: {prompt}")
        
        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(model.device)
        attention_mask = inputs.attention_mask.to(model.device)
        
        logger.info(f"输入长度: {input_ids.shape[1]}")
        
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
                temperature=0.1,
                repetition_penalty=1.1
            )
        
        logger.info(f"生成完成，输出形状: {outputs.shape}")
        
        # 解码完整输出
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"完整输出: {full_response}")
        
        # 修复答案提取逻辑
        if "mistral" in model_name.lower() or "llama" in model_name.lower():
            # 对于Mistral/Llama，查找[/INST]后的内容
            if "[/INST]" in full_response:
                answer = full_response.split("[/INST]")[-1].strip()
            else:
                answer = full_response.strip()
        elif "longalpaca" in model_name.lower():
            # 对于LongAlpaca，查找assistant后的内容
            if "<|im_start|>assistant" in full_response:
                answer = full_response.split("<|im_start|>assistant")[-1].strip()
            else:
                answer = full_response.strip()
        else:
            # 默认提取
            if prompt in full_response:
                answer = full_response[len(prompt):].strip()
            else:
                answer = full_response.strip()
        
        logger.info(f"提取的答案: '{answer}'")
        
        if answer.strip() and answer.strip() != prompt.strip():
            logger.info("✅ 推理成功！")
        else:
            logger.warning("⚠️ 生成了空答案或重复提示")
        
        return answer.strip()
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return ""

def main():
    """主函数"""
    models = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "Yukang/LongAlpaca-70B-16k"
    ]
    
    results = {}
    
    for model_name in models:
        logger.info(f"\n{'='*80}")
        answer = test_model_fixed(model_name)
        results[model_name] = answer
        logger.info(f"{'='*80}")
    
    # 总结结果
    logger.info("\n📊 测试结果总结:")
    for model_name, answer in results.items():
        status = "✅ 成功" if answer and answer != "What is 2 + 2?" else "❌ 失败"
        logger.info(f"{model_name}: {status} - '{answer}'")

if __name__ == "__main__":
    main() 