#!/usr/bin/env python3
"""
最简单的推理测试
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def simple_test():
    # 使用最可靠的模型
    model_name = "microsoft/DialoGPT-medium"
    
    print(f"🧪 最简单测试: {model_name}")
    
    # 1. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to('cuda')
    
    # 3. 最简单测试
    prompt = "Hello"
    print(f"输入: {prompt}")
    
    # 编码
    inputs = tokenizer.encode(prompt, return_tensors="pt").to('cuda')
    print(f"输入tokens: {inputs}")
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=10,
            do_sample=False,
            num_beams=1
        )
    
    # 解码
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"完整输出: {result}")
    
    # 提取新生成的部分
    if prompt in result:
        answer = result[len(prompt):].strip()
    else:
        answer = result.strip()
    
    print(f"生成答案: {answer}")
    
    return True

if __name__ == "__main__":
    simple_test() 